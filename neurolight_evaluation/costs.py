import numpy as np
from scipy.spatial import cKDTree
import rtree
import networkx as nx

import itertools
from typing import List, Hashable, Tuple
import logging

Node = Hashable
Edge = Tuple[Node, Node]

logger = logging.getLogger(__file__)


def get_costs(
    graph: nx.Graph,
    tree: nx.DiGraph,
    location_attr: str,
    penalty_attr: str,
    match_threshold: float,
) -> Tuple[List[Tuple[Node, Node, float]], List[Tuple[Edge, Edge, float]]]:

    graph_nodes = list(graph.nodes)
    graph_edge_list = list(graph.edges)
    tree_nodes = list(tree.nodes)
    tree_edge_list = list(tree.edges)

    # map from node to index. Necessary to vectorize edge operations
    graph_index_map = {u: i for i, u in enumerate(graph_nodes)}
    tree_index_map = {u: i for i, u in enumerate(tree_nodes + [None])}

    # loop over edges to get array of u, v. map u, v to index of u, v in above vectors
    tree_edges = np.array(
        [(tree_index_map[u], tree_index_map[v]) for u, v in tree.edges], dtype=int
    )
    graph_edges = np.array(
        [(graph_index_map[u], graph_index_map[v]) for u, v in graph.edges], dtype=int
    )

    # loop over graph nodes to extract locations
    graph_locations = np.array([graph.nodes[x][location_attr] for x in graph_nodes])

    # loop over graph nodes to extract penalties
    graph_penalties = np.array(
        [graph.nodes[x].get(penalty_attr, 0) for x in graph_nodes]
    )
    graph_edge_penalties = np.array(
        [graph.edges[x].get(penalty_attr, 0) for x in graph_edge_list]
    )

    # loop over tree nodes to extract locations
    # "None" node must have a placeholder location
    tree_locations = np.array(
        [tree.nodes[x][location_attr] for x in tree_nodes] + [(0, 0, 0)]
    )

    # loop over tree nodes to extract penalties
    # "None" node must have a placeholder penalty
    tree_penalties = np.array(
        [tree.nodes[x].get(penalty_attr, 0) for x in tree_nodes] + [0]
    )
    tree_edge_penalties = np.array(
        [tree.edges[x].get(penalty_attr, 0) for x in tree_edge_list]
    )

    # create array of tree_nodes for np.find calls
    tree_nodes = np.array(tree_nodes + [None])

    # initialize kdtrees
    graph_kd = cKDTree(graph_locations)
    tree_kd = cKDTree(tree_locations)

    # get (u, v) index pairs from tree_kd and graph_kd
    # u in tree and v in graph
    close_enough = tree_kd.query_ball_tree(graph_kd, match_threshold)
    # loop over query_ball_tree result
    index_pairs = np.array(
        [(i, x) for i, y in enumerate(close_enough) for x in y]
        + [(-1, i) for i in range(len(graph_nodes))]  # -1 is index of None tree node
    )

    pairs_t = np.take(tree_nodes, index_pairs[:, 0])
    pairs_g = np.take(graph_nodes, index_pairs[:, 1])
    pairs_t_pens = np.take(tree_penalties, index_pairs[:, 0], axis=0)
    pairs_g_pens = np.take(graph_penalties, index_pairs[:, 1], axis=0)

    no_match = index_pairs[:, 0] == -1

    node_costs = (pairs_t_pens + pairs_g_pens) * (1 - no_match)
    node_matchings = np.stack([pairs_g, pairs_t, node_costs], axis=1)

    # Edge matchings

    # What would be faster: rtree containing many small edges and making few large queries
    # or rtree containing few large edges and making many small queries?
    tree_rtree = initialize_rtree(tree_edges, tree_locations)

    possible_edge_matchings = query_rtree(
        tree_rtree, graph_edges, graph_locations, match_threshold
    )
    if len(possible_edge_matchings) < 1:
        return (node_matchings, [])
    tree_e_indices = np.take(tree_edges, possible_edge_matchings[:, 1], axis=0)
    t_e_locs = np.take(tree_locations, tree_e_indices, axis=0)
    graph_e_indices = np.take(graph_edges, possible_edge_matchings[:, 0], axis=0)
    g_e_locs = np.take(graph_locations, graph_e_indices, axis=0)
    t_e_pens = np.take(tree_edge_penalties, possible_edge_matchings[:, 1], axis=0)
    g_e_pens = np.take(graph_edge_penalties, possible_edge_matchings[:, 0], axis=0)

    distances, lengths = edge_dist(g_e_locs, t_e_locs)

    costs = t_e_pens + g_e_pens

    filtered_matchings = possible_edge_matchings[distances < match_threshold]

    filtered_tree_es = np.take(tree_edges, filtered_matchings[:, 1], axis=0)
    tree_us = np.take(tree_nodes, filtered_tree_es[:, 0])
    tree_vs = np.take(tree_nodes, filtered_tree_es[:, 1])
    tree_as = np.concatenate([tree_us, tree_us])
    tree_bs = np.concatenate([tree_vs, tree_vs])
    filtered_graph_es = np.take(graph_edges, filtered_matchings[:, 0], axis=0)
    graph_us = np.take(graph_nodes, filtered_graph_es[:, 0])
    graph_vs = np.take(graph_nodes, filtered_graph_es[:, 1])
    graph_as = np.concatenate([graph_us, graph_vs])
    graph_bs = np.concatenate([graph_vs, graph_us])
    filtered_costs = costs[distances < match_threshold]
    costs = np.concatenate([filtered_costs, filtered_costs])

    edge_matchings = np.stack([graph_as, graph_bs, tree_as, tree_bs, costs], axis=1)

    return node_matchings, edge_matchings


def initialize_rtree(edges, locs):
    p = rtree.index.Property()
    p.dimension = 3
    tree_rtree = rtree.index.Index(properties=p)
    for i, (u, v) in enumerate(edges):
        u_loc = locs[u]
        v_loc = locs[v]
        mins = np.min(np.array([u_loc, v_loc]), axis=0)
        maxs = np.max(np.array([u_loc, v_loc]), axis=0)
        box = tuple(x for x in itertools.chain(mins.tolist(), maxs.tolist()))
        tree_rtree.insert(i, box)

    return tree_rtree


def query_rtree(rtree, edges, locs, radius):

    rects = []
    for u, v in edges:
        s = np.stack([locs[u, :], locs[v, :]])
        lower = np.min(s, axis=0) - radius
        upper = np.max(s, axis=0) + radius
        rects.append(tuple(np.concatenate([lower, upper])))
    possible_tree_edges = [rtree.intersection(rect) for rect in rects]
    # line distances
    possible_matchings = np.array(
        [(i, j) for i, js in enumerate(possible_tree_edges) for j in js]
    )
    return possible_matchings


def edge_dist(a_locs: np.ndarray, b_locs: np.ndarray) -> float:
    """
    psuedo avg distance of a line to another line.
    calculated as the average distance of the end points of a to the line b.
    """
    distance = (
        point_to_edge_dist(a_locs[:, 0], b_locs[:, 0], b_locs[:, 1])
        + point_to_edge_dist(a_locs[:, 1], b_locs[:, 0], b_locs[:, 1])
    ) / 2
    lengths = np.linalg.norm(a_locs[:, 0] - a_locs[:, 1], axis=1)
    return distance, lengths


def point_to_edge_dist(
    centers: np.ndarray, u_locs: np.ndarray, v_locs: np.ndarray
) -> float:
    slope = v_locs - u_locs
    edge_mag = np.linalg.norm(slope, axis=1)
    zero_mag = np.isclose(edge_mag, 0)
    frac = np.clip(
        np.sum((centers - u_locs) * slope, axis=1) / np.sum(slope * slope, axis=1), 0, 1
    )
    frac = np.where(zero_mag, 0, frac)
    min_dist = np.linalg.norm((frac * slope.T).T + u_locs - centers, axis=1)
    return min_dist
