import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import rtree

import itertools
from typing import Tuple, List


def get_edges_xy(
    x: nx.Graph, y: nx.Graph, location_attr: str, node_match_threshold: float,
) -> List[Tuple[int, int]]:

    # setup necessary vectors:
    x_nodes = list(x.nodes)
    y_nodes = list(y.nodes)

    if len(x_nodes) < 1 or len(y_nodes) < 1:
        return []

    # map from node to index. Necessary to vectorize edge operations
    x_index_map = {u: i for i, u in enumerate(x_nodes)}
    y_index_map = {u: i for i, u in enumerate(y_nodes)}

    # get edge vectors
    x_edges = np.array(
        [(x_index_map[u], x_index_map[v]) for u, v in x.edges], dtype=int
    )
    y_edges = np.array(
        [(y_index_map[u], y_index_map[v]) for u, v in y.edges], dtype=int
    )

    # get node location vectors
    x_locations = np.array([x.nodes[node][location_attr] for node in x_nodes])
    y_locations = np.array([y.nodes[node][location_attr] for node in y_nodes])

    # initialize kdtrees
    x_kdtree = cKDTree(x_locations)
    y_kdtree = cKDTree(y_locations)

    # get (u, v) index pairs from y_kdtree and x_kdtree
    close_enough = x_kdtree.query_ball_tree(y_kdtree, node_match_threshold)
    index_pairs = np.array([(i, y) for i, y_nodes in enumerate(close_enough) for y in y_nodes])
    if len(index_pairs) < 1 or len(index_pairs.shape) < 2:
        node_matchings = np.ndarray([0, 2], dtype=np.int64)
    else:
        pairs_x = np.take(x_nodes, index_pairs[:, 0])
        pairs_y = np.take(y_nodes, index_pairs[:, 1])

        node_matchings = np.stack([pairs_x, pairs_y], axis=1)

    # get all nodes close enough to an edge
    x2y_edge_matchings = get_edge_matchings(
        x_edges, x_locations, y_locations, node_match_threshold
    )
    y2x_edge_matchings = get_edge_matchings(
        y_edges, y_locations, x_locations, node_match_threshold
    )

    edge_matchings = np.stack(
        [
            np.concatenate(
                [
                    np.take(x_nodes, x2y_edge_matchings[:, 0]),
                    np.take(x_nodes, y2x_edge_matchings[:, 1]),
                ],
            ),
            np.concatenate(
                [
                    np.take(y_nodes, x2y_edge_matchings[:, 1]),
                    np.take(y_nodes, y2x_edge_matchings[:, 0]),
                ],
            ),
        ],
        axis=1,
    )

    possible_matchings = np.concatenate([node_matchings, edge_matchings])

    if possible_matchings.shape[0] == 0:
        return []
    else:
        return [(a, b) for a, b in np.unique(possible_matchings, axis=0)]


def get_edge_matchings(edges, locations, query_locations, match_threshold):
    
    rtree = initialize_rtree(edges, locations)
    candidate_edge_matchings = query_rtree_points(
        rtree, query_locations, match_threshold
    )
    if len(candidate_edge_matchings) < 1:
        return np.ndarray([0, 2], dtype=np.int64)

    candidate_es = np.take(edges, candidate_edge_matchings[:, 1], axis=0)
    candidate_e_locs = np.take(locations, candidate_es, axis=0)
    candidate_queries = candidate_edge_matchings[:, 0]
    candidate_query_locs = np.take(query_locations, candidate_queries, axis=0)

    distances = point_to_edge_dist(
        candidate_query_locs, candidate_e_locs[:, 0], candidate_e_locs[:, 1]
    )

    filtered_matchings = candidate_edge_matchings[distances < match_threshold]

    filtered_candidate_es = np.take(edges, filtered_matchings[:, 1], axis=0)
    candidate_e_locs = np.take(locations, filtered_candidate_es, axis=0)
    candidate_query_locs = np.expand_dims(
        np.take(query_locations, filtered_matchings[:, 0], axis=0), axis=1
    )
    end_distances = np.linalg.norm(candidate_e_locs - candidate_query_locs, axis=2)
    end_points = np.equal(end_distances, np.min(end_distances, axis=1, keepdims=True))
    num_end_points = end_points.sum(axis=1)
    equal_ends = num_end_points == 2
    end_points[equal_ends] = np.array([True, False])
    if end_points.shape[0] == 0:
        return np.ndarray([0, 2], dtype=np.int64)
    assert max(end_points.sum(axis=1)) == 1, f"{max(end_points.sum(axis=1))}"
    candidate_indices = filtered_candidate_es[end_points]
    query_indices = filtered_matchings[:, 0]

    edge_matchings = np.stack([candidate_indices, query_indices], axis=1)
    return edge_matchings


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


def query_rtree_points(rtree, locs, radius):

    rects = []
    for loc in locs:
        lower = loc - radius
        upper = loc + radius
        rects.append(tuple(np.concatenate([lower, upper])))
    possible_tree_edges = [rtree.intersection(rect) for rect in rects]
    # node i in locs will match to edge j in y
    possible_matchings = np.array(
        [(i, j) for i, js in enumerate(possible_tree_edges) for j in js]
    )
    return possible_matchings


def initialize_kdtrees(x: nx.Graph, y: nx.DiGraph, location_attr: str):
    tree_kd_ids, tree_node_attrs = [list(x) for x in zip(*y.nodes.items())]
    y_kdtree = cKDTree([attrs[location_attr] for attrs in tree_node_attrs])

    graph_kd_ids, graph_node_attrs = [list(x) for x in zip(*x.nodes.items())]
    x_kdtree = cKDTree([attrs[location_attr] for attrs in graph_node_attrs])

    return x_kdtree, graph_kd_ids, y_kdtree, tree_kd_ids


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
