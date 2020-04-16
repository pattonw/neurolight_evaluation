import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

import logging
import itertools

logger = logging.getLogger(__file__)


def fallback_node_penalty():
    return 1


def fallback_edge_penalty():
    return 1


def crossing_edge_penalty():
    return 0.5


def add_fallback(
    graph: nx.DiGraph(),
    fallback: nx.DiGraph(),
    node_offset: int,
    match_threshold: float,
    penalty_attr: str = "penalty",
    location_attr: str = "location",
):
    """
    In the case you are matching a graph G to a tree T, it
    may be the case that G does not contain a subgraph isomorphic
    to G. If you want to prevent failure, you can augment G with T,
    so that there is always a solution, just matching T to T.
    However with sufficient penalties assigned to this matching,
    we can make matching T to T, a last resort, that will only be
    used if matching G to T is impossible.

    T's node id's will be shifted up by node_offset to avoid id conflicts
    """

    fallback_nodes = [n for n in fallback.nodes]
    fallback_kdtree = cKDTree(
        [np.array(fallback.nodes[x][location_attr]) for x in fallback_nodes]
    )

    graph_nodes = [n for n in graph.nodes]
    graph_kdtree = cKDTree(
        [np.array(graph.nodes[x][location_attr]) for x in graph_nodes]
    )

    for node, node_attrs in fallback.nodes.items():
        u = int(node + node_offset)
        graph.add_node(u, **node_attrs)
        graph.nodes[u][penalty_attr] = fallback_node_penalty()

    for u, v in fallback.edges:
        u = int(u + node_offset)
        v = int(v + node_offset)

        graph.add_edge(u, v, **{penalty_attr: fallback_edge_penalty()})

    crossing_edges = fallback_kdtree.query_ball_tree(graph_kdtree, match_threshold)
    criss_crossing_edges = []
    for f_node_index, g_node_indices in enumerate(crossing_edges):
        f_node = fallback_nodes[f_node_index] + node_offset
        f_node_loc = graph.nodes[f_node][location_attr]
        g_nodes = []
        dist = float("inf")
        for g_node_index in g_node_indices:
            candidate = graph_nodes[g_node_index]
            candidate_loc = graph.nodes[candidate][location_attr]
            candidate_dist = np.linalg.norm(candidate_loc - f_node_loc)
            if np.isclose(candidate_dist, dist):
                g_nodes.append(candidate_dist)
            elif candidate_dist < dist:
                g_nodes = [candidate]
                dist = candidate_dist

        for g_node in g_nodes:
            for g_neighbor in neighbors(graph, g_node):
                criss_crossing_edges.append(
                    (f_node, g_neighbor, {penalty_attr: fallback_edge_penalty()})
                )
                criss_crossing_edges.append(
                    (g_neighbor, f_node, {penalty_attr: fallback_edge_penalty()})
                )
            for f_neighbor in neighbors(graph, f_node):
                criss_crossing_edges.append(
                    (g_node, f_neighbor, {penalty_attr: fallback_edge_penalty()})
                )
                criss_crossing_edges.append(
                    (f_neighbor, g_node, {penalty_attr: fallback_edge_penalty()})
                )

            criss_crossing_edges.append((f_node, g_node, {penalty_attr: crossing_edge_penalty()}))
            criss_crossing_edges.append((g_node, f_node, {penalty_attr: crossing_edge_penalty()}))
    
    for u, v, attrs in criss_crossing_edges:
        graph.add_edge(u, v, **attrs)

    return graph


def neighbors(g, n):
    if isinstance(g, nx.DiGraph):
        return itertools.chain(g.predecessors(n), g.successors(n))
    elif isinstance(g, nx.Graph):
        return g.neighbors(n)


def preprocess(graph):
    """
    Take every branch node, b, and split it into d nodes
    where d = degree(b). the nodes b_i create the complete
    graph K_d, and each b_i has 1 aditional edge to a
    neighbor of b.

    This is to help aleviate the following issue: Consider
    ground truth G:
       d
    a--b--c
       e
    and predicted graph T:
    x--y--z
    Then adding G as a fallback to T gives y 3 adjacent edges,
    (xy, yz, yb), however to match to b, a node needs degree at
    least 4. Thus we rely on the fallback, regardless of how close
    y is. This means also that only one of the edges(xy or yz) can
    successfully match to (ab or bc) respectively, simply due to
    the number of edges between b and y being 1.
    """

    max_id = max([node for node in graph.nodes])
    next_id = itertools.count(max_id + 1)
    for node, attrs in list(graph.nodes.items()):
        if graph.degree(node) > 2:
            preds = graph.pred[node]
            succs = graph.succ[node]

            created_nodes = {}
            for p, s in itertools.product(preds.keys(), succs.keys()):
                if p != s:
                    if (p, node) not in created_nodes:
                        p_id = next(next_id)
                        graph.add_node(p_id, **attrs)
                        graph.add_edge(p, p_id)
                        graph.add_edge(p_id, node)
                        graph.remove_edge(p, node)
                        created_nodes[(p, node)] = p_id
                    else:
                        p_id = created_nodes[(p, node)]
                    if (s, node) not in created_nodes:
                        s_id = next(next_id)
                        graph.add_node(s_id, **attrs)
                        graph.add_edge(s_id, s)
                        graph.add_edge(node, s_id)
                        graph.remove_edge(node, s)
                        created_nodes[(s, node)] = s_id
                    else:
                        s_id = created_nodes[(s, node)]
                    graph.add_edge(p_id, s_id)

    return graph
