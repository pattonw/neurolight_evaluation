import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

import logging

logger = logging.getLogger(__file__)


def fallback_node_penalty():
    return 1


def fallback_edge_penalty():
    return 1


def crossing_edge_penalty():
    return 1


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
    for f_node_index, g_node_indices in enumerate(crossing_edges):
        f_node = fallback_nodes[f_node_index] + node_offset
        for g_node_index in g_node_indices:
            g_node = graph_nodes[g_node_index]

            graph.add_edge(f_node, g_node, **{penalty_attr: crossing_edge_penalty()})

    return graph
