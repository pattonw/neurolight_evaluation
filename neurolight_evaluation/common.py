import networkx as nx
import numpy as np

from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__file__)


def recall_precision(
    node_matchings: List[Tuple[int, int]],
    node_x_labels: Dict[int, int],
    node_y_labels: Dict[int, int],
    graph_x: nx.Graph,
    graph_y: nx.Graph,
    location_attr: str,
) -> Tuple[float, float]:
    """
    Calculate recall and precision accross two graphs.
    
    Recall is considered to be the percentage of the cable
    length of graph_x that was successfully matched

    Precision is considered to be the percentage of the cable
    length of graph_y that was successfully matched

    An edge (a, b) is considered successfully matched if a matches
    to c, and b matches to d, where c and d share the same label id.
    Note that it is assumed for edge (a, b) that a and b share a
    label since they are part of the same connected component.

    Note that nodes without adjacent edges will not contribute to
    either metric.

    Args:

        node_matchings: (``list`` of ``tuple`` pairs of ``int``)
    
            A list of tuples containing pairs of nodes that match

        node_x_labels: (``dict`` mapping ``int`` to ``int``)

            A dictionary mapping node_ids in graph_x (assumed to be integers)
            to label ids in graph_x (also assumed to be integers)

        node_y_labels: (``dict`` mapping ``int`` to ``int``)

            A dictionary mapping node_ids in graph_y (assumed to be integers)
            to label ids in graph_y (also assumed to be integers)

        graph_x: (``nx.Graph``)

            The graph_x on which to calculate recall

        graph_y: (``nx.Graph``)

            the graph_y on which to calculate precision

        location_attr: (``str``)

            An attribute that all nodes in graph_x and graph_y have that contains
            a node's location for calculating edge lengths.

    Returns:

        (``tuple`` of ``int``):

            recall and precision
    """

    matched_x = 0
    total_x = 0
    matched_y = 0
    total_y = 0

    x_node_to_y_label = {}
    y_node_to_x_label = {}
    for a, b in node_matchings:
        y_label = x_node_to_y_label.setdefault(a, node_y_labels[b])
        assert y_label == node_y_labels[b], (
            f"node {a} in graph_x matches to multiple labels in graph_y, "
            f"including {(y_label, node_y_labels[b])}!"
        )
        x_label = y_node_to_x_label.setdefault(b, node_x_labels[a])
        assert x_label == node_x_labels[a], (
            f"node {b} in graph_y matches to multiple labels in graph_x, "
            f"including {(x_label, node_x_labels[a])}!"
        )

    for a, b in graph_x.edges():
        a_loc = graph_x.nodes[a][location_attr]
        b_loc = graph_x.nodes[b][location_attr]
        edge_len = np.linalg.norm(a_loc - b_loc)
        if x_node_to_y_label[a] == x_node_to_y_label[b]:
            matched_x += edge_len
        total_x += edge_len

    for a, b in graph_y.edges():
        a_loc = graph_y.nodes[a][location_attr]
        b_loc = graph_y.nodes[b][location_attr]
        edge_len = np.linalg.norm(a_loc - b_loc)
        if y_node_to_x_label[a] == y_node_to_x_label[b]:
            matched_y += edge_len
        total_y += edge_len

    if np.isclose(total_x, 0):
        recall = 0
    else:
        recall = matched_x / (total_x + 1e-4)
    if np.isclose(total_y, 0):
        precision = 0
    else:
        precision = matched_y / (total_y + 1e-4)

    return recall, precision