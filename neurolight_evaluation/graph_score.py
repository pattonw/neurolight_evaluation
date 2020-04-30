import networkx as nx

import comatch

import logging

from .graph_matching.comatch.edges_xy import get_edges_xy
from .graph_metrics import psudo_graph_edit_distance


logger = logging.getLogger(__file__)


def score_graph(
    predicted_tracings: nx.Graph,
    reference_tracings: nx.Graph,
    match_threshold: float,
    location_attr: str,
    node_every: float,
    metric: str,
):

    edges_xy = get_edges_xy(
        predicted_tracings, reference_tracings, location_attr, match_threshold
    )

    nodes_x = list(predicted_tracings.nodes)
    nodes_y = list(reference_tracings.nodes)
    node_labels_x = {
        node: cc
        for cc, cc_nodes in enumerate(nx.connected_components(predicted_tracings))
        for node in cc_nodes
    }
    node_labels_y = {
        node: cc
        for cc, cc_nodes in enumerate(nx.connected_components(reference_tracings))
        for node in cc_nodes
    }

    label_matches, node_matches, splits, merges, fps, fns = comatch.match_components(
        nodes_x, nodes_y, edges_xy, node_labels_x, node_labels_y
    )

    if metric == "graph_edit":
        return psudo_graph_edit_distance(
            node_matches,
            node_labels_x,
            node_labels_y,
            predicted_tracings,
            reference_tracings,
            location_attr,
            node_every,
        )
    else:
        raise NotImplementedError("Only option at the moment is 'graph_edit'")
