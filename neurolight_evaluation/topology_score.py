import numpy as np
import networkx as nx
from sklearn.feature_extraction.image import grid_to_graph
import sklearn
from skimage.morphology import skeletonize as scikit_skeletonize

from funlib.match import GraphToTreeMatcher

from typing import Tuple, List
import logging

from .preprocess import add_fallback
from .costs import get_costs

logger = logging.getLogger(__file__)


def score_tracings(
    predicted_tracings: nx.DiGraph,
    reference_tracings: nx.DiGraph,
    match_threshold: float,
    penalty_attr: str,
    location_attr: str,
) -> Tuple[float, float]:
    if len(predicted_tracings.nodes) < 1:
        return 0, 1
    node_offset = max([node_id for node_id in predicted_tracings.nodes()]) + 1

    # Downsample the nodes in reference tracings by a factor of k to make sure predicted
    # tracings has higher resolution
    k = 10
    i = 0
    for node in list(reference_tracings.nodes()):
        if reference_tracings.degree(node) == 2:
            i += 1
            if i % k != 0:
                a, b = list(reference_tracings.succ[node]) + list(
                    reference_tracings.pred[node]
                )
                reference_tracings.remove_node(node)
                reference_tracings.add_edge(a, b)
    print("adding fallback")
    g = add_fallback(
        predicted_tracings,
        reference_tracings,
        node_offset,
        match_threshold,
        penalty_attr,
        location_attr,
    )
    print("Got fallback, getting costs")
    node_costs, edge_costs = get_costs(
        g, reference_tracings, location_attr, penalty_attr, match_threshold
    )
    edge_costs = [((a, b), (c, d), e) for a, b, c, d, e in edge_costs]
    print("got costs, initializing")

    logger.info(f"Edge costs going into matching: {edge_costs}")

    matcher = GraphToTreeMatcher(
        g, reference_tracings, node_costs, edge_costs, use_gurobi=False
    )
    print("Initialized, matching")
    node_matchings, edge_matchings, _ = matcher.match()
    print("matched!")

    logger.info(f"Final Edge matchings: {edge_matchings}")

    return calculate_recall_precision(
        node_matchings, edge_matchings, g, node_offset, location_attr
    )


def calculate_recall_precision(
    node_matchings: List[Tuple],
    edge_matchings: List[Tuple],
    pred_graph: nx.Graph,
    offset: int,
    location_attr: str,
) -> Tuple[float, float]:

    true_pred = 0
    total_pred = 0
    true_ref = 0
    total_ref = 0

    matched_ref = {}

    for edge_matching in edge_matchings:
        if edge_matching[1] is None:
            continue
        entry = matched_ref.setdefault(edge_matching[1], [])

        edge_pred = edge_matching[0]
        a_pred = pred_graph.nodes[edge_pred[0]][location_attr]
        b_pred = pred_graph.nodes[edge_pred[1]][location_attr]

        edge_ref = edge_matching[1]
        a_ref = pred_graph.nodes[edge_ref[0] + offset][location_attr]
        b_ref = pred_graph.nodes[edge_ref[1] + offset][location_attr]

        if (edge_pred[0] >= offset) or (edge_pred[1] >= offset):
            entry.append(edge_matching[0])
        else:
            true_pred += np.linalg.norm((b_pred - a_pred))

    for u, v in pred_graph.edges:
        u_loc = pred_graph.nodes[u][location_attr]
        v_loc = pred_graph.nodes[v][location_attr]
        if u < offset and v < offset:
            total_pred += np.linalg.norm((u_loc - v_loc))

        elif u >= offset and v >= offset:
            total_ref += np.linalg.norm((u_loc - v_loc))

    true_ref = total_ref
    for edge, failed_matchings in matched_ref.items():
        if len(failed_matchings) > 0:
            a_ref = pred_graph.nodes[edge[0] + offset][location_attr]
            b_ref = pred_graph.nodes[edge[1] + offset][location_attr]
            true_ref -= np.linalg.norm((b_ref - a_ref))

    recall = true_ref / (total_ref)
    precision = true_pred / (total_pred)
    return recall, precision

