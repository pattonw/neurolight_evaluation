import numpy as np
import networkx as nx

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
        node_matchings,
        edge_matchings,
        g,
        reference_tracings,
        node_offset,
        location_attr,
    )


def calculate_recall_precision(
    node_matchings: List[Tuple],
    edge_matchings: List[Tuple],
    pred_graph: nx.Graph,
    ref_graph: nx.Graph,
    offset: int,
    location_attr: str,
) -> Tuple[float, float]:

    true_pred = 0
    total_pred = 0
    true_ref = 0
    total_ref = 0

    matched_ref = {}

    # Edge matchings contains a match for every edge in Pred, not Ref
    for edge_matching in edge_matchings:
        pred_edge = edge_matching[0]
        ref_edge = edge_matching[1]

        a_pred = pred_graph.nodes[pred_edge[0]][location_attr]
        b_pred = pred_graph.nodes[pred_edge[1]][location_attr]

        total_pred += np.linalg.norm(a_pred - b_pred)

        if ref_edge is not None:
            true_pred += np.linalg.norm(a_pred - b_pred)

        reference_matchings = matched_ref.setdefault(ref_edge, [])
        reference_matchings.append(pred_edge)

    for ref_edge in ref_graph.edges():
        pred_edges = matched_ref.get(ref_edge, [])
        failed = len(pred_edge) == 0 or any(
            [a >= offset or b >= offset for a, b in pred_edges]
        )

        a_ref = ref_graph.nodes[ref_edge[0]][location_attr]
        b_ref = ref_graph.nodes[ref_edge[1]][location_attr]
        dist = np.linalg.norm(a_ref - b_ref)
        total_ref += dist
        if not failed:
            true_ref += dist

    recall = true_ref / (total_ref)
    precision = true_pred / (total_pred)
    return recall, precision

