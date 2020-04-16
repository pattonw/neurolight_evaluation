import networkx as nx
import numpy as np

from typing import List, Tuple
import logging

logger = logging.getLogger(__file__)


def calculate_recall_precision_matchings(
    node_matchings: List[Tuple],
    edge_matchings: List[Tuple],
    pred_graph: nx.Graph,
    ref_graph: nx.Graph,
    offset: int,
    location_attr: str,
) -> Tuple[float, float]:

    ref_graph = ref_graph.to_undirected()

    matched_pred = 0
    total_pred = 0
    matched_ref = 0
    total_ref = 0

    reference_targets = {}

    # Edge matchings contains a match for every edge in Pred, not Ref
    accounted_edges = set()
    for edge_matching in edge_matchings:
        pred_edge = edge_matching[0]
        ref_edge = edge_matching[1]

        a_pred = pred_graph.nodes[pred_edge[0]][location_attr]
        b_pred = pred_graph.nodes[pred_edge[1]][location_attr]

        is_pred_edge = pred_edge[0] < offset and pred_edge[1] < offset

        if is_pred_edge and pred_edge not in accounted_edges:
            # keep track of both
            accounted_edges.add((pred_edge[1], pred_edge[0]))
            total_pred += np.linalg.norm(a_pred - b_pred)

        if is_pred_edge and ref_edge is not None:
            matched_pred += np.linalg.norm(a_pred - b_pred)

        reference_matchings = reference_targets.setdefault(ref_edge, [])
        reference_matchings.append(pred_edge)

    for ref_u, ref_v in ref_graph.edges():

        pred_edges = reference_targets.get(
            (ref_u, ref_v), reference_targets.get((ref_v, ref_u), [])
        )
        # Calculating what percent of a failed reference edge was successfully
        # matched gives a more robust estimation of the recall. It is possible
        # that simply adding an edge to the ground truth can decrease your recall
        # without this step by simply making it cheaper to shift a node match
        # onto the fallback to save some fallback edge assignments.
        true_pred_edge_lengths = 0
        total_pred_edge_lengths = 0
        for a, b in pred_edges:
            a_loc = pred_graph.nodes[a][location_attr]
            b_loc = pred_graph.nodes[b][location_attr]
            d = np.linalg.norm(a_loc - b_loc)
            if a < offset and b < offset:
                true_pred_edge_lengths += d
            total_pred_edge_lengths += d
        if total_pred_edge_lengths < 1e-4:
            proportion_true = 0
        else:
            proportion_true = true_pred_edge_lengths / total_pred_edge_lengths

        a_ref = ref_graph.nodes[ref_u][location_attr]
        b_ref = ref_graph.nodes[ref_v][location_attr]
        dist = np.linalg.norm(a_ref - b_ref)
        total_ref += dist
        matched_ref += proportion_true * dist

        print(ref_u, ref_v, proportion_true * dist)

    recall = matched_ref / (total_ref)
    precision = matched_pred / (total_pred)
    logger.debug(
        f"total_ref: {total_ref}, matched_ref: {matched_ref}, "
        f"total_pred: {total_pred}, matched_pred: {matched_pred}"
    )
    return recall, precision, (matched_ref, total_ref, matched_pred, total_pred)
