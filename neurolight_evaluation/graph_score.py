import numpy as np
import networkx as nx
from funlib.match import GraphToTreeMatcher

import logging
import copy

from .preprocess import add_fallback, preprocess
from .costs import get_costs
from .common import calculate_recall_precision_matchings


logger = logging.getLogger(__file__)


def score_graph(
    predicted_tracings: nx.DiGraph,
    reference_tracings: nx.DiGraph,
    match_threshold: float,
    penalty_attr: str,
    location_attr: str,
):
    if len(predicted_tracings.nodes) < 1:
        return 0, 1
    node_offset = max([node_id for node_id in predicted_tracings.nodes()]) + 1

    fallback = preprocess(copy.deepcopy(reference_tracings))

    g = add_fallback(
        predicted_tracings,
        fallback,
        node_offset,
        match_threshold,
        penalty_attr,
        location_attr,
    )
    node_costs, edge_costs = get_costs(
        g, reference_tracings, location_attr, penalty_attr, match_threshold
    )
    edge_costs = [((a, b), (c, d), e) for a, b, c, d, e in edge_costs]

    logger.debug(f"Edge costs going into matching: {edge_costs}")

    matcher = GraphToTreeMatcher(
        g, reference_tracings, node_costs, edge_costs, use_gurobi=False
    )
    node_matchings, edge_matchings, _ = matcher.match()

    logger.debug(f"Final Edge matchings: {edge_matchings}")

    return calculate_recall_precision_matchings(
        node_matchings,
        edge_matchings,
        g,
        reference_tracings,
        node_offset,
        location_attr,
    )
