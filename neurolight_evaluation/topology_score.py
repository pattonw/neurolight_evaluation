import networkx as nx

from typing import Tuple
import logging

from .graph_score import score_graph

logger = logging.getLogger(__file__)


def score_tracings(
    predicted_tracings: nx.DiGraph,
    reference_tracings: nx.DiGraph,
    match_threshold: float,
    penalty_attr: str,
    location_attr: str,
) -> Tuple[float, float]:
    return score_graph(
        predicted_tracings,
        reference_tracings,
        match_threshold,
        penalty_attr,
        location_attr,
    )
