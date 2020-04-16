import numpy as np
import networkx as nx

from funlib.match import GraphToTreeMatcher

from typing import Tuple, List
import logging

from .preprocess import add_fallback
from .costs import get_costs
from .common import calculate_recall_precision_matchings
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
