import networkx as nx
import numpy as np

from typing import Tuple
import logging
import itertools

from .graph_score import score_graph

logger = logging.getLogger(__file__)


def score_tracings(
    predicted_tracings: nx.Graph,
    reference_tracings: nx.Graph,
    match_threshold: float,
    location_attr: str,
    metric: str,
    **metric_kwargs,
) -> Tuple[float, float]:

    return score_graph(
        predicted_tracings,
        reference_tracings,
        match_threshold,
        location_attr,
        metric,
        **metric_kwargs
    )
