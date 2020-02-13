import networkx as nx
from typing import Tuple


def score_tracing(
    predicted_tracings: nx.Graph, reference_tracings: nx.Graph
) -> Tuple[float, float]:
    raise NotImplementedError()
    recall, precision = None, None
    return (recall, precision)

