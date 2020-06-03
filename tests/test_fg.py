import numpy as np
import networkx as nx

from neurolight_evaluation import score_foreground
from neurolight_evaluation.graph_metrics import Metric


def test_simple_match():
    offset = np.array([0, 0, 0])
    scale = np.array([1, 1, 1])

    pred = np.stack((np.eye(10), np.zeros([10, 10])), axis=2)

    ref = nx.Graph()
    a = np.array([0, 0, 0])
    b = np.array([9, 9, 0])
    ref.add_nodes_from([(0, {"loc": a}), (1, {"loc": b})])
    ref.add_edge(0, 1)

    score = score_foreground(
        binary_prediction=pred,
        reference_tracings=ref,
        offset=offset,
        scale=scale,
        match_threshold=0.5,
        location_attr="loc",
        node_spacing=2 ** 0.5,
        metric=Metric.GRAPH_EDIT,
    )

    assert score == 0


def test_simple_wrong():
    offset = np.array([0, 0, 0])
    scale = np.array([1, 1, 1])

    pred = np.stack((np.eye(10), np.zeros([10, 10])), axis=2)

    ref = nx.Graph()
    a = np.array([0, 9, 0])
    b = np.array([9, 0, 0])
    ref.add_nodes_from([(0, {"loc": a}), (1, {"loc": b})])
    ref.add_edge(0, 1)

    score = score_foreground(
        binary_prediction=pred,
        reference_tracings=ref,
        offset=offset,
        scale=scale,
        match_threshold=0.5,
        location_attr="loc",
        node_spacing=2 ** (0.5),
        metric=Metric.GRAPH_EDIT,
    )

    # false pos cost: 1
    # false neg cost: (2*(9**2))**0.5/2**0.5 => 9.0
    # merge cost: 0
    # split cost: 0
    assert score == 10.0
