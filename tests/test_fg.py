import numpy as np
import networkx as nx

from neurolight_evaluation import score_foreground


def test_simple_match():
    offset = np.array([0, 0, 0])
    scale = np.array([1, 1, 1])

    pred = np.stack((np.eye(10), np.zeros([10, 10])), axis=2)

    ref = nx.DiGraph()
    a = np.array([0, 0, 0])
    b = np.array([9, 9, 0])
    ref.add_nodes_from([(0, {"loc": a}), (1, {"loc": b})])
    ref.add_edge(0, 1)

    recall, precision = score_foreground(
        binary_prediction=pred,
        reference_tracings=ref,
        offset=offset,
        scale=scale,
        match_threshold=0.5,
        penalty_attr="penalty",
        location_attr="loc",
    )

    assert recall == 1
    assert precision == 1


def test_simple_wrong():
    offset = np.array([0, 0, 0])
    scale = np.array([1, 1, 1])

    pred = np.stack((np.eye(10), np.zeros([10, 10])), axis=2)

    ref = nx.DiGraph()
    a = np.array([0, 9, 0])
    b = np.array([9, 0, 0])
    ref.add_nodes_from([(0, {"loc": a}), (1, {"loc": b})])
    ref.add_edge(0, 1)

    recall, precision = score_foreground(
        binary_prediction=pred,
        reference_tracings=ref,
        offset=offset,
        scale=scale,
        match_threshold=0.5,
        penalty_attr="penalty",
        location_attr="loc",
    )

    assert recall == 0
    assert precision == 0

