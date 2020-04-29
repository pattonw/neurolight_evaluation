from neurolight_evaluation.graph_matching.comatch.edges_xy import get_edges_xy

import networkx as nx
import numpy as np


def test_get_edges_xy(benchmark):
    """
    x: 0--1--2--3--4--5--6--7--8--9--------------------------10
    y: 0--------------------------1--2--3--4--5--6--7--8--9--10

    (0-4) in x should match to (0) in y
    (5-9) in x should match to (1) in y
    (1-5) in y should match to (9) in x
    (6-10) in y should match to (10) in x
    """
    x_nodes = [(i, {"location": np.array([0, 0, i])}) for i in range(10)]
    x_nodes += [(10, {"location": np.array([0, 0, 18])})]
    x_edges = [(i, i + 1) for i in range(10)]
    x = nx.Graph()
    x.add_nodes_from(x_nodes)
    x.add_edges_from(x_edges)

    y_nodes = [(0, {"location": np.array([0, 0, 0])})]
    y_nodes += [(i + 1, {"location": np.array([0, 0, i + 9])}) for i in range(10)]
    y_edges = [(i, i + 1) for i in range(10)]
    y = nx.Graph()
    y.add_nodes_from(y_nodes)
    y.add_edges_from(y_edges)

    possible_edges = benchmark(get_edges_xy, x, y, "location", 0.5)

    expected_edges = (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (10, 6),
        (10, 7),
        (10, 8),
        (10, 9),
        (10, 10),
    )

    assert set(expected_edges) == set(possible_edges)
