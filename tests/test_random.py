import numpy as np
import networkx as nx
import noise
import pytest

from itertools import product
from neurolight_evaluation import score_foreground

from neurolight_evaluation.fg_score import skeletonize

import logging
import random

logger = logging.getLogger(__file__)


def test_fg_scores():
    logger.info("Starting randomized fg test!")
    shape = [20, 20, 20]
    freq = 16
    frac = 0.3
    offset, scale = np.array([0, 0, 0]), np.array([1, 1, 1])

    fg_pred = np.ndarray(shape)
    for coord in product(*[range(d) for d in shape]):
        fg_pred[coord] = noise.pnoise3(*[c / freq for c in coord]) / 2 + 0.5

    assert fg_pred.min() > 0
    assert fg_pred.max() < 1

    gt_bin_mask = fg_pred > 0.5
    gt_tracings = skeletonize(gt_bin_mask, offset, scale)
    gt_tracings = make_directional(gt_tracings, "location")

    recall, precision = score_foreground(
        gt_bin_mask, gt_tracings, offset, scale, 2, "penalty", "location"
    )
    assert recall == 1
    assert precision == 1

    num_edges = len(gt_tracings.edges())
    to_remove = int(np.ceil(num_edges * frac))

    for edge in random.choices(list(gt_tracings.edges()), k=to_remove):
        gt_tracings.remove_edge(*edge)

    recall, precision = score_foreground(
        gt_bin_mask, gt_tracings, offset, scale, 2, "penalty", "location"
    )
    assert recall == 1
    assert precision < 1


def make_directional(graph: nx.Graph(), location_attr: str):
    g = graph.to_directed()

    for u, v in graph.edges():
        if u == v:
            g.remove_edge(u, v)
            continue
        u_loc, v_loc = g.nodes[u][location_attr], g.nodes[v][location_attr]
        slope = v_loc - u_loc
        neg = slope < 0
        pos = slope > 0
        for n, p in zip(neg, pos):
            if n != p and n:
                g.remove_edge(u, v)
                break
            elif n != p and p:
                g.remove_edge(v, u)
                break

    if not nx.is_directed_acyclic_graph(g):
        cycle = nx.algorithms.find_cycle(g)
        logger.info(cycle)
        for u, v in cycle:
            u_loc, v_loc = g.nodes[u][location_attr], g.nodes[v][location_attr]
            logger.info(v_loc - u_loc)

    return g
