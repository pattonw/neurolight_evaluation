import numpy as np
import networkx as nx
import noise

from itertools import product
from neurolight_evaluation import score_foreground

from neurolight_evaluation.common import make_directional
from neurolight_evaluation.fg_score import skeletonize

import logging
import random
import copy
import pickle

logger = logging.getLogger(__file__)


def test_fg_scores():
    logger.debug("Starting randomized fg test!")
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
    ref_tracings = make_directional(gt_tracings, "location")
    logger.warning(f"pred_cable_len: {cable_len(ref_tracings, 'location')}")

    recall, precision, (true_ref, total_ref, true_pred, total_pred) = score_foreground(
        gt_bin_mask, ref_tracings, offset, scale, 2, "penalty", "location"
    )
    assert recall == 1
    assert precision == 1

    num_edges = len(ref_tracings.edges())
    k = int(np.ceil(num_edges * frac))
    to_remove = random.sample(list(ref_tracings.edges()), k=k)
    to_add = []
    temp = copy.deepcopy(ref_tracings)
    while len(to_add) < len(to_remove):
        a = random.choice(list(ref_tracings.nodes))
        b = random.choice(list(ref_tracings.nodes))
        # make sure line ab does not have the same slope as any neighbors of a or b
        a_loc = ref_tracings.nodes[a]["location"]
        b_loc = ref_tracings.nodes[b]["location"]
        slope = (a_loc - b_loc) / np.linalg.norm(a_loc - b_loc)
        fail = False
        for a_neighbor in ref_tracings.neighbors(a):
            if fail:
                break
            n_loc = ref_tracings.nodes[a_neighbor]["location"]
            n_slope = (a_loc - n_loc) / np.linalg.norm(a_loc - n_loc)
            if abs(np.dot(slope, n_slope)) < 1e-4:
                fail = True
        for b_neighbor in ref_tracings.neighbors(b):
            if fail:
                break
            n_loc = ref_tracings.nodes[b_neighbor]["location"]
            n_slope = (b_loc - n_loc) / np.linalg.norm(b_loc - n_loc)
            if abs(np.dot(slope, n_slope)) < 1e-4:
                fail = True
        if not fail and b not in ref_tracings.neighbors(a) and a not in ref_tracings.neighbors(b):
            temp.add_edge(a, b)
            if nx.is_directed_acyclic_graph(temp):
                to_add.append((a, b))
            else:
                temp.remove_edge(a, b)

    current_total_ref = total_ref
    current_true_ref = true_ref
    current_total_pred = total_pred
    current_true_pred = true_pred

    for edge in to_remove:
        u_loc = ref_tracings.nodes[edge[0]]["location"]
        v_loc = ref_tracings.nodes[edge[1]]["location"]
        edge_len = np.linalg.norm(u_loc - v_loc)

        ref_tracings.remove_edge(*edge)
        logger.info(f"removed edge {edge}")

        (
            recall,
            precision,
            (true_ref, total_ref, true_pred, total_pred),
        ) = score_foreground(
            gt_bin_mask, ref_tracings, offset, scale, 0.05, "penalty", "location"
        )

        assert np.isclose(total_ref, current_total_ref - edge_len)
        assert np.isclose(true_ref, current_true_ref - edge_len)
        assert np.isclose(total_pred, current_total_pred)
        assert np.isclose(true_pred, current_true_pred - edge_len)
        current_total_ref = total_ref
        current_true_ref = true_ref
        current_total_pred = total_pred
        current_true_pred = true_pred

    for edge in to_add:
        ref_tracings.add_edge(*edge)
        u_loc = ref_tracings.nodes[edge[0]]["location"]
        v_loc = ref_tracings.nodes[edge[1]]["location"]
        edge_len = np.linalg.norm(u_loc - v_loc)

        (
            recall,
            precision,
            (true_ref, total_ref, true_pred, total_pred),
        ) = score_foreground(
            gt_bin_mask, ref_tracings, offset, scale, 0.05, "penalty", "location"
        )
        assert np.isclose(total_ref, current_total_ref + edge_len)
        assert np.isclose(true_ref, current_true_ref)
        assert np.isclose(total_pred, current_total_pred)
        assert np.isclose(true_pred, current_true_pred)
        current_total_ref = total_ref
        current_true_ref = true_ref
        current_total_pred = total_pred
        current_true_pred = true_pred


def cable_len(g, location_attr):
    total = 0
    for u, v in g.edges():
        u_loc, v_loc = g.nodes[u][location_attr], g.nodes[v][location_attr]
        total += np.linalg.norm(u_loc - v_loc)
    return total
