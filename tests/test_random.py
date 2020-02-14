import numpy as np
import noise

from itertools import product
from neurolight_evaluation import score_foreground

from neurolight_evaluation.fg_score import skeletonize


def test_fg_scores():
    shape = [50, 50, 50]
    freq = 16
    num_checks = 5
    offset, scale = np.array([0, 0, 0]), np.array([1, 1, 1])

    fg_pred = np.ndarray(shape)
    for coord in product(*[range(d) for d in shape]):
        fg_pred[coord] = noise.pnoise3(*[c / freq for c in coord]) / 2 + 0.5

    assert fg_pred.min() > 0
    assert fg_pred.max() < 1

    gt_bin_mask = fg_pred > 0.5
    gt_tracings = skeletonize(gt_bin_mask, offset, scale)

    predictions = {}
    for threshold in np.linspace(0, 1, num_checks):
        pred_bin_mask = fg_pred > threshold
        predictions[threshold] = pred_bin_mask

    for threshold, prediction in predictions.items():
        recall, precision = score_foreground(prediction, gt_tracings, offset, scale)
        if threshold == 0.5:
            assert recall == 1
            assert precision == 1
        elif threshold < 0.5:
            assert recall < 1
            assert precision == 1
        elif threshold > 0.5:
            assert recall == 1
            assert precision < 1
