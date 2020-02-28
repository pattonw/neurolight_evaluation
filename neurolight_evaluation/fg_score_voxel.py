import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__file__)


def score_foreground_voxel(
    binary_prediction: np.ndarray, ground_truth: np.ndarray,
) -> Tuple[float, float, float]:
    """Computes accuracy metrics for binary segmentations.

    The ground truth segmentation can have a 3rd label that means "ambiguous."
    Segmentation predictions are only evaluated at non-ambiguous locations.

    Args:
        binary_prediction (np.ndarray):

            Binary segmentation to be evaluated. Must have same shape as ground_truth.

        ground_truth (np.ndarray):

            Ground truth segmentation. Can contain labels {-1, 0, 1}.
            Labels of -1 are not used in accuracy evaluation.

    Returns:
        balanced_acc (float):
            
            Balanced accuracy (average of tpr and tnr).

        tpr (float):

            True positive rate.

        tnr (float):

            True negative rate.
    """

    if binary_prediction.shape != ground_truth.shape:
        raise ValueError(
            f"""Image shape mismatch: binary_prediction - {binary_prediction.shape}, 
                          ground_truth - {ground_truth.shape}"""
        )

    matches = (binary_prediction == ground_truth) | (ground_truth == -1)
    ones = ground_truth == 1
    zeros = ground_truth == 0

    true_pos = np.sum((matches & ones))
    if ones.any():
        tpr = true_pos / np.sum(ones)
    else:
        tpr = 1  # is this what we want?

    true_neg = np.sum(matches & zeros)
    if zeros.any():
        tnr = true_neg / np.sum(zeros)
    else:
        tnr = 1

    balanced_acc = 0.5 * (tnr + tpr)

    return balanced_acc, tpr, tnr
