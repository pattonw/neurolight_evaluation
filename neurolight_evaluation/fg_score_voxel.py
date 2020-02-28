import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__file__)

def score_foreground_voxel(
    binary_prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Tuple[float, float, float]:

    matches = (binary_prediction == ground_truth) | (ground_truth == -1)
    ones = ground_truth == 1
    zeros = ground_truth == 0

    true_pos = np.sum((matches & ones))
    if ones.any(): 
        tpr = true_pos / np.sum(ones)
    else:
        tpr = 1 #is this what we want?

    true_neg = np.sum(matches & zeros)
    if zeros.any(): 
        tnr = true_neg / np.sum(zeros)
    else:
        tnr = 1

    balanced_acc = 0.5 * (tnr + tpr)
    
    return balanced_acc, tpr, tnr
