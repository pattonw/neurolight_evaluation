import numpy as np
import networkx as nx

from neurolight_evaluation.fg_score_voxel import score_foreground_voxel

def test_score_foreground():
    gt = np.concatenate((np.ones((4, 4, 4)), np.zeros((4, 4, 4))), axis=2)
    pred = np.concatenate((np.zeros((4, 4, 2)), np.ones((4, 4, 2)), np.zeros((4, 4, 4))), axis=2)
    b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
    assert tpr == 0.5
    assert tnr == 1
    assert b_acc == 0.75

    gt = np.zeros((10))
    pred = np.ones((10))
    b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
    assert tpr == 1
    assert tnr == 0
    assert b_acc == 0.5


    pred = np.concatenate((np.ones((4, 4, 4)), np.zeros((4, 4, 3)), np.ones((4, 4, 1))), axis=2)
    gt = np.concatenate((-np.ones((4, 4, 2)), np.ones((4, 4, 2)), np.zeros((4, 4, 4))), axis=2)
    b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
    assert tpr == 1
    assert tnr == 0.75
    assert b_acc == 0.875

    gt = -np.ones((10))
    pred = np.ones((10))
    b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
    assert tpr == 1
    assert tnr == 1
    assert b_acc == 1

    
    