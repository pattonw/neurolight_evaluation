import numpy as np
import networkx as nx

from neurolight_evaluation.fg_score_voxel import score_foreground_voxel

# iterate over multiple examples

def test_score_foreground_voxel_binary():
    gt = np.concatenate((np.ones((4, 4, 4)), np.zeros((4, 4, 4))), axis=2)
    pred = np.concatenate((np.zeros((4, 4, 2)), np.ones((4, 4, 2)), np.zeros((4, 4, 4))), axis=2)
    b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
    assert tpr == 0.5
    assert tnr == 1
    assert b_acc == 0.75

    tpr_prev = tpr
    tnr_prev = tnr
    b_acc_prev = b_acc

    # adding false positives lowers the tnr
    for i in range(10):
        gt = np.concatenate((gt, np.zeros((4, 4, 1))), axis=2)
        pred = np.concatenate((pred, np.ones((4, 4, 1))), axis=2)
        b_acc, tpr, tnr = score_foreground_voxel(pred, gt)

        assert tpr == tpr_prev
        assert tnr < tnr_prev
        assert b_acc < b_acc_prev

        tpr_prev = tpr
        tnr_prev = tnr
        b_acc_prev = b_acc

    # adding false negatives lowers the tpr
    for i in range(10):
        gt = np.concatenate((gt, np.ones((4, 4, 1))), axis=2)
        pred = np.concatenate((pred, np.zeros((4, 4, 1))), axis=2)
        b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
        
        assert tpr < tpr_prev
        assert tnr == tnr_prev
        assert b_acc < b_acc_prev

        tpr_prev = tpr
        tnr_prev = tnr
        b_acc_prev = b_acc

    # adding true negatives raises tnr
    for i in range(10):
        gt = np.concatenate((gt, np.zeros((4, 4, 1))), axis=2)
        pred = np.concatenate((pred, np.zeros((4, 4, 1))), axis=2)
        b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
        
        assert tpr == tpr_prev
        assert tnr > tnr_prev
        assert b_acc > b_acc_prev

        tpr_prev = tpr
        tnr_prev = tnr
        b_acc_prev = b_acc

    # adding true positives raises tpr
    for i in range(10):
        gt = np.concatenate((gt, np.ones((4, 4, 1))), axis=2)
        pred = np.concatenate((pred, np.ones((4, 4, 1))), axis=2)
        b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
        
        assert tpr > tpr_prev
        assert tnr == tnr_prev
        assert b_acc > b_acc_prev
        
        tpr_prev = tpr
        tnr_prev = tnr
        b_acc_prev = b_acc

    gt = np.zeros((10))
    pred = np.ones((10))
    b_acc, tpr, tnr = score_foreground_voxel(pred, gt)
    assert tpr == 1
    assert tnr == 0
    assert b_acc == 0.5

def test_score_foreground_voxel_ternary():
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

   
def test_rasterize_graph():
    location = 'loc'

    g = nx.Graph()
    g.add_node(0, **{location: (0, 0, 0)})
    g.add_node(1, **{location: (10, 0, 0)})

    radius_inner = 1
    radius_outer = 2