import numpy as np
import networkx as nx

from neurolight_evaluation import score_foreground

def test_simple_match():
    offset = np.array([0,0,0])
    scale = np.array([1,1,1])

    pred = np.stack((np.eye(10), np.zeros([10,10])), axis=2)

    ref = nx.DiGraph()
    a = np.array([0,0,0])
    b = np.array([9,9,0])
    ref.add_nodes_from([('a',{'loc':a}),('b',{'loc':b})])
    ref.add_edge('a','b')

    recall, precision = score_foreground(
        binary_prediction=pred, reference_tracings=ref,
        offset=offset, scale=scale)

def test_simple_wrong():
    offset = np.array([0,0,0])
    scale = np.array([1,1,1])

    pred = np.stack((np.eye(10), np.zeros([10,10])), axis=2)

    ref = nx.Graph()
    a = np.array([0,9,0])
    b = np.array([9,0,0])
    ref.add_nodes_from([('a',{'loc':a}),('b',{'loc':b})])
    ref.add_edge('a','b')

    recall, precision = score_foreground(
        binary_prediction=pred, reference_tracings=ref,
        offset=offset, scale=scale)