import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

from typing import Tuple


def score_foreground(
    binary_prediction: np.ndarray,
    reference_tracings: nx.Graph,
    offset: np.ndarray,
    scale: np.ndarray,
) -> Tuple[float, float]:
    raise NotImplementedError()
    recall, precision = None, None
    return (recall, precision)


def skeletonize(
    binary_prediciton: np.ndarray, offset: np.ndarray, scale: np.ndarray
) -> nx.Graph:
    skeletonized_pred = skeletonize(binary_prediciton)
    s = skeletonized_pred.shape
    # graph with nodes having voxel coordinates
    skeleton = grid_to_graph(n_x=s[0], n_y=s[1], n_z=s[2], mask=skeletonized_pred)
    # scaled into world units
    skeleton = scale_skeleton(offset, scale)
    return skeleton


def scale_skeleton():
    raise NotImplementedError()


def grid_to_graph(binary_prediction: np.ndarray) -> nx.Graph:
    raise NotImplementedError()
