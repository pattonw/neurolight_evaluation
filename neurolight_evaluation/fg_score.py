import numpy as np
import networkx as nx
from sklearn.feature_extraction.image import grid_to_graph
import sklearn
from skimage.morphology import skeletonize as scikit_skeletonize

from typing import Tuple
import logging

from .graph_score import score_graph

logger = logging.getLogger(__file__)


def score_foreground(
    binary_prediction: np.ndarray,
    reference_tracings: nx.Graph,
    offset: np.ndarray,
    scale: np.ndarray,
    match_threshold: float,
    location_attr: str,
    metric: str,
    **metric_kwargs,
) -> Tuple[float, float]:
    predicted_tracings = skeletonize(
        binary_prediction, offset, scale, location_attr=location_attr
    )
    return score_graph(
        predicted_tracings,
        reference_tracings,
        match_threshold,
        location_attr,
        metric,
        **metric_kwargs,
    )


def skeletonize(
    binary_prediction: np.ndarray,
    offset: np.ndarray,
    scale: np.ndarray,
    location_attr: str = "location",
) -> nx.Graph:
    skeletonized_pred = scikit_skeletonize(binary_prediction) == 255
    # graph with nodes having voxel coordinates
    skeleton_graph = grid_to_nx_graph(
        skeleton=skeletonized_pred, location_attr=location_attr
    )
    # scaled into world units
    skeleton_graph = scale_skeleton(
        skeleton=skeleton_graph, offset=offset, scale=scale, location_attr=location_attr
    )
    return skeleton_graph


def scale_skeleton(
    skeleton: nx.Graph, offset: np.ndarray, scale: np.ndarray, location_attr: str
) -> nx.Graph:
    for n, data in skeleton.nodes.items():
        voxel_loc = data[location_attr]
        space_loc = np.multiply(voxel_loc, scale) + offset
        data[location_attr] = space_loc

    return skeleton


def grid_to_nx_graph(skeleton: np.ndarray, location_attr: str) -> nx.Graph:
    # Override with local function
    sklearn.feature_extraction.image._make_edges_3d = _make_edges_3d

    s = skeleton.shape
    # Identify connectivity
    adj_mat = grid_to_graph(n_x=s[0], n_y=s[1], n_z=s[2], mask=skeleton)
    # Identify order of the voxels
    voxel_locs = compute_voxel_locs(mask=skeleton)

    g = nx.Graph()

    nodes = [
        (node_id, {location_attr: voxel_loc})
        for node_id, voxel_loc in enumerate(voxel_locs)
    ]
    g.add_nodes_from(nodes)

    edges = [(a, b) for a, b in zip(adj_mat.row, adj_mat.col) if a != b]
    logger.debug(f"Grid to graph edges: {edges}")
    g.add_edges_from(edges)

    return g


# From https://github.com/neurodata/scikit-learn/blob/tom/grid_to_graph_26/sklearn/feature_extraction/image.py
# Used in grid_to_graph
# automatically set to 26-connectivity
def _make_edges_3d(n_x: int, n_y: int, n_z: int, connectivity=26):
    """Returns a list of edges for a 3D image.
    Parameters
    ----------
    n_x : int
        The size of the grid in the x direction.
    n_y : int
        The size of the grid in the y direction.
    n_z : integer, default=1
        The size of the grid in the z direction, defaults to 1
    connectivity : int in [6,18,26], default=26
        Defines what are considered neighbors in voxel space.
    """
    if connectivity not in [6, 18, 26]:
        raise ValueError("Invalid value for connectivity: %r" % connectivity)

    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))

    edges = []

    edges_self = np.vstack((vertices.ravel(), vertices.ravel()))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))

    edges = [edges_self, edges_deep, edges_right, edges_down]

    # Add the other connections
    if connectivity >= 18:
        edges_right_deep = np.vstack(
            (vertices[:, :-1, :-1].ravel(), vertices[:, 1:, 1:].ravel())
        )
        edges_down_right = np.vstack(
            (vertices[:-1, :-1, :].ravel(), vertices[1:, 1:, :].ravel())
        )
        edges_down_deep = np.vstack(
            (vertices[:-1, :, :-1].ravel(), vertices[1:, :, 1:].ravel())
        )
        edges_down_left = np.vstack(
            (vertices[:-1, 1:, :].ravel(), vertices[1:, :-1, :].ravel())
        )
        edges_down_shallow = np.vstack(
            (vertices[:-1, :, 1:].ravel(), vertices[1:, :, :-1].ravel())
        )
        edges_deep_left = np.vstack(
            (vertices[:, 1:, :-1].ravel(), vertices[:, :-1, 1:].ravel())
        )

        edges.extend(
            [
                edges_right_deep,
                edges_down_right,
                edges_down_deep,
                edges_down_left,
                edges_down_shallow,
                edges_deep_left,
            ]
        )

    if connectivity == 26:
        edges_down_right_deep = np.vstack(
            (vertices[:-1, :-1, :-1].ravel(), vertices[1:, 1:, 1:].ravel())
        )
        edges_down_left_deep = np.vstack(
            (vertices[:-1, 1:, :-1].ravel(), vertices[1:, :-1, 1:].ravel())
        )
        edges_down_right_shallow = np.vstack(
            (vertices[:-1, :-1, 1:].ravel(), vertices[1:, 1:, :-1].ravel())
        )
        edges_down_left_shallow = np.vstack(
            (vertices[:-1, 1:, 1:].ravel(), vertices[1:, :-1, :-1].ravel())
        )

        edges.extend(
            [
                edges_down_right_deep,
                edges_down_left_deep,
                edges_down_right_shallow,
                edges_down_left_shallow,
            ]
        )

    edges = np.hstack(edges)
    return edges


def compute_voxel_locs(mask: np.ndarray) -> np.ndarray:
    locs = np.where(mask == 1)
    locs = np.stack(locs, axis=1)
    return locs
