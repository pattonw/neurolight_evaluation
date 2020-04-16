import numpy as np
import networkx as nx
from sklearn.feature_extraction.image import grid_to_graph
import sklearn
from skimage.morphology import skeletonize as scikit_skeletonize

from funlib.match import GraphToTreeMatcher

from typing import Tuple, List
import logging

from .preprocess import add_fallback
from .costs import get_costs

logger = logging.getLogger(__file__)


def score_foreground(
    binary_prediction: np.ndarray,
    reference_tracings: nx.DiGraph,
    offset: np.ndarray,
    scale: np.ndarray,
    match_threshold: float,
    penalty_attr: str,
    location_attr: str,
) -> Tuple[float, float]:
    predicted_tracings = skeletonize(
        binary_prediction, offset, scale, location_attr=location_attr
    )
    if len(predicted_tracings.nodes) < 1:
        return 0, 1
    node_offset = max([node_id for node_id in predicted_tracings.nodes()]) + 1

    fallback = preprocess(copy.deepcopy(reference_tracings))

    g = add_fallback(
        predicted_tracings,
        fallback,
        node_offset,
        match_threshold,
        penalty_attr,
        location_attr,
    )
    node_costs, edge_costs = get_costs(
        g, reference_tracings, location_attr, penalty_attr, match_threshold
    )
    edge_costs = [((a, b), (c, d), e) for a, b, c, d, e in edge_costs]

    logger.debug(f"Edge costs going into matching: {edge_costs}")

    matcher = GraphToTreeMatcher(
        g, reference_tracings, node_costs, edge_costs, use_gurobi=False
    )
    node_matchings, edge_matchings, _ = matcher.match()

    logger.debug(f"Final Edge matchings: {edge_matchings}")

    return calculate_recall_precision(
        node_matchings, edge_matchings, g, node_offset, location_attr
    )


def calculate_recall_precision(
    node_matchings: List[Tuple],
    edge_matchings: List[Tuple],
    pred_graph: nx.Graph,
    offset: int,
    location_attr: str,
) -> Tuple[float, float]:

    true_pred = 0
    total_pred = 0
    true_ref = 0
    total_ref = 0

    matched_ref = {}

    for edge_matching in edge_matchings:
        if edge_matching[1] is None:
            continue
        entry = matched_ref.setdefault(edge_matching[1], [])

        edge_pred = edge_matching[0]
        a_pred = pred_graph.nodes[edge_pred[0]][location_attr]
        b_pred = pred_graph.nodes[edge_pred[1]][location_attr]

        edge_ref = edge_matching[1]
        a_ref = pred_graph.nodes[edge_ref[0] + offset][location_attr]
        b_ref = pred_graph.nodes[edge_ref[1] + offset][location_attr]

        if (edge_pred[0] >= offset) or (edge_pred[1] >= offset):
            entry.append(edge_matching[0])
        else:
            true_pred += np.linalg.norm((b_pred - a_pred))

    for u, v in pred_graph.edges:
        u_loc = pred_graph.nodes[u][location_attr]
        v_loc = pred_graph.nodes[v][location_attr]
        if u < offset and v < offset:
            total_pred += np.linalg.norm((u_loc - v_loc))

        elif u >= offset and v >= offset:
            total_ref += np.linalg.norm((u_loc - v_loc))

    true_ref = total_ref
    for edge, failed_matchings in matched_ref.items():
        if len(failed_matchings) > 0:
            a_ref = pred_graph.nodes[edge[0] + offset][location_attr]
            b_ref = pred_graph.nodes[edge[1] + offset][location_attr]
            true_ref -= np.linalg.norm((b_ref - a_ref))

    recall = true_ref / (total_ref)
    precision = true_pred / (total_pred)
    return recall, precision


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

    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))

    edges = [edges_deep, edges_right, edges_down]

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
