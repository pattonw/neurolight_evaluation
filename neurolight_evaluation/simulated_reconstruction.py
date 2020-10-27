import networkx as nx
import numpy as np

import itertools
from heapq import heappop, heappush


class Decision:
    """
    A decision is a Node.
    
    """


class Accuracy:
    """
    A class to keep track of prediction accuracy
    """


def simulated_reconstruction(
    gt: nx.Graph, prediction: nx.Graph, seed: np.ndarray, config
):

    outside = set([node for node in prediction])

    # Generate a fake node not in the predictions. This will be placed at the
    # root of the gt neuron and will be "grown" along the prediction until perfect.
    root = min(outside) - 1
    prediction.add_node(root, location=seed)

    # initialization
    accuracy = Accuracy()
    reconstruction_nodes = set()
    visited = []
    to_visit = [(0, root)]

    pred_to_gt, gt_to_pred = match(gt, prediction)

    while len(to_visit) > 0:
        confidence, pred_visit = heappop(to_visit)

        # A single pred node can match to multiple gt nodes
        gt_visits = pred_to_gt[pred_visit]

        # case 1: if not visiting any gt, pred_visit is a false pos
        if len(gt_visits) == 0:
            handle_false_pos_node(prediction, pred_visit, accuracy)
            continue

        # all nodes in prediction that neighbor `pred_visit`
        pred_visit_surface = nx.algorithms.boundary.node_boundary(
            prediction, [pred_visit]
        )
        pred_visit_area = set([pred_visit]).union(pred_visit_surface)

        # all nodes in gt that neighbor a node in `gt_visits`
        gt_visit_surface = nx.algorithms.boundary.node_boundary(gt, gt_visits)
        gt_visit_area = gt_visits.union(gt_visit_surface)

        # nodes A, and B, are diagonal if a neighbor of A matches to B
        # note that this is not comutative
        pred_diagonals = {n: pred_to_gt[n] for n in pred_visit_surface}

        gt_diagonals = {n: gt_to_pred[n] for n in gt_visit_surface}

        # cases for diagonals:

        # If a member of `pred_visit_surface` has no matches, it is a false pos and
        # the edge is not needed.
        # If a member of `pred_visit_surface` has matches that are not part of the gt_visit_area,
        # that member is a false merge and the edge is not needed. (penalizes low res
        # reconstructions since large edges will always be considered false merges if
        # they skip a ground truth node.)

        for pred_neighbor, gt_matchings in pred_diagonals.items():
            if len(gt_matchings) == 0:
                handle_false_pos_edge(prediction, pred_visit, pred_neighbor, accuracy)
            elif any([n not in gt_visit_area for n in gt_matchings]):
                handle_false_merge_edge(prediction, pred_visit, pred_neighbor, accuracy)

        # If a member of `gt_visit_surface` has no matches, it is a false neg and
        # should be reconstructed
        # If a member of `gt_visit_surface` has matches that are not part of the
        # current_reconstruction, that component is a false split and should be merged

        for gt_neighbor, pred_matchings in gt_diagonals.items():
            if len(pred_matchings) == 0:
                handle_false_neg(prediction, gt, pred_visit, gt_neighbor, accuracy)
            elif any([n not in reconstruction_nodes for n in pred_matchings]):
                handle_false_split(
                    prediction,
                    pred_visit,
                    [n for n in pred_matchings if n not in reconstruction_nodes],
                )

    return accuracy


def handle_false_pos_node(prediction, node, accuracy):
    prediction.remove_node(node)
    raise NotImplementedError("Update accuracy. Should increase precision")


def handle_false_pos_edge(prediction, node, neighbor, accuracy):
    prediction.remove_edge((node, neighbor))
    raise NotImplementedError("Update accuracy. Should increase precision")


def handle_false_merge_edge(prediction, node, neighbor, accuracy):
    prediction.remove_edge((node, neighbor))
    raise NotImplementedError(
        "Update accuracy. Should increase precision and reduce merge errors"
    )


def handle_false_neg(prediction, gt, pred_visit, gt_neighbor, accuracy):
    raise NotImplementedError("Add node near gt_neighbor")


def handle_false_split(prediction, contained, not_contained):
    raise NotImplementedError(
        "Add edges to make sure pred_matchings are all part of the same connected component as pred_visit"
    )


def edge_len(gt, e):
    u, v = e
    u_loc = gt.nodes[u]["location"]
    v_loc = gt.nodes[v]["location"]
    return np.linalg.norm(u_loc - v_loc)


def false_neg_cable_len(gt, unmatched_gt, false_neg_roots, gt_pred_matching):
    boundary = nx.algorithms.boundary.node_boundary(gt, false_neg_roots)
    false_neg_boundary = set()
    for node in boundary:
        if gt_pred_matching.get(node) is None:
            false_neg_boundary.add(node)

    false_neg_subtrees = []
    for node in false_neg_boundary:
        false_neg_subtrees.append(
            list(nx.algorithms.traversal.dfs_edges(unmatched_gt, node))
        )

    matched_boundary = set()
    cable_len = 0
    for false_neg_tree in false_neg_subtrees:
        cable_len += sum([edge_len(gt, e) for e in false_neg_tree])
        tree_nodes = set(itertools.chain(*false_neg_tree))
        matched_boundary = matched_boundary.union(
            set(nx.algorithms.boundary.node_boundary(gt, tree_nodes))
        )
    return cable_len, matched_boundary


def match(gt, prediction, config):
    """
    perform comatch
    """
    return matching
