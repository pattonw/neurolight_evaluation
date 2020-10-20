import networkx as nx
import numpy as np

import itertools


class Decision:
    """
    A decision is a Node.
    
    """


def review_error_area_under_curve(gt: nx.Graph, prediction: nx.Graph, config):

    pred_gt_matching, gt_pred_matching = match(gt, prediction, config.matching)

    unmatched_gt = set(
        [node for node in gt.nodes if gt_pred_matching.get(node) is None]
    )

    nodes = list(prediction.nodes)

    decision_points = []
    for node in nodes:
        if pred_gt_matching.get(node) is not None:
            matches_to_gt = True
        else:
            matches_to_gt = False

        if matches_to_gt:
            no_missing_neighbor = True
            for neighbor in gt.neighbors(pred_gt_matching.get(node)):
                if gt_pred_matching.get(neighbor) is None:
                    no_missing_neighbor = False
        else:
            no_missing_neighbor = True

        decision_points.append((matches_to_gt, no_missing_neighbor))

    f = lambda x: 1 - sum(itertools.chain(*x)) / (len(x) * 2)

    accuracy = []
    cost = []

    for i, node, decision in enumerate(zip(nodes, decision_points)):
        accuracy += f([(True, True)] * i + decision_points[i:])
        c = sum([edge_len(gt, e) / 2 for e in prediction.adj(node)])
        false_neg_cost, other_endpoints = false_neg_cable_len(
            gt, unmatched_gt, pred_gt_matching.get(node), gt_pred_matching
        )
        c += false_neg_cost
        cost.append(c)

    return accuracy, cost


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
