import networkx as nx
import numpy as np

import comatch

import itertools
from collections import defaultdict
from heapq import heappop, heappush
import logging
from typing import Set, Tuple
import random

from .graph_matching.comatch.edges_xy import get_edges_xy
from .conf import ReconstructionConfig

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Accuracy:
    """
    A class to keep track of prediction accuracy
    """

    def plot(self):
        x = range(10)
        y = range(10)
        plt.plot(x, y)


class SimulatedTracer:
    def __init__(
        self,
        gt: nx.Graph,
        prediction: nx.Graph,
        seed: np.ndarray,
        config: ReconstructionConfig,
    ):
        self._gt = gt
        self._prediction = prediction.copy()
        self._next_node_id = max([-1] + [node for node in prediction.nodes])
        self._config = config

        self._reconstruction = nx.Graph()

        self.gt_root_node = self.closest_node(gt, seed)

        # track progress
        self.visit_queue = list()
        for node in self._reconstruction.nodes():
            heappush(self.visit_queue, (random.random(), node))

        self.accuracy = Accuracy()

    def next_node_id(self):
        self._next_node_id += 1
        return self._next_node_id

    @property
    def gt(self) -> nx.Graph:
        return self._gt

    @property
    def prediction(self) -> nx.Graph:
        return self._prediction

    @property
    def reconstruction(self) -> nx.Graph:
        return self._reconstruction

    @property
    def config(self) -> ReconstructionConfig:
        return self._config

    def gt_matchings(self, node_id) -> Set[int]:
        return self._gt_to_pred[node_id]

    def pred_matchings(self, node_id) -> Set[int]:
        return self._pred_to_gt[node_id]

    def next_node(self) -> Tuple[float, int]:
        return heappop(self.visit_queue)

    def closest_node(self, graph, location):
        closest_dist = self.config.comatch.match_threshold * 2
        closest_node = None
        for node, attrs in graph.nodes.items():
            dist = np.linalg.norm(attrs[self.config.comatch.location_attr] - location)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist

        return closest_node

    def match_graphs(self):
        # Match the graphs:
        edges_xy = get_edges_xy(
            self.prediction,
            self.gt,
            self.config.comatch.location_attr,
            self.config.comatch.match_threshold,
        )

        nodes_x = list(self.prediction.nodes)
        nodes_y = list(self.gt.nodes)
        node_labels_x = {
            node: cc
            for cc, cc_nodes in enumerate(nx.connected_components(self.prediction))
            for node in cc_nodes
        }
        node_labels_y = {
            node: cc
            for cc, cc_nodes in enumerate(nx.connected_components(self.gt))
            for node in cc_nodes
        }

        (
            label_matches,
            node_matches,
            splits,
            merges,
            fps,
            fns,
        ) = comatch.match_components(
            nodes_x, nodes_y, edges_xy, node_labels_x, node_labels_y
        )

        self.pred_matchings, self.gt_matchings = defaultdict(set), defaultdict(set)
        for pred_node, gt_node in node_matches:
            self.pred_matchings[pred_node].add(gt_node)
            self.gt_matchings[gt_node].add(pred_node)

    def start(self):
        # initialize matching:
        self.match_graphs()

        gt_root = self.gt_root_node
        if gt_root is None:
            raise Exception("Can't start here, too far away from gt!")
        root_matchings = self.pred_matchings[gt_root]
        initial_components = []
        for cc in nx.connected_components(self.prediction):
            if any([n in cc for n in root_matchings]):
                initial_components.append(self.prediction.subgraph(cc).copy())
        if len(initial_components) > 0:
            self._reconstruction = nx.disjoint_union_all(initial_components)
        else:
            root_id = self.next_node_id()
            self._reconstruction.add_node(root_id, **self.gt.nodes[gt_root])
            self.gt_matchings[gt_root].add(root_id)
            self.pred_matchings[root_id].add(gt_root)

        self.roots = []
        for node in self._reconstruction:
            self.roots.append(node)
            heappush(self.visit_queue, (random.random(), node))

        logger.warning(
            f"starting reconstruction with reconstruction with "
            f"{self.reconstruction.number_of_nodes()} nodes and "
            f"{self.reconstruction.number_of_edges()} edges"
        )

        while not self.done():
            self.step()

    def step(self):
        confidence, pred_visit = self.next_node()

        # A single pred node can match to multiple gt nodes
        gt_visits = self.pred_matchings[pred_visit]

        # case 1: if not visiting any gt, pred_visit is a false pos
        if len(gt_visits) == 0:
            self.remove_false_pos_node(pred_visit)

        else:
            self.fix_connectivity(pred_visit, gt_visits)

    def surface_area(self, graph: nx.Graph, inside: Set[int]):
        outside = nx.algorithms.boundary.node_boundary(graph, inside)
        area = outside | inside
        return outside, area

    def fix_connectivity(self, pred_visit, gt_visits):

        # all nodes in prediction that neighbor `pred_visit`
        pred_visit_surface, pred_visit_area = self.surface_area(
            self.reconstruction, set([pred_visit])
        )
        for node in pred_visit_surface:
            assert (node, pred_visit) in self.reconstruction.edges

        # all nodes in gt that neighbor a node in `gt_visits`
        gt_visit_surface, gt_visit_area = self.surface_area(self.gt, gt_visits)

        # nodes A, and B, are diagonal if a neighbor of A matches to B
        # note that this is not comutative
        pred_diagonals = {n: self.pred_matchings[n] for n in pred_visit_surface}

        gt_diagonals = {n: self.gt_matchings[n] for n in gt_visit_surface}

        # cases for diagonals:

        # If a member of `pred_visit_surface` has no matches, it is a false pos and
        # the edge is not needed.
        # If a member of `pred_visit_surface` has matches that are not part of the gt_visit_area,
        # that member is a false merge and the edge is not needed. (penalizes low res
        # reconstructions since large edges will always be considered false merges if
        # they skip a ground truth node.)

        for pred_neighbor, gt_matchings in pred_diagonals.items():
            if len(gt_matchings) == 0:
                self.remove_false_pos_edge(pred_visit, pred_neighbor)
            elif any([n not in gt_visit_area for n in gt_matchings]):
                self.remove_false_merge_edge(pred_visit, pred_neighbor)

        # If a member of `gt_visit_surface` has no matches, it is a false neg and
        # should be reconstructed
        # If a member of `gt_visit_surface` has matches that are not part of the
        # current_reconstruction, that component is a false split and should be merged

        for gt_neighbor, pred_matchings in gt_diagonals.items():
            if len(pred_matchings) == 0:
                self.reconstruct_false_neg(pred_visit, gt_neighbor)
            for pred_match in pred_matchings:
                if pred_match not in self.reconstruction.nodes:
                    self.merge_false_split(
                        pred_visit, pred_match,
                    )

    def done(self):
        logger.info(f"{len(self.visit_queue)} nodes left!")
        return len(self.visit_queue) == 0

    def remove_false_pos_node(self, node):
        self.reconstruction.remove_node(node)
        self.purge_unneeded_components()
        logger.warning("Update accuracy. Should increase precision")

    def purge_unneeded_components(self):
        for cc in list(nx.connected_components(self.reconstruction)):
            if not any([n in cc for n in self.roots]):
                for node in cc:
                    self.reconstruction.remove_node(node)
        self.rebuild_queue()

    def rebuild_queue(self):
        new_queue = []
        num_purged = 0
        for confidence, node in self.visit_queue:
            if node in self.reconstruction:
                heappush(new_queue, (confidence, node))
            else:
                num_purged += 1

        logger.warning(f"purged {num_purged} nodes from queue")
        self.visit_queue = new_queue

    def remove_false_pos_edge(self, node, neighbor):
        self.reconstruction.remove_edge(node, neighbor)
        self.purge_unneeded_components()
        logger.warning("Update accuracy. Should increase precision")

    def remove_false_merge_edge(self, node, neighbor):
        self.reconstruction.remove_edge(node, neighbor)
        logger.warning(
            "Update accuracy. Should increase precision and reduce merge errors"
        )

    def reconstruct_false_neg(self, pred_visit, gt_neighbor):
        logger.warning(f"Fixing False Neg")
        current_loc = self.reconstruction.nodes[pred_visit][
            self.config.comatch.location_attr
        ]
        target_loc = self.gt.nodes[gt_neighbor][self.config.comatch.location_attr]
        distance = np.linalg.norm(target_loc - current_loc)
        offset = target_loc - current_loc
        slope = offset / distance
        last_node_id = pred_visit
        while distance > self.config.comatch.match_threshold:
            new_loc = self.config.new_edge_len * slope + current_loc
            new_node_id = self.next_node_id()
            self.reconstruction.add_node(
                new_node_id, **{self.config.comatch.location_attr: new_loc}
            )
            self.reconstruction.add_edge(new_node_id, last_node_id)
            current_loc = new_loc
            distance -= self.config.new_edge_len
            last_node_id = new_node_id
            logger.warning("Update accuracy. Either stays same or increases accuracy")

        new_loc = distance * slope + current_loc
        new_node_id = self.next_node_id()
        self.reconstruction.add_node(
            new_node_id, **{self.config.comatch.location_attr: new_loc}
        )
        self.reconstruction.add_edge(new_node_id, last_node_id)
        current_loc = new_loc
        distance -= distance
        last_node_id = new_node_id
        logger.warning("Update accuracy. Either stays same or increases accuracy")
        heappush(self.visit_queue, (random.random(), last_node_id))

        self.gt_matchings[gt_neighbor].add(last_node_id)
        self.pred_matchings[last_node_id].add(gt_neighbor)

    def merge_false_split(self, contained, not_contained):
        cc = nx.node_connected_component(self.prediction, not_contained)
        if contained not in cc:
            cc_subgraph = self.prediction.subgraph(cc).copy()
            self._reconstruction = nx.union(self.reconstruction, cc_subgraph)
            for node in cc_subgraph.nodes:
                heappush(self.visit_queue, (random.random(), node))
        self.reconstruction.add_edge(contained, not_contained)


"""
def edge_len(gt, e):
    u, v = e
    u_loc = gt.nodes[u]["location"]
    v_loc = gt.nodes[v]["location"]
    return np.linalg.norm(u_loc - v_loc)
"""


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
