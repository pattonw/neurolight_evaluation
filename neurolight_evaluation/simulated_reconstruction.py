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

        self.gt_root_node = self.closest_node(gt, seed)

        # track progress
        self.visit_queue = list()

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

        for pred_node, gt_nodes in list(self.pred_matchings.items()):
            gt_subgraph = self.gt.subgraph(gt_nodes)
            if len(list(nx.connected_components(gt_subgraph))) > 1:
                self.split_node(pred_node, gt_nodes)

    def start(self):
        # initialize matching:
        self.match_graphs()

        gt_root = self.gt_root_node
        if gt_root is None:
            raise Exception("Can't start here, too far away from gt!")
        root_matchings = self.gt_matchings[gt_root]
        initial_nodes = []
        for cc in nx.connected_components(self.prediction):
            if any([n in cc for n in root_matchings]):
                initial_nodes += list(cc)

        if len(initial_nodes) == 0:
            root_id = self.next_node_id()
            self._prediction.add_node(root_id, **self.gt.nodes[gt_root])
            self.gt_matchings[gt_root].add(root_id)
            self.pred_matchings[root_id].add(gt_root)
            initial_nodes.append(root_id)

        self.roots = []
        for node in initial_nodes:
            self.roots.append(node)
            heappush(self.visit_queue, (random.random(), node))

        logger.warning(
            f"starting reconstruction with "
            f"{self.prediction.number_of_nodes()} nodes and "
            f"{self.prediction.number_of_edges()} edges"
        )

        while not self.done():
            self.step()

    def step(self):
        confidence, pred_visit = self.next_node()
        logger.info(f"Visiting {pred_visit}")

        # A single pred node can match to multiple gt nodes
        gt_visits = self.pred_matchings[pred_visit]

        # case 1: if not visiting any gt, pred_visit is a false pos
        if len(gt_visits) == 0:
            self.remove_false_pos_node(pred_visit)
        elif len(list(nx.connected_components(self.gt.subgraph(gt_visits)))) > 1:
            raise Exception("Should not be reachable!")
            self.split_node(pred_visit, gt_visits)
        else:
            changed = self.fix_connectivity(pred_visit, gt_visits)
            if changed:
                heappush(self.visit_queue, (0, pred_visit))

    def surface_area(self, graph: nx.Graph, inside: Set[int]):
        outside = nx.algorithms.boundary.node_boundary(graph, inside)
        area = outside | inside
        return outside, area

    @property
    def reconstruction_nodes(self):
        root_components = []
        for root in self.roots:
            root_component = nx.node_connected_component(self.prediction, root)
            root_components.append(self.prediction.subgraph(root_component).copy())
        reconstruction = nx.union_all(root_components)
        return reconstruction.nodes

    def fix_connectivity(self, pred_visit, gt_visits):

        # all nodes in prediction that neighbor `pred_visit`
        pred_visit_surface, pred_visit_area = self.surface_area(
            self.prediction, set([pred_visit])
        )

        # all nodes in gt that neighbor a node in `gt_visits`
        gt_visit_surface, gt_visit_area = self.surface_area(self.gt, gt_visits)

        # nodes A, and B, are diagonal if a neighbor of A matches to B
        # note that this is not comutative
        pred_diagonals = {n: self.pred_matchings[n] for n in pred_visit_surface}

        gt_diagonals = {n: self.gt_matchings[n] for n in gt_visit_surface}

        # pred_area_match should be a single connected component in gt. If not, something is wrong
        pred_area_match = set().union(
            *[self.pred_matchings[n] for n in pred_visit_area]
        )
        pred_area_match = self.gt.subgraph(pred_area_match)
        # gt_area_match should be a single connected component in pred, If not something is wrong
        gt_area_match = set().union(*[self.gt_matchings[n] for n in gt_visit_area])
        gt_area_match = self.prediction.subgraph(gt_area_match)
        # cases for diagonals:

        # If a member of `pred_visit_surface` has no matches, it is a false pos and
        # the edge is not needed.
        # If a member of `pred_visit_surface` has matches that are on a different connected
        # components of the gt subgraph containing pred_area_match than visit, then these
        # two nodes are connected locally in the prediction, but not locally in the gt
        # and must be split

        pred_visit_cc = nx.node_connected_component(pred_area_match, list(gt_visits)[0])
        assert all([n in pred_visit_cc for n in gt_visits])

        for pred_neighbor, gt_matchings in pred_diagonals.items():
            if len(gt_matchings) == 0:
                self.remove_false_pos_edge(pred_visit, pred_neighbor)
                return True
            elif all([n not in pred_visit_cc for n in gt_matchings]):
                self.remove_false_merge_edge(pred_visit, pred_neighbor)
                return True
            elif any([n not in pred_visit_cc for n in gt_matchings]):
                # ambiguous case: consider
                # prediction:    a --- b
                # matching:      |    / \
                # gt:            A----B C
                # Splitting and merging both make sense leading to an infinite loop.
                # in this case, first fix b, then come back and fix a.
                raise Exception(
                    "This case should be fixed via preprocessing. "
                    "No node in prediction is allowed to match to multiple nodes in gt."
                )

        # If a member of `gt_visit_surface` has no matches, it is a false neg and
        # should be reconstructed
        # If a member of `gt_visit_surface` has matches that are on a different connected
        # component of the pred subgraph containing gt_area_match than gt_visit, is locally
        # connected to a node in gt, that is not locally connected in pred and must be merged.

        gt_visit_cc = nx.node_connected_component(gt_area_match, pred_visit)

        for gt_neighbor, pred_matchings in gt_diagonals.items():
            if len(pred_matchings) == 0:
                self.reconstruct_false_neg(pred_visit, gt_neighbor)
                return True
            else:
                for pred_match in pred_matchings:
                    if pred_match not in gt_visit_cc:
                        self.merge_false_split(pred_visit, pred_match)
                        return True

        return False

    def done(self):
        logger.info(f"{len(self.visit_queue)} nodes left!")
        return len(self.visit_queue) == 0

    def split_node(self, pred_node, gt_visits):
        gt_subgraph = self.gt.subgraph(gt_visits).copy()
        for cc in nx.connected_components(gt_subgraph):
            cc = list(cc)

            new_loc = self.prediction.nodes[pred_node][
                self.config.comatch.location_attr
            ]
            new_node_id = self.next_node_id()
            self.prediction.add_node(
                new_node_id, **{self.config.comatch.location_attr: new_loc}
            )

            heappush(self.visit_queue, (random.random(), new_node_id))

            for gt_node in cc:
                self.gt_matchings[gt_node].add(new_node_id)
                self.gt_matchings[gt_node].remove(pred_node)
                self.pred_matchings[new_node_id].add(gt_node)

            for neighbor in self.prediction.neighbors(pred_node):
                self.prediction.add_edge(new_node_id, neighbor)
        del self.pred_matchings[pred_node]

    def remove_false_pos_node(self, node):
        logger.info(f"Removing false positive node {node}")
        self.prediction.remove_node(node)
        for gt_match in self.pred_matchings[node]:
            self.gt_matchings[gt_match].remove(node)
        del self.pred_matchings[node]
        self.purge_unneeded_components()
        # logger.warning("Update accuracy. Should increase precision")

    def purge_unneeded_components(self):
        self.rebuild_queue()

    def rebuild_queue(self):
        new_queue = []
        num_purged = 0
        reconstruction_nodes = self.reconstruction_nodes
        for confidence, node in self.visit_queue:
            if node in reconstruction_nodes:
                heappush(new_queue, (confidence, node))
            else:
                num_purged += 1

        logger.warning(f"purged {num_purged} nodes from queue")
        self.visit_queue = new_queue

    def remove_false_pos_edge(self, node, neighbor):
        logger.info(f"Removing false positive edge {(node, neighbor)}")
        self.prediction.remove_edge(node, neighbor)
        self.purge_unneeded_components()
        # logger.warning("Update accuracy. Should increase precision")

    def remove_false_merge_edge(self, node, neighbor):
        logger.info(f"Removing false merge edge {(node, neighbor)}")
        self.prediction.remove_edge(node, neighbor)
        self.purge_unneeded_components()
        # logger.warning(
        #     "Update accuracy. Should increase precision and reduce merge errors"
        # )

    def reconstruct_false_neg(self, pred_visit, gt_neighbor):
        logger.info(f"Reconstructing false negative {(pred_visit, gt_neighbor)}")
        current_loc = self.prediction.nodes[pred_visit][
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
            self.prediction.add_node(
                new_node_id, **{self.config.comatch.location_attr: new_loc}
            )
            self.gt_matchings[gt_neighbor].add(new_node_id)
            self.pred_matchings[new_node_id].add(gt_neighbor)
            self.prediction.add_edge(new_node_id, last_node_id)
            current_loc = new_loc
            distance -= self.config.new_edge_len
            last_node_id = new_node_id
            # logger.warning("Update accuracy. Either stays same or increases accuracy")

        new_loc = distance * slope + current_loc
        new_node_id = self.next_node_id()
        self.prediction.add_node(
            new_node_id, **{self.config.comatch.location_attr: new_loc}
        )
        self.gt_matchings[gt_neighbor].add(new_node_id)
        self.pred_matchings[new_node_id].add(gt_neighbor)
        self.prediction.add_edge(last_node_id, new_node_id)
        # logger.warning("Update accuracy. Either stays same or increases accuracy")
        heappush(self.visit_queue, (random.random(), new_node_id))

    def merge_false_split(self, contained, not_contained):
        logger.info(f"Merging false split {(contained, not_contained)}")
        cc = nx.node_connected_component(self.prediction, not_contained)

        num_added = 0
        if contained not in cc:
            for node in cc:
                num_added += 1
                heappush(self.visit_queue, (random.random(), node))

        logger.info(f"Added {num_added} new nodes to the visit queue")
        self.prediction.add_edge(contained, not_contained)


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
