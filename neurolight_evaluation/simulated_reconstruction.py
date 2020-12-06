import networkx as nx
import numpy as np

import comatch

import itertools
from collections import defaultdict
from heapq import heappop, heappush
import logging
from typing import Set, Tuple, Dict
import random
from enum import Enum
import pickle

from .graph_matching.comatch.edges_xy import get_edges_xy
from .conf import ReconstructionConfig

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Interaction(Enum):
    INIT = 1
    REMOVE_FP = 2
    REMOVE_FM = 3
    RECONSTRUCT_FN = 4
    MERGE_FS = 5
    VISIT = 6


class Accuracy:
    """
    A class to keep track of prediction accuracy
    """

    def __init__(self, gt: nx.Graph, location_attr: str = "location", prediction=None):
        self.location_attr = location_attr

        self._interactions = []
        self._precision = []
        self._recall = []
        self._false_merges = []
        self._false_splits = []
        self._false_pos = []
        self._false_neg = []

        self.total_gt = self.cable_len(gt)
        self.gt = gt
        self.prediction = prediction

        self.reconstructions = []
        self.pred_matchings = None
        self.gt_matchings = None

    def plot(self):
        x = range(len(self.interactions))
        # plt.plot(x, self.interactions)
        plt.plot(x, np.array(self.precision))
        plt.plot(x, np.array(self.recall))
        # plt.plot(x, np.array(self.merges) / max(self.merges))
        # plt.plot(x, np.array(self.splits) / max(self.splits))
        # plt.plot(x, np.array(self.false_pos) / max(self.false_pos))
        # plt.plot(x, np.array(self.false_neg) / self.total_gt)
        plt.legend(
            [
                # "interactions",
                "precision",
                "recall",
                # "merges",
                # "splits",
                # "false_pos",
                # "false_neg",
            ]
        )
        plt.xlabel("interactions")
        plt.ylabel("normalized scores")

    def save(self, filename):
        save_data = {
            "interactions": self.interactions,
            "precision": self.precision,
            "recall": self.recall,
            "merges": self.merges,
            "splits": self.splits,
            "fps": self.false_pos,
            "fns": self.false_neg,
            "graphs": self.reconstructions,
            "pred_matchings": self.pred_matchings,
            "gt_matchings": self.gt_matchings,
            "gt": self.gt,
            "prediction": self.prediction,
        }
        pickle.dump(save_data, open(filename, "wb"))

    def init(self):
        self.interactions.append(Interaction.INIT)
        self.precision.append(1)
        self.recall.append(0)
        self.merges.append(0)
        self.splits.append(0)
        self.false_pos.append(0)
        self.false_neg.append(self.total_gt)

    def visit(self):

        self.interactions.append(self.interactions[-1])
        self.precision.append(self.precision[-1])
        self.recall.append(self.recall[-1])
        self.splits.append(self.splits[-1])
        self.merges.append(self.merges[-1])
        self.false_pos.append(self.false_pos[-1])
        self.false_neg.append(self.false_neg[-1])

    def update(
        self,
        interaction: Interaction,
        pred: nx.Graph,
        gt: nx.Graph,
        reconstructed_pred: Set[int],
        reconstructed_gt: Set[int],
        pred_matchings: Dict[int, int],
        gt_matchings: Dict[int, int],
    ):
        true_gt = self.cable_len(gt.subgraph(reconstructed_gt))
        total_pred = self.cable_len(pred.subgraph(reconstructed_pred))

        self.reconstructions.append(pred.subgraph(reconstructed_pred).copy())
        self.pred_matchings = pred_matchings
        self.gt_matchings = gt_matchings

        matched_reconstruction_nodes = set(
            [n for n in reconstructed_pred if len(pred_matchings[n]) > 0]
        )
        true_pred = self.cable_len(pred.subgraph(matched_reconstruction_nodes))

        fp_nodes = reconstructed_pred - matched_reconstruction_nodes
        fp_components = len(list(nx.connected_components(pred.subgraph(fp_nodes))))
        fn_cable_len = self.total_gt - true_gt
        if all([n in reconstructed_gt for n in gt.nodes]):
            assert fn_cable_len == 0

        num_splits = len(nx.algorithms.node_boundary(gt, reconstructed_gt))
        num_merges = len(
            nx.algorithms.node_boundary(pred, matched_reconstruction_nodes)
        )

        prec, recall, splits, merges, fp, fn = (
            (true_pred + 1e-6) / (total_pred + 1e-6),
            true_gt / self.total_gt,
            num_splits,
            num_merges,
            fp_components,
            fn_cable_len,
        )

        self.interactions.append(interaction)
        self.precision.append(prec)
        self.recall.append(recall)
        self.splits.append(splits)
        self.merges.append(merges)
        self.false_pos.append(fp)
        self.false_neg.append(fn)

    @property
    def interactions(self):
        return self._interactions

    @property
    def precision(self):
        return self._precision

    @property
    def recall(self):
        return self._recall

    @property
    def merges(self):
        return self._false_merges

    @property
    def splits(self):
        return self._false_splits

    @property
    def false_pos(self):
        return self._false_pos

    @property
    def false_neg(self):
        return self._false_neg

    def cable_len(self, graph):
        total = 0
        for u, v in graph.edges:
            u_loc = graph.nodes[u][self.location_attr]
            v_loc = graph.nodes[v][self.location_attr]
            total += np.linalg.norm(u_loc - v_loc)
        return total


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

        self.seed = seed

        # track progress
        self.visit_queue = list()

        self.accuracy = Accuracy(gt.copy(), prediction=prediction.copy())

        self.steps = 0

        self.fixed_nodes = set()

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

        # cleanup matchings:
        # a node in pred that matches to a gt component, will match to all
        # nodes in gt within a radius threshold.
        # This is unnecessary. It should only match to gt nodes that can't be
        # matched elsewhere
        # for pred_node, attrs in self.prediction.nodes.items():
        #     gt_matchings = list(self.pred_matchings[pred_node])
        #     pred_loc = attrs["location"]
        #     gt_locs = [self.gt.nodes[gt_node]["location"] for gt_node in gt_matchings]
        #     gt_matchings = [
        #         x[0]
        #         for x in sorted(list(zip(gt_matchings, gt_locs)), key=lambda x: np.linalg.norm(pred_loc - x[1]))
        #     ]
        #     remaining_matches = []
        #     for gt_node in gt_matchings:
        #         pred_matchings = self.gt_matchings[gt_node]
        #         if len(pred_matchings) < 2 or len(remaining_matches) < 1:
        #             remaining_matches.append(gt_node)
        #         else:
        #             self.gt_matchings[gt_node].remove(pred_node)
        #             self.pred_matchings[pred_node].remove(gt_node)

        num_nodes = self.prediction.number_of_nodes()
        for pred_node, gt_nodes in list(self.pred_matchings.items()):
            gt_subgraph = self.gt.subgraph(gt_nodes)
            if len(list(nx.connected_components(gt_subgraph))) > 1:
                self.split_node(pred_node, gt_nodes)
        num_split = self.prediction.number_of_nodes() - num_nodes
        print(f"Split {num_split} nodes!")

    def init_reconstruction(self):
        gt_root = self.closest_node(self.gt, self.seed)
        if gt_root is None:
            raise Exception(f"Can't start at {self.seed}, too far away from gt!")
        initial_nodes = []

        if len(initial_nodes) == 0:
            root_id = self.next_node_id()
            self._prediction.add_node(root_id, **self.gt.nodes[gt_root])
            self.gt_matchings[gt_root].add(root_id)
            self.pred_matchings[root_id].add(gt_root)
            initial_nodes.append(root_id)

        self.roots = []
        for node in initial_nodes:
            self.roots.append(node)
            heappush(self.visit_queue, (0, node))

        self.accuracy.init()

    def start(self):
        # initialize matching:
        logger.info("Matching graphs")
        self.match_graphs()

        logger.info("initializing reconstruction")
        self.init_reconstruction()

        logger.warning(
            f"starting reconstruction with "
            f"{self.prediction.number_of_nodes()} nodes and "
            f"{self.prediction.number_of_edges()} edges"
        )

        while not self.done():
            self.step()

    def step(self):
        confidence, dp_pred = self.next_node()
        self.fixed_nodes.add(dp_pred)
        self.update_accuracy(Interaction.VISIT)

        # A single pred node can match to multiple gt nodes
        dps_gt = self.pred_matchings[dp_pred]

        # case 1: if not visiting any gt, dp_pred is a false pos
        if len(dps_gt) == 0:
            self.remove_false_pos_node(dp_pred)
        elif len(list(nx.connected_components(self.gt.subgraph(dps_gt)))) > 1:
            raise Exception("Should not be reachable!")
        else:
            self.review_connectivity(dp_pred, dps_gt)

    def local_nodes(self, graph: nx.Graph, inside: Set[int]):
        """
        local_nodes are the set of all nodes in `inside` or adjacent to a
        a node in `inside` given the graph: `graph`.
        """
        boundary = nx.algorithms.boundary.node_boundary(graph, inside)
        local = boundary | inside
        return local

    @property
    def reconstruction_nodes(self):
        root_components = []
        for root in self.roots:
            root_component = nx.node_connected_component(self.prediction, root)
            root_components.append(self.prediction.subgraph(root_component).copy())
        reconstruction = nx.union_all(root_components)
        return reconstruction.nodes

    @property
    def reconstructed_nodes(self):
        reconstructed = set(
            itertools.chain(
                *[self.pred_matchings[n] for n in self.reconstruction_nodes]
            )
        )
        return reconstructed

    def review_connectivity(self, dp_pred, dps_gt):

        # all nodes in prediction that neighbor `dp_pred`
        local_pred = self.local_nodes(self.prediction, set([dp_pred]))

        # all nodes in gt that neighbor a node in `dps_gt`
        local_gt = self.local_nodes(self.gt, dps_gt)

        local_pred = self.prediction.subgraph(local_pred)
        local_gt = self.gt.subgraph(local_gt)

        logger.debug(
            f"iterating over {local_pred.number_of_edges()} local prediction edges"
        )

        for u, v in list(local_pred.edges()):
            if not (u == dp_pred or v == dp_pred):
                # only consider edges adjacent to decision point
                continue
            u_gt = self.pred_matchings[u]
            v_gt = self.pred_matchings[v]
            gt_cc = self.gt.subgraph(u_gt | v_gt)
            gt_ccs = list(nx.connected_components(gt_cc))
            if len(u_gt) == 0:
                self.remove_false_merge_edge(u, v)
            elif len(v_gt) == 0:
                self.remove_false_merge_edge(u, v)
            elif len(gt_ccs) == 2:
                self.remove_false_merge_edge(u, v)
            elif len(gt_ccs) > 2:
                raise NotImplementedError("This should be unreachable!")

        logger.debug(f"iterating over {local_gt.number_of_edges()} local gt edges")

        for g_u, g_v in local_gt.edges():
            assert g_u in dps_gt or g_v in dps_gt
            u_pred = self.gt_matchings[g_u]
            v_pred = self.gt_matchings[g_v]
            pred_cc = self.prediction.subgraph(u_pred | v_pred)
            pred_ccs = list(nx.connected_components(pred_cc))
            if len(u_pred) == 0:
                self.reconstruct_false_neg(dp_pred, g_u)
            elif len(v_pred) == 0:
                self.reconstruct_false_neg(dp_pred, g_v)
            elif len(pred_ccs) > 1:
                self.merge_false_split(g_u, g_v)

    def done(self):
        logger.debug(f"{len(self.visit_queue)} nodes left!")
        return len(self.visit_queue) == 0

    def split_node(self, pred_node, dps_gt):
        gt_subgraph = self.gt.subgraph(dps_gt).copy()
        for cc in nx.connected_components(gt_subgraph):
            cc = list(cc)

            new_loc = self.prediction.nodes[pred_node][
                self.config.comatch.location_attr
            ]
            new_node_id = self.next_node_id()
            self.prediction.add_node(
                new_node_id, **{self.config.comatch.location_attr: new_loc}
            )

            heappush(self.visit_queue, (0, new_node_id))

            for gt_node in cc:
                self.gt_matchings[gt_node].add(new_node_id)
                self.gt_matchings[gt_node].remove(pred_node)
                self.pred_matchings[new_node_id].add(gt_node)

            for neighbor in self.prediction.neighbors(pred_node):
                self.prediction.add_edge(new_node_id, neighbor)
        del self.pred_matchings[pred_node]

    def remove_false_pos_node(self, node):

        logger.debug(f"Removing false positive node {node}")
        self.prediction.remove_node(node)
        for gt_match in self.pred_matchings[node]:
            self.gt_matchings[gt_match].remove(node)
        del self.pred_matchings[node]
        self.rebuild_queue()

        self.update_accuracy(Interaction.REMOVE_FP)

    def rebuild_queue(self):
        new_queue = []
        purged = []
        reconstruction_nodes = self.reconstruction_nodes
        for confidence, node in self.visit_queue:
            if node in reconstruction_nodes:
                heappush(new_queue, (confidence, node))
            else:
                purged.append(node)

        logger.debug(
            f"purged {len(purged)} nodes from queue, queue now has {len(new_queue)} entries!"
        )
        self.visit_queue = new_queue
        return purged

    def remove_false_merge_edge(self, node, neighbor):
        logger.debug(f"Removing false merge edge {(node, neighbor)}")
        self.prediction.remove_edge(node, neighbor)
        self.rebuild_queue()

        self.update_accuracy(Interaction.REMOVE_FM)

    def reconstruct_false_neg(self, dp_pred, gt_neighbor):
        logger.debug(f"Reconstructing false negative {(dp_pred, gt_neighbor)}")
        current_loc = self.prediction.nodes[dp_pred][self.config.comatch.location_attr]
        target_loc = self.gt.nodes[gt_neighbor][self.config.comatch.location_attr]
        distance = np.linalg.norm(target_loc - current_loc)
        offset = target_loc - current_loc
        slope = offset / distance
        last_node_id = dp_pred
        while distance > self.config.comatch.match_threshold:
            new_loc = self.config.new_edge_len * slope + current_loc
            new_node_id = self.next_node_id()
            self.prediction.add_node(
                new_node_id, **{self.config.comatch.location_attr: new_loc}
            )
            self.fixed_nodes.add(new_node_id)
            self.gt_matchings[gt_neighbor].add(new_node_id)
            self.pred_matchings[new_node_id].add(gt_neighbor)
            self.prediction.add_edge(new_node_id, last_node_id)
            current_loc = new_loc
            distance -= self.config.new_edge_len
            last_node_id = new_node_id

            self.update_accuracy(Interaction.RECONSTRUCT_FN)

        new_loc = distance * slope + current_loc
        new_node_id = self.next_node_id()
        self.prediction.add_node(
            new_node_id, **{self.config.comatch.location_attr: new_loc}
        )
        self.gt_matchings[gt_neighbor].add(new_node_id)
        self.pred_matchings[new_node_id].add(gt_neighbor)
        self.prediction.add_edge(last_node_id, new_node_id)
        # logger.warning("Update accuracy. Either stays same or increases accuracy")
        heappush(self.visit_queue, (0, new_node_id))

        self.update_accuracy(Interaction.RECONSTRUCT_FN)

    def merge_false_split(self, g_u, g_v):
        logger.debug(f"merging false split between {g_u} and {g_v}")
        matched_nodes = self.gt_matchings[g_u] | self.gt_matchings[g_v]
        local_predictions = self.prediction.subgraph(matched_nodes)
        pred_ccs = list(nx.connected_components(local_predictions))

        # Greedy inefficient approach:
        edges = [
            (
                a,
                b,
                np.linalg.norm(
                    self.prediction.nodes[a][self.config.comatch.location_attr]
                    - self.prediction.nodes[b][self.config.comatch.location_attr]
                ),
            )
            for a, b in itertools.chain(
                *[
                    itertools.product(x, y)
                    for x, y in itertools.combinations(pred_ccs, 2)
                ]
            )
        ]
        edges = sorted(edges, key=lambda x: x[2])

        while len(pred_ccs) > 1:
            reconstruction_nodes = self.reconstruction_nodes
            u, v, _ = edges[0]
            edges = edges[1:]
            if v not in nx.node_connected_component(local_predictions, u):
                self.prediction.add_edge(u, v)
                total_cable_len = self.accuracy.cable_len(self.prediction)
                if u not in reconstruction_nodes and v in reconstruction_nodes:
                    for node in nx.node_connected_component(self.prediction, u):
                        cost = self.get_cost(self.prediction, node, total_cable_len)
                        heappush(self.visit_queue, (cost, node))
                if v not in reconstruction_nodes and u in reconstruction_nodes:
                    for node in nx.node_connected_component(self.prediction, v):
                        cost = self.get_cost(self.prediction, node, total_cable_len)
                        heappush(self.visit_queue, (cost, node))
            local_predictions = self.prediction.subgraph(matched_nodes)
            pred_ccs = list(nx.connected_components(local_predictions))

        self.update_accuracy(Interaction.MERGE_FS)

    def distance_to_root(self, graph, node):
        root = self.roots[0]

        path = nx.algorithms.shortest_paths.generic.shortest_path(
            graph, source=node, target=root
        )
        return self.accuracy.cable_len(graph.subgraph(path))

    def get_cost(self, graph, node, total_pred_cable_len):
        edges = list(nx.boundary.edge_boundary(graph, set([node]), data=True))
        if len(edges) != 2:
            return self.distance_to_root(graph, node) / total_pred_cable_len
        else:
            return 1 + min([attrs.get("distance", 1) for _, _, attrs in edges])

    def edge_len(self, graph, a, b):
        loc_a = graph.nodes[a][self.config.comatch.location_attr]
        loc_b = graph.nodes[a][self.config.comatch.location_attr]
        return np.linalg.norm(loc_a - loc_b)

    def update_accuracy(self, interaction):
        if interaction == Interaction.VISIT:
            self.accuracy.visit()
        else:
            self.accuracy.update(
                interaction,
                self.prediction,
                self.gt,
                self.reconstruction_nodes,
                self.reconstructed_nodes,
                self.pred_matchings,
                self.gt_matchings,
            )


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
