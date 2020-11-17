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

        for node in self.reconstruction_nodes:
            assert node in self.fixed_nodes
            matches = self.pred_matchings[node]
            _, match_area = self.surface_area(self.gt, matches)
            assert len(list(nx.connected_components(self.gt.subgraph(matches)))) == 1, f"Node {node} is not done"
            gt_matches = set().union(*[self.gt_matchings[gt_node] for gt_node in matches])
            gt_surface_matches = set().union(*[self.gt_matchings[gt_node] for gt_node in match_area])
            num_ccs = len(list(nx.connected_components(self.prediction.subgraph(gt_matches))))
            num_surface_ccs = len(list(nx.connected_components(self.prediction.subgraph(gt_surface_matches))))
            if num_ccs != 1:
                logger.info(f"gt_nodes {gt_matches}, matched to {node} failed and matched to {num_ccs} ccs and {num_surface_ccs} surface_ccs")
                if num_surface_ccs != 1:
                    heappush(self.visit_queue, (0, node))
                    self.step()
                    raise Exception("Done!")


    def step(self):
        confidence, pred_visit = self.next_node()
        logger.info(f"Visiting {pred_visit}")
        logger.info(f"current reconstruction nodes: {self.reconstruction_nodes}")

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
                logger.info(f"pred_visit was changed, check it again:")
                heappush(self.visit_queue, (0, pred_visit))
            elif changed is None:
                logger.info(f"Nothing wrong with {pred_visit}")
                self.fixed_nodes.add(pred_visit)
            else:
                logger.info(f"{pred_visit} was removed")

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

        pred_visit_area = self.prediction.subgraph(pred_visit_area)
        gt_visit_area = self.gt.subgraph(gt_visit_area)

        for u, v in pred_visit_area.edges():
            if not (u == pred_visit or v == pred_visit):
                continue
            u_gt = self.pred_matchings[u]
            v_gt = self.pred_matchings[v]
            gt_cc = self.gt.subgraph(u_gt | v_gt)
            gt_ccs = list(nx.connected_components(gt_cc))
            if len(u_gt) == 0:
                self.remove_false_merge_edge(u, v)
                return True
            elif len(v_gt) == 0:
                self.remove_false_merge_edge(u, v)
                return True
            elif len(gt_ccs) == 2:
                self.remove_false_merge_edge(u, v)
                return True
            elif len(gt_ccs) > 2:
                raise NotImplementedError("This should be unreachable!")

        for g_u, g_v in gt_visit_area.edges():
            assert g_u in gt_visits or g_v in gt_visits
            u_pred = self.gt_matchings[g_u]
            v_pred = self.gt_matchings[g_v]
            pred_cc = self.prediction.subgraph(u_pred | v_pred)
            pred_ccs = list(nx.connected_components(pred_cc))
            if len(u_pred) == 0:
                self.reconstruct_false_neg(pred_visit, g_u)
                return True
            elif len(v_pred) == 0:
                self.reconstruct_false_neg(pred_visit, g_v)
                return True
            elif len(pred_ccs) > 1:
                self.merge_false_split(g_u, g_v)
                return True

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
        self.rebuild_queue()
        # logger.warning("Update accuracy. Should increase precision")

    def rebuild_queue(self):
        new_queue = []
        num_purged = 0
        reconstruction_nodes = self.reconstruction_nodes
        for confidence, node in self.visit_queue:
            if node in reconstruction_nodes:
                heappush(new_queue, (confidence, node))
            else:
                num_purged += 1

        logger.warning(f"purged {num_purged} nodes from queue, queue now has {len(new_queue)} entries!")
        self.visit_queue = new_queue

    def remove_false_pos_edge(self, node, neighbor):
        logger.info(f"Removing false positive edge {(node, neighbor)}")
        self.prediction.remove_edge(node, neighbor)
        self.rebuild_queue()
        # logger.warning("Update accuracy. Should increase precision")

    def remove_false_merge_edge(self, node, neighbor):
        logger.info(f"Removing false merge edge {(node, neighbor)}")
        self.prediction.remove_edge(node, neighbor)
        self.rebuild_queue()
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
            self.fixed_nodes.add(new_node_id)
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

    def merge_false_split(self, g_u, g_v):
        logger.info(f"merging false split between {g_u} and {g_v}")
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
                if u not in reconstruction_nodes and v in reconstruction_nodes:
                    for node in nx.node_connected_component(self.prediction, u):
                        heappush(self.visit_queue, (random.random(), node))
                if v not in reconstruction_nodes and u in reconstruction_nodes:
                    for node in nx.node_connected_component(self.prediction, v):
                        heappush(self.visit_queue, (random.random(), node))
                self.prediction.add_edge(u, v)
            local_predictions = self.prediction.subgraph(matched_nodes)
            pred_ccs = list(nx.connected_components(local_predictions))

        assert len(list(nx.connected_components(self.prediction.subgraph(matched_nodes)))) == 1


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
