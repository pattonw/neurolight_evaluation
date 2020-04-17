import networkx as nx

from typing import Tuple
import logging

from .graph_score import score_graph

logger = logging.getLogger(__file__)


def score_tracings(
    predicted_tracings: nx.DiGraph,
    reference_tracings: nx.DiGraph,
    match_threshold: float,
    penalty_attr: str,
    location_attr: str,
) -> Tuple[float, float]:
    # Downsample the nodes in reference tracings by a factor of k to make sure predicted
    # tracings has higher resolution
    k = 10
    i = 0
    for node in list(reference_tracings.nodes()):
        if reference_tracings.degree(node) == 2:
            i += 1
            if i % k != 0:
                a, b = list(reference_tracings.succ[node]) + list(
                    reference_tracings.pred[node]
                )
                reference_tracings.remove_node(node)
                reference_tracings.add_edge(a, b)

    target_edge_len = 3000

    max_node = max([node for node in predicted_tracings])
    node_ids = itertools.counter(max_node + 1)

    for u, v in predicted_tracings.edges:
        u_loc = predicted_tracings.nodes[u][location_attr]
        v_loc = predicted_tracings.nodes[v][location_attr]

        edge_len = np.linalg.norm(u_loc - v_loc)
        k = edge_len // target_edge_len
        previous = u
        for i in range(k):
            interp_id = next(node_ids)
            interpolated_loc = u_loc + ((i + 0.5) / k) * (v_loc - u_loc)
            predicted_tracings.add_node(interp_id, location=interpolated_loc)
            predicted_tracings.add_edge(previous, interp_id)
            previous = interp_id
        predicted_tracings.remove_edge(u, v)
        predicted_tracings.add_edge(previous, v)

    
    return score_graph(
        predicted_tracings,
        reference_tracings,
        match_threshold,
        penalty_attr,
        location_attr,
    )
