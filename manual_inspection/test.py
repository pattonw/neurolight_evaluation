from volumetric_noise import generate_fractal_noise_3d

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from maximin import maximin_tree_query

from funlib.math import cantor_number
import networkx as nx

from neurolight_evaluation.graph_score import score_graph
from neurolight_evaluation.graph_metrics import Metric
from neurolight_evaluation.simulated_reconstruction import SimulatedTracer
from neurolight_evaluation.conf import ReconstructionConfig

# from neuroglancer_graphs import add_graph
# import neuroglancer

from omegaconf import OmegaConf

from matplotlib import pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)

random_seed = 0
ps = [0.9, 0.7, 0.5, 0.3, 0.1]
voxel_size = (4, 4, 4)
volume_size = (128, 128, 128)
dims = len(voxel_size)

graphs = []

np.random.seed(random_seed)

noisy_vol_1 = generate_fractal_noise_3d(volume_size, voxel_size, octaves=3,)
neighborhood = generate_binary_structure(dims, dims)
peaks = maximum_filter(noisy_vol_1, footprint=neighborhood) == noisy_vol_1

edge_scores = maximin_tree_query(
    noisy_vol_1, peaks.astype(np.uint8), decimate=False, threshold=0.4
)

gt = nx.Graph()
for u, v, score in edge_scores:
    u_id = cantor_number(u)
    u_loc = u * np.array(voxel_size) + np.array(voxel_size) / 2
    v_id = cantor_number(v)
    v_loc = v * np.array(voxel_size) + np.array(voxel_size) / 2

    gt.add_node(u_id, location=u_loc)
    gt.add_node(v_id, location=v_loc)

    gt.add_edge(u_id, v_id, distance=score)

noisy_vol_2 = generate_fractal_noise_3d(volume_size, voxel_size, octaves=3,)
for p in ps:
    noisy_vol = p * noisy_vol_1 + (1 - p) * noisy_vol_2
    neighborhood = generate_binary_structure(dims, dims)
    peaks = maximum_filter(noisy_vol, footprint=neighborhood) == noisy_vol

    edge_scores = maximin_tree_query(
        noisy_vol, peaks.astype(np.uint8), decimate=False, threshold=0.4
    )

    prediction = nx.Graph()
    for u, v, score in edge_scores:
        u_id = cantor_number(u)
        u_loc = u * np.array(voxel_size) + np.array(voxel_size) / 2
        v_id = cantor_number(v)
        v_loc = v * np.array(voxel_size) + np.array(voxel_size) / 2

        prediction.add_node(u_id, location=u_loc)
        prediction.add_node(v_id, location=v_loc)

        prediction.add_edge(u_id, v_id, distance=score)
    graphs.append((gt, prediction))

for gt, prediction in graphs:
    config = OmegaConf.structured(ReconstructionConfig)
    config.comatch.match_threshold = 10

    reconstructions = []
    ccs = nx.connected_components(gt)
    ccs = sorted(ccs, key=lambda x: -len(x))
    for cc in ccs:
        seed_node = min(cc)
        seed_loc = gt.nodes[seed_node]["location"]

        component_subgraph = gt.subgraph(cc).copy()
        sim_tracer = SimulatedTracer(component_subgraph, prediction, seed_loc, config)
        logging.info(f"Starting reconstruction")
        sim_tracer.start()
        logging.info(f"Finished reconstruction")
        reconstructions.append(
            sim_tracer.prediction.subgraph(sim_tracer.reconstruction_nodes).copy()
        )
        print(f"number of nodes in component: {component_subgraph.number_of_nodes()}")
        print(f"number of nodes in prediction: {prediction.number_of_nodes()}")
        logging.info(
            f"Number of nodes in reconstruction: {reconstructions[-1].number_of_nodes()}"
        )
        logging.info(
            f"Number of nodes visited: {len(sim_tracer.fixed_nodes)}"
        )
        # sim_tracer.accuracy.plot()
        # plt.show()

        # check_reconstruction_accuracy:
        edit_dist, (fp, fn, merge, split) = score_graph(
            reconstructions[-1],
            component_subgraph,
            config.comatch.match_threshold,
            "location",
            Metric.GRAPH_EDIT,
            node_spacing=1,
            details=True,
        )
        assert edit_dist == 0
        assert len(sim_tracer.fixed_nodes) >= reconstructions[-1].number_of_nodes(), f"simulated tracer only visited {len(sim_tracer.fixed_nodes)} of {reconstructions[-1].number_of_nodes()} nodes!"

        sim_tracer.accuracy.plot()
        plt.show()
        break

reconstruction = nx.disjoint_union_all(reconstructions)

logging.info(f"Number of nodes in reconstruction: {reconstruction.number_of_nodes()}")

attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": (1, 1, 1)}
# dimensions = neuroglancer.CoordinateSpace(**attrs)
# 
# viewer = neuroglancer.Viewer()
# viewer.dimensions = dimensions
# with viewer.txn() as s:
#     for i, (gt, prediction) in enumerate(graphs):
#         add_graph(s, gt, name=f"gt_{i}", visible=True, graph_dimensions=dimensions)
#         add_graph(
#             s,
#             prediction,
#             name=f"prediction_{i}",
#             visible=True,
#             graph_dimensions=dimensions,
#         )
#         add_graph(
#             s,
#             reconstruction,
#             name="reconstruction_0",
#             visible=True,
#             graph_dimensions=dimensions,
#         )
# 
# print(viewer)
input("Hit ENTER to close!")
