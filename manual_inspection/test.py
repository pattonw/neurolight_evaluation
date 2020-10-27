from volumetric_noise import generate_fractal_noise_3d

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from maximin import maximin_tree_query

from funlib.math import cantor_number
import networkx as nx

from neurolight_evaluation.simulated_reconstruction import simulated_reconstruction

from neuroglancer_graphs import add_graph
import neuroglancer

from matplotlib import pyplot as plt

num_graphs = 1
p = 0.9
voxel_size = (4, 4, 4)
volume_size = (128, 128, 128)
dims = len(voxel_size)

graphs = []

for i in range(num_graphs):
    np.random.seed(i)

    noisy_vol = generate_fractal_noise_3d(volume_size, voxel_size, octaves=3,)
    neighborhood = generate_binary_structure(dims, dims)
    peaks = maximum_filter(noisy_vol, footprint=neighborhood) == noisy_vol

    edge_scores = maximin_tree_query(
        noisy_vol, peaks.astype(np.uint8), decimate=False, threshold=0.4
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
    noisy_vol = p * noisy_vol + (1 - p) * noisy_vol_2
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

gt, prediction = graphs[0]

for cc in nx.connected_components(gt):
    seed_node = min(cc)
    seed_loc = gt.nodes[seed_node]["location"]

    component_subgraph = gt.subgraph(cc).copy()
    reconstruction_accuracy = simulated_reconstruction(gt, prediction, seed_loc, None)
    reconstruction_accuracy.plot()
    plt.show()


attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": (1, 1, 1)}
dimensions = neuroglancer.CoordinateSpace(**attrs)

viewer = neuroglancer.Viewer()
viewer.dimensions = dimensions
with viewer.txn() as s:
    for i, (gt, prediction) in enumerate(graphs):
        add_graph(s, gt, name=f"gt_{i}", visible=True, graph_dimensions=dimensions)
        add_graph(
            s,
            prediction,
            name=f"prediction_{i}",
            visible=True,
            graph_dimensions=dimensions,
        )

print(viewer)
input("Hit ENTER to close!")
