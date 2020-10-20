from volumetric_noise import generate_fractal_noise_3d

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from maximin import maximin_tree_query

from funlib.math import cantor_number
import networkx as nx

from neuroglancer_graphs import add_graph
import neuroglancer

num_graphs = 5
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

    g = nx.Graph()
    for u, v, score in edge_scores:
        u_id = cantor_number(u)
        u_loc = u * np.array(voxel_size) + np.array(voxel_size) / 2
        v_id = cantor_number(v)
        v_loc = v * np.array(voxel_size) + np.array(voxel_size) / 2

        g.add_node(u_id, location=u_loc)
        g.add_node(v_id, location=v_loc)

        g.add_edge(u_id, v_id, distance=score)
    graphs.append(g)


attrs = {"names": ["z", "y", "x"], "units": "nm", "scales": (1, 1, 1)}
dimensions = neuroglancer.CoordinateSpace(**attrs)

viewer = neuroglancer.Viewer()
viewer.dimensions = dimensions
with viewer.txn() as s:
    for i, g in enumerate(graphs):
        add_graph(s, g, name="graph_{i}", visible=True, graph_dimensions=dimensions)

print(viewer)
input("Hit ENTER to close!")