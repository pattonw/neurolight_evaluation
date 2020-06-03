import gunpowder as gp
import numpy as np
# from neurolight.gunpowder import RasterizeSkeleton, SwcFileSource


class NetworkxSource(SwcFileSource):

    def __init__(self, graph, graph_key):

        super(NetworkxSource, self).__init__(None, [graph_key])
        self._graph = graph

    def _read_points(self):
        self._graph_to_kdtree()


def rasterize_graph(
        graph,
        position_attribute,
        radius_pos,
        radius_tolerance,
        roi,
        voxel_size):
    '''Rasterizes a geometric graph into a numpy array.

    For that, the nodes in the graph are assumed to have a position in 3D (see
    parameter ``position_attribute``).

    The created array will have edges painted with 1, background with 0, and
    (optionally) a tolerance region around each edge with -1.

    Args:

        graph (networkx graph):

            The graph to rasterize. Nodes need to have a position attribute.

        position_attribute (string):

            The name of the position attribute of the nodes. The attribute
            should contain tuples of the form ``(z, y, x)`` in world units.

        radius_pos (float):

            The radius of the lines to draw for each edge in the graph (in
            world units).

        radius_tolerance (float):

            The radius of a region around each edge line that will be labelled
            with ``np.uint64(-1)``. Should be larger than ``radius_pos``. If
            set to ``None``, no such label will be produced.

        roi (gp.Roi):

            The ROI of the area to rasterize.

        voxel_size (tuple of int):

            The size of a voxel in the array to create, in world units.
    '''

    graph_key = gp.PointsKey('GRAPH')
    array = gp.ArrayKey('ARRAY')
    array_spec = gp.ArraySpec(voxel_size=voxel_size, dtype=np.uint64)

    pipeline_pos = (
        NetworkxSource(graph, graph_key) +
        RasterizeSkeleton(graph_key, array, array_spec, radius_pos)
        + GrowLabels(array, tolerance, tolerance_spec, radius_tolerance))

    request = gp.BatchRequest()
    request[array] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline_pos):
        batch = pipeline_pos.request_batch(request)
        return batch[array].data
