import typing

import numpy

from .clusters_dist_ward import clusters_dist_ward

# Calculating stage distances (Ward, Centroid)
def calc_stage_distances(features: numpy.ndarray, st_edges: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    # Ward distances
    st_dist_ward = numpy.array([
        clusters_dist_ward(
            features[st_edges[i - 1]:st_edges[i]],
            features[st_edges[i]:st_edges[i + 1]]
        ) for i in range(1, len(st_edges) - 1)
    ])

    # Centroid distance
    st_dist_centr = numpy.array([
        numpy.linalg.norm(
            features[st_edges[i - 1]:st_edges[i]].mean(axis = 0)
            -
            features[st_edges[i]:st_edges[i + 1]].mean(axis = 0)
        ) for i in range(1, len(st_edges) - 1)
    ])

    return st_dist_ward, st_dist_centr
