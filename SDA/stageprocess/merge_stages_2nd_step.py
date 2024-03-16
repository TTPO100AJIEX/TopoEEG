import numpy

from .calc_stage_distances import calc_stage_distances

# Merge stages if length > n_stages
def merge_stages_2nd_step(features: numpy.ndarray, st_edges: numpy.ndarray, dist_threshold: float) -> numpy.ndarray:
    if len(st_edges) <= 2: 
        return st_edges
    
    st_dist_list = calc_stage_distances(features, st_edges)[0]

    while (st_dist_list.min() <= dist_threshold * numpy.mean(st_dist_list)):
        st_min_dist_ind = st_dist_list.argmin()
        st_edges = numpy.delete(st_edges, st_min_dist_ind + 1)
        st_dist_list = calc_stage_distances(features, st_edges)[0]
       
    return st_edges