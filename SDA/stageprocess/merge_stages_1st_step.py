import numpy

from .clusters_dist_ward import clusters_dist_ward

# Merge small stages with neighbours
def merge_stages_1st_step(features: numpy.ndarray, st_edges: numpy.ndarray, len_threshold: int) -> numpy.ndarray:
    if len(st_edges) <= 2: 
        return st_edges
    
    st_lengths = st_edges[1:] - st_edges[:-1]

    while (st_lengths.min() <= len_threshold):
        st_min_len_ind = st_lengths.argmin()

        if (st_min_len_ind == len(st_lengths) - 1):
            st_edges = numpy.delete(st_edges, st_min_len_ind)
        elif (st_min_len_ind == 0):
            st_edges = numpy.delete(st_edges, st_min_len_ind + 1)
        else:
            clust1 = features[st_edges[st_min_len_ind - 1]:st_edges[st_min_len_ind]]
            clust2 = features[st_edges[st_min_len_ind]:st_edges[st_min_len_ind + 1]]
            clust3 = features[st_edges[st_min_len_ind + 1]:st_edges[st_min_len_ind + 2]]
            st_dist_left = clusters_dist_ward(clust1, clust2)
            st_dist_right = clusters_dist_ward(clust2, clust3)
            st_edges = numpy.delete(st_edges, st_min_len_ind + (st_dist_left > st_dist_right))
            
        st_lengths = st_edges[1:] - st_edges[:-1]
        
    return st_edges