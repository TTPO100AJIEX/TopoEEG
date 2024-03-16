import numpy
import pandas

from .cluster_metrics_noground import cluster_metrics_noground

# Calculating clustering noground metrics for adjacent pairs of stages (Silh, Cal-Har, Dav-Bold)
def calc_stage_metr_noground(features: numpy.ndarray, st_edges: numpy.ndarray) -> pandas.DataFrame:
    metrics = [ ]  
    for prev, cur, next in zip(st_edges[:-2], st_edges[1:-1], st_edges[2:]):
        labels = (numpy.arange(prev, next) >= cur).astype(numpy.int64)
        metrics.append(cluster_metrics_noground(features[prev:next], labels))
    return pandas.DataFrame(metrics)