import numpy
import pandas

from .. import stageprocess, clustquality

def edge_statistics(features: numpy.ndarray, st_edges: numpy.ndarray) -> pandas.DataFrame:
    metrics = clustquality.calc_stage_metr_noground(features, st_edges)
    metrics['Ward'], metrics['Centr'] = stageprocess.calc_stage_distances(features, st_edges)
    return metrics