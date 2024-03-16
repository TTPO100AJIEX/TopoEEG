import typing

import numpy
import sklearn.metrics

from .. import stageprocess

# Define the metrics which require true and predicted labels
cluster_metrics = [
    (sklearn.metrics.adjusted_mutual_info_score, 'AMI'),
    (sklearn.metrics.adjusted_rand_score, 'ARI'),
    (sklearn.metrics.fowlkes_mallows_score, 'FMI')
]

def cluster_metrics_ground(edges_true: numpy.ndarray, edges_pred: numpy.ndarray) -> typing.Dict[str, float]:
    _, labels_true = stageprocess.form_stage_bands(edges_true)
    _, labels_pred = stageprocess.form_stage_bands(edges_pred)
    return { name: func(labels_true, labels_pred) for (func, name) in cluster_metrics }