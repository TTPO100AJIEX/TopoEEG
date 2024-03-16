import typing

import numpy
import sklearn.metrics

# Define the metrics which require only data and predicted labels
cluster_metrics = [
    (sklearn.metrics.silhouette_score, 'Silh'),
    (sklearn.metrics.calinski_harabasz_score, 'Cal-Har'),
    (sklearn.metrics.davies_bouldin_score, 'Dav-Bold')
]

def cluster_metrics_noground(data: numpy.ndarray, labels_pred: numpy.ndarray) -> typing.Dict[str, float]:
    return { name: func(data, labels_pred) for (func, name) in cluster_metrics }