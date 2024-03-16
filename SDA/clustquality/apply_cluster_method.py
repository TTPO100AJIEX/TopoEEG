import typing

import numpy
import sklearn.base

from .cluster_metrics_noground import cluster_metrics_noground

def apply_cluster_method(
    data: numpy.ndarray,
    cl_method: typing.Callable[[], sklearn.base.ClusterMixin],
    **kwargs
) -> typing.Tuple[sklearn.base.ClusterMixin, numpy.ndarray, typing.Dict[str, float]]:
    method = cl_method(**kwargs)
    labels = method.fit_predict(data)
    return method, labels, cluster_metrics_noground(data, labels)