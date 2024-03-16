import tqdm
import numpy
import pandas

from .calc_IV import calc_IV

# Calculation of IV and WoE for given DataFrame of features and given clustering labels 
def calc_IV_clust(features: pandas.DataFrame, labels: numpy.ndarray, bins: int = 10) -> pandas.DataFrame:
    IV = [ ]
    for feature_name in tqdm.tqdm(features.columns):
        IVs = [ ]
        feature = features[feature_name].to_numpy()
        for cluster in numpy.unique(labels):
            target = (labels == cluster).astype(int)
            IVs.append(calc_IV(feature, target, bins))
        IV.append({ 'Feature': feature_name, 'IV': numpy.mean(IVs), 'IVs': IVs })
    return pandas.DataFrame(IV)