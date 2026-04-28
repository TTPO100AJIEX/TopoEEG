import os
import typing

import mne
import numpy
import pandas
import IPython.display
import sklearn.metrics
import sklearn.preprocessing
import sklearn.decomposition
import matplotlib.pyplot as plt

import SDA.topology
import SDA.analytics
import SDA.clustquality
import SDA.stageprocess

def process(subj):
    exp = "exp_reduced_flow"
    os.makedirs(f"{subj}/{exp}", exist_ok = True)

    epochs = mne.read_epochs(f"{subj}/epochs.fif")
    data = epochs.get_data(copy = True)
    data = mne.decoding.Scaler(scalings='mean').fit_transform(data)

    per_channel_folder = f"{subj}/{exp}/features/per_channel"
    os.makedirs(per_channel_folder, exist_ok = True)
    per_channel_extractor = SDA.topology.PerChannelFeatureExtractor(n_jobs = -1, folder = per_channel_folder, reduced = True)
    per_channel_features = per_channel_extractor.extract(data)

    dissimilarity_folder = f"{subj}/{exp}/features/dissimilarity"
    os.makedirs(dissimilarity_folder, exist_ok = True)
    dissimilarity_extractor = SDA.topology.DissimilarityFeatureExtractor(n_jobs = -1, folder = dissimilarity_folder, reduced = True)
    dissimilarity_features = dissimilarity_extractor.extract(data)

    overall_folder = f"{subj}/{exp}/features/overall"
    os.makedirs(overall_folder, exist_ok = True)
    overall_extractor = SDA.topology.OverallFeatureExtractor(n_jobs = -1, folder = overall_folder, reduced = True)
    overall_features = overall_extractor.extract(data)

    all_features = pandas.concat([
        per_channel_features,
        dissimilarity_features,
        overall_features
    ], axis = 1)

    all_features.to_feather(f"{subj}/{exp}/features/all_features.feather")

    features = sklearn.preprocessing.StandardScaler().fit_transform(all_features)
    features = pandas.DataFrame(features, columns = all_features.columns)
    features.to_feather(f"{subj}/{exp}/features/features.feather")
    numpy.save(f"{subj}/{exp}/features/features.npy", features.to_numpy())


for i in range(0, 153):
    process(f"sleep-edf/Subj{i}")
