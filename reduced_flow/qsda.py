import os
import sys
sys.path.append(os.getcwd())

import mne
import numpy
import pandas
import sklearn.metrics
import sklearn.preprocessing
import sklearn.decomposition

import SDA.topology
import SDA.analytics
import SDA.clustquality
import SDA.stageprocess

subjs = range(1, 31, 1)

for subj_num in subjs:
    subj = f"phase2/Subj{subj_num}"

    UNIQUE_VALUES_THRESHOLD = 200

    exp = "exp_reduced_flow"
    os.makedirs(f"{subj}/{exp}", exist_ok = True)

    raw_data = mne.io.read_raw_fif(f"{subj}/src/data_rr_filt-raw.fif").drop_channels(ch_names = [ 'IVEOG', 'IHEOG' ])
    epochs = mne.read_epochs(f"{subj}/src/epochs_filt_rr-epo.fif").drop_channels(ch_names = [ 'IVEOG', 'IHEOG' ])
    N_STAGES = int(numpy.loadtxt(f"{subj}/src/n_stages.txt"))
    print('Stages: ', N_STAGES)

    data = epochs.get_data(copy = True)
    print(data.shape)

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

    os.makedirs(f"{subj}/{exp}/qsda", exist_ok = True)
    qsda = SDA.QSDA(
        n_jobs = 1,
        qsda_n_jobs = -1,
        scores_folder = f"{subj}/{exp}/qsda",

        threshold = 1150,
        min_unique_values = UNIQUE_VALUES_THRESHOLD
    )
    best_features, scores = qsda.select(features)

    best_features.to_feather(f"{subj}/{exp}/qsda/best_features.feather")
    numpy.save(f"{subj}/{exp}/qsda/best_features.npy", features.to_numpy())
