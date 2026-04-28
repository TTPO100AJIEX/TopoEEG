import os

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

for subj_num in list(range(27, 28))[::-1]:
    subj = f"phase2/Subj{subj_num}"
    REDUCED = True

# for subj_num in range(2, 4):
    # subj = f"Subj{subj_num}"
    # REDUCED = False

    UNIQUE_VALUES_THRESHOLD = 200

    exp = "exp_shuffled"
    os.makedirs(f"{subj}/{exp}", exist_ok = True)

    epochs = mne.read_epochs(f"{subj}/src/epochs_filt_rr-epo.fif").drop_channels(ch_names = [ 'IVEOG', 'IHEOG' ])
    N_STAGES = int(numpy.loadtxt(f"{subj}/src/n_stages.txt"))
    print('Stages: ', N_STAGES)

    data = epochs.get_data(copy = True)

    numpy.random.seed(42)
    if os.path.exists(f"{subj}/{exp}/shuffle_order.npy"):
        shuffle_order = numpy.load(f"{subj}/{exp}/shuffle_order.npy")
    else:
        shuffle_order = numpy.arange(data.shape[0])
        numpy.random.shuffle(shuffle_order)
        numpy.save(f"{subj}/{exp}/shuffle_order.npy", shuffle_order)
    data = data[shuffle_order]

    per_channel_folder = f"{subj}/{exp}/features/per_channel"
    os.makedirs(per_channel_folder, exist_ok = True)
    per_channel_extractor = SDA.topology.PerChannelFeatureExtractor(n_jobs = -1, folder = per_channel_folder, reduced = REDUCED)
    per_channel_features = per_channel_extractor.extract(data)

    dissimilarity_folder = f"{subj}/{exp}/features/dissimilarity"
    os.makedirs(dissimilarity_folder, exist_ok = True)
    dissimilarity_extractor = SDA.topology.DissimilarityFeatureExtractor(n_jobs = -1, folder = dissimilarity_folder, reduced = REDUCED)
    dissimilarity_features = dissimilarity_extractor.extract(data)

    overall_folder = f"{subj}/{exp}/features/overall"
    os.makedirs(overall_folder, exist_ok = True)
    overall_extractor = SDA.topology.OverallFeatureExtractor(n_jobs = -1, folder = overall_folder, reduced = REDUCED)
    overall_features = overall_extractor.extract(data)

    all_features = pandas.concat([
        per_channel_features,
        dissimilarity_features,
        overall_features
    ], axis = 1)

    all_features.to_feather(f"{subj}/{exp}/features/all_features.feather")
    print('all_features: ', all_features.shape)

    features = sklearn.preprocessing.StandardScaler().fit_transform(all_features)
    features = pandas.DataFrame(features, columns = all_features.columns)

    features.to_feather(f"{subj}/{exp}/features/features.feather")
    print('features: ', features.shape)

    numpy.save(f"{subj}/{exp}/features/features.npy", features.to_numpy())

    os.makedirs(f"{subj}/{exp}/qsda", exist_ok = True)
    qsda = SDA.QSDA(
        n_jobs = 1,
        qsda_n_jobs = 14,
        scores_folder = f"{subj}/{exp}/qsda",

        threshold = 1150,
        min_unique_values = UNIQUE_VALUES_THRESHOLD
    )
    best_features, scores = qsda.select(features)

    best_features.to_feather(f"{subj}/{exp}/qsda/best_features.feather")
    numpy.save(f"{subj}/{exp}/qsda/best_features.npy", features.to_numpy())
    print('best_features: ', best_features.shape)
