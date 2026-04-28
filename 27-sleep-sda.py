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

exp = "exp_reduced_flow"

def match_edges(edges_true: typing.List[int], edges_pred: typing.List[int]) -> typing.List[int]:
    edges_to_choose = set(edges_true)
    edges_to_compare = []
    for edge in edges_pred:
        choices = numpy.array([*edges_to_choose])
        best_idx = numpy.argmin(numpy.abs(choices - edge))
        edges_to_compare.append(int(choices[best_idx]))
        edges_to_choose.remove(choices[best_idx])
    edges_to_compare.sort()
    return edges_to_compare


def analyze(subj, all_features: pandas.DataFrame, n_components: int, folder: str):
    folder = f"{subj}/{exp}/results/{folder}"
    os.makedirs(folder, exist_ok = True)
    
    epochs = mne.read_epochs(f"{subj}/epochs.fif")
    edges_true = numpy.loadtxt(f"{subj}/edges.txt").astype(int)
    N_STAGES = 10

    # Scale features
    all_features = sklearn.preprocessing.StandardScaler().fit_transform(all_features)
    print(all_features.shape)
    numpy.save(f"{folder}/all_features.npy", all_features)
    numpy.savetxt(f"{folder}/all_features_shape.txt", all_features.shape)

    # PCA
    pca = sklearn.decomposition.PCA(n_components = n_components, svd_solver = "full", random_state = 42)
    pca_features = pca.fit_transform(all_features)
    print(pca_features.shape)
    numpy.save(f"{folder}/pca_features.npy", pca_features)
    numpy.savetxt(f"{folder}/pca_features_shape.txt", pca_features.shape)
    
    print('Explained variance', round(pca.explained_variance_ratio_.sum(), 2))
    print([ round(x, 3) for x in pca.explained_variance_ratio_ ])
    numpy.savetxt(f"{folder}/explained_variance.txt", [ pca.explained_variance_ratio_.sum() ])
    numpy.savetxt(f"{folder}/explained_variance_ratios.txt", pca.explained_variance_ratio_)

    # SDA
    sda = SDA.SDA(n_jobs = -1, scale = False, verbose = True)
    results, df_st_edges = sda.apply(pca_features)
    
    metrics = [ ]
    for row in results['St_edges']:
        best_edges_true = match_edges(edges_true, row)
        metrics.append(SDA.clustquality.cluster_metrics_ground(best_edges_true, row))
    results = pandas.concat([ results, pandas.DataFrame(metrics) ], axis = 1)
    
    results.to_csv(f"{folder}/results.csv")
    df_st_edges.to_csv(f"{folder}/df_st_edges.csv")

    # Analyze
    best_results = SDA.analytics.best_results(results, key = 'Avg-Silh')
    best_results.to_csv(f"{folder}/best_results.csv")
    
    for try_n_stages in range(N_STAGES + 1, 0, -1):
        try:
            best_result = SDA.analytics.best_result(results, key = 'Avg-Silh', n_stages = try_n_stages)
            break
        except:
            continue
    best_result_df = pandas.DataFrame([ best_result ])
    best_result_df.to_csv(f"{folder}/best_result.csv")
    
    best_edges = numpy.array(best_result['St_edges'])
    numpy.savetxt(f"{folder}/best_edges.txt", best_edges, fmt = "%d", newline = ' ')

    stage_timing = SDA.analytics.stage_timing(best_edges, epochs)
    stage_timing.to_csv(f"{folder}/stage_timing.csv")

    SDA.analytics.plot_stats(pca_features, epochs, best_result, df_st_edges, edges_true = edges_true).savefig(f"{folder}/stats.svg")
    return best_result

def read_traditional(subj):
    df_ft_psd_loc_db = pandas.read_feather(f'{subj}/{exp}/traditional_features/df_ft_psd_loc_db.feather')
    df_ft_psd_ind_loc_log = pandas.read_feather(f'{subj}/{exp}/traditional_features/df_ft_psd_ind_loc_log.feather')
    df_ft_coh_ind_loc = pandas.read_feather(f'{subj}/{exp}/traditional_features/df_ft_coh_ind_loc.feather')
    df_ft_plv_ind_loc = pandas.read_feather(f'{subj}/{exp}/traditional_features/df_ft_plv_ind_loc.feather')

    result =  pandas.concat([
        df_ft_psd_loc_db,
        df_ft_psd_ind_loc_log,
        df_ft_coh_ind_loc,
        df_ft_plv_ind_loc
    ], axis = 1).fillna(0)

    return result

def read_topological(subj):
    return numpy.load(f"{subj}/{exp}/features/features.npy")

for i in range(0, 153):
    subj = f"sleep-edf/Subj{i}"
    if os.path.exists(f"{subj}/{exp}/results/combined/stats.svg"):
        continue

    topological = read_topological(subj)
    traditional = read_traditional(subj).to_numpy()

    # analyze(subj, topological, 15, "topological")
    # analyze(subj, traditional, 15, "traditional")

    combined_features = numpy.concatenate([ topological, traditional ], axis = 1)
    analyze(subj, combined_features, 15, "combined")
