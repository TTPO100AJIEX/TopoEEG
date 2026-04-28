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

for subj_num in list(range(1, 31)):
    subj = f"phase2/Subj{subj_num}"
    exp = "exp_reduced_flow"
#     print(subj)

# for subj_num in range(1, 4):
#     subj = f"Subj{subj_num}"
#     exp = "exp_full_flow"

    src = f"E:/CourseProject/{subj}"

    epochs = mne.read_epochs(f"{src}/src/epochs_filt_rr-epo.fif")
    N_STAGES = int(numpy.loadtxt(f"{src}/src/n_stages.txt"))
    print('Stages: ', N_STAGES)

    data = epochs.get_data(copy = True)
    features = numpy.load(f"{src}/{exp}/features/features.npy")
    print(features.shape)

    def analyze(all_features: pandas.DataFrame, n_components: int, folder: str):
        folder = f"{src}/{exp}/results/{folder}"
        os.makedirs(folder, exist_ok = True)

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
        sda = SDA.SDA(n_jobs = 14, scale = False, verbose = True)
        results, df_st_edges = sda.apply(pca_features)
        
        results.to_csv(f"{folder}/results.csv")
        df_st_edges.to_csv(f"{folder}/df_st_edges.csv")

        # Analyze
        best_results = SDA.analytics.best_results(results, key = 'Avg-Silh')
        best_results.to_csv(f"{folder}/best_results.csv")
        
        best_result = SDA.analytics.best_result(results, key = 'Avg-Silh', n_stages = N_STAGES)
        best_result_df = pandas.DataFrame([ best_result ])
        best_result_df.to_csv(f"{folder}/best_result.csv")
        
        best_edges = numpy.array(best_result['St_edges'])
        numpy.savetxt(f"{folder}/best_edges.txt", best_edges, fmt = "%d", newline = ' ')
        print(best_edges)

        stage_timing = SDA.analytics.stage_timing(best_edges, epochs)
        stage_timing.to_csv(f"{folder}/stage_timing.csv")
        print(stage_timing)
        
        SDA.analytics.plot_stats(pca_features, epochs, best_result, df_st_edges).savefig(f"{folder}/stats.svg")

    analyze(features, 15, "topological")
