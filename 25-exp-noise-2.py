for i in range(18, 30):
    import os

    subj = f"phase2/Subj{i}"

    UNIQUE_VALUES_THRESHOLD = 200

    exp = "exp_noise"
    os.makedirs(f"{subj}/{exp}", exist_ok = True)

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

    epochs = mne.read_epochs(f"{subj}/src/epochs_filt_rr-epo.fif")# .drop_channels(ch_names = [ 'IVEOG', 'IHEOG' ])
    N_STAGES = int(numpy.loadtxt(f"{subj}/src/n_stages.txt"))
    print('Stages: ', N_STAGES)

    numpy.random.seed(42)
    epochs = mne.simulation.add_noise(epochs, mne.compute_covariance(epochs), verbose = True, random_state = 42)

    # epochs.average().plot_joint().savefig(f"{subj}/{exp}/eeg.svg")
    data = epochs.get_data(copy = True)
    print(data.shape)

    features = numpy.load(f"{subj}/{exp}/qsda/best_features.npy")
    print(features.shape)

    def analyze(all_features: pandas.DataFrame, n_components: int, folder: str):
        folder = f"{subj}/{exp}/results/{folder}"
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

    def read_traditional():
        df_ft_psd_loc_db = pandas.read_feather(f'{subj}/src/df_ft_psd_loc_db.feather')
        df_ft_psd_ind_loc_log = pandas.read_feather(f'{subj}/src/df_ft_psd_ind_loc_log.feather')
        df_ft_coh_ind_loc = pandas.read_feather(f'{subj}/src/df_ft_coh_ind_loc.feather')
        df_ft_plv_ind_loc = pandas.read_feather(f'{subj}/src/df_ft_plv_ind_loc.feather')

        result =  pandas.concat([
            df_ft_psd_loc_db,
            df_ft_psd_ind_loc_log,
            df_ft_coh_ind_loc,
            df_ft_plv_ind_loc
        ], axis = 1)

        if subj == "Subj2":
            result = result # [:-2]
        return result.to_numpy()

    analyze(read_traditional(), 15, "traditional")

    analyze(features, 15, "best_topological")

    combined_features = numpy.concatenate([ read_traditional(), features ], axis = 1)
    analyze(combined_features, 15, "combined")
