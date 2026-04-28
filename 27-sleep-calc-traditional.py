import os

import mne
import numpy
import sklearn.preprocessing

event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5
}
epoch_sec = 15.0

def process(subj):
    print(f'---------------------- Processing {subj} ----------------------')

    exp = "exp_reduced_flow"
    os.makedirs(f"{subj}/{exp}", exist_ok = True)

    annot_train = mne.read_annotations(f"{subj}/annotations.edf")
    annot_train.crop(annot_train[1]["onset"] - 15 * 60, annot_train[-2]["onset"] + 15 * 60)

    raw_data = mne.io.read_raw_edf(f"{subj}/eeg.edf", preload=True, infer_types=True, verbose='error')
    raw_data.set_annotations(annot_train, verbose='error')

    events_train, _ = mne.events_from_annotations(raw_data, event_id=event_id, chunk_duration=epoch_sec, verbose='error')

    tmax = epoch_sec - 1.0 / raw_data.info["sfreq"]
    epochs = mne.Epochs(raw=raw_data, events=events_train, tmin=0.0, tmax=tmax, baseline=None, verbose='error')

    data = epochs.get_data(copy = True)
    data = mne.decoding.Scaler(scalings='mean').fit_transform(data)

    bands = [(0.9, 4, 'Delta (0.9-4 Hz)', 'D'), (4, 8, 'Theta (4-8 Hz)', 'T'), (8, 14, 'Alpha (8-14 Hz)', 'A'), 
            (14, 25, 'Beta (14-25 Hz)', 'B'), (25, 40, 'Gamma (25-40 Hz)', 'G')]

    str_freq = [bands[i][3] for i in range(len(bands))]
    str_freq_rr = str_freq
    n_freq = len(str_freq_rr)

    regions = [(['EEG Fpz-Cz'], 'EEG Fpz-Cz', 'EEG Fpz-Cz'),
            (['EEG Pz-Oz'], 'EEG Pz-Oz', 'EEG Pz-Oz'), 
            (['EOG horizontal'], 'EOG horizontal', 'EOG horizontal'),
            (['Resp oro-nasal'], 'Resp oro-nasal', 'Resp oro-nasal'),
            (['EMG submental'], 'EMG submental', 'EMG submental'),
            (['Temp rectal'], 'Temp rectal', 'Temp rectal'), 
            (['Event marker'], 'Event marker', 'Event marker')]
    regions_plt = [(['EEG Fpz-Cz'], 'EEG Fpz-Cz', 'EEG Fpz-Cz'),
                (['EEG Pz-Oz'], 'EEG Pz-Oz', 'EEG Pz-Oz'), 
                    (['EOG horizontal'], 'EOG horizontal', 'EOG horizontal'),
                    (['Resp oro-nasal'], 'Resp oro-nasal', 'Resp oro-nasal'),
                    (['EMG submental'], 'EMG submental', 'EMG submental'),
                    (['Temp rectal'], 'Temp rectal', 'Temp rectal'), 
                    (['Event marker'], 'Event marker', 'Event marker')]

    n_regions = len(regions)
    n_regions_plt = len(regions_plt)

    sampling_rate = raw_data.info['sfreq']
    ch_names = epochs.ch_names
    n_freq = len(str_freq_rr)
    n_channels = len(ch_names)

    n_samples = epochs.__len__()
    n_times = len(epochs.get_data()[0,0,:])

    import numpy as np
    import pandas as pd

    kwargs = dict(fmin=bands[0][0], fmax=bands[-1][1], sfreq=sampling_rate, bandwidth=None, adaptive=True, n_jobs=1)

    rr_psd_mtaper, rr_freq_mtaper = mne.time_frequency.psd_array_multitaper(epochs.get_data(), verbose='error', **kwargs)
    freq_masks = [(fmin < rr_freq_mtaper) & (rr_freq_mtaper < fmax) for (fmin, fmax, _, _) in bands]
    loc_masks = [[ch_names[i] in reg for i in range(n_channels)] for (reg, _, _) in regions]
    loc_plt_masks = [[ch_names[i] in reg for i in range(n_channels)] for (reg, _, _) in regions_plt]
    # print(rr_freq_mtaper)
    # print(rr_psd_mtaper.shape)

    ft_psd_spectr_raw = np.array([np.mean(rr_psd_mtaper[:,:, _freq_mask], axis=2) for _freq_mask in freq_masks]).transpose(1,2,0)
    ft_psd_sp_loc_raw = np.array([np.mean(ft_psd_spectr_raw[:,_mask,:], axis=1) for _mask in loc_masks]).transpose(1,0,2)
    ft_psd_sp_plt_raw = np.array([np.mean(ft_psd_spectr_raw[:,_mask,:], axis=1) for _mask in loc_plt_masks]).transpose(1,0,2)
    ft_psd_sp_all_raw = np.mean(ft_psd_spectr_raw, axis=1)

    ft_psd_spectr_db = 10 * np.log10(ft_psd_spectr_raw) # Convert psd to dB format
    ft_psd_sp_loc_db = 10 * np.log10(ft_psd_sp_loc_raw) # Convert psd to dB 
    ft_psd_sp_plt_db = 10 * np.log10(ft_psd_sp_plt_raw) # Convert psd to dB format
    ft_psd_sp_all_db = 10 * np.log10(ft_psd_sp_all_raw) # Convert psd to dB format

    df_ft_psd_raw = pd.DataFrame()
    df_ft_psd_db = pd.DataFrame()
    df_ft_psd_loc_raw = pd.DataFrame()
    df_ft_psd_loc_db = pd.DataFrame()
    df_ft_psd_plt_raw = pd.DataFrame()
    df_ft_psd_plt_db = pd.DataFrame()
    df_ft_psd_all_raw = pd.DataFrame()
    df_ft_psd_all_db = pd.DataFrame()

    for i in range(n_freq):
        for j in range(n_channels):
            df_ft_psd_raw[str_freq_rr[i]+'_psd_'+ch_names[j]] = ft_psd_spectr_raw[:,j,i]
            df_ft_psd_db[str_freq_rr[i]+'_psd_'+ch_names[j]] = ft_psd_spectr_db[:,j,i]
        for j in range(n_regions):    
            df_ft_psd_loc_raw[str_freq_rr[i]+'_psd_'+regions[j][1]] = ft_psd_sp_loc_raw[:,j,i]
            df_ft_psd_loc_db[str_freq_rr[i]+'_psd_'+regions[j][1]] = ft_psd_sp_loc_db[:,j,i]
        for j in range(n_regions_plt):    
            df_ft_psd_plt_raw[str_freq_rr[i]+'_psd_'+regions_plt[j][1]] = ft_psd_sp_plt_raw[:,j,i]
            df_ft_psd_plt_db[str_freq_rr[i]+'_psd_'+regions_plt[j][1]] = ft_psd_sp_plt_db[:,j,i]
        df_ft_psd_all_raw[str_freq_rr[i]+'_psd_All'] = ft_psd_sp_all_raw[:,i]
        df_ft_psd_all_db[str_freq_rr[i]+'_psd_All'] = ft_psd_sp_all_db[:,i]
    #print(len(df_ft_psd_db.columns))

    # Scaling dB re-referenced data
    ft_psd_db_sc = sklearn.preprocessing.StandardScaler().fit_transform(df_ft_psd_db.to_numpy())
    df_ft_psd_db_sc = pd.DataFrame(ft_psd_db_sc, columns=df_ft_psd_db.columns)
    ft_psd_loc_db_sc = sklearn.preprocessing.StandardScaler().fit_transform(df_ft_psd_loc_db.to_numpy())
    df_ft_psd_loc_db_sc = pd.DataFrame(ft_psd_loc_db_sc, columns=df_ft_psd_loc_db.columns)
    ft_psd_all_db_sc = sklearn.preprocessing.StandardScaler().fit_transform(df_ft_psd_all_db.to_numpy())
    df_ft_psd_all_db_sc = pd.DataFrame(ft_psd_all_db_sc, columns=df_ft_psd_all_db.columns)

    df_ft_psd_ind = pd.DataFrame()
    df_ft_psd_ind_plt = pd.DataFrame()

    str_psd_ind = ['T_D','A_D','A_T','A_DT','B_D','B_T','B_A','B_DT','B_TA','G_D','G_T','G_A','G_B','G_DT','G_TA','G_AB']

    df_ft_psd_ind_loc = pd.DataFrame()
    df_ft_psd_ind_plt = pd.DataFrame()
    df_ft_psd_ind_all = pd.DataFrame()

    # Indices per region (averaged PSD)
    for _r in range(n_regions):
        for ind in str_psd_ind:
            if (len(ind)==3):
                df_ft_psd_ind_loc[ind+'_psd_'+regions[_r][1]] = (df_ft_psd_loc_raw[ind[0]+'_psd_'+regions[_r][1]] / 
                                                                df_ft_psd_loc_raw[ind[2]+'_psd_'+regions[_r][1]])
            elif (len(ind)==4):
                df_ft_psd_ind_loc[ind+'_psd_'+regions[_r][1]] = (df_ft_psd_loc_raw[ind[0]+'_psd_'+regions[_r][1]] / 
                                                                (df_ft_psd_loc_raw[ind[2]+'_psd_'+regions[_r][1]]+
                                                                df_ft_psd_loc_raw[ind[3]+'_psd_'+regions[_r][1]]))
    # Indices per region for plotting (averaged PSD)
    for _r in range(n_regions_plt):
        for ind in str_psd_ind:
            if (len(ind)==3):
                df_ft_psd_ind_plt[ind+'_psd_'+regions_plt[_r][1]] = (df_ft_psd_plt_raw[ind[0]+'_psd_'+regions_plt[_r][1]] / 
                                                                    df_ft_psd_plt_raw[ind[2]+'_psd_'+regions_plt[_r][1]])
            elif (len(ind)==4):
                df_ft_psd_ind_plt[ind+'_psd_'+regions_plt[_r][1]] = (df_ft_psd_plt_raw[ind[0]+'_psd_'+regions_plt[_r][1]] / 
                                                                    (df_ft_psd_plt_raw[ind[2]+'_psd_'+regions_plt[_r][1]]+
                                                                    df_ft_psd_plt_raw[ind[3]+'_psd_'+regions_plt[_r][1]]))

    # Indices for all channels averaged PSD
    for ind in str_psd_ind:
        if (len(ind)==3):
            df_ft_psd_ind_all[ind+'_psd_All'] = (df_ft_psd_all_raw[ind[0]+'_psd_All'] / 
                                                df_ft_psd_all_raw[ind[2]+'_psd_All'])
        elif (len(ind)==4):
            df_ft_psd_ind_all[ind+'_psd_All'] = (df_ft_psd_all_raw[ind[0]+'_psd_All'] / 
                                                (df_ft_psd_all_raw[ind[2]+'_psd_All']+
                                                df_ft_psd_all_raw[ind[3]+'_psd_All']))

    # Log-scaling PSD indices (dB format)
    df_ft_psd_ind_loc_log = 10 * np.log10(df_ft_psd_ind_loc)
    df_ft_psd_ind_plt_log = 10 * np.log10(df_ft_psd_ind_plt)
    df_ft_psd_ind_all_log = 10 * np.log10(df_ft_psd_ind_all)

    # No need to log-scale A|T & A|(D+T) for m10
    #orig_col = [col for col in df_ft_psd_ind_loc.columns if ('A_T' in col) or ('A_DT' in col)]
    #df_ft_psd_ind_loc_log[orig_col] = df_ft_psd_ind_loc[orig_col]
    #orig_col = [col for col in df_ft_psd_ind_plt.columns if ('A_T' in col) or ('A_DT' in col)]
    #df_ft_psd_ind_plt_log[orig_col] = df_ft_psd_ind_plt[orig_col]
    #orig_col = [col for col in df_ft_psd_ind_all.columns if ('A_T' in col) or ('A_DT' in col)]
    #df_ft_psd_ind_all_log[orig_col] = df_ft_psd_ind_all[orig_col]

    # Scaling
    ft_psd_ind_loc_sc = sklearn.preprocessing.StandardScaler().fit_transform(df_ft_psd_ind_loc_log.to_numpy())
    df_ft_psd_ind_loc_sc = pd.DataFrame(ft_psd_ind_loc_sc, columns=df_ft_psd_ind_loc_log.columns)
    ft_psd_ind_all_sc = sklearn.preprocessing.StandardScaler().fit_transform(df_ft_psd_ind_all_log.to_numpy())
    df_ft_psd_ind_all_sc = pd.DataFrame(ft_psd_ind_all_sc, columns=df_ft_psd_ind_all_log.columns)

    # Averaging epochs by region
    loc_masks = [[ch_names[i] in reg for i in range(n_channels)] for (reg, _, _) in regions]
    loc_plt_masks = [[ch_names[i] in reg for i in range(n_channels)] for (reg, _, _) in regions_plt]

    # Re-referenced data
    ft_epochs = epochs.get_data()
    ft_epochs_loc = np.array([np.mean(ft_epochs[:,_mask,:], axis=1) for _mask in loc_masks]).transpose(1,0,2)
    ft_epochs_plt = np.array([np.mean(ft_epochs[:,_mask,:], axis=1) for _mask in loc_plt_masks]).transpose(1,0,2)

    # print(ft_epochs.shape)
    # print(ft_epochs_loc.shape)
    # print(ft_epochs_plt.shape)
    #print(ch_names)

    # Calculating CSD (Cross-spectral densities), re-referenced data

    ft_csd_matr_sp = []
    ft_csd_matr_loc_sp = []
    ft_csd_matr_plt_sp = []
    kwargs = dict(fmin=bands[0][0], fmax=bands[-1][1], sfreq=sampling_rate, adaptive=True, n_jobs=1)

    # Calculating CSD for each epoch (Multitaper)
    for i in range(n_samples):
        csd_mtaper = mne.time_frequency.csd_array_multitaper(ft_epochs[i].reshape((1, n_channels, n_times)), verbose='error', **kwargs)
        ft_csd_matr_sp.append([csd_mtaper.mean(fmin, fmax).get_data() for (fmin, fmax, _, _) in bands])
        # print(i, np.array(ft_csd_matr_sp[i]).shape)

        csd_mtaper = mne.time_frequency.csd_array_multitaper(ft_epochs_loc[i].reshape((1, n_regions, n_times)), verbose='error', **kwargs)
        ft_csd_matr_loc_sp.append([csd_mtaper.mean(fmin, fmax).get_data() for (fmin, fmax, _, _) in bands])
        # print(i, np.array(ft_csd_matr_loc_sp[i]).shape)

        csd_mtaper = mne.time_frequency.csd_array_multitaper(ft_epochs_plt[i].reshape((1, n_regions_plt, n_times)), verbose='error', **kwargs)
        ft_csd_matr_plt_sp.append([csd_mtaper.mean(fmin, fmax).get_data() for (fmin, fmax, _, _) in bands])
        # print(i, np.array(ft_csd_matr_plt_sp[i]).shape)
        
    ft_csd_matr_sp = np.array(ft_csd_matr_sp)
    ft_csd_matr_loc_sp = np.array(ft_csd_matr_loc_sp)
    ft_csd_matr_plt_sp = np.array(ft_csd_matr_plt_sp)

    # print(ft_csd_matr_sp.shape)
    # print(ft_csd_matr_loc_sp.shape)
    # print(ft_csd_matr_plt_sp.shape)

    # Calculating Coherence, PLV and PSD from CSD, re-referenced data
    SLICE_LEN = 5 #number of epochs to measure cherence and PLV

    df_ft_coh = pd.DataFrame()
    df_ft_plv = pd.DataFrame()
    df_ft_coh_loc = pd.DataFrame()
    df_ft_plv_loc = pd.DataFrame()
    df_ft_coh_plt = pd.DataFrame()
    df_ft_plv_plt = pd.DataFrame()

    for _freq in range(n_freq):
        # By channel pairs
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                coh_list = []
                plv_list = []
                for _samp in range(n_samples):
                    samp_slice = ft_csd_matr_sp[max(_samp-SLICE_LEN//2, 0):min(_samp+SLICE_LEN//2+SLICE_LEN%2, n_samples), 
                                                _freq,:,:]
                    coh = abs(np.mean(samp_slice[:,i,j])) / np.sqrt(np.mean(samp_slice[:,i,i]).real * np.mean(samp_slice[:,j,j]).real)
                    plv = abs(np.mean(samp_slice[:,i,j]/np.abs(samp_slice[:,i,j])))
                    coh_list.append(coh)
                    plv_list.append(plv)
                df_ft_coh[str_freq_rr[_freq]+'_coh_'+ch_names[i]+'_'+ch_names[j]] = np.array(coh_list)
                df_ft_plv[str_freq_rr[_freq]+'_plv_'+ch_names[i]+'_'+ch_names[j]] = np.array(plv_list)
        #print(len(df_ft_plv.columns))
        
        # By region pairs
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                coh_list = []
                plv_list = []
                for _samp in range(n_samples):
                    samp_slice = ft_csd_matr_loc_sp[_samp:min(_samp+SLICE_LEN, n_samples),_freq,:,:]
                    coh = abs(np.mean(samp_slice[:,i,j])) / np.sqrt(np.mean(samp_slice[:,i,i]).real * np.mean(samp_slice[:,j,j]).real)
                    plv = abs(np.mean(samp_slice[:,i,j]/np.abs(samp_slice[:,i,j])))
                    coh_list.append(coh)
                    plv_list.append(plv)
                df_ft_coh_loc[str_freq_rr[_freq]+'_coh_'+regions[i][1]+'_'+regions[j][1]] = np.array(coh_list)
                df_ft_plv_loc[str_freq_rr[_freq]+'_plv_'+regions[i][1]+'_'+regions[j][1]] = np.array(plv_list)   
        #print(len(df_ft_plv_loc.columns))
        
        # By region pairs for plotting
        for i in range(n_regions_plt):
            for j in range(i+1, n_regions_plt):
                coh_list = []
                plv_list = []
                for _samp in range(n_samples):
                    samp_slice = ft_csd_matr_plt_sp[_samp:min(_samp+SLICE_LEN, n_samples),_freq,:,:]
                    coh = abs(np.mean(samp_slice[:,i,j])) / np.sqrt(np.mean(samp_slice[:,i,i]).real * np.mean(samp_slice[:,j,j]).real)
                    plv = abs(np.mean(samp_slice[:,i,j]/np.abs(samp_slice[:,i,j])))
                    coh_list.append(coh)
                    plv_list.append(plv)
                df_ft_coh_plt[str_freq_rr[_freq]+'_coh_'+regions_plt[i][1]+'_'+regions_plt[j][1]] = np.array(coh_list)
                df_ft_plv_plt[str_freq_rr[_freq]+'_plv_'+regions_plt[i][1]+'_'+regions_plt[j][1]] = np.array(plv_list)                   
        # print(len(df_ft_plv_plt.columns))
        

    # Special coherence & PLV features

    df_ft_coh_ind = pd.DataFrame()
    df_ft_plv_ind = pd.DataFrame()
    df_ft_coh_ind_loc = pd.DataFrame()
    df_ft_plv_ind_loc = pd.DataFrame()
    df_ft_coh_ind_plt = pd.DataFrame()
    df_ft_plv_ind_plt = pd.DataFrame()
    df_ft_coh_ind_all = pd.DataFrame()
    df_ft_plv_ind_all = pd.DataFrame()

    for _freq in range(n_freq):
        # By channel pairs
        for _ch in range(n_channels):
            ch_cols = [col for col in df_ft_coh.columns if col.startswith(str_freq_rr[_freq]) and (ch_names[_ch] in col)]
            df_ft_coh_ind[str_freq_rr[_freq]+'_coh_'+ch_names[_ch]+'_06'] = (df_ft_coh[ch_cols] >= 0.6).sum(axis=1)
            df_ft_coh_ind[str_freq_rr[_freq]+'_coh_'+ch_names[_ch]+'_07'] = (df_ft_coh[ch_cols] >= 0.7).sum(axis=1)
            df_ft_coh_ind[str_freq_rr[_freq]+'_coh_'+ch_names[_ch]+'_08'] = (df_ft_coh[ch_cols] >= 0.8).sum(axis=1)
            
            ch_cols = [col for col in df_ft_plv.columns if col.startswith(str_freq_rr[_freq]) and (ch_names[_ch] in col)]
            df_ft_plv_ind[str_freq_rr[_freq]+'_plv_'+ch_names[_ch]+'_06'] = (df_ft_plv[ch_cols] >= 0.6).sum(axis=1)
            df_ft_plv_ind[str_freq_rr[_freq]+'_plv_'+ch_names[_ch]+'_07'] = (df_ft_plv[ch_cols] >= 0.7).sum(axis=1)
            df_ft_plv_ind[str_freq_rr[_freq]+'_plv_'+ch_names[_ch]+'_08'] = (df_ft_plv[ch_cols] >= 0.8).sum(axis=1)
            
        # By region
        for _reg in range(n_regions):
            # Coherence
            reg_cols =[col for col in df_ft_coh_ind.columns if any(ch in col for ch in regions[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('06' in col)]
            df_ft_coh_ind_loc[str_freq_rr[_freq]+'_coh_'+regions[_reg][1]+'_06'] = df_ft_coh_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_coh_ind.columns if any(ch in col for ch in regions[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('07' in col)]
            df_ft_coh_ind_loc[str_freq_rr[_freq]+'_coh_'+regions[_reg][1]+'_07'] = df_ft_coh_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_coh_ind.columns if any(ch in col for ch in regions[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('08' in col)]
            df_ft_coh_ind_loc[str_freq_rr[_freq]+'_coh_'+regions[_reg][1]+'_08'] = df_ft_coh_ind[reg_cols].mean(axis=1)
            # PLV
            reg_cols =[col for col in df_ft_plv_ind.columns if any(ch in col for ch in regions[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('06' in col)]
            df_ft_plv_ind_loc[str_freq_rr[_freq]+'_plv_'+regions[_reg][1]+'_06'] = df_ft_plv_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_plv_ind.columns if any(ch in col for ch in regions[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('07' in col)]
            df_ft_plv_ind_loc[str_freq_rr[_freq]+'_plv_'+regions[_reg][1]+'_07'] = df_ft_plv_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_plv_ind.columns if any(ch in col for ch in regions[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('08' in col)]
            df_ft_plv_ind_loc[str_freq_rr[_freq]+'_plv_'+regions[_reg][1]+'_08'] = df_ft_plv_ind[reg_cols].mean(axis=1)

        # By region for plotting
        for _reg in range(n_regions_plt):
            # Coherence
            reg_cols =[col for col in df_ft_coh_ind.columns if any(ch in col for ch in regions_plt[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('06' in col)]
            df_ft_coh_ind_plt[str_freq_rr[_freq]+'_coh_'+regions_plt[_reg][1]+'_06'] = df_ft_coh_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_coh_ind.columns if any(ch in col for ch in regions_plt[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('07' in col)]
            df_ft_coh_ind_plt[str_freq_rr[_freq]+'_coh_'+regions_plt[_reg][1]+'_07'] = df_ft_coh_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_coh_ind.columns if any(ch in col for ch in regions_plt[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('08' in col)]
            df_ft_coh_ind_plt[str_freq_rr[_freq]+'_coh_'+regions_plt[_reg][1]+'_08'] = df_ft_coh_ind[reg_cols].mean(axis=1)
            # PLV
            reg_cols =[col for col in df_ft_plv_ind.columns if any(ch in col for ch in regions_plt[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('06' in col)]
            df_ft_plv_ind_plt[str_freq_rr[_freq]+'_plv_'+regions_plt[_reg][1]+'_06'] = df_ft_plv_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_plv_ind.columns if any(ch in col for ch in regions_plt[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('07' in col)]
            df_ft_plv_ind_plt[str_freq_rr[_freq]+'_plv_'+regions_plt[_reg][1]+'_07'] = df_ft_plv_ind[reg_cols].mean(axis=1)
            reg_cols =[col for col in df_ft_plv_ind.columns if any(ch in col for ch in regions_plt[_reg][0]) and 
                    col.startswith(str_freq_rr[_freq]) and ('08' in col)]
            df_ft_plv_ind_plt[str_freq_rr[_freq]+'_plv_'+regions_plt[_reg][1]+'_08'] = df_ft_plv_ind[reg_cols].mean(axis=1)
        
        # Averaged by all channels
        reg_cols =[col for col in df_ft_coh_ind.columns if col.startswith(str_freq_rr[_freq]) and ('06' in col)]
        df_ft_coh_ind_all[str_freq_rr[_freq]+'_coh_all_06'] = df_ft_coh_ind[reg_cols].mean(axis=1)
        reg_cols =[col for col in df_ft_coh_ind.columns if col.startswith(str_freq_rr[_freq]) and ('07' in col)]
        df_ft_coh_ind_all[str_freq_rr[_freq]+'_coh_all_07'] = df_ft_coh_ind[reg_cols].mean(axis=1)
        reg_cols =[col for col in df_ft_coh_ind.columns if col.startswith(str_freq_rr[_freq]) and ('08' in col)]
        df_ft_coh_ind_all[str_freq_rr[_freq]+'_coh_all_08'] = df_ft_coh_ind[reg_cols].mean(axis=1)

        reg_cols =[col for col in df_ft_plv_ind.columns if col.startswith(str_freq_rr[_freq]) and ('06' in col)]
        df_ft_plv_ind_all[str_freq_rr[_freq]+'_plv_all_06'] = df_ft_plv_ind[reg_cols].mean(axis=1)
        reg_cols =[col for col in df_ft_plv_ind.columns if col.startswith(str_freq_rr[_freq]) and ('07' in col)]
        df_ft_plv_ind_all[str_freq_rr[_freq]+'_plv_all_07'] = df_ft_plv_ind[reg_cols].mean(axis=1)
        reg_cols =[col for col in df_ft_plv_ind.columns if col.startswith(str_freq_rr[_freq]) and ('08' in col)]
        df_ft_plv_ind_all[str_freq_rr[_freq]+'_plv_all_08'] = df_ft_plv_ind[reg_cols].mean(axis=1)
            
    # print(len(df_ft_plv_ind.columns))
    # print(len(df_ft_plv_ind_loc.columns))
    # print(len(df_ft_plv_ind_plt.columns))
    # print(len(df_ft_plv_ind_all.columns))

    ft_dir_path = f"{subj}/{exp}/traditional_features"
    os.makedirs(ft_dir_path, exist_ok = True)

    df_ft_psd_loc_db.to_feather(os.path.join(ft_dir_path, 'df_ft_psd_loc_db.feather'))
    df_ft_psd_plt_db.to_feather(os.path.join(ft_dir_path, 'df_ft_psd_plt_db.feather'))
    df_ft_psd_all_db.to_feather(os.path.join(ft_dir_path, 'df_ft_psd_all_db.feather'))
    df_ft_psd_ind_loc_log.to_feather(os.path.join(ft_dir_path, 'df_ft_psd_ind_loc_log.feather'))
    df_ft_psd_ind_plt_log.to_feather(os.path.join(ft_dir_path, 'df_ft_psd_ind_plt_log.feather'))
    df_ft_psd_ind_all_log.to_feather(os.path.join(ft_dir_path, 'df_ft_psd_ind_all_log.feather'))

    #df_ft_coh.to_feather(os.path.join(ft_dir_path, 'df_ft_coh.feather'))
    #df_ft_plv.to_feather(os.path.join(ft_dir_path, 'df_ft_plv.feather'))
    df_ft_coh_loc.to_feather(os.path.join(ft_dir_path, 'df_ft_coh_loc.feather'))
    df_ft_plv_loc.to_feather(os.path.join(ft_dir_path, 'df_ft_plv_loc.feather'))
    df_ft_coh_plt.to_feather(os.path.join(ft_dir_path, 'df_ft_coh_plt.feather'))
    df_ft_plv_plt.to_feather(os.path.join(ft_dir_path, 'df_ft_plv_plt.feather'))

    df_ft_coh_ind_loc.to_feather(os.path.join(ft_dir_path, 'df_ft_coh_ind_loc.feather'))
    df_ft_plv_ind_loc.to_feather(os.path.join(ft_dir_path, 'df_ft_plv_ind_loc.feather'))
    df_ft_coh_ind_plt.to_feather(os.path.join(ft_dir_path, 'df_ft_coh_ind_plt.feather'))
    df_ft_plv_ind_plt.to_feather(os.path.join(ft_dir_path, 'df_ft_plv_ind_plt.feather'))
    df_ft_coh_ind_all.to_feather(os.path.join(ft_dir_path, 'df_ft_coh_ind_all.feather'))
    df_ft_plv_ind_all.to_feather(os.path.join(ft_dir_path, 'df_ft_plv_ind_all.feather'))


import tqdm

for subj_name in tqdm.tqdm(os.listdir("sleep-edf")):
    if os.path.exists(f"sleep-edf/{subj_name}/exp_reduced_flow/traditional_features/df_ft_plv_ind_all.feather"):
        continue
    process(f"sleep-edf/{subj_name}")
