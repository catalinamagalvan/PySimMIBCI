# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:45:25 2020

@author: cati9
"""

import numpy as np
from scipy.io import loadmat
import mne
import urllib.request

mne.set_log_level(verbose='warning')


def load_and_epoch_OpenBMI_by_ftp(subject, session, epoch_window=[0, 4],
                                  picks=None):
    """
    Fetch by ftp OpenBMI data and create corresponding MNE Epochs objects.

    Parameters
    ----------
    subject : int
        Subject number in OpenBMI dataset, ranging from 0 to 53.
    session : int
        Session number in OpenBMI dataset, [1, 2].
    epoch_window : list, optional
        List indicating the start and end time of the epochs in seconds,
        relative to the time-locked event. The default is [0, 4].
    picks : list, optional
        Channels to include. Lists of integers will be interpreted
        as channel indices. None (default) will pick all channels.

    Returns
    -------
    epochs : instance of Epochs
        The MNE Epochs corresponding to the loaded file.
    epochs_right : instance of Epochs
        The MNE Epochs corresponding to right hand MI in the loaded file.
    epochs_left : instance of Epochs
        The MNE Epochs corresponding to right hand MI in the loaded file.

    """

    url = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/'\
        'session' + str(session) + '/s' + str(subject+1) + '/ses'\
        's' + '{0:0=2d}'.format(session) + '_subj' + '{0:0=2d}'.format(
                    subject+1) + '_EEG_MI.mat'
    urllib.request.urlretrieve(url, 'EEG_MI.mat')
    # Load .mat data
    data = loadmat('EEG_MI.mat')
    # Create info object
    fs = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    info = create_OpenBMI_info(picks)
    # Training data
    data_train = data['EEG_MI_train'][0, 0]
    t_train = data_train['t'][0]
    eeg_train = data_train['x']
    eeg_train = eeg_train.T
    epochInterval = np.array(range(int(epoch_window[0]*fs),
                                   int(epoch_window[1]*fs)))
    x_train = np.stack([eeg_train[:, epochInterval+event] for event in
                        t_train], axis=2)
    # x (n_trials, n_channels, n_times)
    x_train = np.transpose(x_train, (2, 0, 1))
    y_train = data_train['y_dec'].squeeze().astype(int)-1
    # Test data
    data_test = data['EEG_MI_test'][0, 0]
    t_test = data_test['t'][0]
    eeg_test = data_test['x']
    eeg_test = eeg_test.T
    x_test = np.stack([eeg_test[:, epochInterval+event] for event in
                       t_test], axis=2)
    # x (n_trials, n_channels, n_times)
    x_test = np.transpose(x_test, (2, 0, 1))
    y_test = data_test['y_dec'].squeeze().astype(int)-1
    x = np.concatenate((x_train, x_test), axis=0)
    if picks is not None:
        x = x[:, picks]
    y = np.concatenate((y_train, y_test))
    y[y == 2] = 0
    x_right = x[y == 0]
    x_left = x[y == 1]
    events_matrix = np.zeros((len(y), 3)).astype(int)
    events_matrix[:, 0] = list(range(len(y)))
    events_matrix[:, -1] = y
    epochs_right = mne.EpochsArray(x_right, info, events=events_matrix[y == 0])
    epochs_left = mne.EpochsArray(x_left, info, events=events_matrix[y == 1])
    epochs = mne.EpochsArray(x, info, events=events_matrix)
    return epochs, epochs_right, epochs_left


def create_BCIIV1_info(picks=None):
    """
    Create a basic MNE Info instance suitable for BCI competition IV dataset 1
    data.

    Parameters
    ----------
    picks : list, optional
        Channels to include. Lists of integers will be interpreted
        as channel indices. None (default) will pick all channels.

    Returns
    -------
    info : instance of MNE Info
        Corresponding MNE Info object.

    """
    # Create info object and select channels if requested
    sfreq = 1000
    ch_names = ['AF3', 'AF4', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC5',
                'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'CP5', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'CP6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6',
                'O1', 'O2']
    if picks is not None:
        ch_names_sel = [channel for c, channel in enumerate(ch_names) if c in
                        picks]
    else:
        ch_names_sel = ch_names
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(ch_names_sel, sfreq=sfreq, ch_types='eeg',
                           verbose=None)
    info.set_montage(montage)
    return info


def create_OpenBMI_info(picks=None):
    """
    Create a basic MNE Info instance suitable for OpenBMI dataset data.

    Parameters
    ----------
    picks : list, optional
        Channels to include. Lists of integers will be interpreted
        as channel indices. None (default) will pick all channels.

    Returns
    -------
    info : instance of MNE Info
        Corresponding MNE Info object.

    """
    # Create info object and select channels if requested
    sfreq = 1000
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5',
                'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8',
                'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9',
                'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h',
                'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8',
                'PO3', 'PO4']
    if picks is not None:
        ch_names_sel = [channel for c, channel in enumerate(ch_names) if c in
                        picks]
    else:
        ch_names_sel = ch_names
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(ch_names_sel, sfreq=sfreq, ch_types='eeg',
                           verbose=None)
    info.set_montage(montage)
    return info


def load_and_epoch_OpenBMI_data(fname, epoch_window=[0, 4], picks=None):
    """
    Load OpenBMI data and creates corresponding MNE Epochs objects.

    Parameters
    ----------
    fname : string
        The .mat file to load.
    epoch_window : list, optional
        List indicating the start and end time of the epochs in seconds,
        relative to the time-locked event. The default is [0, 4].
    picks : list, optional
        Channels to include. Lists of integers will be interpreted
        as channel indices. None (default) will pick all channels.

    Returns
    -------
    epochs : instance of Epochs
        The MNE Epochs corresponding to the loaded file.
    epochs_right : instance of Epochs
        The MNE Epochs corresponding to right hand MI in the loaded file.
    epochs_left : instance of Epochs
        The MNE Epochs corresponding to right hand MI in the loaded file.

    """
    # Load .mat data
    data = loadmat(fname)
    # Create info object
    fs = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    info = create_OpenBMI_info(picks)
    # Training data
    data_train = data['EEG_MI_train'][0, 0]
    t_train = data_train['t'][0]
    eeg_train = data_train['x']
    eeg_train = eeg_train.T
    epochInterval = np.array(range(int(epoch_window[0]*fs),
                                   int(epoch_window[1]*fs)))
    x_train = np.stack([eeg_train[:, epochInterval+event] for event in
                        t_train], axis=2)
    # x (n_trials, n_channels, n_times)
    x_train = np.transpose(x_train, (2, 0, 1))
    y_train = data_train['y_dec'].squeeze().astype(int)-1
    # Test data
    data_test = data['EEG_MI_test'][0, 0]
    t_test = data_test['t'][0]
    eeg_test = data_test['x']
    eeg_test = eeg_test.T
    x_test = np.stack([eeg_test[:, epochInterval+event] for event in
                       t_test], axis=2)
    # x (n_trials, n_channels, n_times)
    x_test = np.transpose(x_test, (2, 0, 1))
    y_test = data_test['y_dec'].squeeze().astype(int)-1
    x = np.concatenate((x_train, x_test), axis=0)
    if picks is not None:
        x = x[:, picks]
    y = np.concatenate((y_train, y_test))
    y[y == 2] = 0
    x_right = x[y == 0]
    x_left = x[y == 1]
    events_matrix = np.zeros((len(y), 3)).astype(int)
    events_matrix[:, 0] = list(range(len(y)))
    events_matrix[:, -1] = y
    epochs_right = mne.EpochsArray(x_right, info, events=events_matrix[y == 0])
    epochs_left = mne.EpochsArray(x_left, info, events=events_matrix[y == 1])
    epochs = mne.EpochsArray(x, info, events=events_matrix)
    return epochs, epochs_right, epochs_left


def raw_from_OpenBMI_data_by_ftp(subject, session, picks=None):
    """
    Fetch by ftp OpenBMI data and create corresponding MNE Raw object.

    Parameters
    ----------
    subject : int
        Subject number in OpenBMI dataset, ranging from 0 to 53.
    session : int
        Session number in OpenBMI dataset, [1, 2].
    picks : list, optional
        Channels to include. Lists of integers will be interpreted
        as channel indices. None (default) will pick all channels.

    Returns
    -------
    raw : instance of mne Raw.
        The MNE Raw object corresponding to the loaded file.

    """
    url = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/'\
        'session' + str(session) + '/s' + str(subject+1) + '/ses'\
        's' + '{0:0=2d}'.format(session) + '_subj' + '{0:0=2d}'.format(
                    subject+1) + '_EEG_MI.mat'
    urllib.request.urlretrieve(url, 'EEG_MI.mat')
    # Load .mat data
    data = loadmat('EEG_MI.mat')
    # Create info object
    info = create_OpenBMI_info(picks)
    # Training data
    data_train = data['EEG_MI_train'][0, 0]
    eeg_train = data_train['x']
    eeg_train = eeg_train.T
    raw = mne.io.RawArray(eeg_train, info)
    return raw


def raw_from_OpenBMI_data(fname, picks=None):
    """
    Load OpenBMI data and creates corresponding MNE Raw object.

    Parameters
    ----------
    fname : string
        The .mat file to load.
    picks : list, optional
        Channels to include. Lists of integers will be interpreted
        as channel indices. None (default) will pick all channels.

    Returns
    -------
    raw : instance of mne Raw.
        The MNE Raw object corresponding to the loaded file.

    """
    # Load .mat data
    data = loadmat(fname)
    # Create info object
    info = create_OpenBMI_info(picks)
    # Training data
    data_train = data['EEG_MI_train'][0, 0]
    eeg_train = data_train['x']
    eeg_train = eeg_train.T
    raw = mne.io.RawArray(eeg_train, info)
    return raw
