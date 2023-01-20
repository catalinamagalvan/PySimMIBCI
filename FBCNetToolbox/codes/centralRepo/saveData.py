"""
Data processing functions for EEG data.

@author: Ravikiran Mane. Modified by Catalina Galvan.
"""
import numpy as np
from scipy.io import loadmat, savemat
import os
import pickle
import csv
from shutil import copyfile
import sys
import resampy
import shutil
import urllib.request as request
from contextlib import closing

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
import transforms


def parseKoreaSimFile(dataPath, epochWindow=[0, 2], chans=None,
                      downsampleFactor=4):
    """
    Parse the korea-like sim data file and return an epoched data.

    Parameters
    ----------
    dataPath : str
        path to the mat file.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    """

    offset = 0

    # read the mat file:
    data = loadmat(dataPath)

    nfo = data['nfo'][0][0]
    s = nfo['fs'][0][0]
    s = int(s)
    eeg = data['cnt']
    # eeg = 0.1*np.double(eeg)  # Complete EEG signal
    # eeg = 1e6 * eeg  # In microV
    eeg = eeg.T

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    mrk = data['mrk']  # Samples of start of each trial
    trial = mrk[0][0]['pos'][0]
    epochInterval = np.array(range(epochWindow[0]*s,
                                   int(epochWindow[1]*s)))+int(offset*s)
    x = np.stack([eeg[:, epochInterval+event] for event in trial], axis=2)
    y = mrk[0][0]['y'][0]
    # change the labels from [2, 1] to [0, 1]
    y -= 1
    y = ~y + 2

    channels_names = ['']*len(nfo['clab'])
    aux = nfo['clab']
    for i, _ in enumerate(aux):
        channels_names[i] = str(aux[i])
    if chans is not None:
        channels_names = [channels_names[i] for i in chans]

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((x.shape[0], int(x.shape[1]/downsampleFactor),
                         x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s/downsampleFactor,
                                             axis=1)
        x = xNew
        s = s/downsampleFactor

    data = {'x': x, 'y': y.astype(np.int16), 'c': channels_names, 's': s}
    return data


def parseBci41SimFile(dataPath, epochWindow=[0, 2], chans=None,
                      downsampleFactor=4):
    """
    Parse the korea-like sim data file and return an epoched data.

    Parameters
    ----------
    dataPath : str
        path to the mat file.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    """

    offset = 0

    # read the mat file:
    data = loadmat(dataPath)

    nfo = data['nfo'][0][0]
    s = nfo['fs'][0][0]
    s = int(s)
    eeg = data['cnt']
    # eeg = 0.1*np.double(eeg)  # Complete EEG signal
    # eeg = 1e6 * eeg  # In microV
    eeg = eeg.T

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    mrk = data['mrk']  # Samples of start of each trial
    trial = mrk[0][0]['pos'][0]
    epochInterval = np.array(range(epochWindow[0]*s,
                                   int(epochWindow[1]*s)))+int(offset*s)
    x = np.stack([eeg[:, epochInterval+event] for event in trial], axis=2)
    y = mrk[0][0]['y'][0]
    # change the labels from [2, 1] to [1, 0]
    y -= 1

    channels_names = ['']*len(nfo['clab'])
    aux = nfo['clab']
    for i, _ in enumerate(aux):
        channels_names[i] = str(aux[i])
    if chans is not None:
        channels_names = [channels_names[i] for i in chans]

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((x.shape[0], int(x.shape[1]/downsampleFactor),
                         x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s/downsampleFactor,
                                             axis=1)
        x = xNew
        s = s/downsampleFactor

    data = {'x': x, 'y': y.astype(np.int16), 'c': channels_names, 's': s}
    return data


def parseSimDataset(datasetPath, savePath, subjects, downsampleFactor=None,
                    chans=None):
    """
    Parse the Korea Sim data in a MATLAB format that will be used in the
    next analysis.

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV1 original dataset in mat formate.
    savePath : str
        Path on where to save the epoched eeg data in a mat format.
    subjects: list of str
        IDs of the simulated subjects to be selected. Note: simulated subjects
        can be loaded with this function when they have BCI IV competition
        dataset 1 format.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    """
    if not subjects:
        raise ValueError('In order to load the simulated dataset, artificial '
                         'subjects to be selected must be specficied.')

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for iSub, sub in enumerate(subjects):
        if not os.path.exists(os.path.join(datasetPath, sub+'.mat')):
            raise ValueError('The simulated dataset doesn\'t'
                             'exist at path: ' + os.path.join(datasetPath,
                                                              sub + '.mat'))
        print('Processing subject: ' + sub)
        data = parseBci41SimFile(os.path.join(datasetPath, sub + '.mat'),
                                 epochWindow=[0, 2], chans=chans,
                                 downsampleFactor=4)
        savemat(os.path.join(savePath, sub + '.mat'), data)


def parseBci41SimDataset(datasetPath, savePath, downsampleFactor=4,
                         chans=None, subjects=None):
    """
    .

    Parse the Korea Sim data in a MATLAB format that will be used in the
    next analysis.

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV1 original dataset in mat formate.
    savePath : str
        Path on where to save the epoched eeg data in a mat format.
    subjects: list of int
        IDs of the subjects to be selected. If None, subjects 'b' and 'g'
        are included. Note: Here, 0 refers to subject b, 1 to subject c, 2 to
        subject d, 3 to subject e and 4 to subject g.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    """
    if not subjects:
        # subjects b and g from BCI competition IV,dataset 1
        subjects = [0, 4]
    subs = ['ss_' + str(i) for i in subjects]

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSub, sub in enumerate(subs):
        if not os.path.exists(os.path.join(datasetPath, sub+'.mat')):
            raise ValueError('The BCI IV 1-like augmented dataset doesn\'t'
                             'exist at path: ' + os.path.join(datasetPath,
                                                              sub + '.mat'))
        print('Processing subject: ' + str(subjects[iSub]))
        data = parseBci41SimFile(os.path.join(datasetPath, sub + '.mat'),
                                 epochWindow=[0, 2], chans=chans,
                                 downsampleFactor=downsampleFactor)
        savemat(os.path.join(savePath, 's' + str(subjects[iSub] + 1).zfill(3) +
                             '.mat'), data)


def parseKoreaSimDataset(datasetPath, savePath, verbos=False,
                         downsampleFactor=None, chans=None, subjects=None):
    """
    .

    Parse the Korea Sim data in a MATLAB format that will be used in the
    next analysis.

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV1 original dataset in mat formate.
    savePath : str
        Path on where to save the epoched eeg data in a mat format.
    subjects : list of int
        IDs of the subjects to be selected. If None, the 54 subjects are
        included

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    """
    if not subjects:
        subjects = list(range(54))
    subs = ['ss_' + str(i) for i in subjects]

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSub, sub in enumerate(subs):
        if not os.path.exists(os.path.join(datasetPath, sub+'.mat')):
            raise ValueError('The Korea-like augmented dataset doesn\'t'
                             ' exist at path: ' + os.path.join(datasetPath,
                                                               sub + '.mat'))
        print('Processing subject: ' + sub)
        data = parseKoreaSimFile(os.path.join(datasetPath, sub + '.mat'),
                                 epochWindow=[0, 2], chans=chans,
                                 downsampleFactor=4)
        savemat(os.path.join(savePath, 's'+str(subjects[iSub]+1).zfill(3) +
                             '.mat'), data)


def parseAugEpochedFile(dataPath, downsampleFactor=None):
    """
    Parse the korea-like augmented data file and return an epoched data.

    Parameters
    ----------
    dataPath : str
        path to the mat file.

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    """
    # read the mat file:
    data = loadmat(dataPath)
    x = data['x']
    s = data['s'][0, 0]
    if downsampleFactor is not None:
        xNew = np.zeros((x.shape[0], int(x.shape[1]/downsampleFactor),
                         x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s/downsampleFactor,
                                             axis=1)
        x = xNew
        s = s/downsampleFactor
    data = {'x': x, 'y': data['y'].astype(np.int16), 'c': data['c'],
            's': s}
    return data


def parseSimBci41File(dataPath, epochWindow=[0, 2], chans=list(range(41)),
                      downsampleFactor=None):
    """
    Parse the bci41-like sim data file and return an epoched data.

    Parameters
    ----------
    dataPath : str
        path to the mat file.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    """
    fs = 1000
    offset = 0

    # read the mat file:
    data = loadmat(dataPath)

    nfo = data['nfo'][0][0]
    s = nfo['fs'][0][0]
    s = int(s)
    eeg = data['cnt']
    eeg = 0.1*np.double(eeg)  # Complete EEG signal
    eeg = 1e6 * eeg  # In microV
    eeg = eeg.T

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    mrk = data['mrk']  # Samples of start of each trial
    trial = mrk[0][0]['pos'][0]
    epochInterval = np.array(range(epochWindow[0]*fs,
                                   int(epochWindow[1]*fs)))+int(offset*fs)
    x = np.stack([eeg[:, epochInterval+event] for event in trial], axis=2)
    y = mrk[0][0]['y'][0]
    # change the labels from [2, 1] to [1, 0]
    y -= 1

    channels_names = ['']*len(nfo['clab'])
    aux = nfo['clab']
    for i, _ in enumerate(aux):
        channels_names[i] = str(aux[i])

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((x.shape[0], int(x.shape[1]/downsampleFactor),
                         x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s/downsampleFactor,
                                             axis=1)
        x = xNew
        s = s/downsampleFactor

    data = {'x': x, 'y': y.astype(np.int16), 'c': channels_names, 's': s}
    return data


def parseBci41File(dataPath, epochWindow=[0, 4],
                   chans=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          24, 25, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44,
                          45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58],
                   downsampleFactor=None):
    """
    Parse the bci41 data file and return an epoched data.

    Parameters
    ----------
    dataPath : str
        path to the mat file.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    """
    fs = 1000
    offset = 0.5  # original
    # offset = 0

    # read the mat file:
    data = loadmat(dataPath)

    nfo = data['nfo'][0][0]
    s = nfo['fs'][0][0]
    s = int(s)
    eeg = data['cnt']
    eeg = 0.1*np.double(eeg)  # Complete EEG signal
    eeg = eeg.T

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    mrk = data['mrk']  # Samples of start of each trial
    trial = mrk[0][0]['pos'][0]
    epochInterval = np.array(range(epochWindow[0]*fs,
                                   int(epochWindow[1]*fs))) + int(offset*fs)
    x = np.stack([eeg[:, epochInterval+event] for event in trial], axis=2)
    y = mrk[0][0]['y'][0]
    # change the labels from [-1-1] to [0-1]
    for i, y_i in enumerate(y):
        if y_i == -1:
            y[i] = 0

    channels_names = ['']*len(nfo['clab'][0])
    aux = nfo['clab'][0]
    for i, _ in enumerate(aux):
        channels_names[i] = str(aux[i][0])

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((x.shape[0], int(x.shape[1]/downsampleFactor),
                         x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample can't handle 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s/downsampleFactor,
                                             axis=1)
        x = xNew
        s = s/downsampleFactor

    data = {'x': x, 'y': y, 'c': np.array(channels_names)[chans].tolist(),
            's': s}
    return data


def parseBci41Dataset(datasetPath, savePath, epochWindow=[0, 2], chans=None,
                      subjects=None, downsampleFactor=4):
    """
    Parse the BCI comp. IV-1 data in a MATLAB format that will be used in the
    next analysis.

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV1 original dataset in mat formate.
    savePath : str
        Path on where to save the epoched eeg data in a mat format.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list
        channels to select from the data.
    subjects : list of int
        IDs of the subjects to be selected. If None, subjects 'b' and 'g'
        are included. Note: Here, 0 refers to subject b, 1 to subject c, 2 to
        subject d, 3 to subject e and 4 to subject g.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    """
    if not subjects:
        # subjects b and g from BCI competition IV,dataset 1
        subjects = [0, 4]

    train_subjects = ['s' + str(subject + 1) + 'T' for subject in subjects]
    eval_subjects = ['s' + str(subject + 1) + 'E' for subject in subjects]
    subAll = [train_subjects, eval_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            if not os.path.exists(os.path.join(datasetPath, sub+'.mat')):
                raise ValueError('The BCI-IV-1 original dataset doesn\'t exist'
                                 ' at path: ' + os.path.join(datasetPath,
                                                             sub + '.mat') +
                                 'Please download and copy the extracted '
                                 'dataset at the above path. More details '
                                 'about how to download this data can be '
                                 'found in the Instructions.txt file')

            print('Processing subject: ' + subL[iSubs] + str(subjects[
                iSub]))
            data = parseBci41File(os.path.join(datasetPath, sub+'.mat'),
                                  epochWindow=epochWindow, chans=chans,
                                  downsampleFactor=4)
            savemat(os.path.join(savePath,
                                 subL[iSubs]+str(subjects[iSub]+1).zfill(3) +
                                 '.mat'), data)


def fetchAndParseKoreaFile(dataPath, url=None, epochWindow=[0.5, 2.5],
                           chans=None, downsampleFactor=4):
    """
    Parse one subjects EEG dat from Korea Uni MI dataset.

    Parameters
    ----------
    dataPath : str
        math to the EEG datafile EEG_MI.mat.
        if the file doesn't exists then it will be fetched over FTP using url
    url : str, optional
        FTP URL to fetch the data from. The default is None.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list
        channels to select from the data.
    downsampleFactor  : int
        Data down-sample factor

    Returns
    -------
    data : a eeg structure with following fields:
        x: 3d np array with epoched eeg data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints or channel names. .

    """
    # check if the file exists or fetch it over ftp
    if not os.path.exists(dataPath):
        if not os.path.exists(os.path.dirname(dataPath)):
            os.makedirs(os.path.dirname(dataPath))
        print('fetching data over ftp: ' + dataPath)
        with closing(request.urlopen(url)) as r:
            with open(dataPath, 'wb') as f:
                shutil.copyfileobj(r, f)

    # read the mat file:
    data = loadmat(dataPath)

    s = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    x = np.concatenate((data['EEG_MI_train'][0, 0]['smt'],
                        data['EEG_MI_test'][0, 0]['smt']),
                       axis=1).astype(np.float32)
    start, end = int(epochWindow[0]*s), int(epochWindow[1]*s)
    x = x[start:end]
    y = np.concatenate((data['EEG_MI_train'][0, 0]['y_dec'].squeeze(),
                        data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),
                       axis=0).astype(int)-1
    c = np.array([m.item() for m in data['EEG_MI_train'][0, 0][
        'chan'].squeeze().tolist()])

    del data

    # extract the requested channels:
    if chans is not None:
        x = x[:, :, np.array(chans)]
        c = c[np.array(chans)]

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((int(x.shape[0]/downsampleFactor), x.shape[1],
                         x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s/downsampleFactor,
                                             axis=0)
        x = xNew
        s = s/downsampleFactor

    # change the data dimensions to be in a format: Chan x time x trials
    x = np.transpose(x, axes=(2, 0, 1))
    return {'x': x, 'y': y.astype(np.int16), 'c': c, 's': s}


def parseKoreaDataset(datasetPath, savePath, epochWindow=[0.5, 2.5],
                      chans=None, subjects=None,
                      downsampleFactor=4):
    """
    Parse the Korea Uni. MI data in a MATLAB formate that will be used in the
    next analysis.
    The URL based fetching is a primitive code. So, please make sure not to
    interrupt it.
    Also, if you interrupt the process for any reason, remove the last
    downloaded subjects data.
    This is because, it's highly likely that the downloaded file for that
    subject will be corrupt.

    In spite of all this, make sure that you have close to 100GB free disk
    space and 70GB network bandwidth to properly download and save the MI data.

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV2a original dataset in gdf formate.
    savePath : str
        Path on where to save the epoched EEG data in a mat format.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list of int
        channels to select from the data.
    subjects : list of int
        IDs of the subjects to be selected. If None, the 54 subjects are
        included
    downsampleFactor : int / None, optional
        down-sampling factor to use. The default is 4.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    """
    if not subjects:
        subjects = list(range(54))
    # Base url for fetching any data that is not present!
    fetchUrlBase = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000'\
        '/100542/'
    subAll = [subjects, subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            print('Processing subject: ' + subL[iSubs]+str(sub))
            if not os.path.exists(os.path.join(savePath,
                                               subL[iSubs]+str(subjects[iSub] +
                                                               1).zfill(3) +
                                               '.mat')):
                fileUrl = fetchUrlBase + 'session' + str(
                    iSubs + 1) + '/s' + str(sub + 1) + '/' + 'sess' + str(
                        iSubs+1).zfill(2) + '_'+'subj' + str(sub+1).zfill(
                            2) + '_EEG_MI.mat'
                data = fetchAndParseKoreaFile(os.path.join(
                    datasetPath, 'session'+str(iSubs+1), 's'+str(sub+1),
                    'EEG_MI.mat'), fileUrl, epochWindow=epochWindow,
                    chans=chans, downsampleFactor=downsampleFactor)
                savemat(os.path.join(savePath,
                                     subL[iSubs]+str(sub+1).zfill(3)+'.mat'),
                        data)


def matToPython(datasetPath, savePath, isFiltered=False):
    """
    Convert the mat data to eegdataset and save it.

    Parameters
    ----------
    datasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    isFiltered : bool
        Indicates if the mat data is in the chan*time*trials*FilterBand format.
        default: False

    Returns
    -------
    None.

    """
    print('Creating python eegdataset with raw data.')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # load all the mat files
    data = []
    for root, dirs, files in os.walk(datasetPath):
        files = sorted(files)
        for f in files:
            parD = {}
            parD['fileName'] = f
            parD['data'] = {}
            d = loadmat(os.path.join(root, f),
                        verify_compressed_data_integrity=False)
            if isFiltered:
                parD['data']['eeg'] = np.transpose(d['x'],
                                                   (2, 0, 1, 3)).astype(
                                                       'float32')
            else:
                parD['data']['eeg'] = np.transpose(d['x'], (2, 0, 1)).astype(
                    'float32')

            parD['data']['labels'] = d['y']
            data.append(parD)

    # Start writing the files:
    # save the data in the eegdataloader format.
    # 1 file per sample in a dictionary formate with following fields:
        # id: unique key in 00001 formate
        # data: a 2 dimensional data matrix in chan*time formate
        # label: class of the data
    # Create another separate file to store the epoch info data.
    # This will contain all the intricate data division information.
        # There will be one entry for every data file and will be stored as a
        # 2D array and in csv file.
        # The column names are as follows:
        # id, label -> these should always be present.
        # Optional fields -> subject, session. -> they will be used in data
        # sorting.
    id = 0
    # header row
    dataLabels = [['id', 'relativeFilePath', 'label', 'subject', 'session']]

    for i, d in enumerate(data):
        sub = d['fileName'][-7:-4]  # subject of the data
        # sub = int(d['fileName'][-7:-4])  # subject of the data
        # sub = str(sub).zfill(3)

        if d['fileName'][1] == 'e':
            session = 1
        elif d['fileName'][1] == '-':
            session = int(d['fileName'][2:4])
        else:
            session = 0

        if len(d['data']['labels']) == 1:
            d['data']['labels'] = np.transpose(d['data']['labels'])

        for j, label in enumerate(d['data']['labels']):
            lab = label[0]
            # get the data
            if isFiltered:
                x = {'id': id, 'data': d['data']['eeg'][j, :, :, :],
                     'label': lab}
            else:
                x = {'id': id, 'data': d['data']['eeg'][j, :, :], 'label': lab}

            # dump it in the folder
            with open(os.path.join(savePath, str(id).zfill(5)+'.dat'),
                      'wb') as fp:
                pickle.dump(x, fp)

            # add in data label file
            dataLabels.append([id, str(id).zfill(5) + '.dat', lab, sub,
                               session])

            # increment id
            id += 1
    # Write the dataLabels file as csv
    with open(os.path.join(savePath, "dataLabels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataLabels)

    # write miscellaneous data info as csv file
    dataInfo = [['fs', 250], ['chanName', 'Check Original File']]
    with open(os.path.join(savePath, "dataInfo.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataInfo)


def pythonToMultiviewPython(datasetPath, savePath,
                            filterTransform={
                                'filterBank': {'filtBank': [[4, 8], [8, 12],
                                                            [12, 16], [16, 20],
                                                            [20, 24], [24, 28],
                                                            [28, 32], [32, 36],
                                                            [36, 40]],
                                               'fs': 250,
                                               'filtType': 'filter'}}):
    """
    .

    Convert the raw EEG data into its multi-view representation using a
    filter-bank specified with filterTransform.

    Parameters
    ----------
    datasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    filterTransform: dict
        filterTransform is a transform argument used to define the filter-bank.
        Please use the default settings unless you want to experiment with the
        filters.
        default: {'filterBank':{'filtBank':[[4,8],[8,12],[12,16],[16,20],
                                            [20,24],[24,28],[28,32],[32,36],
                                            [36,40]],'fs':250}}

    Returns
    -------
    None.
    Creates a new dataset and stores it in a savePath folder.

    """
    trasnformAndSave(datasetPath, savePath, transform=filterTransform)


def trasnformAndSave(datasetPath, savePath, transform=None):
    """
    Apply a data transform and save the result as a new eegdataset.

    Parameters
    ----------
    atasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    filterTransform: dict
        A transform to be applied on the data.

    Returns
    -------
    None.

    """
    if transform is None:
        return -1

    # Data options:
    config = {}
    config['preloadData'] = False  # process One by one
    config['transformArguments'] = transform
    config['inDataPath'] = datasetPath
    config['inLabelPath'] = os.path.join(config['inDataPath'],
                                         'dataLabels.csv')

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Outputs will be saved in folder : ' + savePath)

    # Check and compose transforms
    if len(config['transformArguments']) > 1:
        transform = transforms.Compose([transforms.__dict__[key](**value) for
                                        key, value in config[
                                            'transformArguments'].items()])
    else:
        transform = transforms.__dict__[list(config[
            'transformArguments'].keys())[0]](**config['transformArguments'][
                list(config['transformArguments'].keys())[0]])

    # Load the data
    data = eegDataset(dataPath=config['inDataPath'],
                      dataLabelsPath=config['inLabelPath'],
                      preloadData=config['preloadData'], transform=transform)

    # Write the transform applied data
    dLen = len(data)
    perDone = 0

    for i, d in enumerate(data):
        # 1-> realtive-path
        with open(os.path.join(savePath, data.labels[i][1]), 'wb') as fp:
            pickle.dump(d, fp)
        if i/dLen*100 > perDone:
            # print(str(perDone) + '% Completed')
            perDone += 1

    # Copy the labels and config files
    copyfile(config['inLabelPath'], os.path.join(savePath, 'dataLabels.csv'))
    copyfile(os.path.join(config['inDataPath'], "dataInfo.csv"),
             os.path.join(savePath, "dataInfo.csv"))

    # Store the applied transform in the transform . csv file
    with open(os.path.join(config['inDataPath'], "transform.csv"), 'w') as f:
        for key in config['transformArguments'].keys():
            f.write("%s,%s\n" % (key, config['transformArguments'][key]))


def fetchData(dataset, subjects=None):
    """
    .

    Check if the rawMat, rawPython, and multiviewPython data exists.
    If not, then create the above data.

    Parameters
    ----------
    dataset : str
        name of the dataset
    subjects : list
        list of the IDs of the subjects to be selected. If None, all the
        subjects are included
    -------
    None.

    """
    toolboxPath = os.path.dirname(os.path.dirname(os.path.dirname((
        os.path.abspath(__file__)))))
    dataFolder = os.path.join(toolboxPath, 'data', dataset)
    print('Checking data for dataset: ', dataset)
    oDataFolder = 'originalData'
    rawMatFolder = 'rawMat'
    rawPythonFolder = 'rawPython'
    multiviewPythonFolder = 'multiviewPython'
    if os.path.exists(os.path.join(dataFolder, rawMatFolder)):
        shutil.rmtree(os.path.join(dataFolder, rawMatFolder))
    if os.path.exists(os.path.join(dataFolder, rawPythonFolder)):
        shutil.rmtree(os.path.join(dataFolder, rawPythonFolder))
    if os.path.exists(os.path.join(dataFolder, multiviewPythonFolder)):
        shutil.rmtree(os.path.join(dataFolder, multiviewPythonFolder))

    # Generate the processed .mat data:
    if dataset.startswith('simdataset'):
        parseSimDataset(os.path.join(dataFolder, oDataFolder),
                        os.path.join(dataFolder, rawMatFolder),
                        subjects=subjects)
    elif dataset == 'bci41':
        parseBci41Dataset(os.path.join(dataFolder, oDataFolder),
                          os.path.join(dataFolder, rawMatFolder),
                          subjects=subjects)
    elif dataset == 'korea':
        parseKoreaDataset(os.path.join(dataFolder, oDataFolder),
                          os.path.join(dataFolder, rawMatFolder),
                          chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36,
                                 14, 37, 17, 38, 18, 39, 19, 40, 20],
                          subjects=subjects)
    elif dataset == 'koreasim':
        parseKoreaSimDataset(os.path.join(dataFolder, oDataFolder),
                             os.path.join(dataFolder, rawMatFolder),
                             chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36,
                                    14, 37, 17, 38, 18, 39, 19, 40, 20],
                             subjects=subjects)
    elif dataset == 'bci41sim':
        parseBci41SimDataset(os.path.join(dataFolder, oDataFolder),
                             os.path.join(dataFolder, rawMatFolder),
                             subjects=subjects)
    # Convert to eegdataset
    print('Converting to eegdataset.')
    matToPython(os.path.join(dataFolder, rawMatFolder),
                os.path.join(dataFolder, rawPythonFolder))

    # Convert to multiview python
    print('Converting to multi-view eegdataset.')
    pythonToMultiviewPython(os.path.join(dataFolder, rawPythonFolder),
                            os.path.join(dataFolder, multiviewPythonFolder))

    print('All the data you need is present! ')
