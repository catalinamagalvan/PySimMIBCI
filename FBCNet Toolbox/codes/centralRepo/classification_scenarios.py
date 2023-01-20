# Hold out classification analysis of BCI Comp IV-2a and Korea datasets
# @author: Ravikiran Mane
import numpy as np
import torch
import os
import time
import xlwt
import csv
import math
import copy
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from sklearn.model_selection import train_test_split
from statistics import mean

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setRandom(seed):
    """
    Set all the random initializations with a given seed.

    Parameters
    ----------
    seed : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Set np
    np.random.seed(seed)

    # Set torch
    torch.manual_seed(seed)

    # Set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def excelAddData(worksheet, startCell, data, isNpData=False):
    """
    .

    Write the given max 2D data to a given given worksheet
    starting from the start-cell.
    List will be treated as a row.
    List of list will be treated in a matrix format with inner
    list constituting a row.
    will return the modified worksheet which needs to be written
    to a file
    isNpData flag indicate whether the incoming data in the list
    is of np data-type

    Parameters
    ----------
    worksheet : TYPE
        DESCRIPTION.
    startCell : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    isNpData : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    worksheet : TYPE
        DESCRIPTION.

    """
    #  Check the input type.
    if type(data) is not list:
        data = [[data]]
    elif type(data[0]) is not list:
        data = [data]
    else:
        data = data

    # write the data. starting from the given start cell.
    rowStart = startCell[0]
    colStart = startCell[1]

    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if isNpData:
                worksheet.write(rowStart+i, colStart+j, col.item())
            else:
                worksheet.write(rowStart+i, colStart+j, col)

    return worksheet


def dictToCsv(filePath, dictToWrite):
    """
    Write a dictionary to a given csv file.

    Parameters
    ----------
    filePath : TYPE
        DESCRIPTION.
    dictToWrite : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    with open(filePath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictToWrite.items():
            writer.writerow([key, value])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if
               p.requires_grad)


def generateBalancedFolds(idx, label, kFold=5):
    '''
    Generate a class aware splitting of the data index in given number of
    folds.
    Returns list with k sublists corresponding to the k fold splitting.
    '''
    from sklearn.model_selection import StratifiedKFold
    folds = []
    skf = StratifiedKFold(n_splits=kFold)
    for train, test in skf.split(idx, label):
        folds.append([idx[i] for i in list(test)])
    return folds


def splitKfold(idx1, k, doShuffle=True):
    '''
    Split the index from given list in k random parts.
    Returns list with k sublists.
    '''
    idx = copy.deepcopy(idx1)
    lenFold = math.ceil(len(idx)/k)
    if doShuffle:
        np.random.shuffle(idx)
    return [idx[i*lenFold:i*lenFold+lenFold] for i in range(k)]


def loadSplitFold(idx, path, subNo):
    '''
    Load the CV fold details saved in json formate.
    Returns list with k sublists corresponding to the k fold splitting.
    subNo is the number of the subject to load from. starts from 0
    '''
    import json
    with open(path) as json_file:
        data = json.load(json_file)
    data = data[subNo]
    # sort the values in sublists
    folds = []
    for i in list(set(data)):
        folds.append([idx[j] for (j, val) in enumerate(data) if val == i])

    return folds


def set_default_config():
    """
    Define the default model and training related options here.

    Returns
    -------
    config : dict
        The config dict.

    """
    config = {}
    config['batchSize'] = 16
    # Data load options:
    config['preloadData'] = False
    # Training related details
    config['modelTrainArguments'] = {'stopCondi': {'c': {'Or': {'c1': {
        'MaxEpoch': {'maxEpochs': 1500, 'varName': 'epoch'}},
        'c2': {'NoDecrease': {'numEpochs': 200, 'varName': 'valInacc'}}}}},
          'classes': [0, 1], 'sampler': 'RandomSampler',
          'loadBestModel': True, 'bestVarToCheck': 'valInacc', 'lr': 1e-3}
    config['transformArguments'] = None
    config['data'] = 'raw'
    # network initialization details:
    config['loadNetInitState'] = True

    return config


def set_config_model_arguments(dataset):
    if dataset == 'korea':
        config_dict = {'nChan': 20, 'nTime': 500,
                                    'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32,
                                    'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    elif dataset == 'bci41':
        config_dict = {'nChan': 41, 'nTime': 500,
                                    'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32,
                                    'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    elif dataset == 'simdataset':
        config_dict = {'nChan': 41, 'nTime': 500,
                                    'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32,
                                    'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    return config_dict


def print_confusion_matrices(trainResults, valResults, testResults):
    """
    Generates and print confusion matrices from train, validation and test
    results.

    Parameters
    ----------
    trainResults : list of dict
        List of dictionaries, each of which contains training results for one
        subject.
    valResults : list of dict
        List of dictionaries, each of which contains validation results for one
        subject.
    testResults : list of dict
        List of dictionaries, each of which contains test results for one
        subject.

    Returns
    -------
    None.

    """
    # append the confusion matrix
    trainCm = [[r['cm'] for r in result] for result in trainResults]
    trainCm = list(map(list, zip(*trainCm)))
    trainCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for
               cms in trainCm]

    valCm = [[r['cm'] for r in result] for result in valResults]
    valCm = list(map(list, zip(*valCm)))
    valCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for
             cms in valCm]

    testCm = [[r['cm'] for r in result] for result in testResults]
    testCm = list(map(list, zip(*testCm)))
    testCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for
              cms in testCm]
    print('Train confusion matrix:')
    print(trainCm)
    print('Validation confusion matrix:')
    print(valCm)
    print('Test confusion matrix:')
    print(testCm)


def cross_session(dataset, network=None, nGPU=None, random_seed_list=[0]):
    # Set the defaults use these to quickly run the network
    network = network or 'FBCNet'
    nGPU = nGPU or 0
    config = set_default_config()
    config['network'] = network
    # how much of the training data will be used a validation set
    config['validationSet'] = 0.2
    config['pathNetInitState'] = config['network'] + '_' + dataset
    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels',
                                              config['pathNetInitState'] +
                                              '.pth')
    if os.path.exists(os.path.join(masterPath, 'netInitModels')):
        for f in os.listdir(os.path.join(masterPath, 'netInitModels')):
            os.remove(os.path.join(masterPath, 'netInitModels', f))
    else:
        os.makedirs(os.path.join(masterPath, 'netInitModels'))
    # Define data path things
    # Input data base folder:
    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = os.path.join(toolboxPath, 'data')

    # Input data datasetId folders
    if 'FBCNet' in config['network']:
        modeInFol = 'multiviewPython'  # FBCNet uses multi-view data
    else:
        modeInFol = 'rawPython'

    # set final input location
    config['inDataPath'] = os.path.join(config['inDataPath'], dataset,
                                        modeInFol)

    # Path to the input data labels file
    config['inLabelPath'] = os.path.join(config['inDataPath'],
                                         'dataLabels.csv')

    config['modelArguments'] = set_config_model_arguments(dataset)

    for seed in random_seed_list:
        config['randSeed'] = seed
        setRandom(config['randSeed'])

        # Output folder:
        # Lets store all the outputs of the given run in folder.
        config['outPath'] = os.path.join(toolboxPath, 'output')
        config['outPath'] = os.path.join(config['outPath'], 'cross_session',
                                         network, dataset, 'seed' + str(config[
                                             'randSeed']))
        # create output folder
        # create the path
        if not os.path.exists(config['outPath']):
            os.makedirs(config['outPath'])
        print('Outputs will be saved in folder : ' + config['outPath'])

        # Write the config dictionary
        dictToCsv(os.path.join(config['outPath'], 'config.csv'), config)

        # Check and compose transforms
        if config['transformArguments'] is not None:
            if len(config['transformArguments']) > 1:
                transform = transforms.Compose([transforms.__dict__[key](
                    **value) for key, value in config[
                        'transformArguments'].items()])
            else:
                transform = transforms.__dict__[list(config[
                    'transformArguments'].keys())[0]](**config[
                        'transformArguments'][list(config[
                            'transformArguments'].keys())[0]])
        else:
            transform = None

        # Check and Load the data
        data = eegDataset(dataPath=config['inDataPath'],
                          dataLabelsPath=config['inLabelPath'],
                          preloadData=config['preloadData'],
                          transform=transform)
        print('Data loading finished')

        # Check and load the model
        if config['network'] in networks.__dict__.keys():
            netw = networks.__dict__[config['network']]
        else:
            raise AssertionError('No network named ' + config['network'] +
                                 ' is not defined in the networks.py file')

        # Load the net and print trainable parameters:
        net = netw(**config['modelArguments'])
        print('Trainable Parameters in the network are: ' + str(
            count_parameters(net)))

        # Check and load/save the the network initialization.
        if config['loadNetInitState']:
            setRandom(config['randSeed'])
            net = netw(**config['modelArguments'])
            netInitState = net.to('cpu').state_dict()
            torch.save(netInitState, config['pathNetInitState'])

        # Find all the subjects to run
        subs = sorted(set([d[3] for d in data.labels]))

        # Begin training
        trainResults = []
        valResults = []
        testResults = []
        bestEpochResults = []
        for iSub, sub in enumerate(subs):
            if config['loadNetInitState']:
                setRandom(config['randSeed'])
                net = netw(**config['modelArguments'])
                netInitState = net.to('cpu').state_dict()
                torch.save(netInitState, config['pathNetInitState'])

            start = time.time()
            # extract subject data
            subIdx = [i for i, x in enumerate(data.labels) if x[3] in sub]
            subData = copy.deepcopy(data)
            subData.createPartialDataset(subIdx, loadNonLoadedData=True)

            trainData = copy.deepcopy(subData)
            testData = copy.deepcopy(subData)

            # Isolate the train -> session 0 and test data-> session 1
            if len(subData.labels[0]) > 4:
                idxTrain = [i for i, x in enumerate(subData.labels) if x[4] ==
                            '0']
                idxTest = [i for i, x in enumerate(subData.labels) if x[4] ==
                           '1']
            else:
                raise ValueError("The data can not be divided based on the "
                                 "sessions")

            testData.createPartialDataset(idxTest)
            trainData.createPartialDataset(idxTrain)

            # extract the desired amount of train data:
            trainData.createPartialDataset(list(range(0, math.ceil(len(
                trainData)))))
            y = np.array([x[2] for _, x in enumerate(trainData.labels)])
            idxTrain_all = list(range(0, math.ceil(len(trainData))))
            idxTrain, idxVal = train_test_split(idxTrain_all,
                                                stratify=y,
                                                test_size=config[
                                                    'validationSet'],
                                                random_state=0)
            # isolate the train and validation set
            valData = copy.deepcopy(trainData)
            valData.createPartialDataset(idxVal)
            trainData.createPartialDataset(idxTrain)

            # Call the network for training
            net = netw(**config['modelArguments'])
            net.load_state_dict(netInitState, strict=False)

            outPathSub = os.path.join(config['outPath'], 'sub' + str(sub))
            model = baseModel(net=net, resultsSavePath=outPathSub,
                              seed=config['randSeed'],
                              batchSize=config['batchSize'], nGPU=nGPU)
            model.train(trainData, valData, testData,
                        **config['modelTrainArguments'])

            # extract the important results.
            trainResults.append([d['results']['trainBest'] for d in
                                 model.expDetails])
            valResults.append([d['results']['valBest'] for d in
                               model.expDetails])
            testResults.append([d['results']['test'] for d in
                                model.expDetails])
            bestEpochResults.append(model.expDetails[0]['results'][
                'epochBest'])

            # save the results
            results = {'train:': trainResults[-1], 'val: ': valResults[-1],
                       'test': testResults[-1], 'epoch': bestEpochResults[-1]}
            dictToCsv(os.path.join(outPathSub, 'results.csv'), results)

            # Time taken
            print("Time taken = " + str(time.time() - start))

        # Group the results for all the subjects using experiment.
        # the train, test and val accuracy and cm will be written
        trainAcc = [[r['acc'] for r in result] for result in trainResults]
        trainAcc = list(map(list, zip(*trainAcc)))
        valAcc = [[r['acc'] for r in result] for result in valResults]
        valAcc = list(map(list, zip(*valAcc)))
        testAcc = [[r['acc'] for r in result] for result in testResults]
        testAcc = list(map(list, zip(*testAcc)))
        bestEpoch = bestEpochResults
        print('Best epoch: ' + str(bestEpoch))

        print_confusion_matrices(trainResults, valResults, testResults)


def cross_session_aug(dataset, augmentation, nTrialsAugList=[1], network=None,
                      nGPU=None, random_seed_list=[0]):
    # Set the defaults use these to quickly run the network
    network = network or 'FBCNet'
    nGPU = nGPU or 0
    config = set_default_config()
    config['network'] = network
    # how much of the training data will be used a validation set
    config['validationSet'] = 0.2
    config['pathNetInitState'] = config['network'] + '_' + dataset
    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels',
                                              config['pathNetInitState'] +
                                              '.pth')
    if os.path.exists(os.path.join(masterPath, 'netInitModels')):
        for f in os.listdir(os.path.join(masterPath, 'netInitModels')):
            os.remove(os.path.join(masterPath, 'netInitModels', f))
    else:
        os.makedirs(os.path.join(masterPath, 'netInitModels'))
    # Define data path things
    # Input data base folder:
    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = os.path.join(toolboxPath, 'data')
    # Input data datasetId folders
    if 'FBCNet' in config['network']:
        modeInFol = 'multiviewPython'  # FBCNet uses multi-view data
    else:
        modeInFol = 'rawPython'
    # set final input location
    config['inDataPathreal'] = os.path.join(config['inDataPath'],
                                            dataset, modeInFol)
    config['inDataPathsim'] = os.path.join(config['inDataPath'],
                                           dataset + augmentation,
                                           modeInFol)
    # Path to the input data labels file
    config['inLabelPathreal'] = os.path.join(config['inDataPathreal'],
                                             'dataLabels.csv')
    config['inLabelPathsim'] = os.path.join(config['inDataPathsim'],
                                            'dataLabels.csv')

    # Input data datasetId folders
    if 'FBCNet' in config['network']:
        modeInFol = 'multiviewPython'  # FBCNet uses multi-view data
    else:
        modeInFol = 'rawPython'

    if dataset == 'korea':
        config['modelArguments'] = {'nChan': 20, 'nTime': 500,
                                    'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32,
                                    'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    elif dataset == 'bci41':
        config['modelArguments'] = {'nChan': 41, 'nTime': 500,
                                    'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32,
                                    'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    elif dataset == 'simdataset':
        config['modelArguments'] = {'nChan': 41, 'nTime': 500,
                                    'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32,
                                    'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    for seed in random_seed_list:
        config['randSeed'] = seed
        # Output folder:
        # Lets store all the outputs of the given run in folder.
        config['outPath'] = os.path.join(toolboxPath, 'output')
        config['outPath'] = os.path.join(config['outPath'], 'cross_session',
                                         network, dataset + 'aug',
                                         augmentation,
                                         'seed' + str(config['randSeed']))
        # %% create output folder
        if not os.path.exists(config['outPath']):
            os.makedirs(config['outPath'])
        print('Outputs will be saved in folder : ' + config['outPath'])
        # Write the config dictionary
        dictToCsv(os.path.join(config['outPath'], 'config.csv'), config)

        # %% Load the data
        print('Data loading in progress')
        # Check and compose transforms
        if config['transformArguments'] is not None:
            if len(config['transformArguments']) > 1:
                transform = transforms.Compose([transforms.__dict__[key](
                    **value) for key, value in config[
                        'transformArguments'].items()])
            else:
                transform = transforms.__dict__[list(config[
                    'transformArguments'].keys())[0]](**config[
                        'transformArguments'][list(config[
                            'transformArguments'].keys())[0]])
        else:
            transform = None
        data_real = eegDataset(dataPath=config['inDataPathreal'],
                               dataLabelsPath=config['inLabelPathreal'],
                               preloadData=config['preloadData'],
                               transform=transform)
        data_sim = eegDataset(dataPath=config['inDataPathsim'],
                              dataLabelsPath=config['inLabelPathsim'],
                              preloadData=config['preloadData'],
                              transform=transform)
        print('Data loading finished')

        # Check and load the model
        if config['network'] in networks.__dict__.keys():
            netw = networks.__dict__[config['network']]
        else:
            raise AssertionError('No network named ' + config['network'] + ' '
                                 'is not defined in the networks.py file')

        # Load the net and print trainable parameters:
        net = netw(**config['modelArguments'])
        print('Trainable Parameters in the network are: ' + str(
            count_parameters(net)))

        # Check and load/save the the network initialization.
        if config['loadNetInitState']:
            setRandom(config['randSeed'])
            net = netw(**config['modelArguments'])
            netInitState = net.to('cpu').state_dict()
            torch.save(netInitState, config['pathNetInitState'])

        # Find all the subjects to run
        subs_real = sorted(set([d[3] for d in data_real.labels]))

        # Let the training begin
        trainResults = []
        valResults = []
        testResults = []
        bestEpochResults = []
        for iSub, sub in enumerate(subs_real):
            if config['loadNetInitState']:
                setRandom(config['randSeed'])
                net = netw(**config['modelArguments'])
                netInitState = net.to('cpu').state_dict()
                torch.save(netInitState, config['pathNetInitState'])
            # extract simulated subject data
            subIdxSim = [i for i, x in enumerate(data_sim.labels) if
                         x[3] in str(sub).zfill(3)]
            augData = copy.deepcopy(data_sim)
            augData.createPartialDataset(subIdxSim, loadNonLoadedData=True)

            # extract subject data
            subIdx = [i for i, x in enumerate(data_real.labels) if x[3] in
                      sub]
            subData = copy.deepcopy(data_real)
            subData.createPartialDataset(subIdx, loadNonLoadedData=True)

            trainData = copy.deepcopy(subData)
            testData = copy.deepcopy(subData)

            # Isolate the train -> session 0 and test data-> session 1
            if len(subData.labels[0]) > 4:
                idxTrain = [i for i, x in enumerate(subData.labels) if
                            x[4] == '0']
                idxTest = [i for i, x in enumerate(subData.labels) if
                           x[4] == '1']
            else:
                raise ValueError('The data can not be divided based on the'
                                 ' sessions')

            testData.createPartialDataset(idxTest)
            trainData.createPartialDataset(idxTrain)

            # extract the desired amount of train data:
            trainData.createPartialDataset(list(range(0, math.ceil(
                len(trainData)))))
            y = np.array([x[2] for _, x in enumerate(trainData.labels)])
            idxTrain_all = list(range(0, math.ceil(len(trainData))))
            idxTrain, idxVal = train_test_split(idxTrain_all,
                                                stratify=y,
                                                test_size=config[
                                                    'validationSet'],
                                                random_state=0)
            # isolate the train and validation set
            valData = copy.deepcopy(trainData)
            valData.createPartialDataset(idxVal)
            trainData.createPartialDataset(idxTrain)

            for i, N_trials_aug in enumerate(nTrialsAugList):
                # Call the network for training
                setRandom(config['randSeed'])
                net = netw(**config['modelArguments'])
                net.load_state_dict(netInitState, strict=False)

                outPathSub = os.path.join(config['outPath'], 'sub' +
                                          str(sub), 'aug' + str(
                                              N_trials_aug))
                model = baseModel(net=net, resultsSavePath=outPathSub,
                                  seed=config['randSeed'],
                                  batchSize=config['batchSize'],
                                  nGPU=nGPU)
                model.train(trainData, valData, testData, augData,
                            N_trials_aug,
                            **config['modelTrainArguments'])

                # extract the important results.
                trainResults.append([d['results']['trainBest'] for d in
                                     model.expDetails])
                valResults.append([d['results']['valBest'] for d in
                                   model.expDetails])
                testResults.append([d['results']['test'] for d in
                                    model.expDetails])
                bestEpochResults.append(model.expDetails[0]['results'][
                    'epochBest'])

                # save the results
                results = {'train:': trainResults[-1],
                           'val: ': valResults[-1],
                           'test': testResults[-1],
                           'epoch': bestEpochResults[-1]}
                dictToCsv(os.path.join(outPathSub, 'results.csv'),
                          results)

        print('Best epoch: ' + str(bestEpochResults))

        print_confusion_matrices(trainResults, valResults, testResults)


def cross_validation(dataset, network=None, nGPU=None, random_seed_list=[0]):
    # Set the defaults use these to quickly run the network
    network = network or 'FBCNet'
    nGPU = nGPU or 0

    # Define all the model and training related options here.
    config = set_default_config()
    # add some more run specific details.
    config['network'] = network
    config['cv'] = 'subSpecific-Kfold'
    config['kFold'] = 10
    # Network related details
    config['modelArguments'] = set_config_model_arguments(dataset)
    # CV fold details.
    # These files have been written to achieve consistent division of trials in
    # fold across all methods.
    # For random division set config['loadCVFold'] to False
    config['loadCVFold'] = True
    config['pathCVFold'] = {dataset: 'CVIdx-subSpec-' + dataset + '-seq.json'}

    # network initialization details:
    config['loadNetInitState'] = True
    config['pathNetInitState'] = config['network'] + '_' + dataset

    for seed in random_seed_list:
        config['randSeed'] = seed
        setRandom(config['randSeed'])
        # %% Define data path things here. Do it once and forget it!
        # Input data base folder:
        toolboxPath = os.path.dirname(masterPath)
        config['inDataPath'] = os.path.join(toolboxPath, 'data')
        # Input data datasetId folders
        if 'FBCNet' in config['network']:
            modeInFol = 'multiviewPython'  # FBCNet uses multi-view data
        else:
            modeInFol = 'rawPython'
        # set final input location
        config['inDataPath'] = os.path.join(config['inDataPath'],
                                            dataset, modeInFol)
        # Path to the input data labels file
        config['inLabelPath'] = os.path.join(config['inDataPath'],
                                             'dataLabels.csv')
        # give full path to the pathCVFold
        for key, val in config['pathCVFold'].items():
            config['pathCVFold'][key] = os.path.join(masterPath, 'cvFiles',
                                                     val)
        config['pathCVFold'] = config['pathCVFold'][dataset]
        # cv fold divisions are only provided for 10-fold cv.
        if (not os.path.exists(config['pathCVFold'])) or config['kFold'] != 10:
            config['loadCVFold'] = False

        # Output folder:
        # Lets store all the outputs of the given run in folder.
        config['outPath'] = os.path.join(toolboxPath, 'output')
        config['outPath'] = os.path.join(config['outPath'], 'cross_validation',
                                         network, dataset,
                                         'seed' + str(config['randSeed']))
        # Network initialization:
        config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels',
                                                  config['pathNetInitState'] +
                                                  '.pth')
        if not os.path.exists(config['outPath']):
            os.makedirs(config['outPath'])
        print('Outputs will be saved in folder : ' + config['outPath'])

        # Write the config dictionary
        dictToCsv(os.path.join(config['outPath'], 'config.csv'), config)

        # Check and compose transforms
        if config['transformArguments'] is not None:
            if len(config['transformArguments']) > 1:
                transform = transforms.Compose(
                    [transforms.__dict__[key](**value) for key, value in
                     config['transformArguments'].items()])
            else:
                transform = transforms.__dict__[list(
                    config['transformArguments'].keys())[0]](**config[
                        'transformArguments'][list(config[
                            'transformArguments'].keys())[0]])
        else:
            transform = None

        # check and Load the data
        data = eegDataset(dataPath=config['inDataPath'],
                          dataLabelsPath=config['inLabelPath'],
                          preloadData=config['preloadData'],
                          transform=transform)
        print('Data loading finished')

        # Select only the session 1 data
        if len(data.labels[0]) > 4:
            idx = [i for i, x in enumerate(data.labels) if x[4] == '0']
            data.createPartialDataset(idx)

        # Check and load the model
        if config['network'] in networks.__dict__.keys():
            netw = networks.__dict__[config['network']]
        else:
            raise AssertionError('No network named ' + config['network'] + ' '
                                 'defined in the networks.py file')

        # Load the net and print trainable parameters:
        net = netw(**config['modelArguments'])
        print('Trainable Parameters in the network are: ' + str(
            count_parameters(net)))

        # Find all the subjects to run
        subs = sorted(set([d[3] for d in data.labels]))

        # Begin training
        trainResults = []
        valResults = []
        testResults = []

        for i, sub in enumerate(subs):
            # Check and save the the network initialization.
            if config['loadNetInitState']:
                setRandom(config['randSeed'])
                net = netw(**config['modelArguments'])
                netInitState = net.to('cpu').state_dict()
                torch.save(netInitState, config['pathNetInitState'])

            start = time.time()
            # Run the cross-validation over all the folds
            subIdx = [i for i, x in enumerate(data.labels) if x[3] in sub]
            subY = [data.labels[i][2] for i in subIdx]

            if config['loadCVFold']:
                subIdxFold = loadSplitFold(subIdx, config['pathCVFold'], i)
            else:
                subIdxFold = generateBalancedFolds(subIdx, subY,
                                                   config['kFold'])

            trainResultsCV = []
            valResultsCV = []
            testResultsCV = []

            for j, folds in enumerate(subIdxFold):
                # for each fold:
                testIdx = folds
                rFolds = copy.deepcopy(subIdxFold)
                if j+1 < config['kFold']:
                    valIdx = rFolds[j+1]
                else:
                    valIdx = rFolds[0]
                rFolds.remove(testIdx)
                rFolds.remove(valIdx)
                trainIdx = [i for sl in rFolds for i in sl]

                # separate the train, test and validation data
                testData = copy.deepcopy(data)
                testData.createPartialDataset(testIdx, loadNonLoadedData=True)
                trainData = copy.deepcopy(data)
                trainData.createPartialDataset(trainIdx,
                                               loadNonLoadedData=True)
                valData = copy.deepcopy(data)
                valData.createPartialDataset(valIdx, loadNonLoadedData=True)

                # Call the network
                net = netw(**config['modelArguments'])
                net.load_state_dict(netInitState, strict=False)
                outPathSub = os.path.join(config['outPath'], 'sub' + sub,
                                          'fold' + str(j))
                model = baseModel(net=net, resultsSavePath=outPathSub,
                                  batchSize=config['batchSize'], nGPU=nGPU,
                                  seed=config['randSeed'])
                model.train(trainData, valData, testData,
                            **config['modelTrainArguments'])

                # extract the important results.
                trainResultsCV.append([d['results']['trainBest'] for d in
                                       model.expDetails])
                valResultsCV.append([d['results']['valBest'] for d in
                                     model.expDetails])
                testResultsCV.append([d['results']['test'] for d in
                                      model.expDetails])
                # save the results
                results = {'train:': [d['results']['trainBest'] for d in
                                      model.expDetails],
                           'val: ': [d['results']['valBest'] for d in
                                     model.expDetails],
                           'test': [d['results']['test'] for d in
                                    model.expDetails]}
                dictToCsv(os.path.join(outPathSub, 'results.csv'), results)

            # Average the results. : This is only required for excel based
            # reporting
            # You don't need this if you plan to just note the results from a
            # terminal
            trainAccCV = [[r['acc'] for r in result] for result in
                          trainResultsCV]
            trainAccCV = list(map(list, zip(*trainAccCV)))
            trainAccCV = [mean(data) for data in trainAccCV]
            valAccCV = [[r['acc'] for r in result] for result in valResultsCV]
            valAccCV = list(map(list, zip(*valAccCV)))
            valAccCV = [mean(data) for data in valAccCV]
            testAccCV = [[r['acc'] for r in result] for result in
                         testResultsCV]
            testAccCV = list(map(list, zip(*testAccCV)))
            testAccCV = [mean(data) for data in testAccCV]

            # same for CM
            trainCmCV = [[r['cm'] for r in result] for result in
                         trainResultsCV]
            trainCmCV = list(map(list, zip(*trainCmCV)))
            trainCmCV = [np.stack(tuple([cm for cm in cms]), axis=2) for cms in
                         trainCmCV]
            trainCmCV = [np.mean(data, axis=2) for data in trainCmCV]

            valCmCV = [[r['cm'] for r in result] for result in valResultsCV]
            valCmCV = list(map(list, zip(*valCmCV)))
            valCmCV = [np.stack(tuple([cm for cm in cms]), axis=2) for cms in
                       valCmCV]
            valCmCV = [np.mean(data, axis=2) for data in valCmCV]

            testCmCV = [[r['cm'] for r in result] for result in testResultsCV]
            testCmCV = list(map(list, zip(*testCmCV)))
            testCmCV = [np.stack(tuple([cm for cm in cms]), axis=2) for cms in
                        testCmCV]
            testCmCV = [np.mean(data, axis=2) for data in testCmCV]

            # Put everything back.
            temp1, temp2, temp3 = [], [], []

            for iTemp, trainAc in enumerate(trainAccCV):
                temp1.append({'acc': trainAc, 'cm': trainCmCV[iTemp]})
                temp2.append({'acc': valAccCV[iTemp], 'cm': valCmCV[iTemp]})
                temp3.append({'acc': testAccCV[iTemp], 'cm': testCmCV[iTemp]})

            # append to original results
            trainResults.append(temp1)
            valResults.append(temp2)
            testResults.append(temp3)

            # Time taken
            print("Time taken = " + str(time.time() - start))

        # %% Extract and write the results to excel file.
        # You don't need this if you plan to just note the results from a
        # terminal

        # lets group the results for all the subjects using experiment.
        # the train, test and val accuracy and cm will be written
        trainAcc = [[r['acc'] for r in result] for result in trainResults]
        trainAcc = list(map(list, zip(*trainAcc)))
        valAcc = [[r['acc'] for r in result] for result in valResults]
        valAcc = list(map(list, zip(*valAcc)))
        testAcc = [[r['acc'] for r in result] for result in testResults]
        testAcc = list(map(list, zip(*testAcc)))

        print("Results sequence is train, val , test")
        print(trainAcc)
        print(valAcc)
        print(testAcc)

        print_confusion_matrices(trainResults, valResults, testResults)
