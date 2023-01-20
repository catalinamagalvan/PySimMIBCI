#!/usr/bin/env python
# coding: utf-8

"""
     The Base model for any deep learning analysis.
     This class should provide following functionalities for any deep learning
     module
         1. train() -> Train the model
         2. predict() -> Evaluate the train, validation, and test performance
         3. Create train and validation graphs
         4. Run over CPU/ GPU (if available)
     This class needs following things to run:
         1. net -> The architecture of the network. It should inherit nn.Module
             and should define the forward method
         2. trainData, testData and validateData -> these should be eegDatasets
             and data iterators will be forked out of them
             Each sample of these datasets should be a dictionary with two
             fields: 'data' and 'label'
         3. optimizer -> the optimizer of type torch.optim.
         4. outFolder -> the folder where the results will be stored.
         5. preferedDevice -> 'gpu'/'cpu' -> will run on gpu only if it's
             available
    TODO: Include a learning rate scheduler in _trainOE.
    TODO: Add a good hyper-parameter optimizer in the train.
    @author: Ravikiran Mane
"""

# To do deep learning
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
import torch.utils.data.sampler as builtInSampler
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
import pickle
import copy
from sklearn.model_selection import train_test_split

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
import stopCriteria
import samplers

class baseModel():
    def __init__(
        self,
        net,
        resultsSavePath=None,
        seed=3141592,
        setRng=True,
        preferedDevice='gpu',
        nGPU=0,
        batchSize=1):
        self.net = net
        self.seed = seed
        self.preferedDevice = preferedDevice
        self.batchSize = batchSize
        self.setRng = setRng
        self.resultsSavePath = resultsSavePath
        self.device = None

        # Set RNG
        if self.setRng:
            self.setRandom(self.seed)

        # Set device
        self.setDevice(nGPU)
        self.net = net.to(self.device)

        # Check for the results save path.
        if self.resultsSavePath is not None:
            if not os.path.exists(self.resultsSavePath):
                os.makedirs(self.resultsSavePath)
            print('Results will be saved in folder : ' + self.resultsSavePath)

    def train(self, trainData, valData, testData=None, augData=None,
              nTrialsAug=0, classes=None, lossFn='NLLLoss', optimFns='Adam',
              optimFnArgs={}, sampler=None, lr=0.001,
              stopCondi={'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500,
                                                          'varName': 'epoch'}},
                                      'c2': {'NoDecrease': {'numEpochs': 200,
                                                            'varName':
                                                                'valInacc'}}}
                               }}, loadBestModel=True,
              bestVarToCheck='valInacc'):
        """
        Apex function to train and test any network.
        Calls _trainOE for base training and adds the reporting capabilities.

        Parameters
        ----------
        trainData : eegDataset
            dataset used for training.
        valData : eegDataset
            dataset used for validation.
        testData : eegDataset, optional
            dataset to calculate the results on. The default is None.
        classes : list, optional
            List of classes to consider in evaluation matrices.
            None -> all classes.
            The default is None.
        lossFn : string from torch.nn, The default is 'NLLLoss'
            Name of the loss function from torch.nn which will be used for
            training.
        optimFns : string from torch.optim. The default is 'Adam'.
            Name of the optimization function from torch.nn which will be used
            for training.
        optimFnArgs : dict, optional
            Additional arguments to be passed to the optimizer.
            The default is {}.
        sampler : a string specifying sampler to be used in dataloader
            optional
            The sampler to use while training and validation.
            Function with this name will be searched at two places,
                1. torch.utils.data.sampler, 2. samplers
                if not found then error will be thrown.
            The default is None.
        lr : float, optional
            Learning rate. The default is 0.001.
        stopCondi : dict, optional
            Determines when to stop.
            It will be a dictionary which can be accepted by stopCriteria class
            The default is : no decrease in validation Inaccuracy in last
            200 epochs OR epoch > 1500
            This default is represented as:
            {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500,
                                              'varName': 'epoch'}},
                          'c2': {'NoDecrease': {'numEpochs': 200,
                                                'varName': 'valInacc'}}}}}
        loadBestModel : bool, optional
            Whether to load the network with best validation loss/ accuracy
            at the end of training. The default is True.
        bestVarToCheck : 'valInacc'/'valLoss', optional
            the best value to check while determining the best model.
            The default is 'valInacc'.
        Returns
        -------
        None.
        """
        # define the classes
        if classes is None:
            labels = [label[2] for label in trainData.labels]
            classes = list(set(labels))

        # Define the sampler
        if sampler is not None:
            sampler = self._findSampler(sampler)

        # Create the loss function
        lossFn = self._findLossFn(lossFn)(reduction='sum')

        # store the experiment details.
        self.expDetails = []

        # Lets run the experiment
        expNo = 0
        original_net_dict = copy.deepcopy(self.net.state_dict())

        # set the details
        expDetail = {'expNo': expNo, 'expParam': {'optimFn': optimFns,
                                                  'lossFn': lossFn, 'lr': lr,
                                                  'stopCondi': stopCondi}}

        # Reset the network to its initial form.
        self.net.load_state_dict(original_net_dict)

        # Run the training and get the losses.
        trainResults = self._trainOE(trainData, valData, augData, nTrialsAug,
                                     lossFn, optimFns, lr, stopCondi,
                                     optimFnArgs, classes=classes,
                                     sampler=sampler,
                                     loadBestModel=loadBestModel,
                                     bestVarToCheck=bestVarToCheck)

        # store the results and netParm
        expDetail['results'] = {'train': trainResults}
        expDetail['netParam'] = copy.deepcopy(self.net.to('cpu').state_dict())

        self.net.to(self.device)
        # If you are restoring the best model at the end of training then get
        # the final results again.
        pred, act, loss = self.predict(trainData, sampler=sampler,
                                       lossFn=lossFn)
        trainResultsBest = self.calculateResults(pred, act, classes=classes)
        trainResultsBest['loss'] = loss
        pred, act, loss = self.predict(valData, sampler=sampler, lossFn=lossFn)
        valResultsBest = self.calculateResults(pred, act, classes=classes)
        valResultsBest['loss'] = loss
        expDetail['results']['trainBest'] = trainResultsBest
        expDetail['results']['valBest'] = valResultsBest
        expDetail['results']['epochBest'] = trainResults['epoch']
        # if test data is present then get the results for the test data.
        if testData is not None:
            pred, act, loss = self.predict(testData, sampler=sampler,
                                           lossFn=lossFn)
            testResults = self.calculateResults(pred, act, classes=classes)
            testResults['loss'] = loss
            expDetail['results']['test'] = testResults

        # Print the final output to the console:
        print("Exp No. : " + str(expNo + 1))
        print('________________________________________________')
        print("\n Train Results: ")
        print(expDetail['results']['trainBest'])
        print('\n Validation Results: ')
        print(expDetail['results']['valBest'])
        if testData is not None:
            print('\n Test Results: ')
            print(expDetail['results']['test'])

        # save the results
        if self.resultsSavePath is not None:
            # Store the graphs
            self.plotLoss(trainResults['trainLoss'], trainResults['valLoss'],
                          trainResults['epoch'],
                          savePath=os.path.join(self.resultsSavePath,
                                                'exp-'+str(expNo)+'-loss.png'))
            self.plotAcc(trainResults['trainResults']['acc'],
                         trainResults['valResults']['acc'],
                         trainResults['epoch'],
                         savePath=os.path.join(self.resultsSavePath,
                                               'exp-'+str(expNo)+'-acc.png'))

            # Store the data in experimental details.
            with open(os.path.join(self.resultsSavePath, 'expResults' +
                                   str(expNo)+'.dat'), 'wb') as fp:
                pickle.dump(expDetail, fp)

        # Increment the expNo
        self.expDetails.append(expDetail)
        expNo += 1

    def _trainOE(self, trainData, valData, augData=None, nTrialsAug=0,
                 lossFn='NLLLoss', optimFn='Adam', lr=0.001,
                 stopCondi={'c': {'Or': {'c1': {'MaxEpoch': {
                     'maxEpochs': 500, 'varName': 'epoch'}}, 'c2': {
                         'NoDecrease': {'numEpochs': 50,
                                        'varName': 'valLoss'}}}}},
                         optimFnArgs={}, loadBestModel=True,
                         bestVarToCheck='valLoss', classes=None, sampler=None):
        '''
        Internal function to perform the training.
        Do not directly call this function. Use train instead

        Parameters
        ----------
        trainData : eegDataset
            dataset used for training.
        valData : eegDataset
            dataset used for validation.
        lossFn : function handle from torch.nn, The default is NLLLoss
            Loss function from torch.nn which will be used for training.
        optimFn : string from torch.optim. The default is 'Adam'.
            Name of the optimization function from torch.nn which will be used
            for training.
        lr : float, optional
            Learning rate. The default is 0.001.
        stopCondi : dict, optional
            Determines when to stop.
            It will be a dictionary which can be accepted by stopCriteria
            class. The default is : no decrease in validation Inaccuracy in
            last 200 epochs OR epoch > 1500
            This default is represented as:
            {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500,
                                              'varName': 'epoch'}},
                          'c2': {'NoDecrease': {'numEpochs': 200,
                                                'varName': 'valInacc'}}}}}
        optimFnArgs : dict, optional
            Additional arguments to be passed to the optimizer. The default is
            {}.
        loadBestModel: bool, optional
            Whether to load the network with best validation loss/acc at the
            end of training. The default is True.
        bestVarToCheck: 'valInacc'/'valLoss', optional
            the best value to check while determining the best model. The
            default is 'valInacc'.
        continueAfterEarlystop : bool, optional
            Whether to continue training after early stopping has reached. The
            default is False.
        classes: list, optional
            List of classes to consider in evaluation matrices.
            None -> all classes.
            The default is None.
        sampler: function handle to a sampler to be used in dataloader,
                 optional
            The sampler to use while training and validation.
            The default is None.

        Returns
        -------
        dict
            a dictionary with all the training results.
        '''

        # For reporting.
        trainResults = []
        valResults = []
        trainLoss = []
        valLoss = []
        loss = []
        bestNet = copy.deepcopy(self.net.state_dict())
        bestValInaccValue = float('inf')
        bestValLossValue = float('inf')
        earlyStopReached = False

        # Create optimizer
        self.optimizer = self._findOptimizer(optimFn)(self.net.parameters(),
                                                      lr=lr, **optimFnArgs)
        bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())

        # Initialize the stop criteria
        stopCondition = stopCriteria.composeStopCriteria(**stopCondi)

        # lets start the training.
        monitors = {'epoch': 0, 'valLoss': 10000, 'valInacc': 1}
        doStop = False

        while not doStop:
            if not augData:
                # train the epoch.
                loss.append(self.trainOneEpoch(trainData,
                                               lossFn, self.optimizer,
                                               sampler=sampler))
            else:
                # train the epoch.
                loss.append(self.trainOneEpochAug(trainData, augData,
                                                  nTrialsAug, lossFn,
                                                  self.optimizer,
                                                  sampler=sampler))

            # evaluate the training and validation accuracy.
            pred, act, lo = self.predict(trainData, sampler=sampler,
                                         lossFn=lossFn)
            trainResults.append(self.calculateResults(pred, act,
                                                      classes=classes))
            trainLoss.append(lo)
            monitors['trainLoss'] = lo
            monitors['trainInacc'] = 1 - trainResults[-1]['acc']
            pred, act, lo = self.predict(valData, sampler=sampler,
                                         lossFn=lossFn)
            valResults.append(self.calculateResults(pred, act,
                                                    classes=classes))
            valLoss.append(lo)
            monitors['valLoss'] = lo
            monitors['valInacc'] = 1 - valResults[-1]['acc']

            # print the epoch info
            print("\t \t Epoch " + str(monitors['epoch'] + 1))
            print("Train loss = " + "%.3f" % trainLoss[-1] + " Train Acc = " +
                  "%.3f" % trainResults[-1]['acc'] +
                  ' Val Acc = ' + "%.3f" % valResults[-1]['acc'] +
                  " Val loss = " + "%.3f" % valLoss[-1])

            if loadBestModel:
                if monitors['valInacc'] < bestValInaccValue:
                    bestValInaccValue = monitors['valInacc']
                    bestValLossValue = monitors['valLoss']
                    bestEpoch = monitors['epoch']
                    bestNet = copy.deepcopy(self.net.state_dict())
                    bestOptimizerState = copy.deepcopy(
                        self.optimizer.state_dict())
                elif monitors['valInacc'] == bestValInaccValue:
                    if monitors['valLoss'] < bestValLossValue:
                        bestValLossValue = monitors['valLoss']
                        bestEpoch = monitors['epoch']
                        bestNet = copy.deepcopy(self.net.state_dict())
                        bestOptimizerState = copy.deepcopy(
                            self.optimizer.state_dict())
            # Check if to stop
            doStop = stopCondition(monitors)

            # Check if we want to continue the training after the first stop:
            if doStop:
                # first load the best model
                if loadBestModel and not earlyStopReached:
                    self.net.load_state_dict(bestNet)
                    self.optimizer.load_state_dict(bestOptimizerState)
                    bestNet = copy.deepcopy(self.net.state_dict())

            # update the epoch
            monitors['epoch'] += 1
        # Make individual list for components of trainResults and valResults
        t = {}
        v = {}

        for key in trainResults[0].keys():
            t[key] = [result[key] for result in trainResults]
            v[key] = [result[key] for result in valResults]

        return {'trainResults': t, 'valResults': v,
                'trainLoss': trainLoss, 'valLoss': valLoss,
                'epoch': bestEpoch}

    def trainOneEpoch(self, trainData, lossFn, optimizer, sampler=None):
        '''
        Train for one epoch

        Parameters
        ----------
        trainData : eegDataset
            dataset used for training.
        lossFn : function handle of type torch.nn
            the loss function.
        optimizer : optimizer of type torch.optim
            the optimizer.
        sampler : function handle of type torch.utils.data.sampler, optional
            sampler is used if you want to specify any particular data sampling
            in the data loader. The default is None.

        Returns
        -------
        TYPE
            training loss.

        '''

        # Set the network in training mode.
        self.net.train()

        # running loss to zero.
        running_loss = 0

        # set shuffle
        if sampler:
            sampler = sampler(trainData)
        # Stratified mini-batch sampling
        idx_train_0 = [i for i, x in enumerate(trainData.labels) if
                       x[2] == 0]
        idx_train_1 = [i for i, x in enumerate(trainData.labels) if
                       x[2] == 1]
        trainData_0 = copy.deepcopy(trainData)
        trainData_0.createPartialDataset(idx_train_0)
        trainData_1 = copy.deepcopy(trainData)
        trainData_1.createPartialDataset(idx_train_1)
        # Create the dataloader with random shuffle.
        dataloaders0 = DataLoader(trainData_0, batch_size=int(
            self.batchSize/2), shuffle=True, drop_last=True)
        dataloaders1 = DataLoader(trainData_1, batch_size=int(
            self.batchSize/2), shuffle=True, drop_last=True)

        for i, data in enumerate(zip(dataloaders0, dataloaders1)):
            d = {}
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                d['data'] = torch.cat((data[0]['data'],
                                       data[1]['data']), dim=0)
                d['label'] = torch.cat((data[0]['label'],
                                        data[1]['label']), dim=0)
                # forward pass:
                output = self.net(d['data'].unsqueeze(1).to(self.device))

                # calculate loss
                loss = lossFn(output, d['label'].type(
                    torch.LongTensor).to(self.device))
                loss = loss/d['data'].shape[0]
                # backward pass:
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # Return the current loss. This may be helpful to stop or continue the
        # training.
        return running_loss.item()/len(dataloaders0)

    def trainOneEpochAug(self, trainData, augData, nTrialsAug, lossFn,
                         optimizer, sampler=None):
        '''
        Train for one epoch

        Parameters
        ----------
        trainData : eegDataset
            dataset used for training.
        lossFn : function handle of type torch.nn
            the loss function.
        optimizer : optimizer of type torch.optim
            the optimizer.
        sampler : function handle of type torch.utils.data.sampler, optional
            sampler is used if you want to specify any particular data sampling
            in the data loader. The default is None.

        Returns
        -------
        TYPE
            training loss.

        '''
        idx_aug_0 = [i for i, x in enumerate(augData.labels) if
                     x[2] == 0]
        idx_aug_1 = [i for i, x in enumerate(augData.labels) if
                     x[2] == 1]
        augDataTmp_0 = copy.deepcopy(augData)
        augDataTmp_0.createPartialDataset(idx_aug_0)
        augDataTmp_1 = copy.deepcopy(augData)
        augDataTmp_1.createPartialDataset(idx_aug_1)
        # Set the network in training mode.
        self.net.train()

        # running loss to zero.
        running_loss = 0
        idx_train_0 = [i for i, x in enumerate(trainData.labels) if
                       x[2] == 0]
        idx_train_1 = [i for i, x in enumerate(trainData.labels) if
                       x[2] == 1]
        trainData_0 = copy.deepcopy(trainData)
        trainData_0.createPartialDataset(idx_train_0)
        trainData_1 = copy.deepcopy(trainData)
        trainData_1.createPartialDataset(idx_train_1)
        # Create the dataloader with random shuffle.
        dataloaders0 = DataLoader(trainData_0,
                                  batch_size=int((
                                      self.batchSize-2*nTrialsAug)/2),
                                  shuffle=True, drop_last=True)
        dataloaders1 = DataLoader(trainData_1,
                                  batch_size=int((
                                      self.batchSize-2*nTrialsAug)/2),
                                  shuffle=True, drop_last=True)
        dataloaders2 = DataLoader(augDataTmp_0, batch_size=nTrialsAug,
                                  shuffle=True)
        dataloaders3 = DataLoader(augDataTmp_1, batch_size=nTrialsAug,
                                  shuffle=True)
        for i, data in enumerate(zip(dataloaders0, dataloaders1, dataloaders2,
                                     dataloaders3)):
            data_train = {}
            with torch.enable_grad():
                # reset (zero) the gradients of model parameters
                optimizer.zero_grad()
                data_train['data'] = torch.cat((data[0]['data'],
                                                data[1]['data'],
                                                data[2]['data'],
                                                data[3]['data']), dim=0)
                data_train['label'] = torch.cat((data[0]['label'],
                                                 data[1]['label'],
                                                 data[2]['label'],
                                                 data[3]['label']), dim=0)
                # forward pass:
                output = self.net(data_train['data'].unsqueeze(1).to(
                    self.device))

                # calculate loss
                loss = lossFn(output, data_train['label'].type(
                        torch.LongTensor).to(self.device))
                loss = loss/data_train['data'].shape[0]
                # backward pass:
                loss.backward()
                # get the gradient wrt the weights
                # self.net.scb[0].weight.grad
                # update model parameters
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the
        # training.
        return running_loss.item()/len(dataloaders0)

    def predict(self, data, sampler=None, lossFn=None):
        '''
        Predict the class of the input data

        Parameters
        ----------
        data : eegDataset
            dataset of type eegDataset.
        sampler : function handle of type torch.utils.data.sampler, optional
            sampler is used if you want to specify any particular data sampling
            in the data loader. The default is None.
        lossFn : function handle of type torch.nn, optional
            lossFn is not None then function returns the loss. The default is
            None.

        Returns
        -------
        predicted : list
            List of predicted labels.
        actual : list
            List of actual labels.
        loss
            average loss.

        '''

        predicted = []
        actual = []
        loss = 0
        batch_size = self.batchSize
        totalCount = 0
        # set the network in the eval mode
        self.net.eval()

        # initiate the dataloader.
        dataLoader = DataLoader(data, batch_size=batch_size, shuffle=False)

        # with no gradient tracking
        with torch.no_grad():
            # iterate over all the data
            for d in dataLoader:
                preds = self.net(d['data'].unsqueeze(1).to(self.device))
                totalCount += d['data'].shape[0]

                if lossFn is not None:
                    # calculate loss
                    loss += lossFn(preds, d['label'].type(torch.LongTensor).to(self.device)).data

                # Convert the output of soft-max to class label
                _, preds = torch.max(preds, 1)
                predicted.extend(preds.data.tolist())
                actual.extend(d['label'].tolist())

        return predicted, actual, loss.item()/totalCount

    def calculateResults(self, yPredicted, yActual, classes = None):
        '''
        Calculate the results matrices based on the actual and predicted class.

        Parameters
        ----------
        yPredicted : list
            List of predicted labels.
        yActual : list
            List of actual labels.
        classes : list, optional
            List of labels to index the CM.
            This may be used to reorder or select a subset of class labels.
            If None then, the class labels that appear at least once in
            yPredicted or yActual are used in sorted order.
            The default is None.

        Returns
        -------
        dict
            a dictionary with fields:
                acc : accuracy.
                cm  : confusion matrix..

        '''

        acc = accuracy_score(yActual, yPredicted)
        if classes is not None:
            cm = confusion_matrix(yActual, yPredicted, labels= classes)
        else:
            cm = confusion_matrix(yActual, yPredicted)

        return {'acc': acc, 'cm': cm}

    def plotLoss(self, trainLoss, valLoss, bestEpoch, savePath = None):
        '''
        Plot the training loss.

        Parameters
        ----------
        trainLoss : list
            Training Loss.
        valLoss : list
            Validation Loss.
        savePath : str, optional
            path to store the figure. The default is None: figure will be plotted.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.title("Training Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,len(trainLoss)+1),trainLoss,label="Train loss")
        plt.plot(range(1,len(valLoss)+1),valLoss,label="Validation Loss")
        plt.axvline(x=bestEpoch, color="black", linestyle=":")
        plt.legend(loc='upper left')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()
        plt.close()

    def plotAcc(self, trainAcc, valAcc, bestEpoch, savePath= None):
        '''
        Plot the train and validation accuracy.
        '''
        plt.figure()
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,len(trainAcc)+1),trainAcc,label="Train Acc")
        plt.plot(range(1,len(valAcc)+1),valAcc,label="Validation Acc")
        plt.axvline(x=bestEpoch, color="black", linestyle=":")
        plt.ylim((0,1.))
        plt.legend(loc='upper left')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()
        plt.close()

    def setRandom(self, seed):
        '''
        Set all the random initializations with a given seed

        Parameters
        ----------
        seed : int
            seed.

        Returns
        -------
        None.

        '''
        self.seed = seed

        # Set np
        np.random.seed(self.seed)

        # Set torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Set cudnn
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setDevice(self, nGPU = 0):
        '''
        Set the device for training and testing

        Parameters
        ----------
        nGPU : int, optional
            GPU number to train on. The default is 0.

        Returns
        -------
        None.

        '''
        if self.device is None:
            if self.preferedDevice == 'gpu':
                self.device = torch.device("cuda:"+str(nGPU) if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device('cpu')
            print("Code will be running on device ", self.device)

    def _findOptimizer(self, optimString):
        '''
        Look for the optimizer with the given string and then return the function handle of that optimizer.
        '''
        out  = None
        if optimString in optim.__dict__.keys():
            out = optim.__dict__[optimString]
        else:
            raise AssertionError('No optimizer with name :' + optimString + ' can be found in torch.optim. The list of available options in this module are as follows: ' + str(optim.__dict__.keys()))
        return out

    def _findSampler(self, givenString):
        '''
        Look for the sampler with the given string and then return the function handle of the same.
        '''
        out  = None
        if givenString in builtInSampler.__dict__.keys():
            out = builtInSampler.__dict__[givenString]
        elif givenString in samplers.__dict__.keys():
            out = samplers.__dict__[givenString]
        else:
            raise AssertionError('No sampler with name :' + givenString + ' can be found')
        return out

    def _findLossFn(self, lossString):
        '''
        Look for the loss function with the given string and then return the function handle of that function.
        '''
        out  = None
        if lossString in nn.__dict__.keys():
            out = nn.__dict__[lossString]
        else:
            raise AssertionError('No loss function with name :' + lossString + ' can be found in torch.nn. The list of available options in this module are as follows: ' + str(nn.__dict__.keys()))

        return out


