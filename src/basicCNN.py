# -*- coding: utf-8 -*-
"""
Basic CNN

Uses a basic Convolutional Neural Network structure to read hand written letters

@author: Christopher Chang
"""
import numpy as np
# import pandas as pd
import library.imageProcessor as imgProcessor
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

def categorizeLabels(y, numClasses):
    return tf.keras.utils.to_categorical(y, num_classes=numClasses, dtype = "int32")

def unCategorizeLabels(y):
    return [np.argmax(e) for e in y]

def createModel(inputShape, outputNum):
    '''
    Creates the Machine Learning model to test

    Parameters
    ----------
    inputShape : (int, int)
        The shape of the input data
    outputNum : int
        The number of output nodes the model should have.
        If the model is binary, it should use 1 output, not 2.

    Returns
    -------
    model : tf.keras.Model
        The model to use in the machine learning task.
        Model is not compiled so that cross validation may occur.

    '''
    model = None
    
    cnn = keras.Sequential()
    cnn.add(layers.Conv2D(3, 5, (3,3), input_shape=inputShape, activation='relu'))
    cnn.add(layers.Conv2D(1, 6, (2,2), activation='relu'))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(20, activation='relu'))
    cnn.add(layers.Dense(10, activation='relu'))
    cnn.add(layers.Dense(outputNum))     #output
    
    model = cnn
    
    return model

def trainModel(model, imageProc, xTrain, yTrain):
    trainedModel = None
    hyperParameters = []
    trainingScore = 0
    
    yTrainVect = categorizeLabels(yTrain, len(imageProc.characterEnum))
    
    xTrainNew, yTrainNew, xValid, yValid = imageProc.splitData(xTrain, yTrainVect, 123)
    
    #Callback for early stopping
    callbackBest = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    #Hold-out Cross-validation
    scores = {}
    scoreEpoch = {}
    optimizers = ['adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam', 'rmsprop', 'sgd']
    lossFuncs = ['mean_absolute_error', 'mean_squared_error', 'huber_loss']
    metrics = [["categorical_accuracy"]]
    
    for o in optimizers:
        for l in lossFuncs:
            for m in metrics:
                #Debug Logger
                print("{}, {}, {}".format(o, l, m))
                
                currModel = keras.models.clone_model(model)
                currModel.compile(optimizer=o, loss=l, metrics=m)
                
                modelHist = currModel.fit(xTrainNew, yTrainNew, epochs=50, batch_size=100, validation_data=(xValid,yValid), callbacks=[callbackBest])
                
                #Since we have early stopping, we want to get the epoch at which we had the best score
                minScore = min(modelHist.history["val_" + m[0]])
                for i in range(len(modelHist.history["val_" + m[0]])):
                    if (modelHist.history["val_" + m[0]][i] == minScore):
                            scoreEpoch[(o, l, m[0])] = i
                            break
                            
                scores[(o, l, m[0])] = minScore
                
    trainingScore = min(scores.values())
    for k in scores.keys():
        if (trainingScore == scores[k]):
            hyperParameters = k
            break
    
    #Recreate the best model
    trainedModel = model
    trainedModel.compile(optimizer=hyperParameters[0], 
               loss=hyperParameters[1], 
               metrics=[hyperParameters[2]])
    
    #Train the best model on the whole data
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs/basicCNN", histogram_freq=1)
    trainedModel.fit(xTrain, yTrainVect, epochs=scoreEpoch[hyperParameters], batch_size=100, callbacks=[callbackBest, tensorboard_callback])
    
    return trainedModel, hyperParameters, trainingScore

def testModel(model, imageProc, xTest, yTest):
    yPred = []
    f1Score = -1.0
    classReport = ""
    
    yPred = model.predict(xTest)
    
    f1Score = imageProc.calculateF1Score(yPred, yTest, avg='weighted')
    classReport = imageProc.calculateReport(yPred, yTest)
    
    
    return yPred, f1Score, classReport

imageProc = imgProcessor.imageProcessor()

#Get data
xTrain, yTrain = imageProc.getRandomTrainingData(1000, seed=123)
xTest, yTest = imageProc.getRandomTestingData(100, seed=123)

#Create the model
# model = createModel((128,128), 62)

#Train the model, get hyperParameters, and get the training score
# trainedModel, hyperParams, trainingScore = trainModel(model, imageProc, xTrain, yTrain)

#Test the model and calculate stats
# yPred, f1Score, classReport = testModel(trainedModel, imageProc, xTest, yTest)

# print(classReport)
