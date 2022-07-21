# -*- coding: utf-8 -*-
"""
Reduction CNN

Uses a Convolutional Neural Network structure that reduces with each layer 
to read hand written letters

@author: Christopher Chang
"""
import numpy as np
import pandas as pd
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
    inputShape : (int, int, int)
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
    cnn.add(layers.Conv2D(62, 5, (3,3), input_shape=inputShape, activation='relu', padding='valid'))  #(filters, kernel size, stride)
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding="valid", data_format=None))
    cnn.add(layers.Conv2D(124, 3, (1,1), activation='relu', padding="valid"))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", data_format=None))
    cnn.add(layers.Conv2D(248, 3, (1,1), activation='relu', padding="valid"))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", data_format=None))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(512, activation='relu'))
    cnn.add(layers.Dropout(0.3))
    cnn.add(layers.Dense(256, activation='relu'))
    cnn.add(layers.Dense(outputNum, activation='softmax'))     #output
    
    model = cnn
    
    return model

def trainModel(model, imageProc, categoryNum, xTrain, yTrain):
    trainedModel = None
    hyperParameters = []
    trainingScore = 0
    
    yTrainVect = categorizeLabels(yTrain, categoryNum)
    
    xTrainNew, yTrainNew, xValid, yValid = imageProc.splitData(xTrain, yTrainVect, 123)
    
    #Callback for early stopping
    callbackBest = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    #Hold-out Cross-validation
    scores = {}
    scoreEpoch = {}
    optimizers = ['adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam', 'rmsprop', 'sgd']
    lossFuncs = ['categorical_crossentropy', 'kl_divergence']#, 'poisson']  #poisson gives nan values for some reason
    metrics = [['categorical_crossentropy']]
    
    for o in optimizers:
        for l in lossFuncs:
            for m in metrics:
                #Debug Logger
                print("{}, {}, {}".format(o, l, m))
                
                currModel = keras.models.clone_model(model)
                currModel.compile(optimizer=o, loss=l, metrics=m)
                
                modelHist = currModel.fit(xTrainNew, yTrainNew, epochs=50, batch_size=20, validation_data=(xValid,yValid), callbacks=[callbackBest])
                
                #Since we have early stopping, we want to get the epoch at which we had the best score
                minScore = min(modelHist.history["val_" + m[0]])
                for i in range(len(modelHist.history["val_" + m[0]])):
                    if (modelHist.history["val_" + m[0]][i] == minScore):
                            scoreEpoch[(o, l, m[0])] = i + 1
                            break
                            
                scores[(o, l, m[0])] = minScore
                
    trainingScore = min(scores.values())
    for k in scores.keys():
        if (trainingScore == scores[k]):
            hyperParameters = k
            break
    
    #Recreate the best model
    trainedModel = keras.models.clone_model(model)
    trainedModel.compile(optimizer=hyperParameters[0], 
               loss=hyperParameters[1], 
               metrics=[hyperParameters[2]])
    
    #Train the best model on the whole data
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs/reductionCNN", histogram_freq=1)
    trainedModel.fit(xTrain, yTrainVect, epochs=scoreEpoch[hyperParameters], batch_size=20, callbacks=[tensorboard_callback])
    
    return trainedModel, hyperParameters, trainingScore

def testModel(model, imageProc, xTest, yTest):
    yPred = []
    f1Score = -1.0
    confMat = None
    classReport = ""
    
    yPredRaw = model.predict(xTest)
    
    yPred = unCategorizeLabels(yPredRaw)
    
    f1Score = imageProc.calculateF1Score(yPred, yTest, avg='weighted')
    confMat = imageProc.createConfusionMatrix(yPred, yTest)
    classReport = imageProc.calculateReport(yPred, yTest, names=imageProc.characterList)
    
    
    return yPred, f1Score, confMat, classReport

#MAIN ##########################################################

#import the image processor class with all the helper functions
imageProc = imgProcessor.imageProcessor()

#Get data
xTrain, yTrain = imageProc.getRandomTrainingData(200, seed=123)
xTest, yTest = imageProc.getRandomTestingData(10, seed=123)

#Create the model
model = createModel((128,128,1), 62)

#Train the model, get hyperParameters, and get the training score
trainedModel, hyperParams, trainingScore = trainModel(model, imageProc, len(imageProc.characterList), xTrain, yTrain)

#Test the model and calculate stats
yPred, f1Score, confMat, classReport = testModel(trainedModel, imageProc, xTest, yTest)

print(classReport)
print(confMat)

#Save the model
trainedModel.save(imageProc.modelSavesDirectory + '/reductionCNN')

#Use this to generate tensorboard charts on saved models
#%load_ext tensorboard
#%tensorboard --logdir "logs/logs/reductionCNN"
#NN Located at: http://localhost:6006/#scalars

#Use this to specify a port
#%tensorboard --logdir "logs/logs/reductionCNN" --port=6007
#NN Located at: http://localhost:6007/#scalars
