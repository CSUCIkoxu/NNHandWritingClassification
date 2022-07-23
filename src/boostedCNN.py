# -*- coding: utf-8 -*-
"""
Reduction CNN

Uses a Convolutional Neural Network structure that reduces with each layer 
to read hand written letters

@author: Christopher Chang
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import library.imageProcessor as imgProcessor
import library.boostedNNClass as boostedNNClass

def categorizeLabels(y, numClasses):
    return tf.keras.utils.to_categorical(y, num_classes=numClasses, dtype = "int32")

def unCategorizeLabels(y):
    return [np.argmax(e) for e in y]

def createSubModel(inputShape):
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
    cnn.add(layers.Dense(1, activation='softmax'))     #output
    
    model = cnn
    
    return model

def createSuperModel(inputShape, outputNum):
    model = None
    
    cnn = keras.Sequential()
    cnn.add(layers.Dense(62, activation='relu', input_dim=inputShape))
    cnn.add(layers.Dense(124, activation='relu'))
    cnn.add(layers.Dropout(0.3))
    cnn.add(layers.Dense(62, activation='relu'))
    cnn.add(layers.Dense(outputNum, activation='softmax'))     #output
    
    model = cnn
    
    return model

def createModel(inputShape, outputNum):
    model = None
    
    subModels = []
    superModel = None
    
    for i in range(outputNum):
        subModels.append(createSubModel(inputShape))
    
    superModel = createSuperModel(outputNum, outputNum)
    
    model = boostedNNClass.boostedNNClass(superModel, subModels, inputShape, outputNum)
    
    return model
    

def trainModel(model, imageProc, trainingQuantity, categoryNum):
    # trainedModel = None
    subHyperParameters = []
    superHyperParameters = []
    subTrainingScore = []
    superTrainingScore = 0
    
    #Train the Sub-Models
    for c in imageProc.characterList:
        ##Get training data for sub model
        # xTrain, yTrain = imageProc.getTrainingData(c, 123)
        xTrainNew, yTrainNew, xValid, yValid = imageProc.splitData(*imageProc.getTrainingData(c, 123), seed=123)
        
        ##Train one sub-model
        model.trainSubModel(xTrainNew, yTrainNew, imageProc.char2Enum(c), xValid, yValid)
        
    #Train the Super-Model
    ##Get the training data for the super model
    # xTrain, yTrain = imageProc.getRandomTrainingData(trainingQuantity, 123)
    xTrainNew, yTrainNew, xValid, yValid = imageProc.splitData(*imageProc.getRandomTrainingData(trainingQuantity, 123), seed=123)
    
    ##Train the super-model
    model.trainSuperModel(xTrainNew, yTrainNew, categoryNum, xValid, yValid)
    
    return subHyperParameters, superHyperParameters, subTrainingScore, superTrainingScore

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

#Create the model
model = createModel((128,128,1), 62)

#Train the model, get hyperParameters, and get the training score
subHyperParameters, superHyperParameters, subTrainingScore, superTrainingScore = trainModel(model, imageProc, 200, len(imageProc.characterList))

#Get data
xTest, yTest = imageProc.getRandomTestingData(50, seed=123)

#Test the model and calculate stats
yPred, f1Score, confMat, classReport = testModel(model, imageProc, xTest, yTest)

print(classReport)
print(confMat)

#Save the model
# trainedModel.save(imageProc.modelSavesDirectory + '/boostedCNN')


#Use this to generate tensorboard charts on saved models
#%load_ext tensorboard
#%tensorboard --logdir "logs/logs/reductionCNN"
#NN Located at: http://localhost:6006/#scalars

#Use this to specify a port
#%tensorboard --logdir "logs/logs/reductionCNN" --port=6007
#NN Located at: http://localhost:6007/#scalars
