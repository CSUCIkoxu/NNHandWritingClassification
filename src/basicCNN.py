# -*- coding: utf-8 -*-
"""
Basic CNN

Uses a basic Convolutional Neural Network structure to read hand written letters

@author: Christopher Chang
"""
import pandas as pd
import library.imageProcessor as imgProcessor
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers

def createModel(inputShape, outputNum):
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

def trainModel(model, xTrain, yTrain):
    return

def testModel(model, xTest, yTest):
    return