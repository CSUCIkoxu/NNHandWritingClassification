# -*- coding: utf-8 -*-
"""
@author: Christopher Chang
"""

class boostedNNClass:
    def __init__(self, superModel, subModels, inputDim, outputNum):
        #Models and model info
        self.superModel = superModel
        self.subModels = subModels
        self.inputDim = inputDim
        self.outputNum = outputNum
        
    def trainSubModel(self, xTrain, yTrain, subModelNum, xValid=None, yValid=None):
        import tensorflow.keras as keras
        
        if (subModelNum >= len(self.subModels)):
            raise Exception("subModelNum exceeds the number of sub models in the system")
            
        hyperParameters = []
        trainingScore = 0
        
        xTrainNew = None
        yTrainNew = None
        if (xValid == None or yValid == None):
            import sklearn.model_selection as ms
            
            xTrainNew, xValid, yTrainNew, yValid = ms.train_test_split(xTrain, yTrain, test_size=0.3)
        else:
            xTrainNew = xTrain
            yTrainNew = yTrain
            
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
                    
                    currModel = keras.models.clone_model(self.subModels[subModelNum])
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
        self.subModels[subModelNum] = keras.models.clone_model(self.subModels[subModelNum])
        self.subModels[subModelNum].compile(optimizer=hyperParameters[0], 
                   loss=hyperParameters[1], 
                   metrics=[hyperParameters[2]])
        
        #Train the best model on the whole data
        self.subModels[subModelNum].fit(xTrain, yTrain, epochs=scoreEpoch[hyperParameters], batch_size=20)
    
    def trainSuperModel(self, xTrain, yTrain, categoryNum, xValid=None, yValid=None):
        import tensorflow.keras as keras
        import numpy as np
            
        hyperParameters = []
        trainingScore = 0
        
        #Get features from Sub
        subPred = np.empty([len(xTrain), len(self.subModels)], dtype="float")
        
        for i in range(len(xTrain)):
            for j in range(len(self.subModels)):
                subPred[i,j] = self.subModels[j].predict(xTrain[i])
        
        #Put features into training and validation sets
        yTrainVect = self._categorizeLabels(yTrain, categoryNum)
        
        xTrainNew = None
        yTrainNew = None
        xValidNew = None
        yValidNew = None
        if (xValid == None or yValid == None):
            import sklearn.model_selection as ms
            
            xTrainNew, xValidNew, yTrainNew, yValidNew = ms.train_test_split(subPred, yTrainVect, test_size=0.3)
        else:
            xTrainNew = subPred
            yTrainNew = yTrainVect
            
            xValidNew = np.empty([len(xValid), len(self.subModels)], dtype="float")
            yValidNew = self._categorizeLabels(yValid, categoryNum)
            
            for i in range(len(xValid)):
                for j in range(len(self.subModels)):
                    xValidNew[i,j] = self.subModels[j].predict(xValid[i])
        
        #Train Super
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
                    
                    currModel = keras.models.clone_model(self.superModel)
                    currModel.compile(optimizer=o, loss=l, metrics=m)
                    
                    modelHist = currModel.fit(xTrainNew, yTrainNew, epochs=50, batch_size=20, validation_data=(xValidNew,yValidNew), callbacks=[callbackBest])
                    
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
        self.superModel = keras.models.clone_model(self.superModel)
        self.superModel.compile(optimizer=hyperParameters[0], 
                   loss=hyperParameters[1], 
                   metrics=[hyperParameters[2]])
        
        #Train the best model on the whole data
        self.superModel.fit(subPred, yTrainVect, epochs=scoreEpoch[hyperParameters], batch_size=20)
        
    def predict(self, xTest):
        pred = None
        
        import numpy as np
        
        subPred = np.empty([len(xTest), len(self.subModels)], dtype="float")
        
        for i in range(len(xTest)):
            for j in range(len(self.subModels)):
                subPred[i,j] = self.subModels[j].predict(xTest[i])
                
        pred = self.superModel.predict(subPred)
        
        return pred
    
    def _categorizeLabels(self, y, numClasses):
        import tensorflow as tf
        return tf.keras.utils.to_categorical(y, num_classes=numClasses, dtype = "uint8")
    
    def _unCategorizeLabels(self, y):
        import numpy as np
        return [np.argmax(e) for e in y]
        
    
        