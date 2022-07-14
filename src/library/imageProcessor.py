# -*- coding: utf-8 -*-
"""
@author: Christopher Chang
"""

class imageProcessor:
    def __init__(self):
        '''
        Image Processor class
        Used for fetching and processing image data for use in an AI
        
        Expects that the script using this class is in the src folder, other 
        file locations must specify where the data folder is located relative 
        to the script.  Also expects that you are using the "by_Class" version
        of the dataset

        '''
        #Directories
        self.dataDirectory = "../data"
        self.organizationDirectory = "/by_class"
        self.trainDirectory = "/train_"
        self.testDirectory = "/hsf_4"
        
        #Processor Information
        self.characterClasses = []
        
        self.characterList = [str(i) for i in range(10)]
        import string
        for c in list(string.ascii_letters):
            self.characterList.append(c)
        
        self.characterEnum = {}
        for i in range(len(self.characterList)):
            self.characterEnum[self.characterList[i]] = i
        
    def setCharClass(self, chars):
        '''
        Sets the Character Classes to use

        Parameters
        ----------
        chars : list<string>
            A list of characters to read data from

        '''
        self.characterClasses = chars
        
    def char2Enum(self, char):
        return self.characterEnum[char]
    
    def enum2Char(self, enum):
        return self.characterList[enum]
        
    def getTrainingData(self):
        '''
        Gets the training data for the specified character classes

        Returns
        -------
        data : dict<str : list<int, int>>
            A dictionary for each character class with the corresponding training
            data

        '''
        data = {}   #data will be organized in a dictionary
        
        for char in self.characterClasses:
            #The folders in the dataset are labeled by ASCII hex values
            char2Hex = format(ord(char), "x")
            
            charDataDir = self.dataDirectory + self.organizationDirectory + '/' + char2Hex + self.trainDirectory + char2Hex
            
            data[char] = self._loadImages(charDataDir)
        
        return data
    
    def getTestingData(self):
        '''
        Gets the testing data for the specified character classes

        Returns
        -------
        data : dict<str : list<int, int>>
            A dictionary for each character class with the corresponding testing
            data

        '''
        data = {}   #data will be organized in a dictionary
        
        for char in self.characterClasses:
            #The folders in the dataset are labeled by ASCII hex values
            char2Hex = format(ord(char), "x")
            
            charDataDir = self.dataDirectory + self.organizationDirectory + '/' + char2Hex + self.testDirectory
            
            data[char] = self._loadImages(charDataDir)
        
        return data
    
    def getRandomTrainingData(self, quantity, seed=None):
        '''
        Gets a list of random training samples and their labels

        Parameters
        ----------
        quantity : int
            The number of samples to include for each character class
        seed : int, optional
            The seed to use when using random selection. Uses random seed if None.
            The default is None.

        Returns
        -------
        x : list<list<int,int>>
            List of feature data for the 2d images
        y : list<int>
            List of label data. The labels are given as ints which
            corresponds to this classes' characterEnum

        '''
        x = []
        y = []
        
        data = []
        
        import random
        
        if (seed != None):
            random.seed(seed)
        
        #Iterate through all of the characters to generate balanced data
        for char in self.characterList:
            char2Hex = format(ord(char), "x")
            
            charDataDir = self.dataDirectory + self.organizationDirectory + '/' + char2Hex + self.trainDirectory + char2Hex
            
            data.append(((e, self.characterEnum[char]) for e in self._loadRandomImages(charDataDir, quantity)))
            
        #Randomize the elements in the array
    
        random.shuffle(data)
        
        #Split data to its features (x) and labels (y)
        x = data[:][0]
        y = data[:][1]
        
        return x, y
    
    def getRandomTestingData(self, quantity, seed=None):
        '''
        Gets a list of random testing samples and their labels

        Parameters
        ----------
        quantity : int
            The number of samples to include for each character class
        seed : int, optional
            The seed to use when using random selection. Uses random seed if None.
            The default is None.

        Returns
        -------
        x : list<list<int,int>>
            List of feature data for the 2d images
        y : list<int>
            List of label data. The labels are given as ints which
            corresponds to this classes' characterEnum

        '''
        x = []
        y = []
        
        data = []
        
        import random
        
        if (seed != None):
            random.seed(seed)
        
        #Iterate through all of the characters to generate balanced data
        for char in self.characterList:
            char2Hex = format(ord(char), "x")
            
            charDataDir = self.dataDirectory + self.organizationDirectory + '/' + char2Hex + self.testDirectory
            
            data.append(((e, self.characterEnum[char]) for e in self._loadRandomImages(charDataDir, quantity)))
            
        #Randomize the elements in the array
    
        random.shuffle(data)
        
        #Split data to its features (x) and labels (y)
        x = data[:][0]
        y = data[:][1]
        
        return x, y
    
    def splitData(self, x, y, seed=None):
        '''
        Splits the data that is given to training and test sets, 70%-30%.
        Intended to be used for cross-validation.

        Parameters
        ----------
        x : list or panda.dataframe
            The list of features
        y : list
            The list of labels
        seed : int, optional
            An optional seed to use when splitting. The default is None.

        Returns
        -------
        xTrain : list
            The list of features of the training set
        yTrain : list
            The list of labels of the training set
        xTest : list
            The list of features of the testing set
        yTest : list
            The list of labels of the testing set

        '''
        xTrain = None
        yTrain = None
        xTest = None
        yTest = None
        
        import sklearn.model_selection as ms
        
        xTrain, xTest, yTrain, yTest = ms.train_test_split(x, y, test_size=0.3, random_state=seed)
        
        return xTrain, yTrain, xTest, yTest
    
    def calculateF1Score(self, yPred, yTrue, avg='binary'):
        '''
        Calculates the F1-Score of the labels predicted by the model.
        This project uses F1-Score as the standard metric.

        Parameters
        ----------
        yPred : list
            The list of predicted labels
        yTrue : list
            the list of true labels
        avg : str, optional
            The way the average will be calculated for the f1-score. When 
            there is more than 2 labels, use 'weighted'. The default is 'binary'.

        Returns
        -------
        f1Score : float
            The F1-Score of the prediction

        '''
        f1Score = -1.0
        
        import sklearn.metrics as mt
        
        f1Score = mt.f1_score(yTrue, yPred, average=avg)
        
        return f1Score
    
    def calculateReport(self, yPred, yTrue):
        '''
        Returns the classification report for analysis

        Parameters
        ----------
        yPred : list<int>
            The list of predicted labels of the test data
        yTrue : list<int>
            The list of the true labels of the test data

        Returns
        -------
        statsStr : str
            The classification report from the above stats

        '''
        statsStr = ""
        
        import sklearn.metrics as mt
        
        statsStr = mt.classification_report(yTrue, yPred)
        
        return statsStr
        
    def _loadImages(self, path):
        '''
        Loads all of the images in the specified path

        Parameters
        ----------
        path : string
            File path to the directory with images

        Returns
        -------
        data : list<list<int, int>>
            A list of lists containing the image data
                [[image data],
                 [image data],
                 ...]
            where image data is a 2d array (grey scale images)

        '''
        data = []
        
        import os
        import cv2 as opencv
        
        #Get all images in the directory path
        for file in os.listdir(path):
            f = os.path.join(path, file)
            # checking if it is a file
            if os.path.isfile(f):
                data.append(opencv.imread(f, opencv.IMREAD_GRAYSCALE))  #Images are already in black and white, no need to waste space
        
        return data
    
    def _loadRandomImages(self, path, quantity, seed=None):
        data = []
        
        import os
        import random
        import cv2 as opencv
        
        if (seed != None):
            random.seed(seed)
        
        cntr = 0
        maxNumberOfFiles = len(os.listdir(path))
        selectedFileHashes = [] #Ensures that there is no duplicated data
        while (cntr < quantity and len(selectedFileHashes) < maxNumberOfFiles):
            #Grabs a random file from the path
            file = random.choice(os.listdir(path))
            
            f = os.path.join(path, file)
            
            #Checking if it is a file that is not already selected
            if (os.path.isfile(f) and not hash(file) in selectedFileHashes):
                data.append(opencv.imread(f, opencv.IMREAD_GRAYSCALE))
                selectedFileHashes.append(hash(file))
                cntr += 1
        
        return data
        