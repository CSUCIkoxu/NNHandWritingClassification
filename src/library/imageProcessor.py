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
        #self.characterList = ['0','1','2','3','4','5','6','7,8,9,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t','u','v','w','x','y','z']
        
        self.characterList = [str(i) for i in range(10)]
        import string
        self.characterList.append(list(string.ascii_letters))
        
    def setCharClass(self, chars):
        '''
        Sets the Character Classes to use

        Parameters
        ----------
        chars : list<string>
            A list of characters to read data from

        '''
        self.characterClasses = chars
        
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
            
            charDataDir = self.dataDirectory + self.organizationDirectory + char2Hex + self.trainDirectory + char2Hex
            
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
            
            charDataDir = self.dataDirectory + self.organizationDirectory + char2Hex + self.testDirectory
            
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
        y : lsit<str>
            List of label data

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
            
            charDataDir = self.dataDirectory + self.organizationDirectory + char2Hex + self.trainDirectory + char2Hex
            
            data.append(((e, char) for e in self._loadRandomImages(charDataDir, quantity)))
            
        #Randomize the elements in the array
    
        random.shuffle(data, random.random())
        
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
        y : lsit<str>
            List of label data

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
            
            charDataDir = self.dataDirectory + self.organizationDirectory + char2Hex + self.testDirectory
            
            data.append(((e, char) for e in self._loadRandomImages(charDataDir, quantity)))
            
        #Randomize the elements in the array
    
        random.shuffle(data, random.random())
        
        #Split data to its features (x) and labels (y)
        x = data[:][0]
        y = data[:][1]
        
        return x, y
    
    def splitCV(self, x, y):
        trainingX = None
        trainingY = None
        validX = None
        validY = None
        
        return trainingX, validX, trainingY, validY
        
        
    def _loadImages(self, path):
        '''
        Loads all of the images in the specified path

        Parameters
        ----------
        path : string
            File path to the directory with images

        Returns
        -------
        data : list<int, int>
            DESCRIPTION.

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
        