from typing import *
import re
from sys import byteorder
import numpy as np
import os
from sklearn.model_selection import train_test_split


class Prepper:
    def __init__(self, direc : str):
        self._rootDir   : str = direc
        self._folders   : List = os.listdir(direc)
        self._realPairs : Set = set()
        self._fakePairs : Set = set()


    def addNewPair(self, pair : Tuple, real : bool):
        listed_set = list(self._realPairs if real else self._fakePairs)
        if pair[::-1] not in listed_set:
            listed_set.append(pair)
        else:
            return False
        if real:
            self._realPairs = set(listed_set)
        else:
            self._fakePairs = set(listed_set)
        return True

    def findRealPairs(self):
        for folder in self._folders:
            folder_dir : str = self._rootDir + "\\" + folder + "\\"
            file_names : List = os.listdir(folder_dir)
            for file in file_names:
                other_files : List = [f for f in file_names if f != file]
                for f1 in other_files:
                    self.addNewPair((folder_dir + file, folder_dir + f1), True)

    def findFakePairs(self):
        '''
        Randomly select pairs which are not of the same person, this is based on the file
        structure provided by Olivetti dataset
        '''
        folder_names = os.listdir(self._rootDir)
        random_choice_folders : List = []
        while len(random_choice_folders) < 5:
            rand_folder = np.random.choice(folder_names)
            if rand_folder not in random_choice_folders:
                random_choice_folders.append(rand_folder)
        for folder in random_choice_folders:
            file = np.random.choice(os.listdir(self._rootDir+"\\"+folder))
            other_folders = [x for x in folder_names if x != folder]
            for alt_folder in other_folders:
                for alt_file in os.listdir(self._rootDir + "\\" + alt_folder):
                    self.addNewPair((self._rootDir+"\\"+folder+"\\"+file, self._rootDir+"\\"+alt_folder+"\\"+alt_file), False)

    def getPairs(self, real : bool):
        return list(self._realPairs if real else self._fakePairs)

    def aggregateDataset(self, max_size : int, test_dec : float):
        realNPdata = np.zeros([max_size, 2, 112, 92])
        fakeNPdata = np.zeros([max_size, 2, 112, 92])
        data_count = 0
        for realPair, fakePair in zip(self._realPairs, self._fakePairs):
            realNPdata[data_count] = np.array(list(map(Prepper.convertPGMtoNumpy, realPair)))
            fakeNPdata[data_count] = np.array(list(map(Prepper.convertPGMtoNumpy, fakePair)))
            data_count+=1
        labels = np.array([0]*max_size + [1]*max_size)
        return train_test_split(np.concatenate((realNPdata, fakeNPdata))/255, labels, test_size=test_dec)

    @staticmethod
    def convertPGMtoNumpy(img_dir : str):
        with open(img_dir, "rb") as file:
            buffer = file.read()
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        return np.frombuffer(buffer,
                                dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                                count=int(width) * int(height),
                                offset=len(header)
                                ).reshape((int(height), int(width)))

if __name__ == '__main__':

    p = Prepper(PATH_TO_KNOWN_FACES)

    p.findRealPairs()
    print(len(p.getPairs(True)))
    p.findFakePairs()
    print(len(p.getPairs(False)))

    X_train, X_test, Y_train, Y_test = p.aggregateDataset(1800, 0.3)
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)
