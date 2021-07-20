import os.path as osp
import os 
import numpy as np

class NumClassification:
    def __init__(self, trainingPath, testPath) -> None:
        assert osp.exists(trainingPath) and osp.exists(testPath)
        self.trainingPath = trainingPath
        self.testPath = testPath

    def buildTrainingDataset(self):
        filelist = os.listdir(self.trainingPath)
        filelist = [osp.join(self.trainingPath, f) for f in filelist]
        trainingDataset, labels = self._files2array(filelist)
        trainingDataset = np.concatenate([trainingDataset, labels.reshape(-1,1)], axis=1)
        return trainingDataset
            
    def buildTestDataset(self):
        filelist = os.listdir(self.testPath)
        filelist = [osp.join(self.testPath, f) for f in filelist]
        testDataset, labels = self._files2array(filelist)
        return testDataset, labels

    def _files2array(self, filelist):
        dataset = np.zeros([len(filelist), 32 * 32])
        labels = np.zeros(len(filelist))
        for i, file in enumerate(filelist):
            label = file.strip().split("/")[-1].split("_")[0]
            labels[i] = label
            with open(file) as f:
                lines = f.readlines()
                for row in range(32):
                    for col in range(32):
                        dataset[i, row*32 + col] = lines[row][col]
        return dataset, labels
    
    