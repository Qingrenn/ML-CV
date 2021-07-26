import numpy as np
from math import log

class DecisionTree:

    def __init__(self, dataset) -> None:
        assert isinstance(dataset, list) and len(dataset[0]) > 1, "[ERROR] datset format is not suitable"
        self.dataset = dataset

    def __splitDataset(self, dataset, axis, value):
        retDataset = []
        for sample in dataset:
            if sample[axis] == value:
                reducedFeatVec = sample[:axis] + sample[axis+1:]
                retDataset.append(reducedFeatVec)
        return retDataset

    def __calcShannonEnt(self, dataset):
        numEntries = len(dataset)
        labelCounts = {}
        for sample in dataset:
            label = sample[-1]
            labelCounts[label] = labelCounts.get(label, 0) + 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = labelCounts[key]/numEntries
            shannonEnt += -prob * log(prob, 2)
        return shannonEnt

    def __chooseBestFeat(self, dataset):
        baseEnt = self.__calcShannonEnt(dataset)
        bestFeat = 0
        bestInfoGain = 0
        for feat in range(len(dataset[0])-1):
            featList = [sample[feat] for sample in dataset]
            uniqueVals = set(featList)
            newEnt = 0.0
            for val in uniqueVals:
                splitedDataset = self.__splitDataset(dataset, feat, val)
                prob = len(splitedDataset) / len(dataset)
                newEnt += - prob * log(prob, 2)
            infoGain = newEnt - baseEnt
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeat = feat
        return bestFeat
        
    def __majorityCnt(self, classList):
        classCnt = {}
        for vote in classList:
            classCnt[vote] = classCnt.get(vote, 0)
        sortedClassCnt = sorted(classCnt.items(), key=lambda item: item[1], reverse=True)
        return sortedClassCnt[0][0]

    def createTree(self):
        return self.__reverse(self.dataset, ["F1", "F2", "F3", "F4"])

    def __reverse(self, dataset, labels):
        classList = [sample[-1] for sample in dataset]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataset[0]) == 1:
            return self.__majorityCnt(dataset)
        bestFeat = self.__chooseBestFeat(dataset)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del labels[bestFeat]
        featValues = [sample[bestFeat] for sample in dataset]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.__reverse(self.__splitDataset(dataset,bestFeat, value), subLabels)
        return myTree
