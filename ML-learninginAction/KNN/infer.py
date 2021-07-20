from KNN import KNN
from numberClassification import NumClassification
import numpy as np

def acc(pred, label):
    t = np.equal(pred, label)
    return np.sum(t) / len(pred)


nc = NumClassification(trainingPath="digits/trainingDigits", 
                        testPath="digits/testDigits")

trainingDataset = nc.buildTrainingDataset()
testDataset, labels = nc.buildTestDataset()

knn = KNN(trainingDataset, 3, isnorm=False)
pred = knn.infer(testDataset)

print(acc(pred, labels))