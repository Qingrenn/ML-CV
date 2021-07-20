import numpy as np

class KNN:

    def __init__(self, dataset, k, isnorm=True):
        assert isinstance(dataset, np.ndarray) and dataset.shape[1] > 1, "[ERROR] datset format is not suitable"
        if isnorm:
            self.dataset = self._norm(dataset)
        else:
            self.dataset = dataset
        self.k = k

    def _norm(self, dataset):
        data = dataset[:, :-1]
        min = data.min(axis=0)
        max = data.max(axis=0)
        dataset[:, :-1] = (data - min) / (max - min) # ndarray 的广播机制
        return dataset

    def infer(self, sample):
        assert isinstance(sample, np.ndarray) and sample.shape[1] == self.dataset.shape[1] - 1
        res = []
        for i in range(sample.shape[0]):
            err  = (self.dataset[:, :-1] - sample[i, :])**2
            distance = np.sqrt(np.sum(err, axis=1))
            sortedargs = distance.argsort()
            classCount = {}
            for i in range(self.k):
                votedLabel = self.dataset[sortedargs[i], -1]
                classCount[votedLabel] = classCount.get(votedLabel, 0) + 1
            sortedclassCount = sorted(classCount.items(), key=lambda item:item[1], reverse=True)
            res.append(sortedclassCount[0][0])
        return res
            


if __name__ == "__main__":
    test_dataset = np.array([[1,1.1,1], [1,1,1], [0,0,0], [0,0.1,0]])
    knn = KNN(test_dataset , 2)
    res = knn.infer(np.array([[1,0.8], [0, 0.5]]))
    print(res)


