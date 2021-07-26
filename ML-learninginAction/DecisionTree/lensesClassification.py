import os

class lensesClassification:
    def __init__(self, path) -> None:
        assert os.path.exists(path)
        self.trainingPath = path
    
    def buildTrainingDataset(self):
        dataset = []
        with open(self.trainingPath) as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(line.strip().split("\t", 4))
        return dataset
