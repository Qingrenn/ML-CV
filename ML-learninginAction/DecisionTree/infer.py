from DecisionTree import DecisionTree
from lensesClassification import lensesClassification

lc = lensesClassification("lenses.txt")
dataset = lc.buildTrainingDataset()
dt = DecisionTree(dataset)
res = dt.createTree()
print(res)