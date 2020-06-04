import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

def proximityMatrix(model, X):

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:,0]
    proxMat = 1*np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:,i]
        proxMat += 1*np.equal.outer(a, a)

    # to normalize:
        proxMat = proxMat / nTrees

    return proxMat

from sklearn.ensemble import  RandomForestClassifier
from sklearn.datasets import load_breast_cancer
train = load_breast_cancer()

model = RandomForestClassifier(n_estimators=100)
model.fit(train.data, train.target)
print(proximityMatrix(model, train.data, normalize=True))
