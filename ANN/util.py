import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.utils import to_categorical

import logging
logging.getLogger('tensorflow').disabled = True

from variables import*

def get_mnist():
    df = pd.read_csv(data_path)
    colmns = df.columns.values
    Y = df[colmns[0]].values
    X = df[colmns[1:]].values/255.0

    print("Input  data shape: {}".format(X.shape))
    print("Output data shape: {}".format(Y.shape))

    X, Y = shuffle(X, Y)
    Y = to_categorical(Y, num_classes=len(set(Y)) , dtype='float32')

    Ntrain = int(train_split * len(X))

    Xtrain, Xtest = X[:Ntrain,], X[Ntrain:,]
    Ytrain, Ytest = Y[:Ntrain,], Y[Ntrain:,]

    return Xtrain, Ytrain, Xtest, Ytest

def relu(z):
    return z * (z > 0)

def derivative_relu(z):
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    g_z = sigmoid(z)
    return g_z * (1 - g_z)