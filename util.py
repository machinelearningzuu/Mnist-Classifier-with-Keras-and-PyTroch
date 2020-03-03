from variables import *
import numpy as np 
import pandas as pd
import re
import pickle
import os
import matplotlib.pyplot as plt 

def preprocess_data(csv_file,filename):
    if not os.path.exists(filename):
        print('{} data preprocessing and saving !'.format(filename))
        first = True
        X = []
        Y = []
        for line in open(csv_file, encoding="utf8", errors='ignore'):
            if first:
                first = False
            else:
                mnist_row = re.split('[, \n]', line)
                mnist_row = [int(px) for px in mnist_row if len(px) > 0]
                y, x = mnist_row[0], mnist_row[1:]
                X.append(x)
                Y.append(y)
        Xinput = np.array(X)/255
        Yinput = np.array(Y)

        outfile = open(filename,'wb')
        pickle.dump((Xinput,Yinput),outfile)
        outfile.close()
    else:
        print("{} data loading from pickle".format(filename))
        infile = open(filename,'rb')
        Xinput,Yinput = pickle.load(infile)
        infile.close()
    return Xinput, Yinput

def get_data():
    Xtrain, Ytrain = preprocess_data(train_path,train_pickle)    
    Xtest , Ytest   = preprocess_data(test_path,test_pickle)
    return Xtrain, Ytrain, Xtest , Ytest

def plot_image(X,idx=None):
    if  isinstance(X[0], np.ndarray):
        N, K = X.shape
        imgs = X.reshape(N, int(np.sqrt(K)), int(np.sqrt(K)))
        img = imgs[idx]
    else:
        K = X.shape[0]
        img = X.reshape(int(np.sqrt(K)), int(np.sqrt(K)))
    plt.imshow(img, cmap="Greys")
    plt.show()
