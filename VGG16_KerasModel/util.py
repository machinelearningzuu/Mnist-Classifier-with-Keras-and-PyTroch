from variables import *
import numpy as np
import pandas as pd
import re
import pickle
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img

def preprocess_data(csv_file,filename,Train=True):
    if not os.path.exists(filename):
        print('{} data preprocessing and saving !'.format(filename))
        first = True
        X = []
        Y = []
        for l, line in enumerate(open(csv_file, encoding="utf8", errors='ignore')):
            if first:
                first = False
            else:
                mnist_row = re.split('[, \n]', line)
                mnist_row = [int(px) for px in mnist_row if len(px) > 0]
                y, x = mnist_row[0], mnist_row[1:]
                x = np.array(x)
                X.append(x)
                Y.append(y)
            if Train and (l == train_size): # we only load 20000 train images and 4000 test images
                break
            elif (not Train) and (l == test_size):
                break


        Xinput = np.array(X)/255
        Xinput = np.dstack([Xinput] * in_channels) # VGG 16 mush need images with 3 input channels so we stack 3 layers
        Xinput = Xinput.reshape(-1, original_size,original_size,in_channels)
        Yinput = np.array(Y)
        Xinput = upscale_images(Xinput) # vGG 16 need images with minimal width and height as 32 So using this we upscale image to 48 * $8

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
    Xtest , Ytest  = preprocess_data(test_path,test_pickle,False)
    return Xtrain, Ytrain, Xtest , Ytest

def upscale_images(X):
    return np.asarray([img_to_array(array_to_img(im, scale=False).resize((up_scale))) for im in X])