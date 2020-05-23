from variables import *
import numpy as np
import pandas as pd
import re
import pickle
import os
import cv2 as cv
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def get_data(csv_path):
    df = pd.read_csv(csv_path)
    columns = df.columns.values

    labels = df[columns[0]].values
    features = df[columns[1:]].values/255.0

    N = len(labels)
    features = features.reshape(N, img_dim, img_dim, 1)
    return features, labels

def load_data():
    Xtrain, Ytrain = get_data(train_path)
    Xtest , Ytest  = get_data(test_path)

    Xtrain = Xtrain.astype('float32')
    Xtest = Xtest.astype('float32')

    train_datagen = ImageDataGenerator(
                                    width_shift_range=shift,
                                    height_shift_range=shift,
                                    zoom_range = zoom_range
                                    )
    train_datagen.fit(Xtrain)

    test_datagen = ImageDataGenerator()
    test_datagen.fit(Xtest)

    train_generator  = train_datagen.flow(Xtrain, Ytrain, batch_size=batch_size)
    validation_generator = test_datagen.flow(Xtest , Ytest, batch_size=batch_size)

    return train_generator, validation_generator

def get_sequence_digits():
    img = cv.imread('seq_image.jpg', 0)
    img = cv.resize(img, (img_dim*n_digits, img_dim))/255.0
    img_arr = []
    for i in range(5):
        digit = img[:, i*img_dim:(i+1)*img_dim]
        digit = digit.reshape(img_dim,img_dim,1)
        img_arr.append(digit)
    img_arr = np.array(img_arr)
    return img_arr

def plot_data(sequence):
    img = cv.imread('seq_image.jpg', 0)
    print("Predicted Sequence : {}".format(sequence))
    plt.imshow(img)
    plt.show()