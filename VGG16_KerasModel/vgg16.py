import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from util import get_data
from variables import *
from keras import models
from keras.models import Model
from keras import layers
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.995):
            print("\nstop training")
            self.model.stop_training = True

class VGG16Base(object):
    def __init__(self):
        self.vgg_functional = keras.applications.vgg16.VGG16(weights = 'imagenet',
                                                        include_top=False,
                                                        input_shape=input_shape)
        self.vgg_functional.summary()

        Xtrain, Ytrain, Xtest , Ytest = get_data()
        self.Ytrain = Ytrain
        self.Ytest  = Ytest

        print("Shape of Train inputs :",Xtrain.shape)
        print("Shape of Test  inputs :",Xtest.shape)

        train_features  = self.vgg_functional.predict(np.array(Xtrain), batch_size=batch_size, verbose=1)
        test_features   = self.vgg_functional.predict(np.array(Xtest) , batch_size=batch_size, verbose=1)

        train_features_flat = np.reshape(train_features, (train_size, 1*1*512))
        test_features_flat  = np.reshape(test_features , (test_size,  1*1*512))

        self.train_features_flat = train_features_flat
        self.test_features_flat = test_features_flat

        print("Shape of flattened train features :",self.train_features_flat.shape)
        print("Shape of flattened Test  features :",self.test_features_flat.shape)

    def classifier(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=(1*1*512)))
        model.add(Dense(64, activation='relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
        self.model = model

    def train(self):
        callbacks = myCallback()
        self.model.compile(
                          optimizer='Adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.history =  self.model.fit(
                                    self.train_features_flat,
                                    self.Ytrain,
                                    batch_size=batch_size,
                                    epochs=num_epochs,
                                    validation_data=(
                                                    self.test_features_flat,
                                                    self.Ytest
                                                    ),
                                    callbacks= [callbacks]
                                    )

    def save_model_and_history(self):
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)
        print("Model Saved")

        with open(model_history, 'wb') as his:
            pkl.dump(self.history.history, his)

    def load_model_and_history(self):
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        self.model = loaded_model
        self.model.compile(
                          optimizer='Adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.history = pkl.load(open(model_history,"rb"))

    def plot_histroy(self):
        acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.title('Training and validation accuracy')
        plt.plot(epochs, acc, 'red', label='Training acc')
        plt.plot(epochs, val_acc, 'blue', label='Validation acc')
        plt.legend()

        plt.figure()
        plt.title('Training and validation loss')
        plt.plot(epochs, loss, 'red', label='Training loss')
        plt.plot(epochs, val_loss, 'blue', label='Validation loss')

        plt.legend()
        plt.show()

    def predictions(self):
        loss, acc = self.model.evaluate(
                                        self.test_features_flat,
                                        self.Ytest,
                                        batch_size=batch_size
                                        )
        print("Test loss :",loss)
        print("Test accuracy :",acc)

if __name__ == "__main__":
    model = VGG16Base()
    if os.path.exists(os.path.join(os.getcwd(),model_weights)):
        model.load_model_and_history()
    else:
        model.classifier()
        model.train()
        model.save_model_and_history()
    model.predictions()
    model.plot_histroy()
