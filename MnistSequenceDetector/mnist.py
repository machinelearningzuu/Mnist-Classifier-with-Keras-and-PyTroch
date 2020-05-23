import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from variables import *
import numpy as np

import logging
logging.getLogger('tensorflow').disabled = True

from util import load_data, get_sequence_digits, plot_data

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.995):
            print("\nstop training")
            self.model.stop_training = True


class MnistClassifier(object):
    def __init__(self):
        if not os.path.exists(saved_weights):
            train_generator, validation_generator = load_data()
            self.train_generator = train_generator
            self.validation_generator = validation_generator

    def mnist_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        self.model = model

    def train(self):
        callbacks = myCallback()
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model.summary()
        self.model.fit(
            self.train_generator,
            epochs=num_epochs,
            validation_data=self.validation_generator,
            callbacks= [callbacks]
            )

    def save_model(self):
        self.model.save(saved_weights)

    def load_model(self):
        loaded_model = load_model(saved_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                        )
        self.model = loaded_model

    def predict(self):
        img_arr = get_sequence_digits()
        pred = self.model.predict(img_arr)
        P = pred.argmax(axis=-1).tolist()
        sequence = ''.join([str(i) for i in P])
        plot_data(sequence)

    def run(self):
        if os.path.exists(saved_weights):
            print("Loading existing model !!!")
            self.load_model()

        else:
            print("Training the model  and saving!!!")
            self.mnist_model()
            self.train()
            self.save_model()
        self.predict()

if __name__ == "__main__":
    model = MnistClassifier()
    model.run()
