from util import get_data
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from variables import *
import numpy as np
from tensorflow.keras.models import model_from_json

# np.random.seed(42)
# tf.compat.v1.set_random_seed(42)

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.995):
            print("\nstop training")
            self.model.stop_training = True


class MnistClassifier(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest , Ytest = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest


    def mnist_model(self):
        self.model = Sequential([
                    Dense(dense1, input_shape=(tensor_shape,), activation='relu'),
                    Dense(dense1, activation='relu'),
                    Dense(dense1, activation='relu'),
                    Dropout(0.5),
                    Dense(output, activation='softmax'),
                ])

    def train(self):
        callbacks = myCallback()
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model.summary()
        self.model.fit(
            self.Xtrain,
            self.Ytrain,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(self.Xtest,self.Ytest),
            callbacks= [callbacks]
            )

    def save_model(self):
        model_json = self.model.to_json()
        with open(saved_model, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(saved_weights)

    def load_model(self):
        json_file = open(saved_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(saved_weights)
        loaded_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model = loaded_model

    def predict(self,images,labels):
        if not isinstance(images[0], np.ndarray):
            y_pred = np.argmax(self.model.predict(np.array([images]))[0])
            loss, accuracy = self.model.evaluate(images.reshape(1,784),np.array([labels]))
        else:
            y_pred = np.argmax(self.model.predict(images), axis = 1)
            loss, accuracy = self.model.evaluate(images,labels)   
        print("Prediction : {}".format(y_pred))
        print("loss : {}".format(loss))
        print("accuracy : {}".format(accuracy))        
