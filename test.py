from util import get_data, plot_image
from variables import saved_weights
import os
from mnist import MnistClassifier
import numpy as np
current_dir = os.getcwd()
saved_weights = os.path.join(current_dir,saved_weights)

if __name__ == "__main__":
    Xtrain, Ytrain, Xtest , Ytest = get_data()
    classifier = MnistClassifier()
    if os.path.exists(saved_weights):
        print("Loading existing model !!!")
        classifier.load_model()
    else:
        print("Training the model  and saving!!!")
        classifier.mnist_model()
        classifier.train()
        classifier.save_model()
    
    idx = np.random.randint(len(Xtest))
    plot_image(Xtest,idx)
    classifier.predict(Xtest[idx],Ytest[idx])