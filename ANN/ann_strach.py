import numpy as np
from matplotlib import pyplot as plt
from util import*
from variables import*
import math

class MnistANN(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest, Ytest = get_mnist()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest
        self.learning_rate = learning_rate

    def initialize_variables(self):
        D = int(self.Xtrain.shape[1])
        K = int(self.Ytrain.shape[1])

        self.W1 = np.random.randn(D,M) * std
        self.b1 = np.zeros(M)

        self.W2 = np.random.randn(M,K) * std
        self.b2 = np.zeros(K)

    def fordward_prop(self, Xbatch):
        Z1 = np.dot(Xbatch, self.W1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(Z1, self.W2) + self.b2
        A2 = Z2
        return A2

    def loss_function(self, Ypred, Ybatch):
        loss_term = np.square(Ypred - Ybatch).sum() * (1./batch_size)
        reg_term = reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return round(loss_term + reg_term, 3)

    def backward_prop(self, Xbatch, Ybatch):
        Z1 = np.dot(Xbatch, self.W1) + self.b1            # (N, D).(D, M) = (N, M)
        A1 = sigmoid(Z1)                                       # (N, M)
        Ypred = np.dot(Z1, self.W2) + self.b2                  # (N, M).(M, K) = (N, K)

        dYpred = (Ypred - Ybatch) * 2/batch_size          # (N, K)
        self.dW2 = np.dot(A1.T, dYpred) + reg * self.W2        # (M, N).(N, K) = (M, K)
        self.db2 = dYpred.sum(axis=0)                          # (K, )

        dA1 =  np.dot(dYpred, self.W2.T)                       # (N, K).(K, M) = (N, M)
        dZ1 = derivative_sigmoid(Z1) * dA1                     # (N, M) * (N, M) = (N, M)
        self.dW1 = np.dot(Xbatch.T, dZ1) + reg * self.W1  # (D, N).(N, M) = (D, M)
        self.db1 = dZ1.sum(axis=0)                             # (M, )

    def update_parameters(self):
        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.b1 = self.b1 - self.learning_rate * self.db1
        self.W2 = self.W2 - self.learning_rate * self.dW2
        self.b2 = self.b2 - self.learning_rate * self.db2

    @staticmethod
    def accuracy(Ypred, Ybatch):
        P = np.argmax(Ypred, axis=-1)
        Y = np.argmax(Ybatch, axis=-1)
        acc = np.mean(P==Y)
        return round(acc, 3)

    def train(self):
        train_loss = []
        test_loss  = []
        train_acc  = []
        test_acc   = []
        n_batches1 = len(self.Xtrain)//batch_size
        n_batches2 = len(self.Xtest)//batch_size

        self.initialize_variables()
        for i in range(num_epochs+1):
            for j in range(n_batches1):
                Xbatch = self.Xtrain[j*batch_size: (j+1)*batch_size,]
                Ybatch = self.Ytrain[j*batch_size: (j+1)*batch_size,]

                Ypred = self.fordward_prop(Xbatch)

                loss1 = self.loss_function(Ypred, Ybatch)
                acc1 = MnistANN.accuracy(Ypred, Ybatch)

                self.backward_prop(Xbatch, Ybatch)
                self.update_parameters()

            for j in range(n_batches2):
                Xbatch = self.Xtest[j*batch_size: (j+1)*batch_size,]
                Ybatch = self.Ytest[j*batch_size: (j+1)*batch_size,]

                Ypred = self.fordward_prop(Xbatch)

                loss2 = self.loss_function(Ypred, Ybatch)
                acc2 = MnistANN.accuracy(Ypred, Ybatch)

            train_loss.append(loss1)
            test_loss.append(loss2)
            train_acc.append(acc1)
            test_acc.append(acc2)
            print("epoch : {}, train loss : {}, test loss : {}, train acc : {}, test acc : {}".format(i,loss1,loss2,acc1,acc2))

        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.legend()
        plt.show()

        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(test_acc, label='Test Accuracy')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    model = MnistANN()
    model.train()



