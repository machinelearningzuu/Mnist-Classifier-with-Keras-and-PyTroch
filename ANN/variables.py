import os
data_path = os.path.join(os.getcwd(), 'data.csv')

train_split = 0.8
M = 200
std = 1e-6
batch_size = 100
reg = 5e-6
learning_rate = 1.4e-2
num_epochs = 100
verbose = 10