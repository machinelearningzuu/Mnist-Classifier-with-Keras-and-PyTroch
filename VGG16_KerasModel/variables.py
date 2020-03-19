seed = 42
num_classes = 10
batch_size = 16
num_epochs = 100
input_shape = (48, 48, 3)
up_scale = 48,48
train_size = 20000
test_size = 4000
in_channels = 3
original_size = 28
# data paths and model paths
model_architecture = 'mnist_classifier.json'
model_weights = 'mnist_classifier.h5'
model_history = 'ModelHistoryDict'
train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'
train_pickle = 'train'
test_pickle = 'test'