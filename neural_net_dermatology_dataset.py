from L_Layer_Neural_Net import L_layer_model, predict
from random import shuffle
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import time
import datetime


def process_dataset(dataset):
    shuffle(dataset)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    y = y.reshape(y.shape[0], 1)

    no_of_classes = len(np.unique(y))

    # Process Y for multiclass
    Y = np.zeros((1, no_of_classes))
    for i in range(len(y)):
        temp_y = np.zeros((1, no_of_classes))
        temp_y[0, int(y[i]) - 1] = 1
        Y = np.concatenate((Y, temp_y), axis=0)
    Y = Y[1:, :]
    # X = preprocessing.normalize(X)
    X = preprocessing.scale(X)
    return X, Y, no_of_classes

# Variable Parameters
fold = 5
hidden_layer_nodes = range(5, 200, 20)
hidden_layer_activation = 'relu'

# Load the dataset

dermatology_file = 'dermatology.txt'
dataset = np.genfromtxt(dermatology_file, delimiter=',')
dataset = dataset[~np.isnan(dataset).any(axis=1)]  # remove NaN rows

# For learning 3 class classifier, delete other class examples
# dataset = dataset[~(dataset[:, -1] > 3), :]

X, Y, no_of_classes = process_dataset(dataset)


kf = KFold(n_splits=fold)
kf.get_n_splits(X)

Accuracies = []

for iNodes in hidden_layer_nodes:
    accuracy = 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = X[train_index, :].T, X[test_index, :].T
        train_y, test_y = Y[train_index, :].T, Y[test_index, :].T

        layers_dims = [train_x.shape[0], iNodes, no_of_classes]

        parameters = L_layer_model(
            train_x, train_y, layers_dims, no_of_classes, learning_rate=0.0075, num_iterations=2500, print_cost=False, hidden_layer_activation=hidden_layer_activation)

        pred_test, acc = predict(
            test_x, test_y, parameters, no_of_classes, hidden_layer_activation=hidden_layer_activation)
        accuracy += acc
    # print(accuracy/fold)
    Accuracies.append(accuracy/fold)

# Plot the graphs
plt.xlim((0, hidden_layer_nodes[-1] + 2))
plt.ylim((0, 110))
plt.xlabel("Hidden Layer Nodes", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.plot(hidden_layer_nodes, Accuracies, c='b')

plt.title('Classification on dermatology dataset - ' + hidden_layer_activation)
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
plt.savefig('dermatology_dataset' + str(st) + '.jpg')
plt.clf()
print(Accuracies)
