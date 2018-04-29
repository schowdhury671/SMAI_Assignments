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
    # X = preprocessing.scale(X)
    return X, Y, no_of_classes

# Variable Parameters
fold = 5
hidden_layer_nodes = range(10, 80, 20)
hidden_layer_activation = 'relu'

# Load the dataset

pendigits_train_file = 'pendigits.tra'
train_dataset = np.genfromtxt(pendigits_train_file, delimiter=',')
train_dataset = train_dataset[~np.isnan(
    train_dataset).any(axis=1)]  # remove NaN rows

# For learning 3 class classifier, delete other class examples
# train_dataset = train_dataset[~(train_dataset[:, -1] > 4), :]

train_x, train_y, no_of_classes = process_dataset(train_dataset)

pendigits_test_file = 'pendigits.tes'
test_dataset = np.genfromtxt(pendigits_test_file, delimiter=',')
test_dataset = test_dataset[~np.isnan(
    test_dataset).any(axis=1)]  # remove NaN rows

# For learning 3 class classifier, delete other class examples
# test_dataset = test_dataset[~(test_dataset[:, -1] > 4), :]

test_x, test_y, no_of_classes_test = process_dataset(test_dataset)

Accuracies = []

for iNodes in hidden_layer_nodes:
    accuracy = 0

    # Specify layer dimensions
    layers_dims = [train_x.shape[1], iNodes, no_of_classes]

    parameters = L_layer_model(
        train_x.T, train_y.T, layers_dims, no_of_classes, learning_rate=0.0075, num_iterations=3000, print_cost=False, hidden_layer_activation=hidden_layer_activation)

    pred_test, accuracy = predict(
        test_x.T, test_y.T, parameters, no_of_classes, hidden_layer_activation=hidden_layer_activation)

    Accuracies.append(accuracy)


# Plot the graphs
plt.xlim((min(hidden_layer_nodes) - 5, max(hidden_layer_nodes) + 5))
plt.ylim((0, 110))
plt.xlabel("Hidden Layer Nodes", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.plot(hidden_layer_nodes, Accuracies, c='b', label='Accuracy')

# plt.legend(loc='upper center', bbox_to_anchor=(
#     0.5, -0.04), fancybox=True, shadow=True, ncol=5)
plt.title('Classification on pendigit dataset - ' + hidden_layer_activation)
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
plt.savefig('pendigit_dataset' + str(st) + '.jpg')
plt.clf()
print(Accuracies)

