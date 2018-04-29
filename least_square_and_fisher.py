import numpy as np
import csv 
import matplotlib.pyplot as plt

def LSC_binary_classification(X, Y):
	'''
	Find the weight using
	least square method
	as W = (X.T * X)^(-1) * X.T * Y
	'''
	bias = np.ones((X.shape[0],1))
	X_augmented = np.concatenate((bias, X),axis=1)
	W = np.dot(np.linalg.pinv(np.dot(X_augmented.T, X_augmented)), np.dot(X_augmented.T, Y))
	return W

def FisherLDA_binary_classification(X, Y):
	'''
	Find the direction of projection which
	maximises the class separation, the projected points
	and the weights of the classifier
	Direction of projection = S_W ^ -1 * (mean1 - mean0)
	'''
	dataset = np.concatenate((X, Y), axis=1)
	classes = np.unique(Y)
	class_mean = []
	num_of_dim = 1
	for c in classes:
		class_mean.append(((1 / len(Y[Y==c])) * np.sum(dataset[dataset[:, -1]==c], axis = 0))[:-1])
	X_class1 = dataset[dataset[:, -1]==classes[0]]
	X_class2 = dataset[dataset[:, -1]==classes[1]]
	zero_mean_X_1 = X_class1[:, :-1] - class_mean[0]
	zero_mean_X_2 = X_class2[:, :-1] - class_mean[1]
	 # Using Fisher's LDA formula
	within_class_scatter = np.dot(zero_mean_X_1.T, zero_mean_X_1) + np.dot(zero_mean_X_2.T, zero_mean_X_2)
	W = np.dot(np.linalg.inv(within_class_scatter), (class_mean[1] - class_mean[0]))
	W = W / np.linalg.norm(W) # Unit vector along the direction of projection
	projected_data_scalar = np.dot(X, W) # Magnitude of the projected points
	projections = np.multiply(W.reshape(1,W.shape[0]), 
		projected_data_scalar.reshape(projected_data_scalar.shape[0], 1)) # Magnitude * Unit vector gives the actual vector
	weight = LSC_binary_classification(projections, Y) # Classify the projected 1-D points using least square method
	return projections, W, weight

def classify_LSC_LDA(X, Y, dataset_name):
	'''
	Compare two classifiers using 
	1. Least square method
	2. Fisher LDA
	'''
	x_line = np.linspace(-2, 4)
	plt.figure(figsize=(10,10))
	plt.suptitle("Fisher's discriminant vs Least square - C1 vs C2 - " + dataset_name, fontsize = 18)
	plt.xlabel("x1", fontsize = 16)
	plt.ylabel("x2", fontsize = 16)
	plt.scatter(X[:4, 0], X[:4, 1], c='r', marker='x', label='Class C1')
	plt.scatter(X[4:, 0], X[4:, 1], c='b', marker='o', label='Class C2')
	# Plot Least Square Method
	LSC_weights = LSC_binary_classification(X, Y)
	boundary_LSC = -LSC_weights[0] / LSC_weights[2] - (LSC_weights[1] / LSC_weights[2]) * x_line
	plt.plot(x_line, boundary_LSC, color='g', label='Least Square Classifier')
	# Plot LDA
	projections, direction, LDA_weights = FisherLDA_binary_classification(X, Y)
	direction_line = (direction[1] / direction[0]) * x_line
	boundary_LDA = -LDA_weights[0] / LDA_weights[2] - (LDA_weights[1] / LDA_weights[2]) * x_line
	# Plot Projections
	plt.scatter(projections[:4, 0], projections[:4, 1], c='r', marker='^', label = "C1 projection")
	plt.scatter(projections[4:, 0], projections[4:, 1], c='b', marker='<', label = "C2 projection")
	for i in range(len(projections)):
		plt.plot([X[i][0], projections[i][0]], [X[i][1], projections[i][1]], 'y--')
	# Plot Discriminant and Boundary
	plt.plot(x_line, direction_line, '--', color='black', label = "Fisher Discriminant")
	plt.plot(x_line, boundary_LDA, color='m', label = "Fisher LDA")
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, shadow=True, ncol=3)

	plt.savefig(dataset_name + '.jpg', dpi = 150)
	# plt.show()
	plt.clf()

#Table 1
X = np.array([[3, 3], [3, 0], [2, 1], [0, 2], [-1, 1], [0, 0], [-1, -1], [1, 0]])
Y = np.array([[1], [1], [1], [1], [-1], [-1], [-1], [-1]])
classify_LSC_LDA(X, Y, 'Table 1')

#Table 2
X = np.array([[3, 3], [3, 0], [2, 1], [0, 1.5], [-1, 1], [0, 0], [-1, -1], [1, 0]])
Y = np.array([[1], [1], [1], [1], [-1], [-1], [-1], [-1]])
classify_LSC_LDA(X, Y, 'Table 2')