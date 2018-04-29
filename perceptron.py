import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import accuracy_score

def K_Fold_Cross_Validation(X, Y, K):
	'''
	Divide the whole dataset into K-folds and 
	perform testing for each one the fold 
	while train the model using other K-1 folds
	'''
	Y = Y.reshape(Y.shape[0],1)
	sampleSize = int(X.shape[0] / K)
	startPoint = 0
	endPoint = startPoint + sampleSize
	X_cv = X[startPoint:endPoint,:]
	Y_cv = Y[startPoint:endPoint]
	X_train = X[0:startPoint, :]
	X_train = np.append(X_train, X[endPoint:,:],axis=0)
	Y_train = Y[0:startPoint]
	Y_train = np.append(Y_train, Y[endPoint:],axis=0)
	divided_dataset = [[X_train, X_cv, Y_train, Y_cv]]
	for i in range(K-1):
		startPoint += sampleSize
		endPoint=startPoint+sampleSize
		X_cv = X[startPoint:endPoint,:]
		Y_cv = Y[startPoint:endPoint]
		X_train = X[0:startPoint,:]
		X_train = np.append(X_train, X[endPoint:, :],axis=0)
		Y_train = Y[0:startPoint]
		Y_train = np.append(Y_train, Y[endPoint:],axis=0)
		divided_dataset = np.append(divided_dataset, np.array([[X_train, X_cv, Y_train,Y_cv]]),axis=0)
	return divided_dataset

def sign(val):
	return -1 if val < 0 else 1

def votedPerceptron(X, Y, epoch):
	'''
	Train the voted perceptron model
	'''
	W = np.zeros((1, X.shape[1]))
	B = np.zeros((1, 1))
	C = np.zeros((1, 1))
	n = 0
	m = X.shape[0]
	number_of_features = X.shape[1]
	for round in range(epoch):
		for i in range(m):			
			if Y[i] * (np.dot(W[n], X[i].reshape(number_of_features, 1)) + B[n]) <= 0:
				W = np.append(W, np.array([W[n] + Y[i] * X[i]]), axis=0)
				B = np.append(B, np.array([B[n] + Y[i]]), axis=0)
				C = np.append(C, [[1]], axis=0)
				n += 1
			else:
				C[n] += 1
	return W[1:, :], B[1:, :], C[1:, :]

def votedPerceptronPredict(X, Y, W, B, C):
	'''
	Predict the output of the voted perceptron
	using the voted weights found during training
	'''
	Y_predicted = np.zeros((1, 1))
	examples = Y.shape[0]
	shape_w = W.shape
	for e in range(examples):
		temp_val = np.dot(W, X[e].T).reshape(shape_w[0], 1) + B
		temp_val[temp_val>0] = 1
		temp_val[temp_val<0] = -1
		Y_predicted = np.append(Y_predicted, np.asarray([sign(np.dot(C.T, temp_val))]))
	Y_predicted = Y_predicted[1:]
	return Y_predicted.reshape(examples, 1)

def vanillaPerceptron(X, Y, epoch):
	'''
	Train the vanilla perceptron model
	'''
	bias = np.ones((X.shape[0],1))
	X = np.concatenate((X, bias),axis=1) # Append a column of 1s with X (for bias)
	number_of_features = X[0].shape[0]
	W = np.zeros((1, number_of_features))
	m = X.shape[0]
	for round in range(epoch):
		for i in range(m):
			if Y[i] * np.dot(W, X[i].reshape(number_of_features, 1)) <= 0:
				W = W + np.multiply(Y[i], X[i])
	return W

def vanillaPerceptronPredict(X, Y, W):
	'''
	Predict the output of the vanilla perceptron
	using the weights found during training
	'''
	bias = np.ones((X.shape[0],1))
	X = np.concatenate((X, bias),axis=1)
	Y_predicted = np.dot(X, W.T)
	Y_predicted[Y_predicted>0] = 1
	Y_predicted[Y_predicted<0] = -1
	return Y_predicted

def perceptron(X, Y, epochs):
	'''
	Compare two perceptron algorithms
	'''
	K = 10 # For K-fold validation
	accuracy_score_voted = []
	accuracy_score_vanilla = []
	divided_dataset = K_Fold_Cross_Validation(X,Y, K)
	for epoch in epochs:
		print("epoch = ", epoch)
		accuracy_vanilla = 0
		accuracy_voted = 0
		for ij in range(divided_dataset.shape[0]):
			X_train = divided_dataset[ij][0]
			X_cv = divided_dataset[ij][1]
			Y_train = divided_dataset[ij][2]
			Y_cv = divided_dataset[ij][3]
			W, B, C = votedPerceptron(X_train, Y_train, epoch)
			Y_predicted_Voted = votedPerceptronPredict(X_cv, Y_cv, W, B, C)
			accuracy_voted += accuracy_score(Y_cv, Y_predicted_Voted, normalize=True)
			W = vanillaPerceptron(X_train, Y_train, epoch)
			Y_predicted_Vanilla = vanillaPerceptronPredict(X_cv, Y_cv, W)
			accuracy_vanilla += accuracy_score(Y_cv, Y_predicted_Vanilla, normalize=True)
		print("Average Voted Perceptron Accuracy: ", accuracy_voted*100/K)
		print("Average Vanilla Perceptron Accuracy: ", accuracy_vanilla*100/K)
		accuracy_score_voted.append(accuracy_voted*100/K)
		accuracy_score_vanilla.append(accuracy_vanilla*100/K)
	return accuracy_score_voted, accuracy_score_vanilla

epochs = list(range(10, 51, 5))
print('----Breast Cancer Dataset----')
print()
cancer_file = 'breast_cancer_dataset.csv'
dataset = np.genfromtxt(cancer_file, delimiter=',')
dataset = dataset[~np.isnan(dataset).any(axis=1)] #remove NaN rows
shuffle(dataset)
X = dataset[:, 1:-1]
Y = dataset[:, -1]
Y = Y.reshape(Y.shape[0], 1)
Y[Y==2] = -1
Y[Y==4] = 1
accuracy_score_voted, accuracy_score_vanilla = perceptron(X, Y, epochs)
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
plt.plot(epochs, accuracy_score_voted, c='b', label='Voted Preceptron')
plt.plot(epochs, accuracy_score_vanilla, c='r', label='Vanilla Preceptron')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, shadow=True, ncol=5)
plt.title('Classification on breast cancer dataset')
plt.savefig('breast_cancer_dataset.jpg')
plt.clf()

print('----Ionosphere Dataset----')
print()
ionosphere_file = 'ionosphere_dataset.csv'
dataset = np.genfromtxt(ionosphere_file, delimiter=',', dtype = '|U5')
shuffle(dataset)
X = dataset[:, :-1]
Y = dataset[:, -1]
Y = Y.reshape(Y.shape[0], 1)
Y[Y=='b'] = -1
Y[Y=='g'] = 1
X = X.astype(float)
Y = Y.astype(float)
accuracy_score_voted, accuracy_score_vanilla = perceptron(X, Y, epochs)
plt.xlabel("Epoch", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
plt.plot(epochs, accuracy_score_voted, c='b', label='Voted Preceptron')
plt.plot(epochs, accuracy_score_vanilla, c='r', label='Vanilla Preceptron')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, shadow=True, ncol=5)
plt.title('Classification on ionosphere dataset')
plt.savefig('ionosphere_dataset.jpg')