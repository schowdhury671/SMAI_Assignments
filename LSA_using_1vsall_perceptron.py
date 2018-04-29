import numpy as np
import matplotlib.pyplot as plt
import nltk
import codecs
import os
import sys
import sklearn
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from random import shuffle
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfTransformer

def sign(val):
	return -1 if val < 0 else 1

def formTFMatrix(documents, terms):
	'''
	Construct the bag of words matrix for each document
	where each row represents each document and
	columns represent each terms
	'''
	tf_matrix = []
	for words in documents:
		doc_terms = []
		for t in terms:
			doc_terms.append(words.count(t))
		tf_matrix.append(doc_terms)
	return np.array(tf_matrix)

def multiclass_voted_perceptron(X, Y, epoch):
	'''
	Voted percetron training, trained with for class
	'''
	m = X.shape[0]
	number_of_features = X.shape[1]
	W = np.zeros((1, number_of_features))
	B = np.zeros((1, 1))
	C = np.zeros((1, 1))
	n = 0
	for round in range(epoch):
		for i in range(m):			
			if Y[i] * (np.dot(W[n], X[i].reshape(number_of_features, 1)) + B[n]) <= 0:
				W = np.append(W, np.array(W[n] + Y[i] * X[i]), axis=0)
				B = np.append(B, np.array([B[n] + Y[i]]), axis=0)
				C = np.append(C, [[1]], axis=0)
				n += 1
			else:
				C[n] += 1
	return W[1:, :], B[1:, :], C[1:, :]

def votedPerceptronPredict(X, Y, W_dict, B_dict, C_dict, class_labels):
	'''
	Predict the class label using the trained voted perceptron model
	Predict the class for which the absolute vaue prediction is maximum of other class
	'''
	Y_predicted = np.zeros((1, 1))
	examples = Y.shape[0]
	for e in range(examples):
		predict = np.asarray([])
		for x in class_labels:
			W = W_dict[x]
			B = B_dict[x]
			C = C_dict[x]
			shape_w = W.shape
			temp_val = np.dot(W, X[e].T).reshape(shape_w[0], 1) + B
			# temp_val[temp_val>0] = 1
			# temp_val[temp_val<0] = -1
			predict = np.append(predict, np.dot(C.T, temp_val))
		Y_predicted = np.append(Y_predicted, np.asarray([(np.argmax(predict))]))

	Y_predicted = Y_predicted[1:]
	return Y_predicted.reshape(examples, 1)

def one_vs_all_voted(X, Y, X_test, Y_test, epoch):
	'''
	Call voted perceptron for each of the classes
	'''
	class_labels = np.unique(Y)
	weight = {}
	bias = {}
	vote = {}
	Y_original = deepcopy(Y)
	for x in range(class_labels.shape[0]):
		Y = deepcopy(Y_original)
		for i in range(Y.shape[0]):
			if Y[i][0] == x:
				Y[i][0] = 1
			else:
				Y[i][0] = -1
		weight[x], bias[x], vote[x] = multiclass_voted_perceptron(X, Y, epoch)
	Y_predicted = votedPerceptronPredict(X_test, Y_test, weight, bias, vote, class_labels)
	correct_class = 0
	for i in range(Y_predicted.shape[0]):
		if Y_predicted[i][0] == Y_test[i][0]:
			correct_class += 1
	print((correct_class / Y_predicted.shape[0]) * 100)

def extractDocs(dataPath):
	'''
	Extract words from the documents
	'''
	documents = []
	class_labels = []
	stop = set(stopwords.words('english')) # Reomoves the stop words eg the, an, a, and etc.
	stemmer = PorterStemmer() # Converts the words to their root words
	tokenizer = RegexpTokenizer(r'\w+') # Tokenize the document to et rid off non alphabets
	folders = [dataPath + name + '/' for name in os.listdir(dataPath) if os.path.isdir(dataPath + name)]
	class_name = [int(name) for name in os.listdir(dataPath) if os.path.isdir(dataPath + name)]
	it = 0
	for folder in folders:
		docFiles = [folder + docName for docName in os.listdir(folder)]
		doc_words = []
		for f in docFiles:
			file = codecs.open(f, 'r', 'ISO-8859-1')
			file_content = []
			for document in file:
				word_list = [stemmer.stem(line) for line in tokenizer.tokenize(document.lower()) if line not in '']
				file_content.extend([w for w in word_list if w not in stop])
			file.close()
			documents.append(file_content)
			class_labels.append(class_name[it])
		it += 1
	return documents, (np.array([class_labels])).T

def getUniqueWords(documents):
	'''
	Find the dictionary of words
	'''
	words = [word for doc in documents for word in doc]
	unique_words = list(set(words))
	return unique_words

def classify_document_by_perceptron(dataPath, testPath):
	'''
	Classifies the documents after applying PCA
	and then using the voted perceptron model
	'''
	avg_accuracy = []
	documents, classes = extractDocs(dataPath)
	test_documents, test_classes = extractDocs(testPath)
	train_data_length = len(documents)
	for test_doc in test_documents:
		documents.append(test_doc)
	terms = getUniqueWords(documents)
	tf_matrix = formTFMatrix(documents, terms)
	tf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False) 
	tf_idf_matrix = tf.fit_transform(tf_matrix).todense() # Form the tf-idf matrix from bag of words
	test_data = tf_idf_matrix[train_data_length: , :]
	tf_idf_matrix = tf_idf_matrix[:train_data_length, :]
	U, Sigma_temp, V_T = np.linalg.svd(tf_idf_matrix) # Singular Value Decomposition
	Sigma = np.zeros((Sigma_temp.shape[0], Sigma_temp.shape[0]))
	for i in range(Sigma_temp.shape[0]):
		Sigma[i][i] = Sigma_temp[i]
	# k_list = list(range(10, min(tf_idf_matrix.shape[0], tf_idf_matrix.shape[1]), 50))
	k_list = [1110]
	for k in k_list:
		correct_similarity = 0
		# Reducce the dimension using the eigen value which contributes the most
		reduced_data = np.dot(np.dot(U[:, :k], Sigma[:k, :k]), V_T[:k, :])
		epoch = 40 # Epoch for the perceptron gtraing
		print("Accuracy using k = ", k)
		one_vs_all_voted(reduced_data, classes, test_data, test_classes, epoch)

dataPath = str(sys.argv[1]) #'./LSAdata (copy)/train/'
testPath = str(sys.argv[2]) #'./LSAdata (copy)/test/'
dataPath = '../LSAdata (copy)/train/'
testPath = '../LSAdata (copy)/test/'
classify_document_by_perceptron(dataPath, testPath)

# dataPath = str(sys.argv[1])
# testPath = str(sys.argv[2])
