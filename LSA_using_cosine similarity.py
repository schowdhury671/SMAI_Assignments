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

def formTFMatrix(documents, terms):
	'''
	Construct the bag of words matrix for each document
	where each row represents each document and
	columns represent each terms
	'''
	td_matrix = []
	for words in documents:
		doc_terms = []
		for t in terms:
			doc_terms.append(words.count(t))
		td_matrix.append(doc_terms)
	return np.array(td_matrix)

def readtestDoc(testPath):
	'''
	Reading the test document
	'''
	documents = []
	stop = set(stopwords.words('english'))
	stemmer = PorterStemmer()
	tokenizer = RegexpTokenizer(r'\w+')
	file = codecs.open(testPath, 'r', 'ISO-8859-1')
	file_content = []
	for document in file:
		word_list = [stemmer.stem(line) for line in tokenizer.tokenize(document.lower()) if line not in '']
		file_content.extend([w for w in word_list if w not in stop])
	file.close()
	documents.append(file_content)
	return documents

def measure_similarity_by_cosine(dataPath, testPath, testDocLabel):
	avg_accuracy = []
	documents, classes = extractDocs(dataPath)
	# test_documents, test_classes = extractDocs(testPath)
	test_documents = readtestDoc(testPath)
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

	# k_list = list(range(1, min(tf_idf_matrix.shape[0], tf_idf_matrix.shape[1]), 50))
	k_list = [1251] # Best value of k
	print("SVD Accuracy:")
	for k in k_list:
		correct_similarity = 0
		print(tf_idf_matrix.shape, U.shape, Sigma.shape, V_T.shape)
		reduced_data = np.dot(np.dot(U[:, :k], Sigma[:k, :k]), V_T[:k, :])
		cosine_similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(reduced_data, test_data)
		for x in range(test_data.shape[0]):
			augmented_similarity_matrix = np.concatenate(
				(cosine_similarity_matrix[:, x].reshape(cosine_similarity_matrix[:, 0].shape[0], 1), 
					classes.reshape(classes.shape[0], 1)), axis = 1)
			sorted_simliarity_matrix = augmented_similarity_matrix[augmented_similarity_matrix[:, 0].argsort()]
			sorted_simliarity_matrix = sorted_simliarity_matrix[::-1]
			predicted_label = -1
			max_match = -1
			top_10_results = sorted_simliarity_matrix[:10, -1]
			labels = np.unique(top_10_results)
			for l in labels:
				if max_match < list(top_10_results).count(l):
					max_match = list(top_10_results).count(l)
					predicted_label = l
			print("The predicted class label of the document: ", int(predicted_label))
	# 		if int(predicted_label) == int(test_classes[x]):
	# 			correct_similarity += 1
	# 	avg_accuracy.append(correct_similarity * 100 / test_data.shape[0])
	# 	print("Accuracy for k = ", k),
	# 	print(correct_similarity * 100 / test_data.shape[0])
	# return k_list, avg_accuracy

dataPath = str(sys.argv[1]) #'./LSAdata (copy)/train/'
testPath = str(sys.argv[2]) #'./LSAdata (copy)/test/'
testDocLabel = sys.argv[3]
measure_similarity_by_cosine(dataPath, testPath, testDocLabel)
# k, accuracies = measure_similarity_by_cosine(dataPath, testPath, testDocLabel)
# plt.suptitle("Dimension (k) vs Accuracy", fontsize = 18)
# plt.xlabel("k", fontsize = 16)
# plt.ylabel("accuracy", fontsize = 16)
# plt.plot(k, accuracies, color='m', label = "Accracy for different k")
# plt.savefig('KvsAccuracy_compressed.jpg', dpi = 150)