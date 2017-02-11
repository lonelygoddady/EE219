from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
from IPython.display import Image

import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in('ivychen', 'Gol0AP8lJ2GVI7FTaeI7')
import os
import numpy as np

#import TFIDF, LSI function from part b and d
from part_b import TFIDF, preprocess, StopWords_extract
# from part_d import LSI

def Y_data_labeling(Y_data):
	Y_data_after = []
	for y in Y_data:
		if y <= 3:
			Y_data_after.append(0)
		else:
			Y_data_after.append(1)
	return Y_data_after

def target_row_extract(twenty_subset, twenty_all):
	target_file_index = []
	for idx, val in enumerate(twenty_all.target):
	 	if val in twenty_subset.target:
	 		target_file_index.append(idx) #extract row index of our interested documents 
	return target_file_index

def target_tfidf_matrix_build(Target,Origin,row_index):
	for row in range(len(row_index)):
		Target[row,] += Origin[row_index[row],] #copy interested rows to the target matrix

if __name__ == "__main__":
	categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
	'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

	stop_words, NLTK_StopWords = StopWords_extract() #Extract stop words
	twenty_train, X_train_tfidf, X_train_counts, train_count_vect = TFIDF(categories,'train',stop_words,NLTK_StopWords)
	twenty_test, X_test_tfidf, X_test_counts, test_count_vect = TFIDF(categories,'test',stop_words,NLTK_StopWords)

	print X_train_tfidf.shape
	print X_test_tfidf.shape 
	
	svd = TruncatedSVD(n_components=50, random_state=42, algorithm='arpack') #default algorithm will cause segfault
	X_train_LSI = svd.fit_transform(X_train_tfidf) #fit LSI using X_train_tfidf and perform dimension reduction 
	X_test_LSI = svd.transform(X_test_tfidf)

	print X_test_LSI
	print X_train_LSI
	# Y_test = Y_data_labeling(twenty_test.target) #Category values for verification
	# Y_train = Y_data_labeling(twenty_train.target) #Category values for trainning

	# clf = svm.SVC(kernel='linear', probability=True)
	# clf.fit(X_train, Y_train) #train the data over high-dimension data and known Y values
	
	# Y_predict = clf.predict(X_test) #predict the value with the learning algorithm
	
	# #now we need extract those values in our target categories

	# #check prediction results 
	# Recall = []
	# for x in range(len(Y_train)):
	# 	if Y_predict[x] is not Y_test[x]:
	# 		Recall.append(-1)
	# 	else:
	# 		Recall.append(1)

	# print Recall.count(1)
	# print Recall.count(-1) 

	# # print Y_predict 
	# # #for plotting
	# # if not os.path.exists('../Graphs/part_E'):
	# # 	os.makedirs('../Graphs/part_E')