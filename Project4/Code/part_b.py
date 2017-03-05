from __future__ import print_function
import help_functions
from sklearn.cluster import KMeans
from sklearn import metrics


def part_b(data, X_tfidf):

	n_samples, n_features = X_tfidf.shape
	n_digits = 2
	labels = help_functions.data_labeling(data.target)
	# print(labels)

	print("n_digits: %d, \t n_samples %d, \t n_features %d"
	      % (n_digits, n_samples, n_features))

	print(79 * '_')
	print('% 9s' % 'init'
	      '        homo    compl     ARI    AMI')
	help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, 
		n_init=10, max_iter = 200, random_state = 42, tol=1e-5),
              name="k-means++", data=X_tfidf, labels = labels)

	help_functions.bench_k_means(KMeans(init='random', n_clusters=n_digits, 
		n_init=10, max_iter = 200, random_state = 42, tol=1e-5),
	              name="random", data=X_tfidf, labels = labels)

	print(79 * '_')


# 	n_digits: 2,     n_samples 7882,         n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
# k-means++   0.421   0.455   0.440   0.421
#    random   0.456   0.482   0.492   0.455
# _______________________________________________________________________________