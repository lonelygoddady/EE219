from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import help_functions
import numpy as np

n_components = 50
n_iter = 200
random_state = 42
np.random.seed(304145309)

def part_c(data, X_tfidf):

	n_samples, n_features = X_tfidf.shape
	n_digits = 2
	labels = help_functions.data_labeling(data.target)
	# #performing PCA
	# svd = TruncatedSVD(n_components = n_components, n_iter = n_iter)
	normalizer = Normalizer(copy=True)
	# normalizer_pca = Normalizer(copy=False)
	# lsa_pca = make_pipeline(svd, normalizer_pca)
	# X_tfidf_pca = lsa_pca.fit_transform(X_tfidf)

	#performing NMF
	nmf = NMF(n_components = 2, random_state = random_state,
	          alpha = .01, l1_ratio = .2, max_iter = n_iter).fit(X_tfidf)
	lsa_nmf = make_pipeline(nmf, normalizer)
	X_tfidf_nmf = lsa_nmf.fit_transform(X_tfidf)

	# transformer = FunctionTransformer(np.log1p)
	# X_tfidf_nmf = transformer.transform(X_tfidf_nmf)

	print("n_digits: %d, \t n_samples %d, \t n_features %d"
	      % (n_digits, n_samples, n_features))

	print(79 * '_')
	print('% 9s' % 'init'
	      '        homo    compl     ARI    AMI')
	# part_b.bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
 #              name="PCA", data=X_tfidf_pca, labels = labels)
	help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="NMF", data=X_tfidf_nmf, labels = labels)
	
	# # apply kernel ridge
	# kernel_pca = KernelRidge(kernel = "poly", alpha = 1.0)
	# kernel_nmf = KernelRidge(kernel = "poly", alpha = 1.0)
	# _, X_tfidf_pca = kernel_pca.fit(X_tfidf_pca, labels)
	# _, X_tfidf_nmf =kernel_nmf.fit(X_tfidf_nmf, labels)

	# part_b.bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
 #              name="PCA", data=X_tfidf_pca, labels = labels)
	# part_b.bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
 #              name="NMF", data=X_tfidf_nmf, labels = labels)



	# print(79 * '_')

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
	kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
	kmeans.fit(X_tfidf_nmf)
	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = X_tfidf_nmf[:, 0].min() - 1, X_tfidf_nmf[:, 0].max() + 1
	y_min, y_max = X_tfidf_nmf[:, 1].min() - 1, X_tfidf_nmf[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	# # Put the result into a color plot
	# Z = Z.reshape(xx.shape)
	plt.figure(1)
	# plt.clf()
	# plt.imshow(Z, interpolation='nearest',
	#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	#            cmap=plt.cm.Paired,
	#            aspect='auto', origin='lower')

	# plt.plot(X_tfidf_nmf[:, 0], X_tfidf_nmf[:, 1], 'k.', markersize=2)
	plt.scatter(X_tfidf_nmf[:, 0], X_tfidf_nmf[:, 1])

	# # Plot the centroids as a white X
	# centroids = kmeans.cluster_centers_
	# plt.scatter(centroids[:, 0], centroids[:, 1],
	#             marker='x', s=169, linewidths=3,
	#             color='w', zorder=10)
	plt.title('2D plot (NMF-reduced data)\n')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()
# n_digits: 2,     n_samples 7882,         n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#       PCA   0.177   0.538   0.151   0.177
#       NMF   0.176   0.536   0.150   0.176
# _______________________________________________________________________________
# k=50
# n_digits: 2,     n_samples 7882,         n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#       PCA   0.176   0.532   0.154   0.176
#       NMF   0.177   0.536   0.154   0.177
# 		NMF   0.180   0.539   0.159   0.179
# _______________________________________________________________________________