import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

n_components = 50
n_iter = 200
random_state = 42
np.random.seed(304145309)

def part_d(data, X_tfidf):
  n_samples, n_features = X_tfidf.shape
  n_digits = 2
  labels = data.target

  X = X_tfidf
  y = data.target
  svd = TruncatedSVD(n_components = n_components, n_iter = n_iter)
  normalizer = Normalizer(copy=True)
  normalizer_pca = Normalizer(copy=False)
  lsa_pca = make_pipeline(svd, normalizer_pca)
  X_tfidf_pca = lsa_pca.fit_transform(X_tfidf)
  print(X_tfidf_pca.shape)


# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(X_tfidf_pca)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = X_tfidf_pca[:, 0].min() - 1, X_tfidf_pca[:, 0].max() + 1
# y_min, y_max = X_tfidf_pca[:, 1].min() - 1, X_tfidf_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()





#   h = .02  # step size in the mesh

#   #performing PCA
#   svd = TruncatedSVD(n_components = n_components, n_iter = n_iter)
#   normalizer = Normalizer(copy=True)
#   normalizer_pca = Normalizer(copy=False)
#   lsa_pca = make_pipeline(svd, normalizer_pca)
#   X_tfidf_pca = lsa_pca.fit_transform(X_tfidf)

#   #performing NMF
#   nmf = NMF(n_components = n_components, random_state = random_state,
#             alpha = .01, l1_ratio = .2, max_iter = n_iter).fit(X_tfidf)
#   lsa_nmf = make_pipeline(nmf, normalizer)
#   X_tfidf_nmf = lsa_pca.fit_transform(X_tfidf)

#   # create a mesh to plot in
#   x_min, x_max = X_tfidf[:, 0].min() - 1, X_tfidf[:, 0].max() + 1
#   y_min, y_max = target[:, 1].min() - 1, X_tfidf_nmf[:, 1].max() + 1
#   xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                        np.arange(y_min, y_max, h))

#   # title for the plots
#   titles = ['PCA',
#             'NMF']


#   for i, clf in enumerate((pca, nmf)):
#      # Obtain labels for each point in mesh. Use last trained model.
#     Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure(1)
#     plt.clf()
#     plt.imshow(Z, interpolation='nearest',
#                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#                cmap=plt.cm.Paired,
#                aspect='auto', origin='lower')

#     plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
#     # Plot the centroids as a white X
#     centroids = kmeans.cluster_centers_
#     plt.scatter(centroids[:, 0], centroids[:, 1],
#                 marker='x', s=169, linewidths=3,
#                 color='w', zorder=10)
#     plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#               'Centroids are marked with white cross')
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#     plt.xticks(())
#     plt.yticks(())
#   plt.show()