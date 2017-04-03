import numpy as np
import help_functions
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF


def part_d(data, X_tfidf):
    svd = TruncatedSVD(n_components=2, n_iter=200)
    x_tfidf_pca = svd.fit_transform(X_tfidf)

    kmeans_pca = KMeans(init='k-means++', n_clusters=2)
    kmeans_pca.fit(x_tfidf_pca)
    km_labels = kmeans_pca.labels_

    plt.figure(1)
    plt.title("K mean clustering result for PCA reduced data")
    for i in range(len(x_tfidf_pca[:, 0])):
        if km_labels[i] == 0:
            plt.scatter(x_tfidf_pca[i, 0], x_tfidf_pca[i, 1], c='r',
                        marker='x')
        elif km_labels[i] == 1:
            plt.scatter(x_tfidf_pca[i, 0], x_tfidf_pca[i, 1], c='b',
                        marker='o')

    nmf = NMF(n_components=2, random_state=42,
              alpha=.01, l1_ratio=0, max_iter=200).fit(X_tfidf)
    x_tfidf_nmf = nmf.fit_transform(X_tfidf)

    kmeans_nmf = KMeans(init='k-means++', n_clusters=2)
    kmeans_nmf.fit(x_tfidf_nmf)
    km_labels = kmeans_nmf.labels_

    plt.figure(2)
    plt.title("K mean clustering result for NMF reduced data")
    for i in range(len(x_tfidf_nmf[:, 0])):
        if km_labels[i] == 0:
            plt.scatter(x_tfidf_nmf[i, 0], x_tfidf_nmf[i, 1], c='r',
                        marker='x')
        elif km_labels[i] == 1:
            plt.scatter(x_tfidf_nmf[i, 0], x_tfidf_nmf[i, 1], c='b',
                        marker='o')

    x_tfidf_nmf_log = help_functions.logrithm(x_tfidf_nmf)
    kmeans_nmf_log = KMeans(init='k-means++', n_clusters=2)
    kmeans_nmf_log.fit(x_tfidf_nmf_log)
    km_labels_log = kmeans_nmf_log.labels_

    plt.figure(3)
    plt.title("K mean clustering result for NMF log reduced data")
    for i in range(len(x_tfidf_nmf_log[:, 0])):
        if km_labels_log[i] == 0:
            plt.scatter(x_tfidf_nmf_log[i, 0], x_tfidf_nmf_log[i, 1], c='r',
                        marker='x')
        elif km_labels_log[i] == 1:
            plt.scatter(x_tfidf_nmf_log[i, 0], x_tfidf_nmf_log[i, 1], c='b',
                        marker='o')
    plt.show()