from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import help_functions
import numpy as np

k_value = 30
n_iter = 200
random_state = 42
np.random.seed(304145309)


def part_c(data, X_tfidf):
    n_samples, n_features = X_tfidf.shape
    n_clusters = 2
    labels = help_functions.data_labeling(data.target)

    #Check sigma values
    # sigma = help_functions.calculate_sigma(X_tfidf,200)
    # plt.figure(1)
    # plt.title('Sigma Matrix Values')
    # plt.scatter(range(len(sigma)), sigma, marker='o')
    # plt.plot(range(len(sigma)), sigma)
    # plt.show()

    # # performing PCA
    # print("Performing PCA")
    # svd = TruncatedSVD(n_components=k_value, n_iter=n_iter)
    # normalizer_nmf = Normalizer(copy=True)
    # normalizer_pca = Normalizer(copy=True)
    # lsa_pca = make_pipeline(svd, normalizer_pca)
    # X_tfidf_pca = svd.fit_transform(X_tfidf)
    #
    # # performing NMF
    # print("Performing NMF")
    # nmf = NMF(n_components=k_value, random_state=random_state,
    #           alpha=.01, l1_ratio=0, max_iter=n_iter).fit(X_tfidf)
    # lsa_nmf = make_pipeline(nmf, normalizer_nmf)
    # X_tfidf_nmf = nmf.fit_transform(X_tfidf)
    #
    # #print out unnormalized scores for PCA,NMF
    # # print("n_clusters: %d, \t n_samples %d, \t n_features %d"
    # #       % (n_clusters, n_samples, n_features))
    # #
    # # print(79 * '_')
    # # print('% 9s' % 'init'
    # #                '        homo    compl     ARI    AMI')
    # #
    # # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    # #                              name="Original", data=X_tfidf, labels=labels)
    # # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    # #                              name="PCA", data=X_tfidf_pca, labels=labels)
    # # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    # #                              name="NMF", data=X_tfidf_nmf, labels=labels)
    #
    # X_tfidf_pca = help_functions.logrithm(X_tfidf_pca,
    #                                       np.mean(X_tfidf_pca))
    # X_tfidf_nmf = help_functions.logrithm(X_tfidf_nmf,
    #                                       np.mean(X_tfidf_pca))
    #
    # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #                              name="PCA_log", data=X_tfidf_pca, labels=labels)
    # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #                              name="NMF_log", data=X_tfidf_nmf, labels=labels)
    #
    # # print out normalized score for PCA,NMF
    # X_tfidf_pca_norm = lsa_pca.fit_transform(X_tfidf)
    # X_tfidf_nmf_norm = lsa_nmf.fit_transform(X_tfidf)
    #
    # print("n_clusters: %d, \t n_samples %d, \t n_features %d"
    #       % (n_clusters, n_samples, n_features))
    #
    # print(79 * '_')
    # print('% 9s' % 'init'
    #                '        homo    compl     ARI    AMI')
    #
    # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #                              name="PCA_norm", data=X_tfidf_pca_norm, labels=labels)
    # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #                              name="NMF_norm", data=X_tfidf_nmf_norm, labels=labels)
    #
    # X_tfidf_pca_norm = help_functions.logrithm(X_tfidf_pca_norm,
    #                                           10)
    # X_tfidf_nmf_norm = help_functions.logrithm(X_tfidf_nmf_norm,
    #                                            10)
    # #
    # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #                              name="PCA_norm_log", data=X_tfidf_pca_norm, labels=labels)
    # help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #                              name="NMF_norm_log", data=X_tfidf_nmf_norm, labels=labels)

    nmf_2 = NMF(n_components=2, random_state=random_state,
              alpha=.01, l1_ratio=0, max_iter=n_iter).fit(X_tfidf)
    # lsa_nmf = make_pipeline(nmf, normalizer_nmf)
    X_tfidf_nmf_2 = nmf_2.fit_transform(X_tfidf)
    log_base = np.max(X_tfidf_nmf_2) - np.min(X_tfidf_nmf_2)
    X_tfidf_nmf_log = help_functions.logrithm(X_tfidf_nmf_2, log_base)
    plt.figure(2)
    plt.title('NMF_log embedding of the data')
    plt.scatter(X_tfidf_nmf_log[:,0], X_tfidf_nmf_log[:,1], marker='o')
    plt.show()

# k value is 30
# n_clusters: 2, 	 n_samples 7882, 	 n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  Original   0.413   0.450   0.423   0.413
#       PCA   0.430   0.462   0.452   0.430
#       NMF   0.454   0.476   0.503   0.454
#   PCA_log   0.412   0.449   0.424   0.412
#   NMF_log   0.454   0.476   0.503   0.454

# After Normalization
# n_clusters: 2, 	 n_samples 7882, 	 n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  PCA_norm   0.511   0.512   0.613   0.511
#  NMF_norm   0.502   0.504   0.603   0.502
# PCA_norm_log   0.511   0.512   0.613   0.511
# NMF_norm_log   0.504   0.506   0.604   0.504