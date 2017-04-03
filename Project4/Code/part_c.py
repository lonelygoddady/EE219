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


def part_c(data, x_tfidf):
    k_value = 30
    n_iter = 200
    random_state = 42
    np.random.seed(304145309)

    n_samples, n_features = x_tfidf.shape
    n_clusters = 2
    labels = help_functions.data_labeling(data.target)

    # Check sigma values
    # sigma = help_functions.calculate_sigma(x_tfidf,200)
    # plt.figure(1)
    # plt.title('Sigma Matrix Values')
    # plt.scatter(range(len(sigma)), sigma, marker='o')
    # plt.plot(range(len(sigma)), sigma)
    # plt.show()

    # performing PCA
    print("Performing PCA")
    svd = TruncatedSVD(n_components=k_value, n_iter=n_iter)
    normalizer_nmf = Normalizer(copy=True)
    normalizer_pca = Normalizer(copy=True)
    lsa_pca = make_pipeline(svd, normalizer_pca)
    x_tfidf_pca_origin = svd.fit_transform(x_tfidf)

    # performing NMF
    print("Performing NMF")
    nmf = NMF(n_components=k_value, random_state=random_state,
              alpha=.01, l1_ratio=0, max_iter=n_iter)
    lsa_nmf = make_pipeline(nmf, normalizer_nmf)
    x_tfidf_nmf_origin = nmf.fit_transform(x_tfidf)

    # print out unnormalized scores for PCA,NMF
    print("n_clusters: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="Original", data=x_tfidf, labels=labels)
    PCA_est = KMeans(init='random', n_clusters=n_clusters, n_init=10)
    help_functions.bench_k_means(PCA_est,
                                 name="PCA", data=x_tfidf_pca_origin, labels=labels)
    NMF_est = KMeans(init='random', n_clusters=n_clusters, n_init=10)
    help_functions.bench_k_means(NMF_est,
                                 name="NMF", data=x_tfidf_nmf_origin, labels=labels)

    # confusion matrix for PCA and NMF
    help_functions.confusion_matrix_build(PCA_est.labels_, labels)
    help_functions.confusion_matrix_build(NMF_est.labels_, labels)

    x_tfidf_pca = help_functions.logrithm(x_tfidf_pca_origin)
    x_tfidf_nmf = help_functions.logrithm(x_tfidf_nmf_origin)

    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="PCA_log", data=x_tfidf_pca, labels=labels)
    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="NMF_log", data=x_tfidf_nmf, labels=labels)

    # print out normalized score for PCA,NMF
    x_tfidf_pca_norm = lsa_pca.fit_transform(x_tfidf)
    x_tfidf_nmf_norm = lsa_nmf.fit_transform(x_tfidf)

    print("n_clusters: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="PCA_norm", data=x_tfidf_pca_norm, labels=labels)
    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="NMF_norm", data=x_tfidf_nmf_norm, labels=labels)

    x_tfidf_pca_norm = help_functions.logrithm(x_tfidf_pca_norm)
    x_tfidf_nmf_norm = help_functions.logrithm(x_tfidf_nmf_norm)

    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="PCA_norm_log", data=x_tfidf_pca_norm, labels=labels)
    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                                 name="NMF_norm_log", data=x_tfidf_nmf_norm, labels=labels)

    ############ Graph for log and non-log NMF ###############
    nmf_2 = NMF(n_components=2, random_state=random_state,
              alpha=.01, l1_ratio=0, max_iter=n_iter).fit(x_tfidf)
    x_tfidf_nmf_2 = nmf_2.fit_transform(x_tfidf)
    plt.figure(1)
    plt.title('NMF embedding of the data')
    plt.scatter(x_tfidf_nmf_2[:, 0], x_tfidf_nmf_2[:, 1], marker='o')

    x_tfidf_nmf_log = help_functions.logrithm(x_tfidf_nmf_2)
    plt.figure(2)
    plt.title('NMF_log embedding of the data')
    plt.scatter(x_tfidf_nmf_log[:,0], x_tfidf_nmf_log[:,1], marker='o')

    # Graph for log and non-log SVD
    svd_2 = TruncatedSVD(n_components=2, n_iter=n_iter)
    x_tfidf_svd_2 = svd_2.fit_transform(x_tfidf)
    plt.figure(3)
    plt.title('pca embedding of the data')
    plt.scatter(x_tfidf_svd_2[:, 0], x_tfidf_svd_2[:, 1], marker='o')

    x_tfidf_svd_log = help_functions.logrithm(x_tfidf_svd_2)
    plt.figure(4)
    plt.title('pca_log embedding of the data')
    plt.scatter(x_tfidf_svd_log[:,0], x_tfidf_svd_log[:,1], marker='o')
    plt.show()


# k value is 30
# n_clusters: 2, 	 n_samples 7882, 	 n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  Original   0.416   0.451   0.429   0.415
#       PCA   0.430   0.462   0.452   0.430
#       NMF   0.459   0.479   0.511   0.458
#   PCA_log   0.395   0.440   0.386   0.394
#   NMF_log   0.473   0.491   0.531   0.473
# n_clusters: 2, 	 n_samples 7882, 	 n_features 67764
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  PCA_norm   0.511   0.512   0.613   0.511
#  NMF_norm   0.504   0.505   0.604   0.504
# PCA_norm_log   0.518   0.518   0.623   0.518
# NMF_norm_log   0.504   0.506   0.606   0.504
