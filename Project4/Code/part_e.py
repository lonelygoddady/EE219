from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import help_functions
import numpy as np
from scipy.cluster.vq import kmeans

def part_e():
    k_value = 50
    n_iter = 200
    random_state = 42
    np.random.seed(304145309)
    print("Starting part E")
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                  'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                  'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                  'talk.religion.misc']

    stop_words, NLTK_StopWords = help_functions.StopWords_extract()
    twenty_train, x_tfidf, _, _ = help_functions.TFIDF(categories, 'all', stop_words, NLTK_StopWords)

    n_samples, n_features = x_tfidf.shape
    n_clusters = len(categories)

    # Check sigma values
    # sigma = help_functions.calculate_sigma(x_tfidf, 200)
    # plt.figure(1)
    # plt.title('Sigma Matrix Values')
    # plt.scatter(range(len(sigma)), sigma, marker='o')
    # plt.plot(range(len(sigma)), sigma)
    # plt.show()

    # performing PCA
    print("Performing PCA")
    svd = TruncatedSVD(n_components=k_value, n_iter=n_iter, algorithm='arpack')
    normalizer_nmf = Normalizer(copy=True)
    normalizer_pca = Normalizer(copy=True)
    lsa_pca = make_pipeline(svd, normalizer_pca)
    x_tfidf_pca = svd.fit_transform(x_tfidf)

    # performing NMF
    print("Performing NMF")
    nmf = NMF(n_components=k_value, random_state=random_state,
              alpha=.01, max_iter=n_iter, init='nndsvda', solver='cd').fit(x_tfidf)
    lsa_nmf = make_pipeline(nmf, normalizer_nmf)
    x_tfidf_nmf = nmf.fit_transform(x_tfidf)

    # print out unnormalized scores for PCA,NMF
    print("n_cluster: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=50,
                                        max_iter=100, algorithm='full'),
                                 name="Original", data=x_tfidf, labels=twenty_train.target)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="PCA", data=x_tfidf_pca, labels=twenty_train.target)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="NMF", data=x_tfidf_nmf, labels=twenty_train.target)

    x_tfidf_pca = help_functions.logrithm(x_tfidf_pca)
    x_tfidf_nmf = help_functions.logrithm(x_tfidf_nmf)

    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="PCA_log", data=x_tfidf_pca, labels=twenty_train.target)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="NMF_log", data=x_tfidf_nmf, labels=twenty_train.target)

    # print out normalized score for PCA,NMF
    x_tfidf_pca_norm = lsa_pca.fit_transform(x_tfidf)
    x_tfidf_nmf_norm = lsa_nmf.fit_transform(x_tfidf)

    print("n_cluster: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="PCA_norm", data=x_tfidf_pca_norm, labels=twenty_train.target)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="NMF_norm", data=x_tfidf_nmf_norm, labels=twenty_train.target)

    x_tfidf_pca_norm = help_functions.logrithm(x_tfidf_pca_norm)
    x_tfidf_nmf_norm = help_functions.logrithm(x_tfidf_nmf_norm)

    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="PCA_norm_log", data=x_tfidf_pca_norm, labels=twenty_train.target)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=50),
                                 name="NMF_norm_log", data=x_tfidf_nmf_norm, labels=twenty_train.target)


# Starting part E
# _______________________________________________________________________________
# TFIDF matrix constructed
# ('The final number of terms are', 107218)
# ('The final number of docs are', 18846)
# _______________________________________________________________________________
# Performing PCA
# Performing NMF
# n_cluster: 20, 	 n_samples 18846, 	 n_features 107218
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  Original   0.310   0.395   0.079   0.308
#       PCA   0.302   0.373   0.075   0.300
#       NMF   0.276   0.373   0.057   0.274
#   PCA_log   0.298   0.361   0.077   0.296
#   NMF_log   0.281   0.406   0.045   0.279
# n_cluster: 20, 	 n_samples 18846, 	 n_features 107218
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  PCA_norm   0.361   0.375   0.211   0.359
#  NMF_norm   0.335   0.345   0.186   0.333
# PCA_norm_log   0.360   0.372   0.215   0.358
# NMF_norm_log   0.307   0.313   0.173   0.305
