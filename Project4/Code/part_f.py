import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import help_functions


def part_f():
    k_value = 50
    n_iter = 200
    random_state = 42
    np.random.seed(304145309)

    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                  'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                  'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                  'talk.religion.misc']

    stop_words, NLTK_StopWords = help_functions.StopWords_extract()
    twenty_train, x_tfidf, _, _ = help_functions.TFIDF(categories, 'all', stop_words, NLTK_StopWords)

    n_samples, n_features = x_tfidf.shape
    n_clusters = 6

    labels = []
    for y in twenty_train.target:
        if y in [1, 2, 3, 4, 5]:
            labels.append(0)
        elif y in [7, 8, 9, 10]:
            labels.append(1)
        elif y in [11, 12, 13, 14]:
            labels.append(2)
        elif y == 6:
            labels.append(3)
        elif y in [16, 17, 18]:
            labels.append(4)
        elif y in [0, 15, 19]:
            labels.append(5)


    # Check sigma values
    sigma = help_functions.calculate_sigma(x_tfidf, 200)
    plt.figure(1)
    plt.title('Sigma Matrix Values')
    plt.scatter(range(len(sigma)), sigma, marker='o')
    plt.plot(range(len(sigma)), sigma)


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

    help_functions.bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10,
                                        max_iter=100, algorithm='full'),
                                 name="Original", data=x_tfidf, labels=labels)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA", data=x_tfidf_pca, labels=labels)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF", data=x_tfidf_nmf, labels=labels)

    x_tfidf_pca = help_functions.logrithm(x_tfidf_pca)
    x_tfidf_nmf = help_functions.logrithm(x_tfidf_nmf)

    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA_log", data=x_tfidf_pca, labels=labels)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF_log", data=x_tfidf_nmf, labels=labels)

    # print out normalized score for PCA,NMF
    x_tfidf_pca_norm = lsa_pca.fit_transform(x_tfidf)
    x_tfidf_nmf_norm = lsa_nmf.fit_transform(x_tfidf)

    print("n_cluster: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))

    print(79 * '_')
    print('% 9s' % 'init'
                   '        homo    compl     ARI    AMI')

    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA_norm", data=x_tfidf_pca_norm, labels=labels)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF_norm", data=x_tfidf_nmf_norm, labels=labels)

    x_tfidf_pca_norm = help_functions.logrithm(x_tfidf_pca_norm)
    x_tfidf_nmf_norm = help_functions.logrithm(x_tfidf_nmf_norm)

    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="PCA_norm_log", data=x_tfidf_pca_norm, labels=labels)
    help_functions.bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                                 name="NMF_norm_log", data=x_tfidf_nmf_norm, labels=labels)

    plt.show()


# _______________________________________________________________________________
# TFIDF matrix constructed
# ('The final number of terms are', 107218)
# ('The final number of docs are', 18846)
# _______________________________________________________________________________
# _______________________________________________________________________________
# Calculate Sigma Matrix
# [ 17.35063595   9.4278371    7.64959919   7.17296137   7.04231138
#    6.76292606   6.43028694   6.11513569   5.96453741   5.88659376
#    5.6421297    5.48570222   5.44646172   5.37794001   5.30952278
#    5.21257665   5.1205525    5.07819095   5.03043472   5.00545502
#    4.92514153   4.89911124   4.85634427   4.815382     4.77748542
#    4.75227929   4.72559092   4.67582643   4.66348251   4.6283682    4.626819
#    4.5712109    4.55725747   4.52773397   4.52164564   4.48799943
#    4.47249118   4.4615959    4.43733079   4.41006831   4.38868      4.36889676
#    4.36606045   4.35105921   4.33238937   4.31980823   4.30999765
#    4.29465689   4.26706569   4.2544063    4.24416265   4.22419764
#    4.21934336   4.2029136    4.18990526   4.18310533   4.1740628
#    4.1550498    4.13687763   4.12261837   4.10957635   4.10119665
#    4.09008251   4.07139883   4.05595033   4.04384821   4.03356001
#    4.02306554   4.00468433   3.99350904   3.98540062   3.97373087
#    3.96519658   3.94462542   3.93037639   3.92859525   3.90590418
#    3.9000378    3.88957876   3.88374959   3.85711105   3.84786687
#    3.84199926   3.83480122   3.81830954   3.81549954   3.80711137
#    3.80411741   3.78765708   3.78222285   3.77691912   3.76851571
#    3.75330509   3.75084896   3.74024417   3.73545526   3.72474837
#    3.72349392   3.71316809   3.70173401   3.69818544   3.68133147
#    3.66956327   3.66636523   3.65863043   3.65414461   3.64361971
#    3.63756982   3.63443018   3.6280901    3.62432636   3.62008659
#    3.61190861   3.60175712   3.59608267   3.59007576   3.58353622
#    3.58142223   3.57659483   3.57195851   3.55775357   3.55415815
#    3.54730067   3.54392099   3.53629859   3.53480482   3.52936893
#    3.52443732   3.51847378   3.51314371   3.50995036   3.50262713
#    3.49781737   3.49176745   3.48301522   3.47907434   3.47432625
#    3.47312713   3.46931456   3.46179047   3.4587936    3.45417419
#    3.44795508   3.44153482   3.43554007   3.42999799   3.425958
#    3.42406684   3.41883843   3.41390413   3.40759283   3.40543953
#    3.3990947    3.39495731   3.39257451   3.38993415   3.3830874
#    3.38168996   3.37778289   3.37391223   3.37034545   3.36373758
#    3.3598848    3.35386627   3.35062029   3.34400707   3.34239924
#    3.33870339   3.3340973    3.33140391   3.32585838   3.32079304
#    3.31670692   3.31445978   3.31007877   3.30829131   3.30171531
#    3.29828512   3.29363044   3.29025307   3.28661369   3.2812303
#    3.2788328    3.27407859   3.27142668   3.2642599    3.2625827
#    3.25586288   3.25137265   3.24977473   3.24718902   3.24296645
#    3.24066647   3.23961739   3.2341446    3.23017954   3.22625548
#    3.22312786   3.22029215   3.21635732]
# _______________________________________________________________________________
# Performing PCA
# Performing NMF
# n_cluster: 6, 	 n_samples 18846, 	 n_features 107218
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  Original   0.246   0.301   0.095   0.246
#       PCA   0.246   0.314   0.109   0.246
#       NMF   0.216   0.349   0.082   0.216
#   PCA_log   0.236   0.323   0.094   0.236
#   NMF_log   0.236   0.315   0.072   0.236
# n_cluster: 6, 	 n_samples 18846, 	 n_features 107218
# _______________________________________________________________________________
# init        homo    compl     ARI    AMI
#  PCA_norm   0.332   0.331   0.274   0.331
#  NMF_norm   0.297   0.299   0.236   0.297
# PCA_norm_log   0.347   0.347   0.244   0.347
# NMF_norm_log   0.244   0.248   0.166   0.244

