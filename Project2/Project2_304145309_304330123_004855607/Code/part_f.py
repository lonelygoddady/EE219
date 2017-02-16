from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import cross_val_score, cross_val_predict

# import TFIDF, LSI function from part b and d
from part_b import TFIDF, preprocess, StopWords_extract
from part_e import Y_data_labeling, TFIDF_transform




if __name__ == "__main__":
    categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    stop_words, NLTK_StopWords = StopWords_extract()
    twenty_train, X_train_tfidf, X_train_counts, train_count_vect = TFIDF(categories, 'train', stop_words, NLTK_StopWords)
    twenty_test, X_test_tfidf, X_test_counts = TFIDF_transform(categories, 'test', stop_words, NLTK_StopWords,
                                                               train_count_vect)
    # # Performing dimensionality reduction on the TF-IDF matrix
    # svd = TruncatedSVD(n_components=50, random_state=42, algorithm='arpack')
    # X_all = svd.fit_transform(X_all_tfidf)
    # X_all = Normalizer(copy=False).fit_transform(X_all)
    # Y_all = twenty_all.target
    # Y_data_labeling(Y_all)

    svd = TruncatedSVD(n_components=50, random_state=42, algorithm='arpack')    # default algorithm will cause segfault
    X_train_LSI = svd.fit_transform(X_train_tfidf)  # fit LSI using X_train_tfidf and perform dimension reduction
    X_train_LSI = Normalizer(copy=False).fit_transform(X_train_LSI)
    X_test_LSI = svd.transform(X_test_tfidf)
    X_test_LSI = Normalizer(copy=False).fit_transform(X_test_LSI)

    Y_data_labeling(twenty_test.target) # Category values for verification
    Y_data_labeling(twenty_train.target) # Category values for trainning


    # Varying the values of Gamma
    gamma_arr = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    if not os.path.exists('../Graphs/part_F'):
        os.makedirs('../Graphs/part_F')
    highestAccuracy = 0
    highestAccuracyIndex = 0
    highestClf = None
    for i in range(0, len(gamma_arr)):
        clf = svm.SVC(kernel='linear', probability=True, C=gamma_arr[i])
        clf.fit(X_train_LSI, twenty_train.target)
        Y_predict = cross_val_predict(clf, X_test_LSI, twenty_test.target, cv=5)
        print "\n***************************************************************************\n"
        accuracy_svm = np.mean(twenty_test.target == Y_predict)
        print 'The accuracy for the model under gamma {0} is {1}'.format(gamma_arr[i], accuracy_svm)
        print '\'0\' is Computer Technology and \'1\' is Recreational Activity'
        print "The precision and recall values are:"
        print metrics.classification_report(twenty_test.target, Y_predict)
        print 'The confusion matrix is as shown below:'
        print metrics.confusion_matrix(twenty_test.target, Y_predict)
        if highestAccuracy < accuracy_svm:
            highestAccuracy = accuracy_svm
            highestAccuracyIndex = i
            highestClf = clf

    # # Make the Pool of workers
    # pool = ThreadPool(4)
    # # Open the urls in their own threads
    # # and return the results
    # func = partial(gamma_vary, cv, X_all, Y_all)
    # results = pool.map(func, gamma_arr)
    # # close the pool and wait for the work to finish
    # pool.close()
    # pool.join()
    print '\n\n\nAll DONE!!!\nThe highest Accuracy is {0} at gamma {1}'.format(highestAccuracy,
                                                                               gamma_arr[highestAccuracyIndex])
    probas_ = highestClf.predict_proba(X_test_LSI)
    fpr, tpr, thresholds = metrics.roc_curve(twenty_test.target, probas_[:, 1])
    plt.plot(fpr, tpr, lw=1, label="SVM ROC")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.savefig('../Graphs/part_F/ROC.png')
    plt.show()




