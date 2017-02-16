from sklearn import svm
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer

from matplotlib import pyplot as plt
import numpy as np

# import TFIDF, LSI function from part b and d
from part_b import TFIDF, preprocess, StopWords_extract
# from part_d import LSI

def Y_data_labeling(Y_data):
    for y in range(len(Y_data)):
        if Y_data[y] <= 3:      # for Computer Technology category
            Y_data[y] = 0
        else:                   # for Recreation Activity
            Y_data[y] = 1

# this is the TFIDF function variant which reuse the fitted count_vect instead of refitting it


def TFIDF_transform(categories,train_or_test,stop_words,NLTK_StopWords,count_vect):
    twenty_data = fetch_20newsgroups(subset=train_or_test,categories=categories, remove=('headers','footers','quotes'))

    # Stores the size of the dataset
    size, = twenty_data.filenames.shape

    # Performing preprocessing on every document
    for item in range(0,size):
        # print twenty_data.filenames[item]
        sentence = twenty_data.data[item]
        twenty_data.data[item] = preprocess(sentence, stop_words, NLTK_StopWords)

    # Transform instead of fit_transform!!!
    X_counts = count_vect.transform(twenty_data.data)

    # Calculating the TF-IDF values for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_counts)
    X_tfidf = tf_transformer.transform(X_counts)
    # docs,terms = X_tfidf.shape
    return twenty_data, X_tfidf, X_counts

if __name__ == "__main__":
    categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    stop_words, NLTK_StopWords = StopWords_extract()    # Extract stop words
    twenty_train, X_train_tfidf, X_train_counts, train_count_vect = TFIDF(categories,'train',stop_words,NLTK_StopWords)
    twenty_test, X_test_tfidf, X_test_counts = TFIDF_transform(categories,'test',stop_words,NLTK_StopWords, train_count_vect)

    print X_train_tfidf.shape
    print X_test_tfidf.shape

    svd = TruncatedSVD(n_components=50, random_state=42, algorithm='arpack')    # default algorithm will cause segfault
    X_train_LSI = svd.fit_transform(X_train_tfidf)  # fit LSI using X_train_tfidf and perform dimension reduction
    X_train_LSI = Normalizer(copy=False).fit_transform(X_train_LSI)
    X_test_LSI = svd.transform(X_test_tfidf)
    X_test_LSI = Normalizer(copy=False).fit_transform(X_test_LSI)

    Y_data_labeling(twenty_test.target) # Category values for verification
    Y_data_labeling(twenty_train.target) # Category values for trainning

    clf = svm.SVC(kernel='linear', probability=True)
    LSI_model = clf.fit(X_train_LSI, twenty_train.target) # train the data over high-dimension data and known Y values

    Y_predict = LSI_model.predict(X_test_LSI) # predict the value with the learning algorithm

    # Calculating the accuracy, recall, precision and confusion matrix
    accuracy_svm = np.mean(twenty_test.target == Y_predict)
    print 'The accuracy for the model is %f' % accuracy_svm
    print '\'0\' is Computer Technology and \'1\' is Recreational Activity'
    print "The precision and recall values are:"
    print metrics.classification_report(twenty_test.target, Y_predict)
    print 'The confusion matrix is as shown below:'
    print metrics.confusion_matrix(twenty_test.target, Y_predict)

    # Plotting the ROC
    probas_ = LSI_model.predict_proba(X_test_LSI)
    fpr, tpr, thresholds = metrics.roc_curve(twenty_test.target, probas_[:, 1])
    plt.plot(fpr, tpr, lw=1, label="SVM ROC")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.show()
