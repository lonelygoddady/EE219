from sklearn import svm
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import GaussianNB
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
    categories = ['comp.sys.ibm.pc.hardware' , 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

    stop_words, NLTK_StopWords = StopWords_extract()    # Extract stop words
    twenty_train, X_train_tfidf, X_train_counts, train_count_vect = TFIDF(categories,'train',stop_words,NLTK_StopWords)
    twenty_test, X_test_tfidf, X_test_counts = TFIDF_transform(categories,'test',stop_words,NLTK_StopWords, train_count_vect)

    # print X_train_tfidf.shape
    # print X_test_tfidf.shape

    svd = TruncatedSVD(n_components=50, random_state=42, algorithm='arpack')    # default algorithm will cause segfault
    X_train_LSI = svd.fit_transform(X_train_tfidf)  # fit LSI using X_train_tfidf and perform dimension reduction
    X_train_LSI = Normalizer(copy=False).fit_transform(X_train_LSI)
    X_test_LSI = svd.transform(X_test_tfidf)
    X_test_LSI = Normalizer(copy=False).fit_transform(X_test_LSI)

    clf_OVO = svm.SVC(kernel='linear', probability=True, decision_function_shape = 'ovo')
    OVO_model = clf_OVO.fit(X_train_LSI, twenty_train.target) # train the data over high-dimension data and known Y values

    clf_OVR = svm.SVC(kernel='linear', probability=True, decision_function_shape = 'ovr')
    OVR_model = clf_OVR.fit(X_train_LSI, twenty_train.target) # train the data over high-dimension data and known Y values    

    
    clf_NB = GaussianNB()
    NB_model = clf_NB.fit(X_train_LSI, twenty_train.target) # train the data over high-dimension data and known Y values 

    # predict the value with the learning algorithm
    Y_predict_ovo = OVO_model.predict(X_test_LSI) 
    Y_predict_ovr = OVR_model.predict(X_test_LSI)
    Y_predict_nb = NB_model.predict(X_test_LSI)
    Y_predict = [Y_predict_ovo, Y_predict_ovr, Y_predict_nb]

    # Calculating the accuracy, recall, precision and confusion matrix
    accuracy = np.zeros(3)
    for i in range(len(Y_predict)):
        accuracy[i] = np.mean(twenty_test.target == Y_predict[i])

    model_name = ['SVM_OVO ', 'SVM_OVR ', 'NB ']


    for i in range(len(accuracy)):
        s = 'The accuracy for the ' + model_name[i] + 'model is ' + str(accuracy[i])
        print(s)
        print ('\'0\' is comp.sys.ibm.pc.hardware, \'1\' is comp.sys.mac.hardware, \'2\' is misc.forsale and \'3\' is soc.religion.christian')
        print ("The precision and recall values are:")
        print (metrics.classification_report(twenty_test.target, Y_predict[i]))
        print ('The confusion matrix is as shown below:')
        print (metrics.confusion_matrix(twenty_test.target, Y_predict[i]))
        print("\n")

