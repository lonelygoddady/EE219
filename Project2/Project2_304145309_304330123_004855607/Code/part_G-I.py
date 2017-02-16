from sklearn import svm
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from IPython.display import Image
import numpy as np
import os

# import TFIDF, LSI function from part b and d
from part_b import TFIDF, preprocess, StopWords_extract
# from part_d import LSI


def Y_data_labeling(Y_data):
    for y in range(len(Y_data)):
        if Y_data[y] <= 3:
            Y_data[y] = 0
        else:
            Y_data[y] = 1

# this is the TFIDF function variant which reuse the fitted count_vect instead of refitting it


def TFIDF_transform(categories,train_or_test,stop_words,NLTK_StopWords,count_vect):
    twenty_data = fetch_20newsgroups(subset=train_or_test,categories=categories, remove=('headers','footers','quotes'))

    # Stores the size of the dataset
    size, = twenty_data.filenames.shape

    # Performing preprocessing on every document
    for item in range(0,size):
        # print (twenty_data.filenames[item])
        sentence = twenty_data.data[item]
        twenty_data.data[item] = preprocess(sentence, stop_words, NLTK_StopWords)

    # Transform instead of fit_transform!!!
    X_counts = count_vect.transform(twenty_data.data)

    # Calculating the TF-IDF values for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_counts)
    X_tfidf = tf_transformer.transform(X_counts)
    docs,terms = X_tfidf.shape
    return twenty_data, X_tfidf, X_counts

def infor_print(accuracy, X_test, Y_predict):
    print ('The accuracy for the multinomial naive bayes model is %f' % accuracy)
    print ('\'0\' is Computer Technology and \'1\' is Recreational Activity')
    print ("The precision and recall values are:")
    print (metrics.classification_report(X_test.target, Y_predict))
    print ('The confusion matrix is as shown below:')
    print (metrics.confusion_matrix(X_test.target, Y_predict))

def roc_print(dirLoc, probas_, fpr, tpr, thresholds, Label):
    if not os.path.exists(dirLoc):
        os.makedirs(dirLoc)
    plt.figure(1)
    plt.plot(fpr, tpr, lw=1, label = Label)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    s = dirLoc+ '/ROC'
    plt.savefig(s)


if __name__ == "__main__":
    categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    stop_words, NLTK_StopWords = StopWords_extract() # Extract stop words
    twenty_train, X_train_tfidf, X_train_counts, train_count_vect = TFIDF(categories,'train',stop_words,NLTK_StopWords)
    twenty_test, X_test_tfidf, X_test_counts = TFIDF_transform(categories,'test',stop_words,NLTK_StopWords, train_count_vect)

    Y_data_labeling(twenty_test.target) # Category values for verification
    Y_data_labeling(twenty_train.target) # Category values for trainning

    #part_G

    clf_multiNB = MultinomialNB()
    bayes_model = clf_multiNB.fit(X_train_tfidf.toarray(), twenty_train.target) # train the data over high-dimension data and known Y values

    Y_predict_multiNB = bayes_model.predict(X_test_tfidf) # predict the value with the learning algorithm
    accuracy_bayes = np.mean(twenty_test.target == Y_predict_multiNB)
    probas_ = bayes_model.predict_proba(X_test_tfidf)
    fpr, tpr, thresholds = metrics.roc_curve(twenty_test.target, probas_[:, 1])
    print ('Part_G')
    infor_print(accuracy_bayes, twenty_test, Y_predict_multiNB)
    roc_print('../Graphs/part_G', probas_, fpr, tpr, thresholds, "Multinomial Naive Bayes ROC")

    #part_H
    clf_log = LogisticRegression()
    logistic_model = clf_log.fit(X_train_tfidf.toarray(), twenty_train.target) # train the data over high-dimension data and known Y values
    Y_predict_log = logistic_model.predict(X_test_tfidf) # predict the value with the learning algorithm
    accuracy_logistic = np.mean(twenty_test.target == Y_predict_log)
    probas_ = logistic_model.predict_proba(X_test_tfidf)
    fpr, tpr, thresholds = metrics.roc_curve(twenty_test.target, probas_[:, 1])
    print("\n")
    print("\n")
    print ('Part_H')
    infor_print(accuracy_logistic, twenty_test, Y_predict_log)
    roc_print('../Graphs/part_H', probas_, fpr, tpr, thresholds, "Logistic Regression ROC")

    #part_I
    print("\n")
    print("\n")
    print ('Part_I')
    for i, C in enumerate((100,1,0.01)):
        clf_L1_LR = LogisticRegression(C=C, penalty = 'l1', tol = 0.01)
        clf_L2_LR = LogisticRegression(C=C, penalty = 'l2', tol = 0.01)
        clf_L1_LR.fit(X_train_tfidf.toarray(), twenty_train.target)
        clf_L2_LR.fit(X_train_tfidf.toarray(), twenty_train.target)

        Y_predict_L1_LR = clf_L1_LR.predict(X_test_tfidf)
        Y_predict_L2_LR = clf_L2_LR.predict(X_test_tfidf)

        # Calculating the accuracy, recall, precision and confusion matrix

        print ('When the coefficient is %f' % C)
        accuracy_L1 = np.mean(twenty_test.target == Y_predict_L1_LR)
        accuracy_L2 = np.mean(twenty_test.target == Y_predict_L2_LR)  
              
        print ('The accuracy for the model with L1 regularization is %f' % accuracy_L1)
        print ("The coefficient for the model is :")
        print (clf_L1_LR.score(X_train_tfidf.toarray(), twenty_train.target))

        print ('The accuracy for the model with L2 regularization is %f' % accuracy_L2)
        print ("The coefficient for the model is :")        
        print (clf_L2_LR.score(X_train_tfidf.toarray(), twenty_train.target))

