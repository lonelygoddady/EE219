import json
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pprint import pprint
import numpy as np
import os
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
from sklearn import metrics
from nltk import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split



input_files = ['superbowl']  # tweet-data file names
tweets = []
location = []
t_text = []
tweet_text = ['tweet']
WA_LOC = ['seattle', 'washington', 'kirkland', ', wa', ',wa']
MA_LOC = ['massachusetts', 'mass', ',ma',', ma']
tweet_data = ['tweet']
labels = []

def loc_extraction(loc):
    if 'WA' in loc:
        return 'wa'
    elif 'MA' in loc:
        return 'ma'
    else:
        for i in WA_LOC:
            if i in loc:
                return 'wa'
        for i in MA_LOC:
            if i in loc:
                return 'ma'
        return 'false'

def StopWords_extract():
    stop_words = [text.ENGLISH_STOP_WORDS]
    with open('NLTK_StopWords.txt', 'r') as file:
        # NLTK_StopWords = file.read().splitlines()
        lists = file.readlines()

    # NLTK_StopWords = []
    for i in range(len(lists)):
        lists[i] = lists[i].rstrip()  # remove trailing spaces

    NLTK_StopWords = lists

    return stop_words, NLTK_StopWords

def preprocess(sentence, stop_words, NLTK_StopWords):
    sentence = sentence.lower()  # transfers each word to lower case
    tokenizer = RegexpTokenizer(r'\w+')  # Tokenizer to remove punctuation marks
    stemmer = SnowballStemmer("english")  # Stemmer to perform stemming
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if
                      not w in NLTK_StopWords and w not in stop_words and len(w) is not 1]  # Removes stop words
    filtered_words_mid = [stemmer.stem(plural) for plural in filtered_words]
    filtered_words_final = [i for i in filtered_words_mid if not i.isdigit()]  # Removes numbers and digits
    return " ".join(filtered_words_final)

def TFIDF(tweet_data):
    # Fetching the data set
    stop_words, NLTK_StopWords = StopWords_extract()

    # Performing preprocessing on every document
    for item in range(len(tweet_data)):
        sentence = tweet_data[item]
        tweet_data[item] = preprocess(sentence, stop_words, NLTK_StopWords)

    # Transferring the modified dataset into a Term Document Matrix
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(tweet_data)

    # Calculating the TF-IDF values for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    docs, terms = X_train_tfidf.shape
    print(79 * '_')
    print("TFIDF matrix constructed")
    print ("The final number of terms are", terms)
    print ("The final number of docs are", docs)
    print(79 * '_')
    return tweet_data, X_train_tfidf

def acurracy_check(labels, Y_predict):
    # Calculating the accuracy, recall, precision and confusion matrix
    accuracy_svm = np.mean(labels == Y_predict)
    print ('The accuracy for the model is %f' % accuracy_svm)
    print ('\'0\' is from WA and \'1\' is from MA')
    print ("The precision and recall values are:")
    print (metrics.classification_report(labels, Y_predict))
    print ('The confusion matrix is as shown below:')
    print (metrics.confusion_matrix(labels, Y_predict))


def ROC_plot(model, X_test_LSI, labels, title):
    # Plotting the ROC
    probas_ = model.predict_proba(X_test_LSI)
    fpr, tpr, thresholds = metrics.roc_curve(labels, probas_[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, lw=1, label=title)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.savefig(title)
    print("figure_saved")


for file_name in input_files:
    file = './tweet_data (1)/tweets_#' + file_name + '.txt'
    f = open(file)
    line = f.readline()
    
    for i in range(100000):
        tweet = json.loads(line)
        tweets.append(tweet)
        line = f.readline()

for i, tweet in enumerate(tweets):
    # print(i)
    # tweet = tweets[i]
    location.append(tweet['tweet']['user']['location'])
    # pprint(tweet['tweet']['user']['location'])
    tweet_text.append(tweet['tweet']['text'])
    # print(tweet_text[i+1])
    # pprint(tweet['tweet']['text'])
    if 'media' in tweet['tweet']['entities']:
        url = tweet['tweet']['entities']['media'][0]['url']
        # pprint(tweet['tweet']['entities']['media'][0]['url'])
        tweet_text[i+1] = tweet_text[i+1].strip(url)
        # print(tweet_text[i+1])
    if 'url' in tweet['tweet']['entities']['urls']:
    	url = tweet['tweet']['entities']['urls']['url']
    	tweet_text[i+1] = tweet_text[i+1].strip(url)
del tweet_text[0]


for i, loc in enumerate(location):
    loc_def = loc_extraction(loc)
    if loc_def == 'wa':
        labels.append(0)
        tweet_data.append(tweet_text[i])
    elif loc_def == 'ma':
        labels.append(1)
        tweet_data.append(tweet_text[i])
del tweet_data[0]

X_test, X_train, y_test, y_train = train_test_split(tweet_data, labels, test_size=0.8)
X_train, X_train_tfidf = TFIDF(X_train)
X_test, X_test_tfidf = TFIDF(X_test)

#LSI
svd = TruncatedSVD(n_components=50, random_state=42, algorithm='arpack')
X_train_LSI = svd.fit_transform(X_train_tfidf)
X_train_LSI = Normalizer(copy=False).fit_transform(X_train_LSI)
X_test_LSI = svd.fit_transform(X_test_tfidf)
X_test_LSI = Normalizer(copy=False).fit_transform(X_test_LSI)

print(X_train_LSI.shape)
print(X_test_LSI.shape)

#SVM
clf_ovr = svm.SVC(kernel='linear', probability=True, C = 0.01)
ovr_model = clf_ovr.fit(X_train_LSI, y_train)


#Logistic
clf_log = LogisticRegression()
logistic_model = clf_log.fit(X_train_LSI, y_train)

#Naive Bayes
clf_multiNB = GaussianNB()
bayes_model = clf_multiNB.fit(X_train_LSI, y_train)

#value prediction
print("performing SVM prediction")
Y_predict_ovr = ovr_model.predict(X_test_LSI)
acurracy_check(y_test, Y_predict_ovr)
ROC_plot(ovr_model, X_test_LSI, y_test, "SVM ROC")


print("performing Logistic prediction")
Y_predict_log = clf_log.predict(X_test_LSI)
acurracy_check(y_test, Y_predict_log)
ROC_plot(logistic_model, X_test_LSI, y_test, "Logistic ROC")

print("performing Multi-NB prediction")
Y_predict_NB = clf_multiNB.predict(X_test_LSI)
acurracy_check(y_test, Y_predict_NB)
ROC_plot(bayes_model, X_test_LSI, y_test, "Naive Bayes ROC")





    

