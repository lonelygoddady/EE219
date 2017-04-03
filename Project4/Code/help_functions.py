import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_extraction import text
from sklearn import metrics
from nltk import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


def data_labeling(Y_data):
    labels = []
    for y in Y_data:
        labels.append(0 if (y <= 3) else 1)
    return labels


# two stop words lists
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


# The function below performs the pre-processing and cleaning on the data
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


def TFIDF(categories, train_or_test, stop_words, NLTK_StopWords):
    # Fetching the data set
    twenty_data = fetch_20newsgroups(subset=train_or_test, categories=categories,
                                     remove=('headers', 'footers', 'quotes'))

    # Stores the size of the dataset
    size, = twenty_data.filenames.shape

    # Performing preprocessing on every document
    for item in range(0, size):
        # print (twenty_data.filenames[item])
        sentence = twenty_data.data[item]
        twenty_data.data[item] = preprocess(sentence, stop_words, NLTK_StopWords)

    # Transferring the modified dataset into a Term Document Matrix
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_data.data)

    # Calculating the TF-IDF values for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    # vectorizer = TfidfVectorizer(max_df=0.8,
    #                              min_df=2, stop_words=stop_words,
    #                              use_idf=True, sublinear_tf=True)
    # X_train_tfidf = vectorizer.fit_transform(twenty_data.data)
    docs, terms = X_train_tfidf.shape
    print(79 * '_')
    print("TFIDF matrix constructed")
    print ("The final number of terms are", terms)
    print ("The final number of docs are", docs)
    print(79 * '_')
    return twenty_data, X_train_tfidf, X_train_counts, count_vect


def bench_k_means(estimator, name, data, labels):
    estimator.fit(data)
    print('% 9s   %.3f   %.3f   %.3f   %.3f'
          % (name,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             ))


def data_load():
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    stop_words, NLTK_StopWords = StopWords_extract()
    twenty_train, X_tfidf, X_counts, count_vect = TFIDF(categories, 'all', stop_words, NLTK_StopWords)
    return twenty_train, X_tfidf, X_counts, count_vect


def calculate_sigma(X_tfidf, k_values):
    print(79 * '_')
    print("Calculate Sigma Matrix")
    U, Sigma, VT = randomized_svd(X_tfidf, n_components=k_values,
                                  n_iter=200,
                                  random_state=42)
    print(Sigma)
    print(79 * '_')
    return Sigma


def logrithm(tfidf):
    matrix_val = sorted(set(tfidf.flatten()))
    const = matrix_val[1]
    mean = np.mean(matrix_val)
    for (x, y), value in np.ndenumerate(tfidf):
        # tfidf[x,y] = np.log(value+(1+abs(mean)))
        tfidf[x,y] = np.log(1+np.power(0.4, value))
    return tfidf


def confusion_matrix_build(km_labels, labels):
    tn, fp, fn, tp = 0, 0, 0, 0
    correct = 0
    wrong = 0
    for i in range(len(km_labels)):
        if km_labels[i] == labels[i]:
            correct = correct + 1
        else:
            wrong = wrong + 1

        if km_labels[i] == 0 and labels[i] == 0:
            tn = tn + 1
        elif km_labels[i] == 1 and labels[i] == 0:
            fp = fp + 1
        elif km_labels[i] == 0 and labels[i] == 1:
            fn = fn + 1
        elif km_labels[i] == 1 and labels[i] == 1:
            tp = tp + 1
    print([[tn, fn], [fp, tp]])
    print(correct,wrong)
    return np.array([[tn, fn], [fp, tp]])
