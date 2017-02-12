from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from nltk import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import os

#two stop words lists
def StopWords_extract():
    stop_words = [text.ENGLISH_STOP_WORDS]
    with open('NLTK_StopWords.txt', 'r') as file:
            # NLTK_StopWords = file.read().splitlines()
            lists = file.readlines()

    # NLTK_StopWords = []
    for i in range(len(lists)):
        lists[i] = lists[i].rstrip()# remove trailing spaces

    NLTK_StopWords = lists

    return stop_words, NLTK_StopWords

#The function below performs the pre-processing and cleaning on the data
def preprocess(sentence, stop_words, NLTK_StopWords):
    sentence = sentence.lower() #transfers each word to lower case
    tokenizer = RegexpTokenizer(r'\w+') #Tokenizer to remove punctuation marks
    stemmer = SnowballStemmer("english") #Stemmer to perform stemming
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in NLTK_StopWords and w not in stop_words and len(w) is not 1] #Removes stop words
    filtered_words_mid = [stemmer.stem(plural) for plural in filtered_words]
    filtered_words_final = [i for i in filtered_words_mid if not i.isdigit()] #Removes numbers and digits
    return " ".join(filtered_words_final)

def TFIDF(categories, train_or_test, stop_words, NLTK_StopWords):
    #Fetching the data set
    twenty_data = fetch_20newsgroups(subset=train_or_test,categories=categories, remove=('headers','footers','quotes'))

    #Stores the size of the dataset
    size, = twenty_data.filenames.shape

    #Performing preprocessing on every document
    for item in range(0,size):
        print twenty_data.filenames[item]
        sentence = twenty_data.data[item]
        twenty_data.data[item] = preprocess(sentence, stop_words, NLTK_StopWords)

    #Transferring the modified dataset into a Term Document Matrix    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_data.data)

    #Calculating the TF-IDF values for every term in the document
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    docs,terms = X_train_tfidf.shape
    print "The final number of terms are", terms
    print "The final number of docs are", docs
    return twenty_data, X_train_tfidf, X_train_counts, count_vect

if __name__ == "__main__":
    # perform tfidf
    categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',   'comp.sys.mac.hardware','comp.windows.x',
    'misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space',
    'soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']

    stop_words, NLTK_StopWords = StopWords_extract()
    twenty_train, X_train_tfidf, X_train_counts, count_vect = TFIDF(categories,'train',stop_words,NLTK_StopWords)



