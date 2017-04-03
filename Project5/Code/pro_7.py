import datetime
import json
import re
import unirest
import json
import ast
import numpy
from sklearn.feature_extraction import text
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as pyplot

WA_loc = ['seattle', ' washington', ',wa', ', wa']
MA_loc = ['boston', 'massachusetts', 'mass', ',ma', ', ma']

regex_str = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

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
    sentence = " ".join(sentence)
    sentence = sentence.lower()  # transfers each word to lower case
    tokens = tokenize(sentence)
    filtered_tokens = [w for w in tokens if not w.startswith('@')
                       and not w.startswith('#') and not w.startswith('http')]

    new_sentence = " ".join(filtered_tokens)

    tokenizer = RegexpTokenizer(r'\w+')  # Tokenizer to remove punctuation marks
    # stemmer = SnowballStemmer("english")  # Stemmer to perform stemming
    tokens = tokenizer.tokenize(new_sentence)
    filtered_words = [w for w in tokens if
                      not w in NLTK_StopWords and w not in stop_words and len(w) is not 1]  # Removes stop words
    # filtered_words_mid = [stemmer.stem(plural) for plural in filtered_words]
    filtered_words_final = [i for i in filtered_words if not i.isdigit()]  # Removes numbers and digits
    new_sentence = " ".join(filtered_words_final)
    return new_sentence

def check_list(tokens, loc):
    for i in tokens:
        if i in loc: return 1
    return 0

def modified_time(end_time):
    # get the data's time in datetime format and slice them into hours
    date_time = datetime.datetime.fromtimestamp(end_time)
    actual_time = datetime.datetime(date_time.year, date_time.month, date_time.day,
                                    date_time.hour, int(date_time.minute/6)*6, 0)
    return unicode(actual_time), actual_time

def tweet_extract(tweets):

    # Structure that holds hour wise features
    data_WA, data_MA = {}, {}

    for i in range(24):
        for j in range(10):
            data_WA[unicode(datetime.datetime(2015, 2, 1, i, j*6, 0))] = []
            data_MA[unicode(datetime.datetime(2015, 2, 1, i, j*6, 0))] = []

    ####### Loop through each tweet and calculate statistics #######
    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        # create the data for this hour if not exist
        modified_actual_time, actual_time = modified_time(end_time)

        #only process data in this window
        if modified_actual_time in data_WA.keys():
            if check_list(WA_loc, tweet_data['tweet']['user']['location'].lower()):
                data_WA[modified_actual_time].append(tweet_data['title'])
                # print(tweet_data['tweet']['user']['location'].lower())

            elif check_list(MA_loc, tweet_data['tweet']['user']['location'].lower()):
                data_MA[modified_actual_time].append(tweet_data['title'])
                # print(tweet_data['tweet']['user']['location'].lower())

    return data_WA, data_MA


def store_infor():
    tweets = open('./tweet_data/tweets_#superbowl.txt', 'rb')

    #   Extract all tweets in 10-minute windows and tokenize them
    tweets_WA, tweets_MA = tweet_extract(tweets)
    tweets_WA_tok, tweets_MA_tok = [], []
    stop_words, NLTK_StopWords = StopWords_extract()

    for key in tweets_WA.keys():
        tweets_WA_tok.append(preprocess(tweets_WA[key], stop_words, NLTK_StopWords))
        tweets_MA_tok.append(preprocess(tweets_MA[key], stop_words, NLTK_StopWords))

    WA_sentiments, MA_sentiments = [], []
    WA_label, MA_label = [], []

    for item in tweets_WA_tok:
        # print(item)
        if (item != ''):
            response = unirest.post("https://japerk-text-processing.p.mashape.com/sentiment/",
                                    headers={
                                        "X-Mashape-Key": "xoP8DthHv5msh1AkGSIu6vZDCsNTp1ptAwBjsn0uX4sghhWpnK",
                                        "Content-Type": "application/x-www-form-urlencoded",
                                        "Accept": "application/json"
                                    },
                                    params={
                                        "language": "english",
                                        "text": item
                                    }
                                    )
            result = response.body
            WA_sentiments.append(result['probability'])
            WA_label.append(result['label'])
        else:
            WA_sentiments.append({'neg':0, 'neutral':0, 'pos':0})
            WA_label.append('N/A')

    for item in tweets_MA_tok:
        # print(item)
        if (item != ''):
            response = unirest.post("https://japerk-text-processing.p.mashape.com/sentiment/",
                                    headers={
                                        "X-Mashape-Key": "xoP8DthHv5msh1AkGSIu6vZDCsNTp1ptAwBjsn0uX4sghhWpnK",
                                        "Content-Type": "application/x-www-form-urlencoded",
                                        "Accept": "application/json"
                                    },
                                    params={
                                        "language": "english",
                                        "text": item
                                    }
                                    )
            result = response.body
            MA_sentiments.append(result['probability'])
            MA_label.append(result['label'])
        else:
            MA_sentiments.append({'neg': 0, 'neutral': 0, 'pos': 0})
            MA_label.append('N/A')

    # Write APi results into txt file
    WA_sent = open('./pro_7_results/WA_sentiments.txt', 'wb')
    MA_sent = open('./pro_7_results/MA_sentiments.txt', 'wb')
    WA_l = open('./pro_7_results/WA_label.txt', 'wb')
    MA_l = open('./pro_7_results/MA_label.txt', 'wb')

    for i in range(len(WA_sentiments)):
        WA_sent.write(str(WA_sentiments[i])+'\n')
        WA_l.write(WA_label[i]+'\n')

    for i in range(len(MA_sentiments)):
        MA_sent.write(str(MA_sentiments[i])+'\n')
        MA_l.write(MA_label[i]+'\n')

if __name__ == '__main__':
    store_infor()
    WA_sent = open('./pro_7_results/WA_sentiments.txt', 'rb')
    MA_sent = open('./pro_7_results/MA_sentiments.txt', 'rb')
    WA_l = open('./pro_7_results/WA_label.txt', 'rb')
    MA_l = open('./pro_7_results/MA_label.txt', 'rb')

    WA_pos, MA_pos, WA_neg, MA_neg = [], [], [], []

    for line in WA_sent:
        print line
        tweet_data = ast.literal_eval(line)
        WA_pos.append(tweet_data['pos'])
        WA_neg.append(tweet_data['neg'])

    for line in MA_sent:
        print line
        tweet_data = ast.literal_eval(line)
        MA_pos.append(tweet_data['pos'])
        MA_neg.append(tweet_data['neg'])

    WA_mean = numpy.mean(WA_pos)
    WA_pos = [w if w != 0 else WA_mean for w in WA_pos]
    WA_mean_neg = numpy.mean(WA_neg)
    WA_neg = [w if w != 0 else WA_mean_neg for w in WA_neg]

    MA_mean = numpy.mean(MA_pos)
    MA_pos = [w if w != 0 else MA_mean for w in MA_pos]
    MA_mean_neg = numpy.mean(MA_neg)
    MA_neg = [w if w != 0 else MA_mean_neg for w in MA_neg]

    pyplot.figure(1)
    # pyplot.bar(range(len(WA_pos)), WA_pos, color='r', label='WA POS')
    # pyplot.bar(range(len(WA_neg)), WA_neg, color='b', label='WA NEG')
    pyplot.plot(range(len(WA_pos)), WA_pos)
    z = numpy.polyfit(range(len(WA_pos)), WA_pos, 10)
    p = numpy.poly1d(z)
    pyplot.plot(range(len(WA_pos)), p(range(len(WA_pos))), "r--")
    pyplot.plot(range(len(WA_neg)), WA_neg)
    z = numpy.polyfit(range(len(WA_neg)), WA_neg, 10)
    p = numpy.poly1d(z)
    pyplot.plot(range(len(WA_neg)), p(range(len(WA_neg))), "r--", color='b')
    pyplot.xlabel('Time bin')
    pyplot.ylabel('Sentiment Score')
    pyplot.title('WA Sentiments')
    pyplot.legend(loc='upper right')

    pyplot.figure(2)
    # pyplot.bar(range(len(MA_pos)), MA_pos, color='r', label='MA POS')
    # pyplot.bar(range(len(MA_neg)), MA_neg, color='b', label='MA NEG')
    pyplot.plot(range(len(MA_pos)), MA_pos)
    z = numpy.polyfit(range(len(MA_pos)), MA_pos, 10)
    p = numpy.poly1d(z)
    pyplot.plot(range(len(MA_pos)), p(range(len(MA_pos))), "r--")
    pyplot.plot(range(len(MA_neg)), MA_neg)
    z = numpy.polyfit(range(len(MA_neg)), MA_neg, 10)
    p = numpy.poly1d(z)
    pyplot.plot(range(len(MA_neg)), p(range(len(MA_neg))), "r--", color='b')
    pyplot.xlabel('Time bin')
    pyplot.ylabel('Sentiment Score')
    pyplot.title('MA Sentiments')
    pyplot.legend(loc='upper right')
    pyplot.show()


