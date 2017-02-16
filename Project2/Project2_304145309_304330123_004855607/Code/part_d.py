from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

# import the tfidf matrix from b
from part_b import TFIDF,preprocess,StopWords_extract


def LSI(X_train_tfidf, k):
    svd = TruncatedSVD(n_components=k, random_state=42, algorithm='arpack')     # default algorithm will cause segfault
    LSI_tfidf = svd.fit_transform(X_train_tfidf)
    print LSI_tfidf.shape
    return LSI_tfidf

if __name__ == "__main__":
    # categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
    # 'comp.sys.mac.hardware','comp.windows.x', 'misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball',
    # 'rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space', 'soc.religion.christian',
    # 'talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']

    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

    stop_words, NLTK_StopWords = StopWords_extract()                            # Extract stop words
    twenty_train, X_train_tfidf, X_train_counts, Train_count_vect = TFIDF(categories, 'train',
                                                                          stop_words, NLTK_StopWords)
    LSI_TFIDF = LSI(X_train_tfidf, 50)
