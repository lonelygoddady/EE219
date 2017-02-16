from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
import numpy as np

# import processed data from part b
from part_b import TFIDF,preprocess,StopWords_extract

if __name__ == "__main__":
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                  'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                  'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
                  'talk.religion.misc']

    stop_words, NLTK_StopWords = StopWords_extract() # Extract stop words
    twenty_train, X_train_tfidf, X_train_counts, Train_count_vect = TFIDF(categories, 'train',
                                                                          stop_words, NLTK_StopWords)

    docs, terms = X_train_counts.shape
    TF_ICF = np.zeros(shape=(20,terms))

    # iterate through documents and check their category
    for item in range(docs):
        item_cat = twenty_train.target[item] # extract the catogory index of this document
        TF_ICF[item_cat,] += X_train_counts[item,] # build the TFICF matrix
        # Note TF_ICF is a VERY SPARSE matrix. So you will mostly see zeros with print

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    tf_icf_final = tf_transformer.transform(TF_ICF) # calculate this TF_ICF matrix
    features = Train_count_vect.get_feature_names() # get the term names from the term vector

    # extract 10 most significant terms from following class
    target_cat_index = [twenty_train.target_names.index("comp.sys.ibm.pc.hardware"),
                        twenty_train.target_names.index("comp.sys.mac.hardware"),
                        twenty_train.target_names.index("misc.forsale"),
                        twenty_train.target_names.index("soc.religion.christian")]

    for x in range(len(target_cat_index)):
        row = tf_icf_final[target_cat_index[x]].toarray()[0]  # get all term values in tf_icf matrix for that category
        zipped_row = zip(row,features) # combine term frequency and names
        sort_row = sorted(zipped_row)
        print 'The 10 most significant terms in ' + str(twenty_train.target_names[target_cat_index[x]]) \
              + ' is' + str(zip(*sort_row[-10:])[1])


