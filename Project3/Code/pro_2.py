from __future__ import print_function
import pandas as ps
import numpy as np
import os 
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matrixNMF

k_val = [10, 50, 100]

# np.random.seed(304145309)


def load_data():
    data = ps.read_csv('../ml-latest-small/ratings.csv', usecols=['userId', 'movieId', 'rating'])
    R = ps.pivot_table(data, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
    R_mat = R.as_matrix()
    print("R is ..")
    print(R.shape)
    data_mat = data.as_matrix()
    return data_mat, R_mat


def cross_validation():
    Threshold = [1, 2, 3, 4, 5]
    # data_mat, R_mat = load_data()
    data = ps.read_csv('../ml-latest-small/ratings.csv', usecols=['userId', 'movieId', 'rating'])
    R = ps.pivot_table(data, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
    # R_mat = R.as_matrix()
    R_index = R.index
    R_column = R.columns
    data_mat = data.as_matrix()
    print("R is ..")
    print(R.shape)

    N = len(data_mat)
    split_N = np.arange(N)
    kf = KFold(n_splits=10)
    Avg_error_train = np.zeros((3,10))
    Avg_error_test = np.zeros((3,10))
    counter = 0
    k_counter = 0

    precision_tot = np.zeros((3,5))
    precision_real = np.zeros((3,5))
    recall_tot = np.zeros((3,5))
    recall_real = np.zeros((3,5))

    for train_index, test_index, in kf.split(split_N):
        # num_train = len(train_index)
        # num_test = len(test_index)
        data_train, data_test = data_mat[train_index], data_mat[test_index]
        train_df = ps.DataFrame(data_train, columns=['userId', 'movieId', 'rating'])
        test_df = ps.DataFrame(data_test, columns=['userId', 'movieId', 'rating'])

        R_train = ps.pivot_table(train_df, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
        R_test = ps.pivot_table(test_df, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
        R_train_new = R_train.reindex(index=R_index, columns=R_column, fill_value=0)
        # R_test = R_test.reindex(index = R_index, columns = R_column, fill_value = 0)
        print("train and test dimensions")
        print(R_train_new.shape)

        train_mat = R_train_new.as_matrix()
        for w in k_val:
            model = NMF(n_components=w, init='random', random_state=42)
        # userID = R_train_new['userId']
        # movieID = R_train_new['movieId']
        # data_rating = R_train_new['rating']
        # train_mat = sparse.coo_matrix((data_rating, (userID, movieID)))
            model.fit(train_mat)
            V = model.components_
            U = model.fit_transform(train_mat)
        # U, V, err = NMF(train_mat, 100)
            R_predict = np.dot(U, V)

            R_predict = ps.DataFrame(R_predict, index=R_index, columns=R_column)
            R_predict_train = R_predict.reindex(index=R_train.index, columns=R_train.columns, fill_value=0)
            R_predict_test = R_predict.reindex(index=R_test.index, columns=R_test.columns, fill_value=0)

            y_train_real = R_train.as_matrix()
            y_train_pre = R_predict_train.as_matrix()
            y_test_real = R_test.as_matrix()
            y_test_pre = R_predict_test.as_matrix()

            Avg_error_train[k_counter][counter] = (np.sum(sum(abs(y_train_pre - y_train_real)))) / (np.sum(sum(y_train_real)))
            Avg_error_test[k_counter][counter] = (np.sum(sum(abs(y_test_pre - y_test_real)))) / (np.sum(sum(y_test_real)))
        

        # part 3
            for i in Threshold:
                for m in range(len(y_train_pre)):
                    for n in range(len(y_train_pre[0])):
                        if y_train_pre[m][n] >= i:
                            precision_tot[k_counter][i-1] = precision_tot[k_counter][i-1] + 1
                            if y_train_real[m][n] >= i:
                                precision_real[k_counter][i-1] = precision_real[k_counter][i-1] + 1

                for m in range(len(y_test_pre)):
                    for n in range(len(y_test_pre[0])):
                        if y_test_pre[m][n] >= i:
                            recall_tot[k_counter][i-1] = recall_tot[k_counter][i-1] + 1
                            if y_test_real[m][n] >= i:
                                recall_real[k_counter][i-1] = recall_real[k_counter][i-1] + 1
        
        counter = counter + 1

    for k in range(len(k_val)):
        print ("When k equals to ", k_val[k])
        max_error_train = np.amax(Avg_error_train[k])
        min_error_train = np.min(Avg_error_train[k])
        mean_error_train = np.mean(Avg_error_train[k])
        max_error_test = np.amax(Avg_error_test[k])
        min_error_test = np.min(Avg_error_test[k])
        mean_error_test = np.mean(Avg_error_test[k])
        print("The average error for the 10 train group is ", mean_error_train)
        print("The max average error for the 10 train group is ", max_error_train)
        print("The min average error for the 10 train group is ", min_error_train)
        print("The average error for the 10 train group is ", mean_error_test)
        print("The max average error for the 10 train group is ", max_error_test)
        print("The min average error for the 10 train group is ", min_error_test)


    #ROC plot
    # if not os.path.exists('../Graphs/pro_3'):
    #     os.makedirs('../Graphs/pro_3')

    # plt.figure()
    # for i in range(len(like_train_predict)):
    #     fpr, tpr, thresholds = roc_curve(y_score = like_test_predict[i], y_true = like_train_predict[i])
    #     plt.plot(fpr, tpr, lw = lw, color = color, label = 'Threshold = %d' % (i+1))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve (precision over recall)')
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('../Graphs/pro_3/ROC')

        print("predic", precision_tot[k])
        print("real", precision_real[k])
        for i in range(len(precision_tot[k])):
            precision_real[k][i] = precision_real[k][i] / precision_tot[k][i]
            recall_real[k][i] = recall_real[k][i] / recall_tot[k][i]

        print(recall_real)
        print(precision_real)

        if not os.path.exists('../Graphs/pro_3'):
            os.makedirs('../Graphs/pro_3')

        plt.figure()
        plt.plot(recall_real[k], precision_real[k])
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        s = 'ROC curve (precision over recall), k = ' + str(k_val[k])
        plt.title(s)
        plt.show()
        s = "../Graphs/pro_3/ROC_k=" + str(k_val[k])
        plt.savefig(s)



if __name__ == "__main__":
    cross_validation()
