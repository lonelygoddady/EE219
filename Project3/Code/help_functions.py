from __future__ import print_function
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
from scipy import linalg
import math

from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold


def fetch_valid_data():
    links = ps.read_csv('../ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    table = links.pivot_table(index=['user_id'], columns=['movie_id'], values='rating', fill_value=0)
    movieID_index = list(set(links['movie_id']))
    # print("R shape is: ", R.shape)
    return table, movieID_index
    # movieID_index return original indexes of all movies (column 1 to 9066)


def calculate_error(predicted, actual, weights):
    error = actual - predicted
    squared_error = np.multiply(error,error)
    squared_error = np.multiply(weights, squared_error)
    sum_squared_error = sum(sum(squared_error))
    return sum_squared_error


def nmf(X, k, weight=None, max_iter=100, reg_lambda=0):
    print("performing WH decomposition...")
    eps = 1e-5  # to avoid 0 entries
    # X = X.toarray()
    if weight is None:
        weight = np.sign(X)  # this is W
    # initial matrices
    rows, columns = X.shape
    W = np.random.rand(rows, k)
    W = np.maximum(W, eps)

    H = linalg.lstsq(W, X)[0]
    H = np.maximum(H, eps)

    weighted_X = weight * X
    for i in range(1, max_iter + 1):
        top = dot(weighted_X, H.T)
        bottom = dot((weight * dot(W, H)), H.T) + reg_lambda * W + eps
        W *= top / bottom
        W = np.maximum(W, eps)

        top = dot(W.T, weighted_X)
        bottom = dot(W.T, weight * dot(W, H)) + reg_lambda * H + eps
        H *= top / bottom
        H = np.maximum(H, eps)

    X_predict = dot(W, H)
    total_error = X - X_predict
    # print('real error sum is: ', np.sum(real_error))
    total_residual = np.sum(weight * (total_error * total_error))
    return W, H, total_residual


def ROC_curve_plotter(predicted, actual, k, reg_lambda=0, reverse=False):
    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    tn = 0  # true negative

    if reverse is True:
        threshold_value = np.arange(0.8, 1, 0.01)
    else:
        threshold_value = np.arange(1, 5.5, 0.5)
    
    fp_rate = np.zeros(len(threshold_value))
    tp_rate = np.zeros(len(threshold_value))

    for i, t in enumerate(threshold_value):
        tp = np.sum(actual[predicted >= t] >= t)
        fp = np.sum(actual[predicted >= t] < t)
        fn = np.sum(actual[predicted < t] >= t)
        tn = np.sum(actual[predicted < t] < t)

        # fp_rate[i] = fp / (float(fp + tn) + 1e-6)  # calculating TPR
        # tp_rate[i] = tp / (float(tp + fn) + 1e-6)  # calculating FPR
        fp_rate[i] = tp / (float(fp + tp) + 1e-6)  # calculating precision
        tp_rate[i] = tp / (float(tp + fn) + 1e-6)  # calculating recall
        
        print (tp, fp, fn, tn)

    if reg_lambda is not 0:
        plt.figure(k)
        plt.title('ROC Curve k={0} lambda=0.01,0.1,1'.format(k))
    else:
        plt.figure(1)
        plt.title('ROC Curve k=10,50,100')

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.scatter(fp_rate, tp_rate, s=60, marker='o')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.scatter(tp_rate, fp_rate, s=60, marker='o')
    plt.plot(tp_rate, fp_rate)


def cross_validation():
    Threshold = [1, 2, 3, 4, 5]
    data_mat, R_mat = fetch_valid_data()
    data = ps.read_csv('../ml-latest-small/ratings.csv', usecols=['userId', 'movieId', 'rating'])
    R = ps.pivot_table(data, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
    R_index = R.index
    R_column = R.columns
    print("R is ..")
    print(R.shape)

    N = len(data_mat)
    split_N = np.arange(N)
    kf = KFold(n_splits=10)
    Avg_error_train = np.zeros(10)
    Avg_error_test = np.zeros(10)
    counter = 0

    like_train_true = np.zeros((10, 5))
    like_train_predict = np.zeros((10, 5))
    like_test_true = np.zeros((10, 5))
    like_test_predict = np.zeros((10, 5))
    train_test_num = np.zeros((10, 2))

    for train_index, test_index, in kf.split(split_N):
        num_train = len(train_index)
        num_test = len(test_index)
        train_test_num[counter] = [num_train, num_test]
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
        # model = NMF(n_components=k, init='random', random_state=42)
        # model.fit(train_mat)
        # V = model.components_
        # U = model.fit_transform(train_mat)
        # U, V, err = NMF(train_mat, 100)
        U, V, err = nmf(train_mat, k=100)
        R_predict = np.dot(U, V)

        R_predict = ps.DataFrame(R_predict, index=R_index, columns=R_column)
        R_predict_train = R_predict.reindex(index=R_train.index, columns=R_train.columns, fill_value=0)
        R_predict_test = R_predict.reindex(index=R_test.index, columns=R_test.columns, fill_value=0)

        y_train_real = R_train.as_matrix()
        y_train_pre = R_predict_train.as_matrix()
        y_test_real = R_test.as_matrix()
        y_test_pre = R_predict_test.as_matrix()

        Avg_error_train[counter] = (np.sum(sum(abs(y_train_pre - y_train_real)))) / (np.sum(sum(y_train_real)))
        Avg_error_test[counter] = (np.sum(sum(abs(y_test_pre - y_test_real)))) / (np.sum(sum(y_test_real)))
        counter = counter + 1

        # part 3
        # for i in Threshold:
        #     like_train_true[counter][i - 1] = np.sum(sum(x >= Threshold for x in y_train_real)) / num_train
        #     like_train_predict[counter][i - 1] = np.sum(sum(x >= Threshold for x in y_train_pre)) / num_train
        #     like_test_true[counter][i - 1] = np.sum(sum(x >= Threshold for x in y_test_real)) / num_test
        #     like_test_predict[counter][i - 1] = np.sum(sum(x >= Threshold for x in y_test_pre)) / num_test

    max_error_train = np.amax(Avg_error_train)
    min_error_train = np.min(Avg_error_train)
    mean_error_train = np.mean(Avg_error_train)
    max_error_test = np.amax(Avg_error_test)
    min_error_test = np.min(Avg_error_test)
    mean_error_test = np.mean(Avg_error_test)

    print("The average error for the 10 train group is ", mean_error_train)
    print("The max average error for the 10 train group is ", max_error_train)
    print("The min average error for the 10 train group is ", min_error_train)
    print("The average error for the 10 train group is ", mean_error_test)
    print("The max average error for the 10 train group is ", max_error_test)
    print("The min average error for the 10 train group is ", min_error_test)
