from __future__ import print_function
import pandas as ps
import numpy as np
import os 
from sklearn.decomposition import NMF
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import help_functions

k = 100



def pro_3():
    Threshold = np.arange(1, 5.5, 0.5)
    # data_mat, R_mat = load_data()
    data = ps.read_csv('../ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    R = data.pivot_table(values='rating', columns=['movieId'], index=['userId'], fill_value=0)
    R_mat = R.as_matrix()
    R_index = R.index
    R_column = R.columns
    data_mat = data.as_matrix()

    N = len(data_mat)
    split_N = np.arange(N)
    kf = KFold(n_splits=10)
    Avg_error_train = np.zeros(10)
    Avg_error_test = np.zeros(10)
    counter = 0
    
    fp_rate_train = np.zeros((len(Threshold), 10))
    tp_rate_train = np.zeros((len(Threshold), 10))
    fp_rate_test = np.zeros((len(Threshold), 10))
    tp_rate_test = np.zeros((len(Threshold), 10))
    precision_tot = np.zeros((len(Threshold), 10))
    recall_tot = np.zeros((len(Threshold), 10))

    for train_index, test_index, in kf.split(split_N):

        data_train, data_test = data_mat[train_index], data_mat[test_index]
        train_df = ps.DataFrame(data_train, columns=['userId', 'movieId', 'rating', 'timestamp'])
        test_df = ps.DataFrame(data_test, columns=['userId', 'movieId', 'rating', 'timestamp'])
        R_train = ps.pivot_table(train_df, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
        R_test = ps.pivot_table(test_df, values='rating', columns=['movieId'], index=['userId'], fill_value=0)
        R_train_new = R_train.reindex(index=R_index, columns=R_column, fill_value=0)
        train_mat = R_train_new.as_matrix()

        W_mat = train_mat.copy()
        W_mat[W_mat > 0] = 1

        U, V, _ = help_functions.nmf(R_mat, 100, weight=W_mat)

        R_predict = np.dot(U, V)
        R_predict = ps.DataFrame(R_predict, index=R_index, columns=R_column)
        R_predict_train = R_predict.reindex(index=R_train.index, columns=R_train.columns, fill_value=0)
        R_predict_test = R_predict.reindex(index=R_test.index, columns=R_test.columns, fill_value=0)


        y_train_real = R_train.as_matrix()
        y_train_pre = R_predict_train.as_matrix()
        y_test_real = R_test.as_matrix()
        y_test_pre = R_predict_test.as_matrix()

        Avg_error_train[counter] = mean_absolute_error(y_train_real, y_train_pre)
        Avg_error_test[counter] = mean_absolute_error(y_test_real, y_test_pre)


    #     #part 3
        for i, t in enumerate(Threshold):
            tp_train = np.sum(y_train_real[y_train_pre >= t] >= t)
            fp_train = np.sum(y_train_real[y_train_pre >= t] < t)
            fn_train = np.sum(y_train_real[y_train_pre < t] >= t)
            tn_train = np.sum(y_train_real[y_train_pre < t] < t)

            tp_test = np.sum(y_test_real[y_test_pre >= t] >= t)
            fp_test = np.sum(y_test_real[y_test_pre >= t] < t)
            fn_test = np.sum(y_test_real[y_test_pre < t] >= t)
            tn_test = np.sum(y_test_real[y_test_pre < t] < t)

            fp_rate_train[i][counter] = fp_train / (float(fp_train + tn_train) + 1e-6)
            tp_rate_train[i][counter] = tp_train / (float(tp_train + fn_train) + 1e-6)
            fp_rate_test[i][counter] = fp_test / (float(fp_test + tn_test) + 1e-6)
            tp_rate_test[i][counter] = tp_test / (float(tp_test + fn_test) + 1e-6)

            precision_tot[i][counter] = tp_train / (float(tp_train + fp_train) + 1e-6)  # calculating precision
            recall_tot[i][counter] = tp_test / (float(tp_test + fp_test) + 1e-6)  # calculating recall

            print (tp_train, fp_train, fn_train, tn_train)


        counter = counter + 1

    max_error_train = np.amax(Avg_error_train)
    min_error_train = np.min(Avg_error_train)
    mean_error_train = np.mean(Avg_error_train)
    max_error_test = np.amax(Avg_error_test)
    min_error_test = np.min(Avg_error_test)
    mean_error_test = np.mean(Avg_error_test)

    precision = np.zeros (len(Threshold))
    recall = np.zeros (len(Threshold))
    fp_train_f = np.zeros (len(Threshold))
    fp_test_f = np.zeros (len(Threshold))
    tp_train_f = np.zeros (len(Threshold))
    tp_test_f = np.zeros (len(Threshold))

    for i in range(len(Threshold)):
        fp_train_f[i] = np.mean(fp_rate_train[i])
        fp_test_f[i] = np.mean(fp_rate_test[i])
        tp_train_f[i] = np.mean(tp_rate_train[i])
        tp_test_f[i] = np.mean(tp_rate_test[i])
        precision[i] = np.mean(precision_tot[i])
        recall[i] = np.mean(recall_tot[i])

    print(precision)
    print(recall)

    print("The average error for the 10 train group is ", mean_error_train)
    print("The max average error for the 10 train group is ", max_error_train)
    print("The min average error for the 10 train group is ", min_error_train)
    print("The average error for the 10 test group is ", mean_error_test)
    print("The max average error for the 10 test group is ", max_error_test)
    print("The min average error for the 10 test group is ", min_error_test)


    #ROC plot

    if not os.path.exists('../Graphs/pro_3'):
        os.makedirs('../Graphs/pro_3')

    plt.figure(1)
    # plt.plot(fp_train_f, tp_train_f)
    plt.plot(fp_train_f, tp_train_f)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False_Positive_Rate')
    plt.ylabel('True_Positive_Rate')
    plt.title('ROC curve (train_data)')
    plt.show()
    plt.savefig('../Graphs/pro_3/ROC_train.png')    

    plt.figure(2)
    plt.plot(fp_test_f, tp_test_f)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False_Positive_Rate')
    plt.ylabel('True_Positive_Rate')
    plt.title('ROC curve (test_data)')
    plt.show()
    plt.savefig('../Graphs/pro_3/ROC_test.png')

    plt.figure(3)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision over recall')
    plt.show()
    plt.savefig('../Graphs/pro_3/PR.png')







