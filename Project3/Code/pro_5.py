from __future__ import print_function
import os
from sklearn.model_selection import KFold
import numpy as np
import help_functions
import pandas as ps
import matplotlib.pyplot as plt


def pro_5(table):
    L = np.arange(1, 100)
    Threshold = 3
    R = table.as_matrix()
    hitRate = np.zeros((len(L), R.shape[0]))
    falseRate = np.zeros((len(L), R.shape[0]))
    kf = KFold(n_splits=10)
    N = len(R)
    split_N = np.arange(N)
    # print(R.shape[0])
    for train_index, test_index, in kf.split(split_N):
        data_train = np.zeros(R.shape)
        data_test = np.zeros(R.shape)

        data_train[train_index], data_test[test_index] = R[train_index], R[test_index]
        train_weight = np.sign(data_train)
        U, V, _ = help_functions.nmf(X=train_weight, k=100, weight=R)
        R_predict = np.dot(U, V)
        R_predict = R * R_predict
        R_predict = ps.DataFrame(R_predict, index=table.index, columns=table.columns)
        data_predicted_test = R_predict.loc[test_index + 1]

        # now we need to numerator over each user
        for index, row in data_predicted_test.iterrows():
            row = row.sort_values(0, ascending=False)
            for l in L:
                k = 0
                hit = 0
                for rate_column in row.index.values:
                    if R[index-1][rate_column-1] != 0:
                        if R[index-1][rate_column-1] > Threshold:
                            hit += 1
                        k += 1
                    if k == l:
                        break
                hitR = hit / l
                hitRate[l-1][index-1] = hitR
                falseRate[l-1][index-1] = 1 - hitR

    hitArray = []
    falseArray = []
    if not os.path.exists('../Graphs/pro_3'):
        os.makedirs('../Graphs/pro_3')

    for l in L:
        hitArray.append(np.mean(hitRate[l-1]))
        falseArray.append(np.mean(falseRate[l-1]))
        if l == 5:
            print("hitRate for L = 5 is: ", hitArray[l-1])
            print("falseRate for L = 5 is: ", falseArray[l-1])

    plt.figure(24)
    plt.scatter(falseArray, hitArray)
    plt.plot(falseArray, hitArray)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False-Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.title('Hit Rate VS False Alarm Rate')
    plt.show()
    plt.savefig('../Graphs/pro_5/Rate.png')



