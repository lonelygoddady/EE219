import csv
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm


# change Sunday-1, Monday-2, Tuesday-3....Saturday-7
def Day_change(data):
    for x in range(len(data)):
        if data[x][1] == "Sunday":
            data[x][1] = 1
        elif data[x][1] == "Monday":
            data[x][1] = 2
        elif data[x][1] == "Tuesday":
            data[x][1] = 3
        elif data[x][1] == "Wednesday":
            data[x][1] = 4
        elif data[x][1] == "Thursday":
            data[x][1] = 5
        elif data[x][1] == "Friday":
            data[x][1] = 6
        elif data[x][1] == "Saturday":
            data[x][1] = 7
        else:
            print "Day Datatype error"
            data[x][1] = 8

#change work_flow number
def Work_flow_change(data):
    for x in range(len(data)):
        data[x][3] = int(data[x][3][10:])

#file name update
def File_name_change(data):    
    for x in range(len(data)):
        data[x][4] = int(data[x][4][5:])   

#convert string to number for matrix        
def int_convert(data):
    for x in range(len(data)):
        for y in range(len(data[0])-2):
            data[x][y] = int(data[x][y])
        data[x][5] = float(data[x][5])
        data[x][6] = int(data[x][6])
    return data
    
def data_converter(data):
    Day_change(data)
    Work_flow_change(data)
    File_name_change(data)
    data = int_convert(data)
    return data

def linear_regression(features, target, file_name):
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=42)
    Net_lr = LinearRegression()
    Net_lr.fit(X_train, y_train)
    Predict = cross_val_predict(Net_lr,features,target,cv=10)

    buf = 'Linear regression coefficient for network data is: '
    print(buf)
    print(Net_lr.coef_)

    # calculate the 10-fold cross validation rmse
    score = (Predict - target)**2
    buf = '10 Fold validate RMSE over data set is: '
    print(buf + str(np.mean(score) ** 0.5))
    print('\n')

    ##### Linear Regression stats summary and graph plotting ####
    # est = sm.OLS(target, features).fit()  # panda OLS library used to build model on entire dataset and provide stats on variable
    # print est.summary()

    # plt.figure(1)
    # plt.plot(Predict, target, '.')
    # plt.xlabel("Predicted Backup size")
    # plt.ylabel("Actual Backup Size")
    # plt.title('Predicted values vs. Actual values')
    # # plt.plot([min(f), max(features)], [min(features), max(features)], 'k--', lw=4)
    # plt.plot(Predict, Predict, 'k--', lw=1, c='red')
    # lin_file_name = file_name + ".png"
    # plt.savefig(lin_file_name)

    # plt.figure(2)
    # residual = Predict - target
    # plt.plot(Predict, residual, '.')
    # plt.xlabel("Predicted values")
    # plt.ylabel("Residual values")
    # plt.title('Predicted values vs. residual values')
    # plt.hlines(y=0, xmin=min(Predict), xmax=max(Predict))
    # residual_file_name = file_name + "_residual.png"
    # plt.savefig(residual_file_name)

    return (np.mean(score) ** 0.5)

def poly_regression(features,target,degree_N, file_name):

    poly_RMSEs = [] 
    
    poly = LinearRegression()
    HX_train, HX_test, Hy_train, Hy_test = train_test_split(features, target, train_size=0.9)

    print("Processing Polynomial Analysis...")
    for degree in range(1,degree_N+1):
        kfold = KFold(n_splits = 10, shuffle = True)
        poly_F = PolynomialFeatures(degree = degree, interaction_only = True, include_bias = False)
        X_poly_all = poly_F.fit_transform(features)
        score = cross_val_score(poly, X_poly_all, target, cv = kfold, scoring = 'neg_mean_squared_error')
        poly_RMSEs.append(math.sqrt(np.mean(abs(score))))

    print(poly_RMSEs)

    min_degree = 1 + np.argmin(poly_RMSEs)
    buffer = 'The min RMSE over all data in the polynomial regression is ' + str(min(poly_RMSEs)) + ' degree: ' \
            + str(min_degree)
    print(buffer)

    # ###### Polynomial regression plotting code #######
    # plt.figure(3)
    # CV_RMSE, = plt.plot(range(1,degree_N+1),poly_RMSEs, '-o', c = "green", label = "Cross Valid")
    # # plt.legend([Test_set,Train_set,Overall,CV_RMSE], ["Test_set","Train_set","Overall", "CV"], loc = 1)
    # plt.xlabel("degree number")
    # plt.ylabel("RMSE values")
    # plt.title("RMSE values vs. Degree number")
    # plt.savefig(file_name + '.png')

    # poly_F = PolynomialFeatures(degree = min_degree, interaction_only = True, include_bias = False)
    # X_poly_train = poly_F.fit_transform(HX_train)
    # poly.fit(X_poly_train, Hy_train)
    # X_poly_all = poly_F.fit_transform(features)
    # Y_poly_all = poly.predict(X_poly_all)

    # plt.figure(4)
    # plt.plot(Y_poly_all, target, '.')
    # plt.xlabel("Predicted Backup size")
    # plt.ylabel("Actual Backup Size")
    # plt.title('Predicted values vs. Actual values')
    # plt.plot(Y_poly_all, Y_poly_all, 'k--', lw=1, c='red')
    # plt.savefig(file_name + 'Predicted_vs_Actual.png')
    
    # plt.figure(5)
    # plt.plot(Y_poly_all, Y_poly_all - target, '.')
    # plt.xlabel("Predicted Backup size")
    # plt.ylabel("Residuals")
    # plt.title('Predicted values vs. Residuals values')
    # plt.hlines(y=0, xmin=min(Y_poly_all), xmax=max(Y_poly_all))
    # plt.savefig(file_name + 'Predicted_vs_Residuals.png')

    return min(poly_RMSEs)

######################## Problem 2 ################################

# data processing
with open('network_backup_dataset.csv') as csvfile:
    reader = csv.reader(csvfile)
    network_orign = list(reader)
    network_data = list(network_orign[1:])
    network_data = data_converter(network_data)

    ## transfer list to panda object
    # network_data = zip(*network_data)
    # Network_data = pd.DataFrame({'Week': network_data[0],
    #                              'Day': network_data[1],
    #                              'Hour': network_data[2],
    #                              'WorkFlow': network_data[3],
    #                              'FileName': network_data[4],
    #                              'SizeofBackup': network_data[5],
    #                              'BackupTime': network_data[6],
    #                             })

    # print Network_data.dtypes
    # target = Network_data['SizeofBackup']
    # del Network_data['SizeofBackup']
    # features = Network_data

    target = []
    for i in range(len(network_data)):
        target.append(network_data[i][5])

    features = np.zeros((len(network_data),6))
    for x in range(len(network_data)):
        features[x][0] = network_data[x][0]
        features[x][1] = network_data[x][1]
        features[x][2] = network_data[x][2]
        features[x][3] = network_data[x][3]
        features[x][4] = network_data[x][4]
        features[x][5] = network_data[x][6]

# build the linear regression model 
RMSE_lin_net = linear_regression(features, target, "../Graphs/Problem2a/LR_vs_Actual")

######################### Problem 3 ############################## 

#separate features and target into 5 workflow subsets 
WF0_features = []
WF0_T = []
WF1_features = []
WF1_T = []
WF2_features = [] 
WF2_T = []
WF3_features = []
WF3_T = []
WF4_features = [] 
WF4_T = []

for i in range(len(features)):
    if int(features[i][3]) == 0:
        WF0_features.append(features[i])
        WF0_T.append(target[i])
    elif int(features[i][3]) == 1:
        WF1_features.append(features[i])
        WF1_T.append(target[i])
    elif int(features[i][3]) == 2:
        WF2_features.append(features[i])
        WF2_T.append(target[i])
    elif int(features[i][3]) == 3:
        WF3_features.append(features[i])
        WF3_T.append(target[i])
    elif int(features[i][3]) == 4:
        WF4_features.append(features[i])
        WF4_T.append(target[i])

RMSE0 = linear_regression(WF0_features,WF0_T, "../Graphs/Problem3/WF0_LR_vs_Actual")
RMSE1 = linear_regression(WF1_features,WF1_T, "../Graphs/Problem3/WF1_LR_vs_Actual")
RMSE2 = linear_regression(WF2_features,WF2_T, "../Graphs/Problem3/WF2_LR_vs_Actual")
RMSE3 = linear_regression(WF3_features,WF3_T, "../Graphs/Problem3/WF3_LR_vs_Actual")
RMSE4 = linear_regression(WF4_features,WF4_T, "../Graphs/Problem3/WF4_LR_vs_Actual")

RMSE_avg = np.mean([RMSE0,RMSE1,RMSE2,RMSE3,RMSE4])
print('The avg RMSE for pice-wise linear regression is: ' + str(RMSE_avg))

poly_degree = 10
RMSE_poly_net = poly_regression(features, target, poly_degree, "../Graphs/Problem3/Multi_Poly_comp") 


########################### Problem 4 ############################## 

#data processing 
Housing = pd.read_csv('housing_data.csv',header = None)
Housing.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGS','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
Housing_T = Housing['MEDV']
del Housing['MEDV']

#perform linear and polynomial regress
poly_degree = 10
RMSE_lin_housing = linear_regression(Housing, Housing_T, "../Graphs/Problem4/LR_vs_Actual")
RMSE_poly_housing = poly_regression(Housing, Housing_T, poly_degree, "../Graphs/Problem4/Multi_Poly_comp")
print('Line regression RMSE: ' + str(RMSE_lin_housing))
print('Polynomial regression RMSE' + str(RMSE_poly_housing))
 
    
    
