import math
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor

path_problem1 = r'../Graphs/Problem1/'
path_problem2a = r'../Graphs/Problem2a/'
path_problem2b = r'../Graphs/Problem2b/'
path_problem2c = r'../Graphs/Problem2c/'
path_problem3 = r'../Graphs/Problem3/'
path_problem4 = r'../Graphs/Problem4/'

def plot_generation(network_data):
    if not os.path.exists(path_problem1):
        os.makedirs(path_problem1)
    backup_size = np.zeros((5,22))
    for i in range(len(network_data)):
        if network_data[i][0] < 4:
            workflow = network_data[i][3]
            day = (network_data[i][0]-1)*7+network_data[i][1]
            backup_size[workflow][day] = backup_size[workflow][day] + network_data[i][5]

    backup_size = backup_size[0:20]
    day_axis = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    for i in range(len(backup_size)):
        backup_size_single = np.transpose(backup_size[i])
        backup_size_single = backup_size_single[0:20]
        plt.figure(i)
        plt.plot(day_axis, backup_size_single)
        s = 'workflow_'+repr(i)
        plt.title(s)
        plt.xlabel('Period')
        plt.ylabel('Back_up_size')
        s = '../Graphs/Problem1/workflow'+repr(i)+'_backup_size'
        plt.savefig(s)

        
def linear_regression(features, target, file_name):    
    if not os.path.exists(file_name):
        os.makedirs(file_name)
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

    plt.figure(1)
    plt.plot(Predict, target, '.')
    plt.xlabel("Predicted Backup size")
    plt.ylabel("Actual Backup Size")
    plt.title('Predicted values vs. Actual values')
    # plt.plot([min(f), max(features)], [min(features), max(features)], 'k--', lw=4)
    plt.plot(Predict, Predict, 'k--', lw=1, c='red')
    lin_file_name = file_name + ".png"
    plt.savefig(lin_file_name)

    plt.figure(2)
    residual = Predict - target
    plt.plot(Predict, residual, '.')
    plt.xlabel("Predicted values")
    plt.ylabel("Residual values")
    plt.title('Predicted values vs. residual values')
    plt.hlines(y=0, xmin=min(Predict), xmax=max(Predict))
    residual_file_name = file_name + "_residual.png"
    plt.savefig(residual_file_name)

    return (np.mean(score) ** 0.5)

def poly_regression(features,target,degree_N, file_name):

    poly_RMSEs = [] 
    poly_RMSEs_test = []
    poly_RMSEs_train = []
    poly_RMSEs_all = []
    
    poly = LinearRegression()
    HX_train, HX_test, Hy_train, Hy_test = train_test_split(features, target, train_size=0.9)

    print("Processing Polynomial Analysis...")
    for degree in range(1,degree_N+1):
        kfold = KFold(n_splits = 10, shuffle = True)
        poly_F = PolynomialFeatures(degree = degree, interaction_only = True, include_bias = False)
        X_poly_all = poly_F.fit_transform(features)
        score = cross_val_score(poly, X_poly_all, target, cv = kfold, scoring = 'neg_mean_squared_error')
        poly_RMSEs.append(math.sqrt(np.mean(abs(score))))

    print('MSEs for degree 1 to 10: ')
    print(poly_RMSEs)

    min_degree = 1 + np.argmin(poly_RMSEs)
    buffer = 'The min RMSE over all data in the polynomial regression is ' + str(min(poly_RMSEs)) + ' degree: ' \
            + str(min_degree)
    print(buffer)

    # ###### Polynomial regression plotting code #######
    plt.figure(3)
    CV_RMSE, = plt.plot(range(1,degree_N+1),poly_RMSEs, '-o', c = "green", label = "Cross Valid")
    # plt.legend([Test_set,Train_set,Overall,CV_RMSE], ["Test_set","Train_set","Overall", "CV"], loc = 1)
    plt.xlabel("degree number")
    plt.ylabel("RMSE values")
    plt.title("RMSE values vs. Degree number")
    plt.savefig(file_name + '.png')

    poly_F = PolynomialFeatures(degree = min_degree, interaction_only = True, include_bias = False)
    X_poly_train = poly_F.fit_transform(HX_train)
    poly.fit(X_poly_train, Hy_train)
    X_poly_all = poly_F.fit_transform(features)
    Y_poly_all = poly.predict(X_poly_all)

    plt.figure(4)
    plt.plot(Y_poly_all, target, '.')
    plt.xlabel("Predicted Backup size")
    plt.ylabel("Actual Backup Size")
    plt.title('Predicted values vs. Actual values')
    plt.plot(Y_poly_all, Y_poly_all, 'k--', lw=1, c='red')
    plt.savefig(file_name + 'Predicted_vs_Actual.png')
    
    plt.figure(5)
    plt.plot(Y_poly_all, Y_poly_all - target, '.')
    plt.xlabel("Predicted Backup size")
    plt.ylabel("Residuals")
    plt.title('Predicted values vs. Residuals values')
    plt.hlines(y=0, xmin=min(Y_poly_all), xmax=max(Y_poly_all))
    plt.savefig(file_name + 'Predicted_vs_Residuals.png')

    return min(poly_RMSEs)

def random_forest(feature, target):
    if not os.path.exists(path_problem2b):
        os.makedirs(path_problem2b)
    print("\nQuestion 2b: Processing Random Forest Analysis...")
    depth = range(4, 16)
    clfResult = []
    for d in range(len(depth)):
        clf = RandomForestRegressor(n_estimators=20, max_depth=depth[d])
        score = cross_val_score(clf, feature, target, cv=10, scoring="neg_mean_squared_error")
        clfResult.append((score.mean()*-1)**.5)
        # print('Depth vs RMSR {0}/{1} finished'.format(d+1, len(depth)))

    plt.figure(101)
    plt.plot(depth, clfResult)
    plt.title('RMSE vs Depth of Tree')
    plt.xlabel('Depth')
    plt.ylabel('Root Mean Square Error')

    bestDepth = depth[clfResult.index(min(clfResult))]

    plt.savefig('../Graphs/Problem2b/RMSE vs Depth of Tree.png')

    # finding different # of trees

    treeRange = range(20, 200, 10)
    clfResult = []
    for t in range(len(treeRange)):
        clf = RandomForestRegressor(n_estimators=treeRange[t], max_depth=4)
        score = cross_val_score(clf, feature, target, cv=10, scoring="neg_mean_squared_error")
        clfResult.append((score.mean()*-1)**.5)
        # print('Tree vs RMSR {0}/{1} finished'.format(t+1, len(treeRange)))

    bestTree = treeRange[clfResult.index(min(clfResult))]

    plt.figure(102)
    plt.plot(treeRange, clfResult)
    plt.title('RMSE vs Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Root Mean Square Error')

    plt.savefig('../Graphs/Problem2b/RMSE vs No of Trees.png')

    # getting the best estimate RMSE
    bestRF = RandomForestRegressor(n_estimators=bestTree, max_depth=bestDepth)
    predictedRF = cross_val_predict(bestRF, feature, target, cv=10)
    bestScore = cross_val_score(bestRF, feature, target, cv=10, scoring="neg_mean_squared_error")

    plt.figure(103)
    plt.scatter(target, predictedRF)
    plt.xlabel("Actual Median Value")
    plt.ylabel("Predicted Median Value")
    plt.title('Fitted values vs Actual Values')
    plt.plot([min(target), max(target)], [min(target), max(target)], 'r-.', lw=3)

    plt.savefig('../Graphs/Problem2b/Fitted vs Actual.png')

    plt.figure(104)
    plt.scatter(predictedRF, predictedRF - target, c='b', s=20, alpha=0.5)
    plt.xlabel('Predicted Median Value')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Median Value')
    plt.hlines(y=0, xmin=min(predictedRF), xmax=max(predictedRF))

    plt.savefig("../Graphs/Problem2b/Residuals vs Predicted Median Value.png")

    print ("******RANDOM FOREST REGRESSION RESULT******")
    print ("Optimized Max Depth: " + str(bestDepth))
    print ("Optimized of Maximum Tree: " + str(bestTree))
    print ("Root Mean Squared Error: " + str((bestScore.mean()*-1)**.5))

    # plt.show()


def neuralNetworkRegression(features, target):
    if not os.path.exists(path_problem2c):
        os.makedirs(path_problem2c)
    # x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.9, random_state=42)
    print("\nQuestion 2c: Processing Neural Network Regression...")
    print("This step may take up to 10 mins to finish, please be patient...")
    neuralNetworkError = []
    # unitSet = range(2, 40, 2)
    unitSet = range(100, 1000, 50)

    for x in range(len(unitSet)):
        neuralSet = MLPRegressor(alpha=1e-4, random_state=42, hidden_layer_sizes=(unitSet[x],))
        score = cross_val_score(neuralSet, features, target, cv=10, scoring="neg_mean_squared_error")
        neuralNetworkError.append((score.mean()*-1)**.5)
        # print('data number {0}/{1} finished'.format(x+1, len(unitSet)))


    plt.figure(105)
    plt.plot(unitSet, neuralNetworkError)
    plt.title('RMSE vs Number of Hidden Units')
    plt.xlabel('Hidden Units')
    plt.ylabel('Root Mean Square Error')
    plt.savefig("../Graphs/Problem2c/RMSE vs Number of Hidden Units1.png")

    bestUnit = unitSet[neuralNetworkError.index(min(neuralNetworkError))]
    bestNeural = MLPRegressor(alpha=1e-4, random_state=42, hidden_layer_sizes=(bestUnit, ))
    predictedNeural = cross_val_predict(bestNeural, features, target, cv=10)

    plt.figure(106)
    plt.scatter(target, predictedNeural)
    plt.xlabel("Target Value")
    plt.ylabel("Predicted Value")
    plt.title('Predicted Value vs Target Values')
    plt.plot([min(target), max(target)], [min(target), max(target)], 'r-.', lw=3)

    plt.savefig('../Graphs/Problem2c/Fitted vs Actual1.png')

    plt.figure(107)
    plt.scatter(predictedNeural, predictedNeural - target, c='b', s=20, alpha=0.5)
    plt.xlabel('Predicted Value')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Value')
    plt.hlines(y=0, xmin=min(predictedNeural), xmax=max(predictedNeural))

    plt.savefig('../Graphs/Problem2c/Residuals vs Predicted1.png')

    print ("\n********Neural Network Regression Result********")
    print ("Optimized Hidden Unit: " + str(bestUnit))
    print ("Root Mean Squared Error: " + str(min(neuralNetworkError)))
    # plt.show()


def boston_housing_pr4():

    # data processing
    Housing = pd.read_csv('housing_data.csv', header=None)
    Housing.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGS', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                       'MEDV']
    Housing_T = Housing['MEDV']
    del Housing['MEDV']
    HX_train, HX_test, Hy_train, Hy_test = train_test_split(Housing, Housing_T, train_size=0.9, random_state=42)

    # build the linear regression model
    H_lr = LinearRegression()
    H_lr.fit(HX_train, Hy_train)
    H_pre_test = H_lr.predict(HX_test)
    H_pre_all = H_lr.predict(Housing)
    H_lin_rmse = math.sqrt(mean_squared_error(Hy_test, H_pre_test))

    buf = 'Linear regression coefficient for Housing data is: '
    print(buf)
    print(H_lr.coef_)

    buf = 'RMSE for Housing test data is: '
    print(buf + str(H_lin_rmse))

    # calculate the 10-fold linear regression cross validation rmse
    score = cross_val_score(H_lr, Housing, Housing_T, cv=10, scoring='neg_mean_squared_error')
    buf = '10 Fold validate RMSE for Housing data is: '
    print(buf + str(((-1) * score.mean()) ** 0.5))
    print('\n')

    # polynomial regression
    poly = LinearRegression()
    poly_RMSEs = []
    print("Processing Polynomial Analysis...")
    for degree in range(10):
        poly_F = PolynomialFeatures(degree=degree, interaction_only=True)
        X_poly_train = poly_F.fit_transform(HX_train)
        poly.fit(X_poly_train, Hy_train)
        X_poly_test = poly_F.fit_transform(HX_test)
        Y_poly_test = poly.predict(X_poly_test)
        Housing_poly = poly_F.fit_transform(Housing)
        H_poly_test = poly.predict(Housing_poly)
        # print(Y_poly_test)
        RMSE_poly_test = math.sqrt(mean_squared_error(Y_poly_test, Hy_test))

        buf = 'Poly RMSE for Housing test data is: '
        print(buf + str(RMSE_poly_test))

        # calculate the 10-fold poly regression cross validation rmse
        score = cross_val_score(poly, Housing_poly, Housing_T, cv=10, scoring='neg_mean_squared_error')
        RMSE_score = str(((-1) * score.mean()) ** 0.5)
        buf = '10 Fold validate RMSE for Housing data is: '
        print(buf + RMSE_score)
        poly_RMSEs.append(float(RMSE_score))

    buf = 'The min RMSE in the polynomial regression is ' + str(min(poly_RMSEs)) + ' degree: ' + str(
        1 + np.argmin(poly_RMSEs))
    print(poly_RMSEs)
    print(buf)


def boston_housing_pr5(features, target):
##part a)

    alpha_range = [1,0.1,0.01,0.001]


    clf = RidgeCV(normalize = True,scoring ='mean_squared_error',alphas = alpha_range, cv=10)
    clf.fit(features,target)
    scores = clf.predict(features)
    RMSE_ridge=np.sqrt(mean_squared_error(target,scores))
    alpha_min = clf.alpha_
    print('Problem5a: Ridge regression via 10-fold cross validation')
    s = 'When alpha = ' + repr(alpha_min) + ', the ridge regression reaches the minimum RMSE = ' + repr(RMSE_ridge)
    print(s)
    s = 'its coefficients are ['
    for i in range(len(clf.coef_)-1):
        s = s + repr(clf.coef_[i]) + ', '
    s = s + repr(clf.coef_[len(clf.coef_)-1]) + ']'
    print(s)


##part b)
    alpha_range = [1,0.1,0.01,0.001]


    clf = LassoCV(normalize = True,alphas = alpha_range, cv=10)
    clf.fit(features,target)
    scores = clf.predict(features)
    RMSE_lasso=np.sqrt(mean_squared_error(target,scores))
    alpha_min = clf.alpha_
    print('Problem5b: Lasso regression via 10-fold cross validation')
    s = 'When alpha = ' + repr(alpha_min) + ', the lasso regression reaches the minimum RMSE = ' + repr(RMSE_lasso)
    print(s)
    s = 'its coefficients are ['
    for i in range(len(clf.coef_)-1):
        s = s + repr(clf.coef_[i]) + ', '
    s = s + repr(clf.coef_[len(clf.coef_)-1]) + ']'
    print(s)

def problem2a(features,target):
    # build the linear regression model 
    RMSE_lin_net = linear_regression(features, target, "../Graphs/Problem2a/LR_vs_Actual")

def problem3(features,target):
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

def problem4():
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