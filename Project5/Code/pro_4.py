"""
PROBLEM 4 Objective
apply 10-fold-cross-validattion linear regression on three period:
1. Before Feb. 1, 8:00 a.m.
2. Between Feb. 1, 8:00 a.m. and 8:00 p.m.
3. After Feb. 1, 8:00 p.m.

New Fetures from the paper in the preview:
1. number of tweets
2. number of retweets
3. number of followers
4. max numbber of followers
5. time of the data (24 hours represent the day)
6. ranking score
7. impression count
# 8. max favorite count
9. favorite count
10.  number of user
11.  number of verified user
12.  user mention
13.  URL mention
# 14.  max list
15.  number of long tweets

(feature 8 and 14 are temporarily excluded)

Explain your model's training accuracy and the significance of each feature using
the t-test and P-value results of fitting the model.
"""

import datetime
import statsmodels.api as sm
import help_functions
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline


def predictor_label_extraction(hour_wise_data):
    x, y = [], []

    # print (hour_wise_data.values()[0])
    for prev_hour in hour_wise_data.keys():
        hour = datetime.datetime.strptime(prev_hour, "%Y-%m-%d %H:%M:%S")
        next_hour = unicode(hour + datetime.timedelta(hours=1))

        if next_hour in hour_wise_data.keys():
            y.append(hour_wise_data[next_hour]['tweets_count'])
            x.append(hour_wise_data[prev_hour].values())
    return x, y


def cross_validation(x, y):
    all_prediction_errors = []
    kf = KFold(n_splits=10, random_state=42)

    X, Y = np.array(x), np.array(y)

    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = Y[train_index], Y[test_index]

        # Ridge/Lasso regression
        # train_x = sm.add_constant(train_x)
        # test_x = sm.add_constant(test_x, has_constant='add')
        # model = sm.OLS(train_y, train_x)
        # results = model.fit_regularized(L1_wt=1, alpha=0.01)

        # neural net
        # model = MLPRegressor(random_state=42)
        # results = model.fit(train_x, train_y)

        # polynomial ridge
        # model = make_pipeline(PolynomialFeatures(3), Ridge(alpha=0.01, random_state=42))
        # results = model.fit(train_x, train_y)

        #random forest
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        results = model.fit(train_x, train_y)

        # prediction
        test_y_predicted = results.predict(test_x)

        prediction_error = abs(test_y_predicted - test_y)
        prediction_error = np.mean(prediction_error)
        print('Prediction error for this fold is: ' + str(prediction_error))
        all_prediction_errors.append(prediction_error)

    return np.mean(all_prediction_errors)

input_files = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']  # tweet-data file names
# input_files = ['patriots', 'sb49', 'superbowl']  # tweet-data file names

for file_name in input_files:
    tweets = open('./tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')

    ######### Break hourwise data into three periods #######
    hour_wise_data = help_functions.features_extraction(tweets)
    hourwise_period1, hourwise_period2, hourwise_period3 = {}, {}, {}
    break_point1 = datetime.datetime(2015, 2, 1, 8, 0, 0)
    break_point2 = datetime.datetime(2015, 2, 1, 20, 0, 0)

    for i in hour_wise_data:
        data_time = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        if data_time < break_point1:
            hourwise_period1[i] = hour_wise_data[i]
        elif data_time >= break_point1 and data_time <= break_point2:
            hourwise_period2[i] = hour_wise_data[i]
        elif data_time > break_point2:
            hourwise_period3[i] = hour_wise_data[i]

    ######### perform 10-fold-cross-validation over period wise data #######
    X1, Y1 = predictor_label_extraction(hourwise_period1)
    X2, Y2 = predictor_label_extraction(hourwise_period2)
    X3, Y3 = predictor_label_extraction(hourwise_period3)

    error1 = cross_validation(X1, Y1)
    error2 = cross_validation(X2, Y2)
    error3 = cross_validation(X3, Y3)

    print('Average prediction errors for period 1 2 and 3 of hashtag ' + str(file_name))
    print(error1, error2, error3)

    ######### perform 10-fold-cross-validation over all data #######
    x_total, y_total = predictor_label_extraction(hour_wise_data)
    total_error = cross_validation(x_total, y_total)
    print('Average prediction error for the whole period is: ' + str(total_error))