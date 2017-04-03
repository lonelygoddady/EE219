"""
PROBLEM 3 Objective
Fit a linear regression model with new features to predict # of tweets in the next
hour, from data in the PREVIOUS hour.

New Fetures from the paper in the preview:
1. number of tweets
2. number of retweets
3. number of followers
4. max numbber of followers
5. time of the data (24 hours represent the day)

6. ranking score
7. impression count
8.  number of user
9.  number of verified user
10.  user mention
11.  URL mention
12.  list count
13.  Max list
14.  Friends count
15.  number of long tweets

Attention: This is NOT the order the following program uses

Model: Random Forest

Explain your model's training accuracy and the significance of each feature using
the t-test and P-value results of fitting the model.
"""

import datetime
import help_functions
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

# Remember this order!
features = ['ranking', 'friends_count', 'impression', 'URL', 'list_num', 'long_tweet', 'day_hour', 'tweets_count',
 'verified_user', 'user_count', 'Max_list','retweets_count', 'followers', 'user_mention', 'max_followers']

input_files = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']  # tweet-data file names
#input_files = ['gohawks']  # tweet-data file names

for file_name in input_files:
    tweets = open('./tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')

    ######### Perform linear regression for each hashtag file #######
    hour_wise_data = help_functions.features_extraction(tweets)
    X, Y = [], []

    print (hour_wise_data.values()[0])
    for prev_hour in hour_wise_data.keys():
        hour = datetime.datetime.strptime(prev_hour, "%Y-%m-%d %H:%M:%S")
        next_hour = unicode(hour + datetime.timedelta(hours=1))

        if next_hour in hour_wise_data.keys():
            Y.append(hour_wise_data[next_hour]['tweets_count'])
            X.append(hour_wise_data[prev_hour].values())

    # random forest regression analysis
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    results = model.fit(X, Y)
    print('The accuracy is: ' + str(results.score(X, Y)))
    print('Feature importance are: ' + str(results.feature_importances_))
    t_val = results.feature_importances_

    # lasso/ridge regression analysis
    # X_const = sm.add_constant(X)
    # LR_model = sm.OLS(Y, X_const)
    # LR_results = LR_model.fit_regularized(L1_wt=0, alpha=0.01)
    # print(LR_results.summary())
    # print('P values: ' + str(LR_results.pvalues))
    # print('T values: ' + str(LR_results.tvalues))
    # t_val = LR_results.tvalues.tolist()
    # t_val.pop(0) # pop the constant item
    #
    # t_val = [abs(i) for i in t_val]

    # retrive indeces of top three T values in the order of third, second, first
    indeces = sorted(range(len(t_val)), key=lambda i: t_val[i])[-3:]
    print('top three features are:')
    print(features[indeces[2]], features[indeces[1]], features[indeces[0]])

    # Extract top three features
    first_fea, sec_fea, third_fea = [], [], []

    for i in range(len(X)):
        first_fea.append(X[i][indeces[2]])
        sec_fea.append(X[i][indeces[1]])
        third_fea.append(X[i][indeces[0]])

    # Plot diagrams:
    help_functions.plot_feature(first_fea, Y, file_name, features[indeces[2]])
    help_functions.plot_feature(sec_fea, Y, file_name, features[indeces[1]])
    help_functions.plot_feature(third_fea, Y, file_name, features[indeces[0]])

    print('-'*50)

    # Top three features for different topics:
    # gohawks:        X1, X4, X12
    # ('ranking', 'URL', 'user_mention')

    # gopatriots:     X1, X4, X13
    # ('ranking', 'URL', 'max_followers')

    # nfl:            X12, X9, X7
    # ('user_mention', 'user_count', 'tweets_count')

    # patriots:       X1, X11, X4
    # ('ranking', 'followers', 'URL')

    # sb49:           X12, X11, X10
    # ('user_mention', 'followers', 'retweets_count')

    # Superbowl:      X10, X4, X13
    # ('retweets_count', 'URL', 'max_followers')

