"""
PROBLEM 2 Objective
Fit a linear regression model with 5 features to predict # of tweets in the next
hour, from data in the PREVIOUS hour.

Fetures:
1. number of tweets
2. number of retweets
3. sum of followers of the users posting the hashtag
4. max numbber of followers of the user posting the hashtag
5. time of the dat (24 hours represent the day)

Explain your model's training accuracy and the significance of each feature using
the t-test and P-value results of fitting the model.
"""
import json
import datetime
import statsmodels.api as sm

input_files = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']  # tweet-data file names
# input_files = ['patriots', 'sb49', 'superbowl']  # tweet-data file names

for file_name in input_files:
    tweets = open('./tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')
    first_tweet = json.loads(tweets.readline())  # make tweets into dictionary

    tweets.seek(0, 0)  # set start point to 0 again

    start_time = first_tweet['firstpost_date']  # get the first tweet start time
    st_data_time = datetime.datetime.fromtimestamp(start_time)
    start_time -= (
        st_data_time.minute * 60 + st_data_time.second)  # rectify the time offset to make the hour window align with day hours
    end_time_of_window = start_time + 3600  # window to keep track of number of hours of data

    # regression features
    user_ids_hour = []  # who has tweeted in this hour
    current_hour_count = 0
    number_hour_retweets = 0
    number_hour_followers = 0
    number_max_followers = 0

    # Structure that holds hour wise features
    hour_wise_data = {}

    ####### Loop through each tweet and calculate statistics #######
    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        if end_time < end_time_of_window:
            #   calculate features in the previous hour
            current_hour_count += 1
            number_hour_retweets += tweet_data['metrics']['citations']['total']

            if tweet_data['tweet']['user']['id'] not in user_ids_hour:
                user_ids_hour.append(tweet_data['tweet']['user']['id'])
                number_hour_followers += tweet_data['author']['followers']
                number_max_followers = tweet_data['author']['followers'] if \
                    number_max_followers < tweet_data['author']['followers'] else number_max_followers
        else:
            # get the data's time in datetime format and slice them into hours
            date_time = datetime.datetime.fromtimestamp(end_time)
            actual_time = datetime.datetime(date_time.year, date_time.month, date_time.day,
                                            date_time.hour, 0, 0)
            modified_actual_time = unicode(actual_time)

            if modified_actual_time not in hour_wise_data.keys():
                hour_wise_data[modified_actual_time] = {'tweets_count': 0, 'retweets_count': 0,
                                                        'followers': 0, 'max_followers': 0, 'day_hour': 0}

            hour_wise_data[modified_actual_time]['tweets_count'] += current_hour_count
            hour_wise_data[modified_actual_time]['retweets_count'] += number_hour_retweets
            hour_wise_data[modified_actual_time]['followers'] += number_hour_followers
            hour_wise_data[modified_actual_time]['max_followers'] += number_max_followers
            hour_wise_data[modified_actual_time]['day_hour'] = actual_time.hour

            # reinitialize variables (*** this part is tricky ****)
            user_ids_hour = [tweet_data['tweet']['user']['id']]
            current_hour_count = 1
            number_hour_retweets = tweet_data['metrics']['citations']['total']
            number_hour_followers = tweet_data['author']['followers']
            number_max_followers = tweet_data['author']['followers']

            st_data_time = datetime.datetime.fromtimestamp(end_time)
            start_time = end_time - (st_data_time.minute * 60 + st_data_time.second)  # rectify the time offset
            end_time_of_window = start_time + 3600

    ######### Perform linear regression for each hashtag file #######
    X, Y = [], []

    print (hour_wise_data.values()[0])
    for prev_hour in hour_wise_data.keys():
        hour = datetime.datetime.strptime(prev_hour, "%Y-%m-%d %H:%M:%S")
        next_hour = unicode(hour + datetime.timedelta(hours=1))

        if next_hour in hour_wise_data.keys():
            Y.append(hour_wise_data[next_hour]['tweets_count'])
            X.append(hour_wise_data[prev_hour].values())

    X = sm.add_constant(X)
    LR_model = sm.OLS(Y, X)
    LR_results = LR_model.fit()
    print(LR_results.summary())
    print('P values: ' + str(LR_results.pvalues))
    print('T values: ' + str(LR_results.tvalues))
