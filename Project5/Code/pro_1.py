"""
PROBLEM 1 Objective
1.  average number of tweets per hour
2.  average number of followers of users posting the tweets
3.  average number of retweets
4.  Plot "number of tweets in hour" over time for #SuperBowl and #NFL
"""

import json
import time
import datetime
from matplotlib import pyplot as plt

start = time.clock()
input_files = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']  # tweet-data file names
# input_files_plot = ['nfl', 'superbowl']  # tweet-data file names

# loop through each file in tweet-data folder
for file_name in input_files:
    tweets = open('./tweet_data/tweets_#{0}.txt'.format(file_name), 'rb')
    first_tweet = json.loads(tweets.readline())  # make tweets into dictionary
    # first_tweet = json.load(tweets)
    # print first_tweet
    user_ids = {}  # this dict structure store users' followers number
    number_of_tweets = len(tweets.readlines())  # get number of tweets
    number_of_retweets = 0
    tweets.seek(0, 0)  # set start point to 0 again

    current_window = 1
    start_time = first_tweet['firstpost_date']  # get the first tweet start time

    # assign the first user_id and its no. of followers
    user_ids[first_tweet['tweet']['user']['id']] = first_tweet['author']['followers']
    st_data_time = datetime.datetime.fromtimestamp(start_time)
    start_time -= (
        st_data_time.minute * 60 + st_data_time.second)  # rectify the time offset to make the hour window align with day hours
    end_time_of_window = start_time + 3600  # window to keep track of number of hours of data

    number_of_tweets_hour, number_of_retweets = [], []
    current_hour_count = 0

    # loop through each tweet and calculate statistics
    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data['firstpost_date']

        user_id = tweet_data['tweet']['user']['id']
        number_of_followers = tweet_data['author']['followers']
        user_ids[user_id] = number_of_followers
        number_of_retweets.append(tweet_data['metrics']['citations']['total'])

        if end_time < end_time_of_window:
            current_hour_count += 1
            # number_of_retweets += tweet_data['metrics']['citations']['total']
        else:
            number_of_tweets_hour.append(current_hour_count)
            current_window += 1
            current_hour_count = 1 #0
            # number_of_retweets = tweet_data['metrics']['citations']['total']
            end_time_of_window = start_time + current_window * 3600

    # print(number_of_tweets_hour)
    # print(number_of_retweets)
    print ('Averge number of tweets: ', (sum(number_of_tweets_hour) / len(number_of_tweets_hour)))
    print ('Averge number of followers: ', sum(user_ids.values()) / len(user_ids))
    print ('Averge number of retweets: ', float(sum(number_of_retweets)) / float(sum(number_of_tweets_hour)))

    # plot graph for #superbowl and #nfl
    # if file_name == 'superbowl' or file_name == "nfl":
    #     plt.figure(2 if file_name == 'superbowl' else 1)
    #     plt.ylabel('Number of Tweets')
    #     plt.xlabel('Hour')
    #     plt.title('Number of Tweets per hour for {}'.format(file_name))
    #     plt.bar(range(len(number_of_tweets_hour)), number_of_tweets_hour)

end = time.clock();
print("Runtime: ", end - start)
plt.show()
