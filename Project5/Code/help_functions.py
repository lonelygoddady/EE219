import json
import datetime
import matplotlib.pyplot as plt
import os

def plot_feature(x, y, hashtag, feature_name):
    folder = '/Graph/Pro3/'
    fig = plt.figure()
    plt.scatter(y, x)
    plt.xlabel(feature_name)
    plt.ylabel('voltage (mV)')
    plt.title('Next hour tweet count vs '+feature_name)
    plt.grid(True)
    filename = hashtag+'_'+feature_name+'.png'
    fullpath = os.path.join(folder, filename)
    plt.savefig(filename)

def modified_time(end_time):
    # get the data's time in datetime format and slice them into hours
    date_time = datetime.datetime.fromtimestamp(end_time)
    actual_time = datetime.datetime(date_time.year, date_time.month, date_time.day,
                                    date_time.hour, 0, 0)
    return unicode(actual_time), actual_time


def features_extraction(tweets):
    first_tweet = json.loads(tweets.readline())  # make tweets into dictionary

    tweets.seek(0, 0)  # set start point to 0 again

    start_time = first_tweet['firstpost_date']  # get the first tweet start time
    st_data_time = datetime.datetime.fromtimestamp(start_time)
    start_time -= (
        st_data_time.minute * 60 + st_data_time.second)  # rectify the time offset to make the hour window align with day hours
    end_time_of_window = start_time + 3600  # window to keep track of number of hours of data

    # regression features
    user_ids_hour = []  # who has tweeted in this hour

    # Structure that holds hour wise features
    hour_wise_data = {}

    ####### Loop through each tweet and calculate statistics #######
    for tweet in tweets:
        tweet_data = json.loads(tweet)
        end_time = tweet_data.get('firstpost_date')

        # create the data for this hour if not exist
        modified_actual_time, actual_time = modified_time(end_time)
        if modified_actual_time not in hour_wise_data.keys():

            # hour_wise_data[modified_actual_time] = {'tweets_count': 0, 'retweets_count': 0,
            #                                         'followers': 0, 'max_followers': 0,
            #                                         'day_hour': 0, 'ranking': 0, 'impression': 0,
            #                                         'favo': 0, 'user_count': 0, 'verified_user': 0,
            #                                         'user_mention': 0, 'URL': 0, 'long_tweet': 0}
            hour_wise_data[modified_actual_time] = {'tweets_count': 0, 'retweets_count': 0,
                                                    'followers': 0, 'max_followers': 0,
                                                    'day_hour': actual_time.hour,
                                                    'user_count': 0, 'ranking': 0,
                                                    'user_mention': 0, 'URL': 0, 'impression': 0,
                                                    'verified_user': 0, 'list_num': 0, 'Max_list': 0,
                                                    'long_tweet': 0,'friends_count': 0}

            #   Features to be excluded: 'favo': 0,  'statuses_count': 0, 'Max_fav': 0,}


        if end_time < end_time_of_window:
            #   calculate features in the previous hour
            hour_wise_data[modified_actual_time]['tweets_count'] += 1
            hour_wise_data[modified_actual_time]['retweets_count'] += tweet_data['metrics']['citations']['total']

            if tweet_data['tweet']['user']['id'] not in user_ids_hour:
                user_ids_hour.append(tweet_data['tweet']['user']['id'])
                hour_wise_data[modified_actual_time]['user_count'] += 1
                hour_wise_data[modified_actual_time]['followers'] += tweet_data['author']['followers']
                hour_wise_data[modified_actual_time]['max_followers'] = tweet_data['author']['followers'] if \
                    hour_wise_data[modified_actual_time]['max_followers'] < tweet_data['author']['followers'] \
                    else hour_wise_data[modified_actual_time]['max_followers']
                hour_wise_data[modified_actual_time]['URL'] += len(tweet_data['tweet']['entities']['urls'])
                hour_wise_data[modified_actual_time]['ranking'] += tweet_data['metrics']['ranking_score']
                hour_wise_data[modified_actual_time]['verified_user'] += 1 \
                    if tweet_data['tweet']['user']['verified'] == 'true' else 0
                hour_wise_data[modified_actual_time]['user_mention'] += \
                    len(tweet_data["tweet"]["entities"]["user_mentions"])
                # hour_wise_data[modified_actual_time]['favo'] += tweet_data['tweet']['favorite_count']
                hour_wise_data[modified_actual_time]['impression'] += tweet_data['metrics']['impressions']
                hour_wise_data[modified_actual_time]['long_tweet'] += 1 if len(tweet_data['title']) > 100 else 0
                hour_wise_data[modified_actual_time]['friends_count'] += tweet_data['tweet']['user']['friends_count']
                # hour_wise_data[modified_actual_time]['statuses_count'] += tweet_data['tweet']['user']['statuses_count']
                hour_wise_data[modified_actual_time]['list_num'] += 0 if tweet_data["tweet"]["user"]["listed_count"] \
                                                                         == None else tweet_data["tweet"]["user"]["listed_count"]
                # hour_wise_data[modified_actual_time]['Max_fav'] = tweet_data['tweet']['favorite_count'] if \
                #     hour_wise_data[modified_actual_time]['Max_fav']  < tweet_data['tweet']['favorite_count'] else \
                #     hour_wise_data[modified_actual_time]['Max_fav']
                hour_wise_data[modified_actual_time]['Max_list'] = tweet_data["tweet"]["user"]["listed_count"] if \
                    tweet_data["tweet"]["user"]["listed_count"] != None and hour_wise_data[modified_actual_time]['Max_list'] \
                    < tweet_data["tweet"]["user"]["listed_count"] else hour_wise_data[modified_actual_time]['Max_list']

        else:
            # reinitialize variables for the first item in the window
            # (*** this part is tricky ****)
            user_ids_hour = [tweet_data['tweet']['user']['id']]
            hour_wise_data[modified_actual_time]['tweets_count'] = 1
            hour_wise_data[modified_actual_time]['retweets_count'] = tweet_data['metrics']['citations']['total']
            hour_wise_data[modified_actual_time]['followers'] = tweet_data['author']['followers']
            hour_wise_data[modified_actual_time]['max_followers'] = tweet_data['author']['followers']
            hour_wise_data[modified_actual_time]['ranking'] = tweet_data['metrics']['ranking_score']
            hour_wise_data[modified_actual_time]['impression'] = tweet_data['metrics']['impressions']
            # hour_wise_data[modified_actual_time]['favo'] = tweet_data['tweet']['favorite_count']
            hour_wise_data[modified_actual_time]['user_count'] = 1
            hour_wise_data[modified_actual_time]['verified_user'] = 1 if tweet_data['tweet']['user']['verified'] \
                                                                         == 'true' else 0
            hour_wise_data[modified_actual_time]['user_mention'] = len(tweet_data["tweet"]["entities"]["user_mentions"])
            hour_wise_data[modified_actual_time]['URL'] = len(tweet_data['tweet']['entities']['urls'])
            hour_wise_data[modified_actual_time]['long_tweet'] = 1 if len(tweet_data['title']) > 100 else 0
            hour_wise_data[modified_actual_time]['friends_count'] = tweet_data['tweet']['user']['friends_count']
            # hour_wise_data[modified_actual_time]['statuses_count'] = tweet_data['tweet']['user']['statuses_count']
            hour_wise_data[modified_actual_time]['list_num'] = 0 if tweet_data["tweet"]["user"]["listed_count"] == None \
                else tweet_data["tweet"]["user"]["listed_count"]
            # hour_wise_data[modified_actual_time]['Max_fav'] = tweet_data['tweet']['favorite_count']
            hour_wise_data[modified_actual_time]['Max_list'] = 0 if tweet_data["tweet"]["user"]["listed_count"] == None \
                else tweet_data["tweet"]["user"]["listed_count"]

            # update time window
            st_data_time = datetime.datetime.fromtimestamp(end_time)
            start_time = end_time - (st_data_time.minute * 60 + st_data_time.second)  # rectify the time offset
            end_time_of_window = start_time + 3600

    return hour_wise_data
