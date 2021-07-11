# Author: Markus Böck

import numpy as np
import pandas as pd
import os
import re
import csv
import datetime

class IICF:
    def __init__(self, path_to_data, dataset_type):
        self.path_to_data = path_to_data
        
        expanded_path = os.path.expanduser(path_to_data)
        part_files = [os.path.join(expanded_path, f) for f in os.listdir(expanded_path) if dataset_type in f]
        self.part_files = sorted(part_files, key = lambda x:x[-5:])
        print("Found files:", self.part_files)
        
        self.initialise_target_names()
        self.initialise_ratings_matrix()
        self.initialise_tweet_features()
        
        print("\nFinished initialisation!")
        
    def initialise_target_names(self):
        
        self.all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                        "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
                       "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                       "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
                       "enaging_user_account_creation", "engagee_follows_engager", "reply", "retweet", "quote", "like"]

        self.all_features_to_idx = dict(zip(self.all_features, range(len(self.all_features))))

        self.feature_names = [
            "photo",
            "video",
            "link",
            "retweet",
            "quote",
            "toplevel",
            "follower_count",
            "following_count",
            "verified"
        ]
        self.target_names = [
            "reply",
            "retweet",
            "quote",
            "like"
        ]

        self.feature_to_col = dict(zip(self.feature_names, range(len(self.feature_names))))
        print("Tweet Features:", self.feature_to_col)
        self.target_to_col = dict(zip(self.target_names, range(len(self.target_names))))
        print("Tweet Targets:", self.target_to_col)
        
        
    # Calculates ratings matrix,a pandas dataframe of form
    # user_id, tweet_id, reply, retweet, quote, like, follow
    # where follow indicates whether user_id follows author of tweet_id
    # this may take a while
    def initialise_ratings_matrix(self):
        print("Initialising \"Ratings Matrix\" ...")
        self.tweet_ids = {} # map tweet ids to row index
        i = 0
        total_entries = 0

        # collect relevant data for "Ratings Matrix"
        tids = []
        uids = []
        replies = []
        retweets = []
        quotes = []
        likes = []
        follows = []

        print("Reading", self.part_files, "...")
        
        for file in self.part_files:
            with open(file, 'r') as f:
                linereader = csv.reader(f, delimiter='\x01')
                for row in linereader:
                    tweet_id = row[self.all_features_to_idx["tweet_id"]]
                    if tweet_id not in self.tweet_ids:
                        self.tweet_ids[tweet_id] = i # assign tweet_id row i
                        i += 1

                    user_id = row[self.all_features_to_idx["engaging_user_id"]]

                    total_entries += 1

                    reply = int(row[self.all_features_to_idx["reply"]] != "")
                    retweet = int(row[self.all_features_to_idx["retweet"]] != "")
                    quote = int(row[self.all_features_to_idx["quote"]] != "")
                    like = int(row[self.all_features_to_idx["like"]] != "")

                    # follow relationship is not a tweet feature and is collected here also
                    follow = bool(row[self.all_features_to_idx["engagee_follows_engager"]])

                    replies.append(reply)
                    retweets.append(retweet)
                    quotes.append(quote)
                    likes.append(like)
                    follows.append(follow)
                    tids.append(tweet_id)
                    uids.append(user_id)

                    if total_entries % 10000 == 0:
                        print(f"Read {total_entries} entries.", end="\r")
                    
        self.total_entries = total_entries
        print("Read", self.total_entries, "entries.")
        
        print("Convert to DataFrame ...") # maybe this is a bottleneck and could be sped up with sparse matrices
        
        self.R = pd.DataFrame(
            {"reply": replies,
             "retweet": retweets,
             "quote": quotes,
             "like":likes,
             "follow":follows},
            index=pd.MultiIndex.from_arrays((uids, tids), names=["user_id", "tweet_id"]))
        
        del replies
        del retweets
        del quotes
        del likes
        del follows
        
        #print("Sort index ...")
        #self.R.sort_index(inplace=True)
        
        print("Done!")
       
        # calculate average engagements
        self.target_means = self.R[["reply", "retweet", "quote", "like"]].mean(axis=0)
        print("Engagement means:")
        print(self.target_means)
        
        
        
    def parse_input_features(self, row):
        tweet_id = row[self.all_features_to_idx['tweet_id']]
        user_id = row[self.all_features_to_idx['engaging_user_id']]

        input_feats = np.zeros((self.tweet_features.shape[1],),dtype=np.float32)

        # one hot encode media (photo, video, link)
        media = row[self.all_features_to_idx["present_media"]]
        if "Photo" in media:
            input_feats[self.feature_to_col["photo"]] = 1

        if "Video" in media or "GIF" in media:
            input_feats[self.feature_to_col["video"]] = 1

        if row[self.all_features_to_idx["present_links"]] != "":
            input_feats[self.feature_to_col["link"]] = 1

        # one hot encode tweet type (toplevel, quote, retweet)
        tweet_type = row[self.all_features_to_idx["tweet_type"]]
        if tweet_type == "TopLevel":
            input_feats[self.feature_to_col["toplevel"]] = 1
        elif tweet_type == "Quote":
            input_feats[self.feature_to_col["quote"]] = 1
        elif tweet_type == "Retweet":
            input_feats[self.feature_to_col["retweet"]] = 1

        # log10 follower count of tweet author
        input_feats[self.feature_to_col["follower_count"]] = np.log10(int(row[self.all_features_to_idx["engaged_with_user_following_count"]])+1)
        # log10 following count of tweet author
        input_feats[self.feature_to_col["following_count"]] = np.log10(int(row[self.all_features_to_idx["engaged_with_user_follower_count"]])+1)
        input_feats[self.feature_to_col["verified"]] = bool(row[self.all_features_to_idx["engaged_with_user_is_verified"]])
        

        input_feats /= np.linalg.norm(input_feats) # normalise
        
        # following relationship is not a tweet feature
        follow = bool(row[self.all_features_to_idx["engagee_follows_engager"]])
        tweet_timestamp = int(row[self.all_features_to_idx['tweet_timestamp']])

        return tweet_id, user_id, input_feats, follow, tweet_timestamp
        
    
    # calculates and stores tweet features and average tweet engagements of training data
    # in self.tweet_features and self.tweet_targets
    # this may take a while
    def initialise_tweet_features(self):
        print("Calculate tweet features ...")
        
        # precompute all tweet features
        self.tweet_features = np.zeros((len(self.tweet_ids), len(self.feature_names)), dtype=np.float32)
        # precompute engagement means for each tweet
        self.tweet_targets = np.zeros((len(self.tweet_ids), len(self.target_names)), dtype=np.float32)
        
        # count tweets for averaging
        tweet_counts = np.zeros((len(self.tweet_ids),), dtype=np.float32)

        # collect timestamps for consistency check
        tweet_timestamps = np.zeros((len(self.tweet_ids)), dtype=np.int)
        
        
        for file in self.part_files:
            with open(file, 'r') as f:
                linereader = csv.reader(f, delimiter='\x01')
                j = 0
                for row in linereader:
                    tweet_id, user_id, input_feats, follow, timestamp = self.parse_input_features(row)

                    tweet_index = self.tweet_ids[tweet_id] # get row for tweet id
                    
                    if tweet_timestamps[tweet_index] != 0:
                        assert timestamp == tweet_timestamps[tweet_index], "Found tweet with different timestamps!"
                    else:
                        tweet_timestamps[tweet_index] = timestamp

                    self.tweet_features[tweet_index,:] = input_feats # store tweet features


                    # count engagements
                    if row[self.all_features_to_idx["reply"]]:
                        self.tweet_targets[tweet_index, self.target_to_col["reply"]] += 1

                    if row[self.all_features_to_idx["retweet"]]:
                        self.tweet_targets[tweet_index, self.target_to_col["retweet"]] += 1

                    if row[self.all_features_to_idx["quote"]]:
                        self.tweet_targets[tweet_index, self.target_to_col["quote"]] += 1

                    if row[self.all_features_to_idx["like"]]:
                        self.tweet_targets[tweet_index, self.target_to_col["like"]] += 1


                    # count occurences of tweet
                    tweet_counts[tweet_index] += 1

                    j += 1
                    if j % 10000 == 0:
                        print(f"{j}/{self.total_entries}", end="\r")


        print(f"{j}/{self.total_entries} Done!")

        # average engagements
        self.tweet_targets /= tweet_counts.reshape(-1,1)
        

    # gets row of data and predicts all features simultaneously
    # this is faster than predicting each target alone
    def predict(self, tweet_id, user_id, features, follow):                    
        try:
            # neighbourhood: get all tweets from engaging user
            # throws KeyError if user_id is unknown
            rated_tweets = self.R.loc[user_id,:]
            
            # filter for tweets with the same follow relationship
            rated_tweets = rated_tweets[rated_tweets.follow == follow] 

            # transform tweet_ids to row indexes
            rated_tweets_ids = rated_tweets.index.values
            rated_tweets_indexes = [self.tweet_ids[tid] for tid in rated_tweets_ids]

            # similiartiy is the angle between features (features are normalised)
            similarities = self.tweet_features[rated_tweets_indexes,:].dot(features)

            # calculate weights as usual
            weights = similarities / np.sum(np.abs(similarities))

            # get engagement means for tweets in neighbourhood
            item_means = self.tweet_targets[rated_tweets_indexes,:]
            
            # transform user engagments to np array
            user_ratings = np.array(rated_tweets)[:,0:4] # 5th column is follow status

            # make predictions according to formula
            target_prediction = self.target_means + weights.dot(user_ratings - item_means)
            
            # restrict the predictions to the interval [0,1]
            target_prediction = target_prediction.clip(0,1)

            reply_pred, retweet_pred, quote_pred, fav_pred = target_prediction
            
            return reply_pred, retweet_pred, quote_pred, fav_pred

        except KeyError:
            # user not known => predict average in training data
            reply_pred, retweet_pred, quote_pred, fav_pred = self.target_means
            
            return reply_pred, retweet_pred, quote_pred, fav_pred
        
        
        
        