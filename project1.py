#!/usr/bin/env python
# coding: utf-8

# # Project
# 
# Welcome to the group project! The project is based on the [ACM RecSys 2021 Challenge](https://recsys-twitter.com/).
# 
# - Detailed information about the task, submission and grading can be found in a [dedicates site on TUWEL](https://tuwel.tuwien.ac.at/mod/page/view.php?id=1217340).
# - Information about the dataset structure [on this site on TUWEL](https://tuwel.tuwien.ac.at/mod/page/view.php?id=1218810).

# In[10]:


team_name = "" # Team 15 e.g. 'team_1'
team_members = [("Reetu Rani 11843424",""),
                ("","")] # [("Jane Doe","012345678"), ("John Doe","012345678")]


# In[11]:


print(team_name)
print(team_members)


# In[12]:


path_to_data = '~/shared/data/project/training/'
dataset_type = 'one_hour' # all_sorted, one_day, one_hour, one_week


# In[ ]:





# In[13]:


import os
import re
import csv
import datetime

from model import reply_pred_model, retweet_pred_model, quote_pred_model, fav_pred_model 

all_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",               "engaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))

def parse_input_line(line):
    features = line #.split("\x01")
    tweet_id = features[all_features_to_idx['tweet_id']]
    user_id = features[all_features_to_idx['engaging_user_id']]
    input_feats = features[all_features_to_idx['text_tokens']]
    tweet_timestamp = features[all_features_to_idx['tweet_timestamp']]
    return tweet_id, user_id, input_feats, tweet_timestamp


def evaluate_test_set():
    expanded_path = os.path.expanduser(path_to_data)
    part_files = [os.path.join(expanded_path, f) for f in os.listdir(expanded_path) if dataset_type in f]
    part_files = sorted(part_files, key = lambda x:x[-5:]) 
        
    with open('results.csv', 'w') as output:
        for file in part_files:
            with open(file, 'r') as f:
                linereader = csv.reader(f, delimiter='\x01')
                last_timestamp = None
                for row in linereader:
                    tweet_id, user_id, features, tweet_timestamp = parse_input_line(row)                                                           
                    reply_pred = reply_pred_model(features) # reply_model
                    retweet_pred = retweet_pred_model(features) # retweet_model
                    quote_pred = quote_pred_model(features) # pred_model
                    fav_pred = fav_pred_model(features) # fav_model
                    
                    # print(str(tweet_timestamp))
                    # print(str(reply_pred)+" "+str(retweet_pred)+" "+str(quote_pred)+" "+str(fav_pred))
                    
                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')


# In[14]:


#evaluate_test_set()


# In[15]:


# hidden


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

vectorizer = TfidfVectorizer() 

tfidf_text_tokens = vectorizer.fit_transform(map(str, df.text_tokens))


# In[17]:


tweet_engaged_with_user_like = r_df.loc[(r_df['user'] == 4) & (r_df['like'] == True)]['tweet']
tweet_engaged_with_user_reply = r_df.loc[(r_df['user'] == 4) & (r_df['reply'] == True)]['tweet']
tweet_engaged_with_user_retweet = r_df.loc[(r_df['user'] == 4) & (r_df['retweet'] == True)]['tweet']
tweet_engaged_with_user_retweet_wc = r_df.loc[(r_df['user'] == 4) & (r_df['retweet_wc'] == True)]['tweet']


# In[18]:


def get_tweet_ids_engaged_by_user_id(user_id):
    return np.array(r_df.loc[ r_df['user'] == user_id ].index)
    
def get_item_vector(user_id):
    tweet_ids_engaged_by_user_id = get_tweet_ids_engaged_by_user_id(user_id)
    return tfidf_text_tokens[tweet_ids_engaged_by_user_id]

def get_user_engagements(user_id, engagement_type):
    return np.array( r_df.loc[ r_df['user'] == user_id ][engagement_type] )


import sklearn.preprocessing as pp

def compute_user_profile_by_rating(user_ratings):
    user_rating_weight = tfidf_vector.T.multiply(user_ratings)
    user_profile = user_rating_weight.mean(axis=1).T
    return pp.normalize(user_profile)

def compute_user_profile(user_id, engagement_type):
    user_ratings = get_user_engagements(user_id, engagement_type)
    return compute_user_profile_by_rating(user_ratings)


# In[19]:


user_id = 3
tweet_ids_engaged_by_user_id = get_tweet_ids_engaged_by_user_id(user_id)
tfidf_vector = get_item_vector(user_id)
user_like_engagements = get_user_engagements(user_id, 'like')

print(tweet_ids_engaged_by_user_id)
print(user_like_engagements)


# In[20]:


user_profile = compute_user_profile(user_id, 'like')


# In[21]:


print(user_profile[user_profile.nonzero()])


# In[22]:


def recommend(user_profile, topN=20):
    sims = linear_kernel(user_profile, tfidf_text_tokens).flatten()
    sims = sims.argsort()[::-1]
    sim_item_ids = np.array(r_df.iloc[sims]['tweet'])

    return list(filter(
        (lambda item_id: item_id not in tweet_ids_engaged_by_user_id), sim_item_ids   
    ))[:topN]

recommendations = recommend(user_profile)
print(recommendations)


# In[23]:


def map_tweetIDX_to_tweetID(ids):
    tweet_id_map = pd.DataFrame()
    tweet_id_map['tweet_id'] = df['tweet_id']
    tweet_id_map['tweet'] = df['tweet_id'].map(tweetId_to_tweetIDX)
    return tweet_id_map.loc[tweet_id_map['tweet'].isin(ids)]['tweet_id']


# In[24]:


recommended_tweet_ids = map_tweetIDX_to_tweetID(recommendations)


# In[25]:


columns = ['tweet_id', 'like_timestamp']
gt_predictions = df.loc[df['tweet_id'].isin(recommended_tweet_ids)][columns]
hit = gt_predictions['like_timestamp'].count()
n = len(gt_predictions.index)
ap = hit / n
print(ap)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




