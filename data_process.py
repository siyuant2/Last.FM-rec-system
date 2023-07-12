import numpy as np
import pandas as pd
import os
from datetime import datetime
from utilities import get_timestamp



def load_data(filename):
    path = os.path.join('./data/' + filename + '.csv')
    df = pd.read_csv(path)
    return df


def user_taggedartists_timestamps():
    user_taggedartists_timestamps = load_data('user_taggedartists_timestamps')
    tags = load_data('tags')
    user_artists = load_data('user_artists')
    user_taggedartists_timestamps = user_taggedartists_timestamps.merge(tags, how='left', on='tagID')
    user_taggedartists_timestamps['year'] = user_taggedartists_timestamps['timestamp'].apply(get_timestamp)
    user_taggedartists_timestamps = user_taggedartists_timestamps[user_taggedartists_timestamps['year'] >= 2000]
    user_taggedartists_timestamps = user_taggedartists_timestamps.sort_values(by=['userID', 'timestamp'])
    user_taggedartists_timestamps = user_taggedartists_timestamps.merge(user_artists, how='left',
                                                                        on=['userID', 'artistID'])
    user_taggedartists_timestamps['weight'] = user_taggedartists_timestamps['weight'].fillna(0)
    return user_taggedartists_timestamps

def user_count(user_taggedartists_timestamps):
    user_count = user_taggedartists_timestamps.groupby('userID')['tagID'].agg(['count']).reset_index()
    user_count['20%'] = (user_count['count'] * 0.2).apply(round)
    return user_count

def get_label():
    user_artists = load_data('user_artists')
    user_artists['positive'] = np.where(user_artists['weight'] <= 260, 0, 1)
    key = list(user_artists['userID'].astype(str) + ',' + user_artists['artistID'].astype(str))
    user_artists_label = dict(zip(key, user_artists['positive']))
    return user_artists_label
def data_svd():
    user_artists = load_data('user_artists')
    user_artists['weight'] = user_artists['weight'] ** 0.25
    user_artists['weight'] = (user_artists['weight'] - min(user_artists['weight'])) / (
            max(user_artists['weight']) - min(user_artists['weight']))
    return user_artists



