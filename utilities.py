import pandas as pd
from datetime import datetime


def train_test_split(user_taggedartists_timestamps, user_count):
    test_data = []
    train_data = []
    for id in list(user_count['userID']):
        idx = user_count[user_count['userID'] == id]['20%'].iloc[0]
        train_sampe = user_taggedartists_timestamps[user_taggedartists_timestamps['userID'] == id].iloc[:-idx]
        test_sample = user_taggedartists_timestamps[user_taggedartists_timestamps['userID'] == id].iloc[-idx:]
        train_data.append(train_sampe)
        test_data.append(test_sample)
    test = pd.concat(test_data).reset_index(drop=True)
    train = pd.concat(train_data).reset_index(drop=True)
    return train, test


def get_timestamp(st):
    if st < 0:
        return 1900
    date = datetime.fromtimestamp(int(st)/1000)
    return int(date.strftime("%Y"))


def jointag(tag):
    return ' '.join(tag)

def MinMaxScaler(data):
    min_val = min(data)
    max_val = max(data)
    data = data - min_val
    data = data/(max_val + 1e-7)
    return data