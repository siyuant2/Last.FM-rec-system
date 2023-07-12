from sklearn.feature_extraction.text import TfidfVectorizer
from utilities import jointag, MinMaxScaler
from scipy.sparse import coo_matrix
import numpy as np
from datetime import datetime
import time
from sklearn.neighbors import NearestNeighbors


class CandidateEmb:
    def __init__(self, data, max_features):
        self.data = data
        self.max_features = max_features

    def tag_concat(self):
        tag_value_artist = self.data.groupby(['artistID'])['tagValue'].apply(list).reset_index()
        tag_value_artist['tagValue'] = tag_value_artist['tagValue'].apply(jointag)
        return tag_value_artist, list(tag_value_artist['artistID'])

    def build_tfidf(self, data):
        corpus = data['tagValue']
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        X = vectorizer.fit_transform(corpus)
        X = coo_matrix(X, dtype=np.float32).toarray()
        return X, vectorizer

    def get_tfidf_vec(self, data, vectorizer):
        corpus = data['tagValue']
        X = vectorizer.transform(corpus)
        return coo_matrix(X, dtype=np.float32).toarray()

    def get_freshness_artist(self):
        date_string = "2011/12/31"
        date = datetime.strptime(date_string, "%Y/%m/%d")
        timestamp = int(time.mktime(date.timetuple()))
        example_age_artist = self.data.groupby(['artistID'])['timestamp'].agg(['max']).reset_index()
        example_age_artist['example_age'] = timestamp * 1000 - example_age_artist['max']
        m,n = min(example_age_artist['example_age']), max(example_age_artist['example_age'])
        example_age_artist['example_age'] = (example_age_artist['example_age'] - m)/(n-m)
        example_age_artist['example_age_square'] = example_age_artist['example_age'] ** 2
        example_age_artist = example_age_artist[['example_age', 'example_age_square']].to_numpy()
        return example_age_artist

    def get_freshness_user(self):
        user_last_tagged_time = self.data.groupby('userID')['timestamp'].agg(['max']).reset_index()
        date_string = "2011/12/31"
        date = datetime.strptime(date_string, "%Y/%m/%d")
        timestamp = int(time.mktime(date.timetuple()))
        user_last_tagged_time['example_age'] = timestamp * 1000 - user_last_tagged_time['max']
        m,n = min(user_last_tagged_time['example_age']), max(user_last_tagged_time['example_age'])
        user_last_tagged_time['example_age'] = (user_last_tagged_time['example_age'] - m) / (n-m)
        user_last_tagged_time['example_age_square'] = user_last_tagged_time['example_age'] ** 2
        user_last_tagged_time['example_age_root'] = user_last_tagged_time['example_age'] ** 0.5
        user_example_age = user_last_tagged_time[['example_age', 'example_age_square', 'example_age_root']].to_numpy()
        user_example_age_dict = dict(zip(list(user_last_tagged_time['userID']), user_example_age))
        return user_example_age_dict

    def get_popularity(self):
        pop = self.data.groupby('artistID')['weight'].agg(['sum']).reset_index()
        pop['sum'] = np.where(pop['sum'] >= 3000, 3000, pop['sum'])
        m,n = min(pop['sum']), max(pop['sum'])
        pop['sum'] = (pop['sum'] - m)/ (n-m)
        pop = pop['sum'].to_numpy().reshape(-1, 1)
        return pop

    def get_embedding_artistlist(self, example_age_artist, pop, X, artistID):
        concat_emb = np.append(example_age_artist, pop, axis=1)
        artist2vec = np.concatenate((X, concat_emb), axis = 1)
        artist_emb_dict = dict(zip(artistID, artist2vec))
        user_artistlist_emd = {}
        for id in list(self.data['userID'].unique()):
            artist_list = list(self.data[self.data['userID'] == id]['artistID'].unique())
            emb = np.zeros(503)
            for artist in artist_list:
                emb = np.sum([artist_emb_dict[artist], emb], axis=0)
            user_artistlist_emd[id] = emb
        return user_artistlist_emd, artist2vec, artist_emb_dict


def get_n_neibors(vector, n_neigh):
    _, neigh = n_neigh.kneighbors([vector])
    return neigh


class CandidateGen:
    def __init__(self, n, metric, vectors):
        self.n_neighbor = n
        self.metric = metric
        self.artist_vec = vectors

    def build_knn(self):
        n_neigh = NearestNeighbors(n_neighbors=self.n_neighbor, metric = self.metric)
        n_neigh.fit(self.artist_vec)
        return n_neigh





