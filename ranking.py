import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
from candidate_generation import get_n_neibors


def combine_data(user_artistlist_emd, n_neigh, user_interact_dict, user_example_age_dict, artist_emb,
                 artist_interact_dict, user_artists_label):
    X = []
    y = []
    artists = []
    for user in user_artistlist_emd.keys():
        candidates = []
        neigh = get_n_neibors(user_artistlist_emd[user], n_neigh)
        # _, neigh = n_neigh.kneighbors([user_artistlist_emd[user]])
        neigh = neigh.reshape(-1)
        user_inter_emb = user_interact_dict[user]
        user_example_age_emb = user_example_age_dict[user]
        for nei in list(neigh):
            key = str(user) + ',' + str(list(artist_emb.keys())[nei])
            nei_emb = artist_emb[list(artist_emb.keys())[nei]]
            artists.append(list(artist_emb.keys())[nei])
            if key in artist_interact_dict:
                artist_inter_emb = artist_interact_dict[list(artist_emb.keys())[nei]]
            else:
                artist_inter_emb = np.zeros(10)
            user_artist_inter_emb = np.multiply(user_inter_emb, artist_inter_emb)
            concat_emb = np.concatenate(
                (user_inter_emb, user_example_age_emb, nei_emb, artist_inter_emb, user_artist_inter_emb))
            candidates.append(concat_emb)
            if key in user_artists_label:
                y.append(user_artists_label[key])
            else:
                y.append(0)
        X.append(candidates)
    X = np.array(X)
    X = X.reshape(len(y), -1)
    return X, y, artists, list(user_artistlist_emd.keys())


class LogisticsReg:
    def __init__(self, c, n_iter, X_tr, y_tr,X_te, y_te, random_state):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.c = c
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = LogisticRegression(random_state=self.random_state)

    # def hyperparam_tunning(self, random_search_n_iter, random_search_c):
    #    random_grid = {'max_iter': random_search_n_iter, 'C': random_search_c}
    #    logisticsReg_random = RandomizedSearchCV(estimator=self.model, param_distributions=random_grid,
    #                                             n_iter=10, cv=3, verbose=2,
    #                                             random_state=self.random_state, n_jobs=-1)
    #    logisticsReg_random.fit(self.X_tr, self.y_tr)
    #    n_iter = logisticsReg_random.best_params_['max_iter']
    #    c = logisticsReg_random.best_params_['C']

    def train(self):
        lr_params = [{'max_iter': [self.n_iter - 100, self.n_iter, self.n_iter + 100], 'C': [self.c - 0.1, self.c, self.c + 0.1]}]
        lr = GridSearchCV(self.model, lr_params, cv=5, scoring='f1')
        lr.fit(self.X_tr, self.y_tr)
        filename = './model/finalized_model.sav'
        pickle.dump(lr, open(filename, 'wb'))
        return lr

    def test(self, model):
        return model.score(self.X_te, self.y_te)

    def get_recommendation(self, model):
        scores = model.predict_proba(self.X_te)
        return scores[:, 0]



