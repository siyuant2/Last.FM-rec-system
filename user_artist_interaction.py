from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
# from surprise import accuracy
# from surprise.model_selection import GridSearchCV
import numpy as np


class BiasedSVD:
    def __init__(self, data, random_state):
        self.data = data
        self.random_state = random_state
        self.n_factors = 10
        self.n_epoch = 100
        self.lr_all = 0.005
        self.reg_all = 0.08
        self.biased = True
        self.cv = 5

    def buildSVD(self):
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(self.data[['userID', 'artistID', 'weight']], reader)
        # param_grid = {'n_factors': [10, 16, 100], 'n_epochs': [10, 13, 100], 'lr_all': [0.002, 0.005, 0.01],
        #              'reg_all': [0.05, 0.08, 0.1]}
        # gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
        # gs.fit(data)
        biasedSVD = SVD(n_factors=self.n_factors, n_epochs=self.n_epoch, biased=self.biased, lr_all=self.lr_all, reg_all=self.reg_all, random_state=self.random_state)
        cross_validate(biasedSVD, data, measures=['RMSE', 'MAE'], cv=self.cv, verbose=True)
        trainset = data.build_full_trainset()
        biasedSVD.fit(trainset)
        np.save('user_pu', biasedSVD.pu)
        np.save('artist_qi', biasedSVD.qi)
        return biasedSVD, biasedSVD.pu, biasedSVD.qi