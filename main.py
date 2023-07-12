import pandas as pd

from data_process import user_taggedartists_timestamps, user_count, load_data, get_label, data_svd
from candidate_generation import CandidateEmb, CandidateGen
from utilities import train_test_split
from args import rec_sys_args_parser
from ranking import combine_data, LogisticsReg
import numpy as np
from user_artist_interaction import BiasedSVD
import pickle

args = rec_sys_args_parser()
user_artists = load_data('user_artists')
user_taggedartists_timestamps = user_taggedartists_timestamps()
user_count = user_count(user_taggedartists_timestamps)
train, test = train_test_split(user_taggedartists_timestamps, user_count)

candidate_emb_train = CandidateEmb(train, args.max_features)
tag_value_artist, artist_id = candidate_emb_train.tag_concat()

X, vectorizer = candidate_emb_train.build_tfidf(tag_value_artist)

example_age_artist = candidate_emb_train.get_freshness_artist()
pop = candidate_emb_train.get_popularity()
user_artistlist_emd, artist2vec, artist_emb_dict = candidate_emb_train.get_embedding_artistlist(example_age_artist,
                                                                                                    pop, X, artist_id)

candidate_gen_train = CandidateGen(args.n_neighbors, args.metric, artist2vec)
n_neigh = candidate_gen_train.build_knn()

if args.runsvd == False:
    user_interact = np.load('./data/user_pu.npy')
    artist_interact = np.load('./data/artist_qi.npy')
else:
    user_artists = data_svd()
    biassvd = BiasedSVD(user_artists, args.random_state)
    _, user_interact, artist_interact = biassvd.buildSVD()

user_interact_dict = dict(zip(list(user_artists['userID'].unique()), user_interact))
artist_interact_dict = dict(zip(list(user_artists['artistID'].unique()), artist_interact))

user_example_age_dict = candidate_emb_train.get_freshness_user()
user_artists_label = get_label()

X_train, y_train, _, _ = combine_data(user_artistlist_emd, n_neigh, user_interact_dict, user_example_age_dict,
                                artist_emb_dict, artist_interact_dict, user_artists_label)



candidate_emb_test = CandidateEmb(test, args.max_features)
tag_value_artist, artist_id = candidate_emb_test.tag_concat()

X = candidate_emb_test.get_tfidf_vec(tag_value_artist, vectorizer)

example_age_artist = candidate_emb_test.get_freshness_artist()
pop = candidate_emb_test.get_popularity()
user_artistlist_emd, artist2vec, artist_emb_dict = candidate_emb_test.get_embedding_artistlist(example_age_artist,
                                                                                                    pop, X, artist_id)

candidate_gen_test = CandidateGen(args.n_neighbors, args.metric, artist2vec)
n_neigh = candidate_gen_test.build_knn()

user_example_age_dict = candidate_emb_test.get_freshness_user()
user_artists_label = get_label()

X_test, y_test, artistID_test, userID_test= combine_data(user_artistlist_emd, n_neigh, user_interact_dict, user_example_age_dict,
                              artist_emb_dict, artist_interact_dict, user_artists_label)

model = LogisticsReg(args.c, args.n_iter, X_train, y_train, X_test, y_test, args.random_state)

if args.judge == True:
    lr = model.train()
    # model.test(lr)
    score = model.get_recommendation(lr)
else:
    lr = pickle.load(open('./model/finalized_model.sav', 'rb'))
    # model.test(lr)
    score = model.get_recommendation(lr)

result = pd.DataFrame()
result['userid'] = np.repeat(userID_test, args.n_neighbors)
result['artistid'] = artistID_test
result['score'] = score
result.to_csv('./result/rec+result.csv')








