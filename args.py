import argparse

def rec_sys_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--c', type=int, default=1.25, help='penalty term')
    parser.add_argument('--n_iter', type=float, default=600, help='number of iteration of logistics model')
    parser.add_argument('--random_state', type=int, default=2023, help='random state')
    parser.add_argument('--judge', type=bool, default=False, help='train again or use saved model')
    parser.add_argument('--runsvd', type=bool, default=False, help='run svd model to get user, artist interaction data')
    parser.add_argument('--metric', type=str, default='cosine', help='metric of similarities')
    parser.add_argument('--n_neighbors', type=int, default=20, help='find n neighbors')
    parser.add_argument('--max_features', type=float, default=500, help='max_feature of each document tf-idf')

    args = parser.parse_args()

    return args