import argparse
import activetesting
import json
import numpy as np
import openml
import os
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogated Active Testing')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/active_testing',
                        help='directory to store cache')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--flow_id', type=int, default=6952, help='openml flow id')
    parser.add_argument('--relevant_parameters', type=json.loads,
                        default='{"C": "numeric", "gamma": "numeric", "kernel": "categorical", "coef0": "numeric", "tol": "numeric"}')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cache_directory = args.cache_directory + '/' + str(args.flow_id)
    study_cache_path = cache_directory + '/study.pkl'
    num_runs = 500
    all_scores = []
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    if not os.path.isfile(study_cache_path):
        study = openml.study.get_study(args.study_id, 'tasks')
        with open(study_cache_path, 'wb') as fp:
            pickle.dump(study, fp, 0)

    with open(study_cache_path, 'rb') as fp:
        study = pickle.load(fp)

    for task_id in study.tasks:
        X, y = activetesting.utils.get_X_y_from_openml(task_id, args.flow_id, num_runs, args.relevant_parameters, cache_directory)
        clf = RandomForestRegressor()
        scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_absolute_error')
        all_scores.append(scores.mean())
        print("Task %d; Ranges: [%2f-%2f]; MSE: %0.4f (+/- %0.4f)" % (task_id, min(y), max(y), scores.mean(), scores.std() * 2))

        # TODO: correlation coefficient :)
    print("ALL: ", np.mean(all_scores))