import argparse
import activetesting
import json
import numpy as np
import openml
import os
import pickle
import random

from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogated Active Testing')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/active_testing',
                        help='directory to store cache')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--flow_id', type=int, default=6952, help='openml flow id')
    parser.add_argument('--relevant_parameters', type=json.loads, default='{"C": "numeric", "gamma": "numeric", "kernel": "categorical", "coef0": "numeric", "tol": "numeric"}')
    parser.add_argument('--scoring', type=str, default='neg_mean_absolute_error')
    parser.add_argument('--num_runs', type=int, default=500, help='max runs to obtain from openml')
    parser.add_argument('--prevent_model_cache', action='store_true', help='prevents loading old models from cache')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--num_tasks', type=int, default=None, help='limit number of tasks (for testing)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cache_directory = args.cache_directory + '/' + str(args.flow_id)
    study_cache_path = cache_directory + '/study.pkl'
    cache_controller = activetesting.utils.ModelCacheController()
    all_scores = []
    loss_curves = {}
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
    all_tasks = study.tasks
    if args.num_tasks:
        all_tasks = random.sample(all_tasks, args.num_tasks)

    for task_id in all_tasks:
        X, y, column_names, categoricals = activetesting.utils.get_X_y_from_openml(task_id=task_id,
                                                                                   flow_id=args.flow_id,
                                                                                   num_runs=args.num_runs,
                                                                                   relevant_parameters=args.relevant_parameters,
                                                                                   cache_directory=cache_directory)
        X, cat_mapping = activetesting.utils.encode_categoricals(X, categoricals)

        model_cache_directory = cache_directory + '/' + str(task_id)
        model_cache_filename = 'RandForest_' + str(args.num_runs) + '.pkl'

        clf = cache_controller.retrieve(X=X, y=y,
                                        cat_indices=categoricals,
                                        cache_directory=model_cache_directory,
                                        filename=model_cache_filename,
                                        prevent_cache=args.prevent_model_cache)

        y_hat = cross_val_predict(clf, X, y, cv=10)
        scores = cross_val_score(clf, X, y, cv=10, scoring=args.scoring)
        spearman = pearsonr(y, y_hat)
        all_scores.append(scores.mean())
        print("Task %d; Ranges: [%2f-%2f]; Pearson Spearman Correlation: %f; MSE: %0.4f (+/- %0.4f)" %(task_id,
                                                                                                       min(y), max(y),
                                                                                                       spearman[0],
                                                                                                       scores.mean(),
                                                                                                       scores.std() * 2))

        average_rank = activetesting.strategies.modelbased_tablelookup_average_ranking(task_ids=all_tasks,
                                                                                       holdout_task_id=task_id,
                                                                                       flow_id=args.flow_id,
                                                                                       num_runs=args.num_runs,
                                                                                       relevant_parameters=args.relevant_parameters,
                                                                                       cache_controller=cache_controller,
                                                                                       cache_directory=cache_directory,
                                                                                       prevent_model_cache=args.prevent_model_cache)
        ar_spearman = pearsonr(y, average_rank)
        best_index = ar_spearman.index(min(ar_spearman))
        regret = max(y) - y[best_index]
        loss_curves[task_id] = activetesting.utils.ranks_to_losscurve(average_rank, y)
        print("Task %d; Average Rank: Pearson Spearman Correlation: %f; Regret@1: %f" % (task_id, ar_spearman[0], regret))

        # TODO: correlation coefficient :)
    print("ALL: ", np.mean(all_scores))
    activetesting.utils.plot_loss_curves(loss_curves, args.cache_directory, 'avg_rank.pdf')
