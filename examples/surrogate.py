import argparse
import activetesting
import numpy as np
import openml
import os
import pickle
import random
import traceback

from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogated Active Testing')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/active_testing',
                        help='directory to store cache')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='openml flow name')
    parser.add_argument('--scoring', type=str, default='neg_mean_absolute_error')
    parser.add_argument('--evaluation_measure', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=500, help='max runs to obtain from openml')
    parser.add_argument('--prevent_model_cache', action='store_true', help='prevents loading old models from cache')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--num_tasks', type=int, default=None, help='limit number of tasks (for testing)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    openml.config.apikey = None
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://www.openml.org/api/v1/'
    cache_directory = args.cache_directory + '/' + args.classifier
    study_cache_path = cache_directory + '/study.pkl'
    cache_controller = activetesting.utils.ModelCacheController()
    all_scores = []
    task_losscurve = {}

    if args.classifier == 'random_forest':
        flow_id = 6969
        config_space = activetesting.config_spaces.get_random_forest_default_search_space()
    elif args.classifier == 'adaboost':
        flow_id = 6970
        config_space = activetesting.config_spaces.get_adaboost_default_search_space()
    elif args.classifier == 'libsvm_svc':
        flow_id = 7707
        config_space = activetesting.config_spaces.get_libsvm_svc_default_search_space()
    else:
        raise ValueError()

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
        try:
            X, y, column_names, categoricals = activetesting.utils.get_X_y_from_openml(task_id=task_id,
                                                                                       flow_id=flow_id,
                                                                                       num_runs=args.num_runs,
                                                                                       config_space=config_space,
                                                                                       evaluation_measure=args.evaluation_measure,
                                                                                       cache_directory=cache_directory)
        except ValueError as e:
            # TODO: change error
            traceback.print_tb(e.__traceback__)
            continue
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
        print("Task %d; Ranges: [%2f-%2f]; Pearson Spearman Correlation: %f; MSE: %0.4f (+/- %0.4f)" % (task_id,
                                                                                                        min(y), max(y),
                                                                                                        spearman[0],
                                                                                                        scores.mean(),
                                                                                                        scores.std() * 2))

        # average_rank = activetesting.strategies.modelbased_tablelookup_average_ranking(task_ids=all_tasks,
        #                                                                                holdout_task_id=task_id,
        #                                                                                flow_id=flow_id,
        #                                                                                num_runs=args.num_runs,
        #                                                                                config_space=config_space,
        #                                                                                evaluation_measure=args.evaluation_measure,
        #                                                                                cache_controller=cache_controller,
        #                                                                                cache_directory=cache_directory,
        #                                                                                prevent_model_cache=args.prevent_model_cache)
        # ar_spearman = pearsonr(y, average_rank)
        # best_index = ar_spearman.index(min(ar_spearman))
        # regret = max(y) - y[best_index]
        # task_losscurve[task_id] = activetesting.utils.ranks_to_losscurve(average_rank, y)
        # print("Task %d; Average Rank: Pearson Spearman Correlation: %f; Regret@1: %f" % (task_id, ar_spearman[0], regret))

        # TODO: correlation coefficient :)
    print("ALL: ", np.mean(all_scores))
    activetesting.utils.plot_loss_curves(task_losscurve, args.cache_directory, 'avg_rank.pdf')
    print(activetesting.utils.task_losscurve_to_avg_losscurve(task_losscurve, args.num_runs))
