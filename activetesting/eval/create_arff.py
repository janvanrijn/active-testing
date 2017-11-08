import activetesting
import arff
import argparse
import numpy as np
import openml
import json
import os


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
    parser.add_argument('--num_tasks', type=int, default=5, help='limit number of tasks (for testing)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.study_id, 'tasks')
    X_all = None
    y_all = None
    column_names_all = None
    for task in study.tasks:
        print(task)
        X, y, column_names, categoricals = activetesting.utils.get_X_y_from_openml(task_id=task,
                                                                                   flow_id=args.flow_id,
                                                                                   num_runs=args.num_runs,
                                                                                   relevant_parameters=args.relevant_parameters,
                                                                                   cache_directory=args.cache_directory)
        if X_all is None:
            X_all = X
            y_all = y
            column_names_all = column_names
        else:
            X_all = np.concatenate((X_all, X))
            y_all = np.concatenate((y_all, y))
            if list(column_names) != list(column_names_all):
                raise ValueError()

    if len(y_all) < args.num_runs * len(study.tasks) * 0.25:
        raise ValueError('Num results suspiciously low. Please check.')


    arff_dict = activetesting.utils.X_and_y_to_arff(X_all, y_all, column_names, categoricals)
    filename = 'res.arff'
    with open(filename, 'w') as fp:
        arff.dump(arff_dict, fp)
