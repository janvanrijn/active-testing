import argparse
import json
import numpy as np
import openml
import os
import pandas as pd
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

    relevant_parameters = list(args.relevant_parameters.keys())

    for task_id in study.tasks:
        try:
            os.makedirs(cache_directory + '/' + str(task_id))
        except FileExistsError:
            pass

        # grab 200 random evaluations
        evaluations_cache_path = cache_directory + '/' + str(task_id) + '/evaluations.pkl'
        setups_cache_path = cache_directory + '/' + str(task_id) + '/setups.pkl'
        if not os.path.isfile(evaluations_cache_path) or not os.path.isfile(setups_cache_path):
            study = openml.study.get_study(args.study_id, 'tasks')
            evaluations = openml.evaluations.list_evaluations('predictive_accuracy', size=num_runs, task=[task_id],
                                                              flow=[args.flow_id])
            with open(evaluations_cache_path, 'wb') as fp:
                pickle.dump(evaluations, fp)

            # setups
            setup_ids = []
            for run_id, evaluation in evaluations.items():
                setup_ids.append(evaluation.setup_id)
            setups = openml.setups.list_setups(setup=setup_ids)

            with open(setups_cache_path, 'wb') as fp:
                pickle.dump(setups, fp)

        with open(evaluations_cache_path, 'rb') as fp:
            evaluations = pickle.load(fp)
        with open(setups_cache_path, 'rb') as fp:
            setups = pickle.load(fp)

        setup_parameters = {}

        for setup_id, setup in setups.items():
            hyperparameters = {}
            for pid, hyperparameter in setup.parameters.items():
                name = hyperparameter.parameter_name
                value = hyperparameter.value
                if name not in relevant_parameters:
                    continue

                if name in hyperparameters:
                    # duplicate parameter name, this can happen due to subflows.
                    # when this happens, we need to fix
                    raise ValueError('Duplicate hyperparameter:', name, 'Values:', value, hyperparameters[name])
                hyperparameters[name] = value
            setup_parameters[setup_id] = hyperparameters

        y = []
        dataframe = pd.DataFrame(columns=relevant_parameters)
        categoricals = set()
        for idx, param in enumerate(relevant_parameters):
            if args.relevant_parameters[param]:
                categoricals.add(idx)

        for run_id, evaluation in evaluations.items():
            currentX = {}
            for idx, param in enumerate(relevant_parameters):
                currentX[param] = json.loads(setup_parameters[evaluation.setup_id][param])

            dataframe = dataframe.append(currentX, ignore_index=True)
            y.append(float(evaluation.value))
        X = np.array(pd.get_dummies(dataframe).as_matrix())
        y = np.array(y)

        if X.shape[0] > num_runs:
            raise ValueError()
        if y.shape[0] > num_runs:
            raise ValueError()

        clf = RandomForestRegressor()
        scores = cross_val_score(clf, X, y, cv=10, scoring='neg_mean_absolute_error')
        all_scores.append(scores.mean())
        print("Task %d; Ranges: [%2f-%2f]; MSE: %0.4f (+/- %0.4f)" % (task_id, min(y), max(y), scores.mean(), scores.std() * 2))

        # TODO: correlation coefficient :)
    print("ALL: ", np.mean(all_scores))