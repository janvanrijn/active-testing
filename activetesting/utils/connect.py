import json
import numpy as np
import openml
import os
import pandas as pd
import pickle


def get_X_y_from_openml(task_id, flow_id, num_runs, relevant_parameters, cache_directory):
    try:
        os.makedirs(cache_directory + '/' + str(task_id))
    except FileExistsError:
        pass

    # grab num_runs random evaluations
    evaluations_cache_path = cache_directory + '/' + str(task_id) + '/evaluations.pkl'
    setups_cache_path = cache_directory + '/' + str(task_id) + '/setups.pkl'
    if not os.path.isfile(evaluations_cache_path) or not os.path.isfile(setups_cache_path):
        evaluations = openml.evaluations.list_evaluations('predictive_accuracy', size=num_runs, task=[task_id], flow=[flow_id])
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
            if name not in relevant_parameters.keys():
                continue

            if name in hyperparameters:
                # duplicate parameter name, this can happen due to subflows.
                # when this happens, we need to fix
                raise ValueError('Duplicate hyperparameter:', name, 'Values:', value, hyperparameters[name])
            hyperparameters[name] = value
        setup_parameters[setup_id] = hyperparameters

    y = []
    dataframe = pd.DataFrame(columns=relevant_parameters.keys())
    categoricals = set()
    for idx, param in enumerate(relevant_parameters):
        if relevant_parameters[param]:
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

    return X, y
