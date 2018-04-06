import activetesting
import arff
import argparse
import json
import numpy as np
import openml
import os
import pandas
import sklearn


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogated Active Testing')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/active_testing',
                        help='directory to store cache')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='random_forest', help='openml flow id')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=250, help='max runs to obtain from openml')
    parser.add_argument('--normalize', action='store_true', help='normalizes y values per task')
    parser.add_argument('--prevent_model_cache', action='store_true', help='prevents loading old models from cache')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--num_tasks', type=int, default=None, help='limit number of tasks (for testing)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    study = openml.study.get_study(args.study_id, 'tasks')
    setup_data_all = None
    scaler = sklearn.preprocessing.MinMaxScaler()

    if args.classifier == 'random_forest':
        flow_id = 6969
        relevant_parameters = {"bootstrap": "nominal", "max_features": "numeric", "min_samples_leaf": "numeric",
         "min_samples_split": "numeric", "criterion": "nominal", "strategy": "nominal"}
    elif args.classifier == 'adaboost':
        flow_id = 6970
        relevant_parameters = {"algorithm": "nominal", "learning_rate": "numeric", "max_depth": "numeric",
                               "n_estimators": "numeric", "strategy": "nominal"}
    elif args.classifier == 'libsvm_svc':
        flow_id = 7707
        relevant_parameters = {"C": "numeric", "gamma": "numeric", "kernel": "categorical", "coef0": "numeric", "tol": "numeric"}
    elif args.classifier == 'ranger':
        flow_id = 5965
        relevant_parameters = {"min.node.size": "numeric", "num.trees": "numeric"} # TODO: extend!
    else:
        raise ValueError()

    relevant_tasks = study.tasks
    if args.num_tasks:
        relevant_tasks = study.tasks[:args.num_tasks]

    for task_id in relevant_tasks:
        print("Currently processing task", task_id)
        try:
            setup_data = activetesting.utils.get_dataframe_from_openml(task_id=task_id,
                                                                      flow_id=flow_id,
                                                                      num_runs=args.num_runs,
                                                                      relevant_parameters=relevant_parameters,
                                                                      evaluation_measure=args.scoring,
                                                                      cache_directory=args.cache_directory)
        except ValueError as e:
            print('Problem in task %d:' %task_id, e)
            continue
        setup_data['task_id'] = task_id
        if setup_data_all is None:
            setup_data_all = setup_data
        else:
            if list(setup_data.columns.values) != list(setup_data_all.columns.values):
                raise ValueError()
            if args.normalize:
                setup_data[['y']] = scaler.fit_transform(setup_data[['y']])

            setup_data_all = pandas.concat((setup_data_all, setup_data))

    if len(setup_data_all) < args.num_runs * len(relevant_tasks) * 0.25:
        raise ValueError('Num results suspiciously low. Please check.')

    task_qualities = {}
    for task_id in relevant_tasks:
        task = openml.tasks.get_task(task_id)
        task_qualities[task_id] = task.get_dataset().qualities
    # index of qualities: the task id
    qualities_with_na = pandas.DataFrame.from_dict(task_qualities, orient='index', dtype=np.float)
    qualities = pandas.DataFrame.dropna(qualities_with_na, axis=1, how='any')

    meta_data = setup_data_all.join(qualities, on='task_id', how='inner')

    arff_dict = activetesting.utils.dataframe_to_arff(meta_data)
    filename = 'meta_%s.arff' %args.classifier
    with open(filename, 'w') as fp:
        arff.dump(arff_dict, fp)
