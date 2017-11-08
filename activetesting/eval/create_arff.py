import activetesting
import arff
import argparse
import json
import numpy as np
import openml
import os
import pandas


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
    setup_data_all = None

    for task_id in study.tasks:
        print("Currently processing task", task_id)
        setup_data = activetesting.utils.get_dataframe_from_openml(task_id=task_id,
                                                                  flow_id=args.flow_id,
                                                                  num_runs=args.num_runs,
                                                                  relevant_parameters=args.relevant_parameters,
                                                                  cache_directory=args.cache_directory)
        setup_data['task_id'] = task_id
        if setup_data_all is None:
            setup_data_all = setup_data
        else:
            if list(setup_data.columns.values) != list(setup_data_all.columns.values):
                raise ValueError()

            setup_data_all = pandas.concat((setup_data_all, setup_data))

    if len(setup_data_all) < args.num_runs * len(study.tasks) * 0.25:
        raise ValueError('Num results suspiciously low. Please check.')

    task_qualities = {}
    for task_id in study.tasks:
        task = openml.tasks.get_task(task_id)
        task_qualities[task_id] = task.get_dataset().qualities
    # index of qualities: the task id
    qualities_with_na = pandas.DataFrame.from_dict(task_qualities, orient='index', dtype=np.float)
    qualities = pandas.DataFrame.dropna(qualities_with_na, axis=1, how='any')

    meta_data = setup_data_all.join(qualities, on='task_id', how='inner')
    print(meta_data)

    arff_dict = activetesting.utils.dataframe_to_arff(meta_data)
    filename = 'res.arff'
    with open(filename, 'w') as fp:
        arff.dump(arff_dict, fp)
