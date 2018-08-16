import activetesting
import arff
import argparse
import json
import numpy as np
import openml
import openmlcontrib
import os
import pandas
import sklearn


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache',
                        help='directory to store cache')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/active_testing',
                        help='directory to store output')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='openml flow id')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=500, help='max runs to obtain from openml')
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
        config_space = activetesting.config_spaces.get_random_forest_default_search_space()
    elif args.classifier == 'adaboost':
        flow_id = 6970
        config_space = activetesting.config_spaces.get_adaboost_default_search_space()
    elif args.classifier == 'libsvm_svc':
        flow_id = 7707
        config_space = activetesting.config_spaces.get_libsvm_svc_default_search_space()
    else:
        raise ValueError()

    relevant_tasks = study.tasks
    if args.num_tasks:
        relevant_tasks = study.tasks[:args.num_tasks]

    for task_id in relevant_tasks:
        print("Currently processing task", task_id)
        try:
            setup_data = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id=task_id,
                                                                               flow_id=flow_id,
                                                                               num_runs=args.num_runs,
                                                                               configuration_space=config_space,
                                                                               parameter_field='parameter_name',
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

    setup_data_with_meta_features = setup_data_all.join(qualities, on='task_id', how='inner')

    os.makedirs(args.output_directory, exist_ok=True)
    # create the task / parameters / performance arff
    filename = os.path.join(args.output_directory, 'pertask_%s.arff' % args.classifier)
    relation_name = 'openml-meta-flow-%d' % flow_id
    json_meta = {'flow_id': flow_id, 'openml_server': openml.config.server}
    with open(filename, 'w') as fp:
        arff.dump(openmlcontrib.meta.dataframe_to_arff(setup_data_all,
                                                       relation_name,
                                                       json.dump(json_meta)), fp)

    # create the task / meta-features / parameters / performance arff
    filename = os.path.join(args.output_directory, 'meta_%s.arff' % args.classifier)
    with open(filename, 'w') as fp:
        arff.dump(openmlcontrib.meta.dataframe_to_arff(setup_data_with_meta_features,
                                                       relation_name,
                                                       json.dump(json_meta)), fp)
