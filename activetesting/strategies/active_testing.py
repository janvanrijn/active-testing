import activetesting
import numpy as np
import warnings


def modelbased_tablelookup_active_testing(task_ids, holdout_task_id, flow_id, num_runs, relevant_parameters, cache_controller, cache_directory, prevent_model_cache=False):
    X_test, y_test, column_names_test, cat_test = activetesting.utils.get_X_y_from_openml(task_id=holdout_task_id,
                                                                                          flow_id=flow_id,
                                                                                          num_runs=num_runs,
                                                                                          relevant_parameters=relevant_parameters,
                                                                                          cache_directory=cache_directory)
    X_test_list_dicts = activetesting.utils.X_data_to_list_of_dicts(X_test, column_names_test)
    task_configuration_estimation = {}

    for task_id in task_ids:
        if task_id == holdout_task_id:
            warnings.warn('Holdout task present in task_ids. Skipping it. ')
            continue
        model_cache_directory = cache_directory + '/' + str(task_id)
        model_cache_filename = 'RandForest_' + str(num_runs) + '.pkl'
        X_train, y_train, column_names_train, cat_train = activetesting.utils.get_X_y_from_openml(task_id=task_id,
                                                                                                  flow_id=flow_id,
                                                                                                  num_runs=num_runs,
                                                                                                  relevant_parameters=relevant_parameters,
                                                                                                  cache_directory=cache_directory)
        if cat_train != cat_test:
            raise ValueError()
        if not np.array_equal(column_names_test, column_names_train):
            raise ValueError()

        X_train, cat_mapping = activetesting.utils.encode_categoricals(X_train, cat_train)

        clf = cache_controller.retrieve(X=X_train, y=y_train,
                                        cat_indices=cat_train,
                                        cache_directory=model_cache_directory,
                                        filename=model_cache_filename,
                                        prevent_cache=prevent_model_cache)
        y_test_hat = clf.predict(X_test)

        for idx in range(len(y_test_hat)):
            if task_id not in task_configuration_estimation:
                task_configuration_estimation[task_id] = {}
            task_configuration_estimation[task_id][X_test_list_dicts[idx]] = y_test_hat[idx]

