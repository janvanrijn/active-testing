import activetesting
import numpy as np
import warnings

from scipy.stats import rankdata


def modelbased_tablelookup_average_ranking(task_ids, holdout_task_id, flow_id, num_runs, relevant_parameters, cache_controller, cache_directory, prevent_model_cache=False):
    X_test, y_test, column_names_test, cat_test = activetesting.utils.get_X_y_from_openml(task_id=holdout_task_id,
                                                                                          flow_id=flow_id,
                                                                                          num_runs=num_runs,
                                                                                          relevant_parameters=relevant_parameters,
                                                                                          cache_directory=cache_directory)
    X_test, cat_mapping = activetesting.utils.encode_categoricals(X_test, cat_test)
    total_ranks = np.zeros(len(X_test))

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

        task_ranks = rankdata(y_test_hat, method='average')
        # TODO: high rank implies a good classifier
        total_ranks = np.add(total_ranks, task_ranks)

    # total ranks should be devided by len(X_test)
    total_ranks = np.divide(total_ranks, len(X_test))
    return total_ranks

