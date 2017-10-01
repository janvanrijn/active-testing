import activetesting
import numpy as np
import warnings

from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor


def modelbased_tablelookup_average_ranking(task_ids, holdout_task_id, flow_id, num_runs, relevant_parameters, cache_directory):
    X_test, y_test = activetesting.utils.get_X_y_from_openml(holdout_task_id, flow_id, num_runs, relevant_parameters, cache_directory)
    total_ranks = np.zeros(len(X_test))

    for task_id in task_ids:
        if task_id == holdout_task_id:
            warnings.warn('Holdout task present in task_ids. Skipping it. ')
            continue

        X_train, y_train = activetesting.utils.get_X_y_from_openml(task_id, flow_id, num_runs, relevant_parameters, cache_directory)
        clf = RandomForestRegressor()
        clf.fit(X_train, y_train)
        y_test_hat = clf.predict(X_test)
        task_ranks = rankdata(y_test_hat, method='average')
        # TODO: high rank implies a good classifier
        total_ranks = np.add(total_ranks, task_ranks)

    # total ranks should be devided by len(X_test)
    total_ranks = np.divide(total_ranks, len(X_test))
    return total_ranks

