import activetesting
import numpy as np
import sklearn
import warnings

from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor


def modelbased_tablelookup_average_ranking(task_ids, holdout_task_id, flow_id, num_runs, relevant_parameters, cache_directory):
    X_test, y_test, cat_test = activetesting.utils.get_X_y_from_openml(holdout_task_id, flow_id, num_runs, relevant_parameters, cache_directory)
    X_test, cat_mapping = activetesting.utils.encode_categoricals(X_test, cat_test)
    total_ranks = np.zeros(len(X_test))

    for task_id in task_ids:
        if task_id == holdout_task_id:
            warnings.warn('Holdout task present in task_ids. Skipping it. ')
            continue

        X_train, y_train, cat_train = activetesting.utils.get_X_y_from_openml(task_id, flow_id, num_runs, relevant_parameters, cache_directory)
        if cat_train != cat_test:
            raise ValueError()
        X_train, cat_mapping = activetesting.utils.encode_categoricals(X_train, cat_train)
        clf = sklearn.pipeline.Pipeline(
            steps=[('encoder', sklearn.preprocessing.OneHotEncoder(categorical_features=list(cat_train), handle_unknown='ignore')),
                   ('classifier', RandomForestRegressor())])
        clf.fit(X_train, y_train)
        y_test_hat = clf.predict(X_test)

        task_ranks = rankdata(y_test_hat, method='average')
        # TODO: high rank implies a good classifier
        total_ranks = np.add(total_ranks, task_ranks)

    # total ranks should be devided by len(X_test)
    total_ranks = np.divide(total_ranks, len(X_test))
    return total_ranks

