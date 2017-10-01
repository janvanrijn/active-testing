import os
import sklearn

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor


def cache_model(X, y, cat_indices, cache_directory, filename, n_trees=16):
    try:
        os.makedirs(cache_directory)
    except FileExistsError:
        pass

    cache_location = cache_directory + '/' + filename

    clf = sklearn.pipeline.Pipeline(
        steps=[('encoder', sklearn.preprocessing.OneHotEncoder(
            categorical_features=list(cat_indices),
            handle_unknown='ignore')),
               ('classifier', RandomForestRegressor(n_estimators=n_trees))])
    clf.fit(X, y)
    joblib.dump(clf, cache_location)