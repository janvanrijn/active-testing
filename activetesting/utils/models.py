import os
import sklearn

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

class ModelCacheController:

    def __init__(self):
        self.models = {}

    def retrieve(self, X, y, cat_indices, cache_directory, filename, n_trees=16, prevent_cache=False):
        filepath = cache_directory + '/' + filename

        if not os.path.isfile(filepath) or prevent_cache:
            clf = self._cache_model(X, y, cat_indices, n_trees)
            self.models[filepath] = clf

            try:
                os.makedirs(cache_directory)
            except FileExistsError:
                pass
            joblib.dump(clf, filepath)

        elif filepath not in self.models:
            self.models[filepath] = joblib.load(filepath)

        return self.models[filepath]

    def _cache_model(self, X, y, cat_indices, n_trees=16):
        clf = sklearn.pipeline.Pipeline(
            steps=[('encoder', sklearn.preprocessing.OneHotEncoder(
                categorical_features=list(cat_indices),
                handle_unknown='ignore')),
                   ('classifier', RandomForestRegressor(n_estimators=n_trees))])
        clf.fit(X, y)
        return clf
