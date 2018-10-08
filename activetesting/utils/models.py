import os
import sklearn
import sklearn.compose
import sklearn.ensemble
import sklearn.impute

from sklearn.externals import joblib


class ModelCacheController:

    def __init__(self):
        self.models = {}

    def retrieve(self, X, y, cat_indices, cache_directory, filename, n_trees=16, prevent_cache=False):
        filepath = cache_directory + '/' + filename

        if not os.path.isfile(filepath) or prevent_cache:
            clf = ModelCacheController._cache_model(X, y, cat_indices, n_trees)
            self.models[filepath] = clf

            try:
                os.makedirs(cache_directory)
            except FileExistsError:
                pass
            joblib.dump(clf, filepath)

        elif filepath not in self.models:
            self.models[filepath] = joblib.load(filepath)

        return self.models[filepath]

    @staticmethod
    def _cache_model(X, y, cat_indices, n_trees=16):
        num_indices = set(range(X.shape[1])) - cat_indices
        numeric_transformer = sklearn.pipeline.make_pipeline(
            sklearn.impute.MissingIndicator(error_on_new=False),
            sklearn.impute.SimpleImputer(strategy='median'),
            sklearn.preprocessing.StandardScaler())

        # note that the dataset is encoded numerically, hence we can only impute
        # numeric values, even for the categorical columns.
        categorical_transformer = sklearn.pipeline.make_pipeline(
            sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1),
            sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))

        transformer = sklearn.compose.ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, list(num_indices)),
                ('nominal', categorical_transformer, list(cat_indices))],
            remainder='passthrough')

        clf = sklearn.pipeline.make_pipeline(transformer,
                                             sklearn.ensemble.RandomForestRegressor(n_estimators=n_trees))
        clf.fit(X, y)
        return clf
