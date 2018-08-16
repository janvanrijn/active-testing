import numpy as np
import warnings


def encode_categoricals(X, categoricals, mapping_orig=None):
    if mapping_orig is None:
        mapping = dict()
        for categorical in categoricals:
            mapping[categorical] = dict()
    else:
        mapping = np.copy(mapping_orig)

    for obs_idx in range(len(X)):

        for feat_idx in categoricals:
            value = X[obs_idx][feat_idx]
            if value not in mapping[feat_idx]:
                if mapping_orig is not None:
                    warnings.warn('Could not find value of attribute #%d: %d' %(feat_idx, value))
                mapping[feat_idx][value] = len(mapping[feat_idx])
            X[obs_idx][feat_idx] = mapping[feat_idx][value]
    return np.array(X, dtype=np.float64), mapping


def X_data_to_list_of_dicts(X, column_names):
    result = []
    for _, observation in enumerate(X):
        current_dict = {}
        for feature_idx, column in enumerate(column_names):
            current_dict[column_names[feature_idx]] = observation[feature_idx]
        result.append(current_dict)
    return result
