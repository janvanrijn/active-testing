import arff
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


def X_and_y_to_arff(X, y, column_names, categoricals):
    y = np.reshape(y, (len(y), 1))
    data = np.concatenate((X, y), axis=1)
    attributes = []
    for idx, column_name in enumerate(column_names):
        attributes.append((column_name, 'NUMERIC'))
    attributes.append(('y', 'NUMERIC'))

    arff_dict = {}
    arff_dict['data'] = data
    arff_dict['attributes'] = attributes
    arff_dict['description'] = 'emm'
    arff_dict['relation'] = 'emm'

    return arff_dict