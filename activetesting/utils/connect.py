import numpy as np
import openmlcontrib


def get_X_y_from_openml(task_id, flow_id, num_runs, relevant_parameters, cache_directory):

    dataframe = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id, flow_id, num_runs, relevant_parameters,
                                                                      cache_directory)

    categorical_columns = set(dataframe.columns) - set(dataframe._get_numeric_data().columns)
    categorical_indices = {dataframe.columns.get_loc(col_name) for col_name in categorical_columns}

    y = np.array(dataframe['y'], dtype=np.float)

    dataframe.drop('y', 1, inplace=True)
    return dataframe.as_matrix(), y, dataframe.columns.values, categorical_indices
