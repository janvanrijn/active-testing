import argparse
import json
import openml


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogated Active Testing')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--flow_id', type=int, default=6952, help='openml flow id')
    parser.add_argument('--relevant_parameters', type=json.loads, default='{"C": 0, "gamma": 0, "kernel": 0, "coef0": 0, "tol": 0}')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    study = openml.study.get_study(args.study_id, 'tasks')
    relevant_parameters = list(args.relevant_parameters.keys())

    for task_id in study.tasks:
        # grab 200 random evaluations
        evaluations = openml.evaluations.list_evaluations("predictive_accuracy", size=200, task=[task_id], flow=[args.flow_id])

        # setups
        setup_ids = []
        for run_id, evaluation in evaluations.items():
            setup_ids.append(evaluation.setup_id)

        setups = openml.setups.list_setups(setup=setup_ids)
        setup_parameters = {}

        for setup_id, setup in setups.items():
            hyperparameters = {}
            for pid, hyperparameter in setup.parameters.items():
                name = hyperparameter.parameter_name
                value = hyperparameter.value
                if name not in relevant_parameters:
                    continue

                if name in hyperparameters:
                    # duplicate parameter name, this can happen due to subflows.
                    # when this happens, we need to fix
                    raise ValueError('Duplicate hyperparameter:', name, 'Values:', value, hyperparameters[name])
                hyperparameters[name] = value
            setup_parameters[setup_id] = hyperparameters

        X = []
        y = []
        for run_id, evaluation in evaluations.items():
            currentX = []
            for param in relevant_parameters:
                currentX.append(setup_parameters[evaluation.setup_id][param])

            X.append(currentX)
            y.append(evaluation.value)

        print(X)
        print(y)