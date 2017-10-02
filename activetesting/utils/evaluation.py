import collections
import copy


def ranks_to_losscurve(ranks, y):
    if len(ranks) != len(y):
        raise ValueError()

    rank_loss = {}
    for i in range(len(ranks)):
        rank_loss[ranks[i]] = 1 - y[i]
    rank_loss = collections.OrderedDict(sorted(rank_loss.items()))

    loss_curve = [1.0]
    previous_score = 1.0

    for idx, (key, value) in enumerate(rank_loss.items()):
        score = min(previous_score, value)
        loss_curve.append(score)
        previous_score = score
    return loss_curve


def task_losscurve_to_avg_losscurve(task_losscurves, max_size):
    total_losscurve = []
    for _, losscurve in task_losscurves.items():
        if len(losscurve) < max_size:
            losscurve_prime = copy.deepcopy(losscurve)
            losscurve_prime.extend([losscurve[-1]] * (max_size - len(losscurve)))
        else:
            losscurve_prime = losscurve

        for idx, value in enumerate(losscurve_prime):
            if idx >= max_size:
                break
            if idx < len(total_losscurve):
                total_losscurve[idx] += value
            else:
                total_losscurve.append(value)

    avg_losscurve = []
    for idx, loss in enumerate(total_losscurve):
        avg_losscurve.append(total_losscurve[idx] / len(task_losscurves))
    return avg_losscurve
