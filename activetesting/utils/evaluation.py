import collections


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
