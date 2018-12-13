import itertools
import torch.optim as optim


def float_range(start, end, step):
    x = start
    if step > 0:
        while x < end:
            yield x
            x += step
    else:
        while x > end:
            yield x
            x += step


def lr_schedule(start, end, steps_no):
    start, end, steps_no = float(start), float(end), float(steps_no)
    step = (end - start) / (steps_no - 1.0)
    schedules = [float_range(start, end, step), itertools.repeat(end)]
    return itertools.chain(*schedules)


def get_optimizer(name, estimator, lr=0.000235, eps=0.0003, alpha=None):
    weights = estimator.parameters()
    if name == "Adam":
        return optim.Adam(weights, lr=lr, eps=eps)
    elif name == "RMSprop":
        return optim.RMSprop(weights, lr=lr, eps=eps, alpha=alpha)
