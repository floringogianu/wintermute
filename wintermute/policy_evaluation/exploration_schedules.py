""" Various exploration schedules.

    * constant_schedule(value)
        constant_schedule(.1)   =>   .1, .1, .1, .1, .1, ...

    * linear_schedule(start, end, steps_no)
        linear_schedule(.5, .1, 5)  =>  .5, .4, .3, .2, .1, .1, .1, .1, ...

    * log_schedule(start, end, steps_no)
        log_schedule(1, 0.001, 3)   =>   1., .1, .01, .001, .001, .001, ...
"""

import itertools


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


def constant_schedule(epsilon=0.05):
    return itertools.repeat(epsilon)


def linear_schedule(start, end, steps_no):
    start, end, steps_no = float(start), float(end), float(steps_no)
    step = (end - start) / (steps_no - 1.0)
    schedules = [float_range(start, end, step), itertools.repeat(end)]
    return itertools.chain(*schedules)


def log_schedule(start, end, steps_no):
    from math import log, exp

    log_start, log_end = log(start), log(end)
    log_step = (log_end - log_start) / (steps_no - 1)
    log_range = float_range(log_start, log_end, log_step)
    return itertools.chain(map(exp, log_range), itertools.repeat(end))


SCHEDULES = {"linear": linear_schedule, "log": log_schedule}


def get_schedule(name="linear", start=1, end=0.01, steps=None):
    if name == "constant":
        return constant_schedule(start)
    return SCHEDULES[name](start, end, steps)


def get_random_schedule(args, probs):
    assert len(args) == len(probs)
    import numpy as np

    return get_schedule(*args[np.random.choice(len(args), p=probs)])


if __name__ == "__main__":
    import sys

    const = get_schedule("constant", [0.1])
    sys.stdout.write("Constant(0.1):")
    for _ in range(10):
        sys.stdout.write(" {:.2f}".format(next(const)))
    sys.stdout.write("\n")

    linear = get_schedule("linear", [0.5, 0.1, 5])
    sys.stdout.write("Linear Schedule(.5, .1, 5):")
    for _ in range(10):
        sys.stdout.write(" {:.2f}".format(next(linear)))
    sys.stdout.write("\n")

    logarithmic = get_schedule("log", [1, 0.001, 4])
    sys.stdout.write("Logarithmic Schedule(1, .001, 4):")
    for _ in range(10):
        sys.stdout.write(" {:.3f}".format(next(logarithmic)))
    sys.stdout.write("\n")
