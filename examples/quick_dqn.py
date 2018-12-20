""" A DQN example using wintermute that should train fast and show some results
    quickly on `Pong`. The setup is fairly different from the original paper.
"""
import time
import random
from functools import partial
from types import SimpleNamespace
from datetime import datetime

import torch
from torch import optim
from termcolor import colored as clr
from rl_logger import Logger

from wintermute.env_wrappers import get_wrapped_atari
from wintermute.estimators import get_estimator
from wintermute.policy_evaluation import EpsilonGreedyPolicy
from wintermute.policy_evaluation import get_epsilon_schedule as get_epsilon

# from wintermute.policy_improvement import get_optimizer
from wintermute.policy_improvement import DQNPolicyImprovement
from wintermute.replay import NaiveExperienceReplay as ER
from wintermute.replay.prioritized_replay import ProportionalSampler as PER

# from wintermute.replay import FlatExperienceReplay as ER

from utils import get_parser, print_namespace


def priority_update(mem, dqn_loss):
    """ Callback for updating priorities in the proportional-based experience
    replay and for computing the importance sampling corrected loss.
    """
    losses = dqn_loss.loss
    mem.update([loss.item() for loss in losses.detach().abs()])
    return (losses * mem.weights.to(losses.device).view_as(losses)).mean()


def train(args):
    """ Here we do the training.
    """
    env = args.env
    train_log = args.log.groups["training"]

    state, reward, done = env.reset(), 0, False
    warmed_up = False
    ep_cnt = 0
    for step in range(1, args.step_no + 1):

        # take action and save the s to _s and a to _a to be used later
        pi = args.policy_evaluation(state)
        _state, _action = state, pi.action
        state, reward, done, _ = env.step(pi.action)

        # add a (_s, _a, r, d) transition
        args.experience_replay.push((_state, _action, reward, done))
        # args.experience_replay.push(_state[0, 3], _action, reward, done)

        # sample a batch and do some learning
        do_training = (step % args.update_freq == 0) and warmed_up

        if do_training:
            batch = args.experience_replay.sample()
            if args.prioritized:
                args.policy_improvement(batch, cb=args.priority_update)
            else:
                args.policy_improvement(batch)

        if step % 1000 == 0:
            args.policy_improvement.update_target_estimator()

        # do some logging
        train_log.update(
            ep_cnt=(1 if done else 0),
            rw_per_ep=(reward, (1 if done else 0)),
            rw_per_step=reward,
            max_q=pi.q_value,
            sampling_fps=1,
            training_fps=32 if do_training else 0,
        )

        if done:
            state, reward, done = env.reset(), 0, False
            ep_cnt += 1

            if ep_cnt % args.log_freq == 0:
                args.log.log(train_log, step)
                train_log.reset()

        warmed_up = len(args.experience_replay) > args.learn_start
    args.log.log(train_log, step)
    train_log.reset()


def main(args):
    """ Here we initialize stuff.
    """
    args.seed = random.randint(0, 1e4) if args.seed == 42 else args.seed
    print(f"torch manual seed={args.seed}.")
    torch.manual_seed(args.seed)

    # wrap the gym env
    env = get_wrapped_atari(
        f"{args.game}NoFrameskip-v4", mode="training", hist_len=4
    )
    print(env)

    # construct an estimator to be used with the policy
    action_no = env.action_space.n
    estimator = get_estimator(
        "atari", hist_len=4, action_no=action_no, hidden_sz=256
    )
    estimator = estimator.cuda()

    # construct an epsilon greedy policy
    # also: epsilon = {'name':'linear', 'start':1, 'end':0.1, 'steps':1000}
    epsilon = get_epsilon(steps=args.epsilon_steps)
    policy_evaluation = EpsilonGreedyPolicy(estimator, action_no, epsilon)

    # construct a policy improvement type
    # optimizer = get_optimizer('Adam', estimator, lr=0.0001, eps=0.0003)
    optimizer = optim.Adam(
        estimator.parameters(), lr=args.lr, eps=args.adam_eps
    )
    policy_improvement = DQNPolicyImprovement(
        estimator, optimizer, gamma=0.99, is_double=args.double_dqn
    )

    # we also need an experience replay
    if args.prioritized:
        experience_replay = PER(
            args.mem_size,
            batch_size=32,
            alpha=0.6,
            optim_steps=((args.step_no - args.learn_start) / args.update_freq),
        )
        priority_update_cb = partial(priority_update, experience_replay)
    else:
        experience_replay = ER(args.mem_size, batch_size=32)
        # experience_replay = ER(100000, batch_size=32, hist_len=4)  # flat

    # construct a tester
    tester = None

    # construct a logger
    if not args.label:
        sampling = "prioritized" if args.prioritized else "uniform"
        label = f"{datetime.now():%Y%b%d-%H%M%S}_{args.game}_{sampling}"

    log = Logger(label=label, path=f"./results/{label}")
    train_log = log.add_group(
        tag="training",
        metrics=(
            log.SumMetric("ep_cnt", resetable=False),
            log.AvgMetric("rw_per_ep", emph=True),
            log.AvgMetric("rw_per_step"),
            log.MaxMetric("max_q"),
            log.FPSMetric("training_fps"),
            log.FPSMetric("sampling_fps"),
        ),
        console_options=("white", "on_blue", ["bold"]),
    )
    log.log_info(train_log, "date: %s." % time.strftime("%d/%m/%Y | %H:%M:%S"))
    log.log_info(train_log, "pytorch v%s." % torch.__version__)

    # Add the created objects in the args namespace
    args.env = env
    args.policy_evaluation = policy_evaluation
    args.policy_improvement = policy_improvement
    args.experience_replay = experience_replay
    args.tester = tester
    args.log = log
    if args.prioritized:
        args.priority_update = priority_update_cb

    # print the args
    print_namespace(args)

    # start the training
    train(args)


if __name__ == "__main__":
    main(get_parser(
        game="Pong",
        step_no=4_000_000,
        update_freq=1,
        learn_start=256,
        mem_size=100_000,
        epsilon_steps=30_000,
        lr=0.0001,
        log_freq=5
    ))
