""" A DQN example using wintermute. """
import time
from types import SimpleNamespace

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
from wintermute.data_structures import NaiveExperienceReplay as ER

# from wintermute.data_structures import FlatExperienceReplay as ER


def train(args):
    """ Here we do the training.
    """
    env = args.env
    train_log = args.log.groups["training"]

    state, reward, done = env.reset(), 0, False
    warmed_up = False
    for step in range(1, args.training_steps + 1):

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
            args.log.log(train_log, step)
            train_log.reset()

        warmed_up = len(args.experience_replay) > args.start_learning_after
        """
        # testing
        if tester.is_test_time(step):
            estimator = policy_evaluation.get_estimator()
            tester.test(step, estimator)
        """
    args.log.log(train_log, step)
    train_log.reset()


def main(
    seed=42,
    label="results",
    training_steps=10_000_000,
    lr=0.0001,
    is_double=False,
):
    """ Here we initialize stuff.
    """
    print(f"torch manual seed={seed}.")
    torch.manual_seed(seed)

    # wrap the gym env
    env = get_wrapped_atari("PongNoFrameskip-v4", mode="training", hist_len=4)
    print(env)

    # construct an estimator to be used with the policy
    action_no = env.action_space.n
    estimator = get_estimator(
        "atari", hist_len=4, action_no=action_no, hidden_sz=512
    )
    estimator = estimator.cuda()

    # construct an epsilon greedy policy
    # also: epsilon = {'name':'linear', 'start':1, 'end':0.1, 'steps':1000}
    epsilon = get_epsilon(steps=30000)
    policy_evaluation = EpsilonGreedyPolicy(estimator, action_no, epsilon)

    # construct a policy improvement type
    # optimizer = get_optimizer('Adam', estimator, lr=0.0001, eps=0.0003)
    optimizer = optim.Adam(estimator.parameters(), lr=lr)
    policy_improvement = DQNPolicyImprovement(
        estimator, optimizer, gamma=0.99, is_double=is_double
    )

    # we also need an experience replay
    experience_replay = ER(100_000, batch_size=32)
    # experience_replay = ER(100000, batch_size=32, hist_len=4)

    # construct a tester
    tester = None

    # construct a logger
    log = Logger(label=label, path=f"./{label}")
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

    # construct a structure for easily accessing objects and settings
    args = SimpleNamespace(
        env=env,
        policy_evaluation=policy_evaluation,
        policy_improvement=policy_improvement,
        experience_replay=experience_replay,
        tester=tester,
        log=log,
        training_steps=training_steps,
        start_learning_after=256,
        update_freq=1,
    )
    for k, v in args.__dict__.items():
        if k != "env":
            k = clr(k, attrs=["bold"])
            print(f"{k}: {v}")

    # start the training
    train(args)


if __name__ == "__main__":
    import fire

    # create command line arguments from the function's arguments
    fire.Fire(main)
