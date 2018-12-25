""" Utils for example code.
"""

import argparse
from termcolor import colored as clr


def get_parser(**kwargs):
    """ Configure a parser with custom defaults.

    Returns:
        Namespace: Contains the parsed arguments.
    """

    prs = argparse.ArgumentParser(description=kwargs.get("description", "DQN"))
    prs.add_argument("--seed", type=int, default=42, help="RNG seed.")
    prs.add_argument(
        "--game",
        type=str,
        default=kwargs.get("game", "pong"),
        help="ATARI game in the ALE format, eg.: space_invaders, seaquest.",
    )
    prs.add_argument(
        "--label",
        type=str,
        default="",
        help="Experiment label, used in the naming of folders.",
    )
    prs.add_argument(
        "--step_no",
        type=int,
        default=kwargs.get("step_no", 50_000_000),
        help="Total no of training steps.",
    )
    prs.add_argument(
        "--double-dqn",
        action=kwargs.get("double_dqn", "store_true"),
        help="Train with Double-DQN.",
    )
    prs.add_argument(
        "--prioritized",
        action=kwargs.get("prioritized", "store_true"),
        help="Train with Prioritized Experience Replay.",
    )
    prs.add_argument(
        "--lr",
        type=float,
        default=kwargs.get("lr", 0.00025),
        help="Adam learning rate.",
    )
    prs.add_argument(
        "--adam-eps",
        type=float,
        default=1.5e-4,
        metavar="ε",
        help="Adam epsilon",
    )
    prs.add_argument(
        "--update-freq", type=int, default=kwargs.get("update_freq", 4)
    )
    prs.add_argument(
        "--epsilon-steps",
        type=int,
        default=kwargs.get("epsilon_steps", 1_000_000),
    )
    prs.add_argument(
        "--learn-start", type=int, default=kwargs.get("learn_start", 80000)
    )
    prs.add_argument(
        "--mem-size", type=int, default=kwargs.get("mem_size", 1_000_000)
    )
    prs.add_argument("--log-freq", type=int, default=kwargs.get("log_freq", 10))
    prs.add_argument(
        "--no-gym",
        action=kwargs.get("no_gym", "store_true"),
        help="Use directly the ALE-wrapper instead of OpenAI Gym.",
    )
    return prs.parse_args()


def print_namespace(args):
    for k, v in args.__dict__.items():
        if k != "env":
            k = clr(k, "green", attrs=["bold"])
            print(f"{k}: {v}")
