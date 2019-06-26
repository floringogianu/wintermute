from .deterministic import DeterministicPolicy, CategoricalDeterministicPolicy
from .epsilon_greedy import EpsilonGreedyPolicy

# from .categorical import CategoricalPolicyEvaluation
from .exploration_schedules import get_schedule as get_epsilon_schedule
