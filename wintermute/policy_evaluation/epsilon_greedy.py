""" Epsilon Greedy. """

from typing import NamedTuple
from numpy import random

from .deterministic import DeterministicPolicy
from .exploration_schedules import get_schedule as get_epsilon_schedule


class EpsilonGreedyOutput(NamedTuple):
    """ The output of the epsilon greedy policy. """

    action: int
    q_value: float
    full: object


class EpsilonGreedyPolicy(object):
    

    def __init__(self, estimator, action_space, epsilon):
        """ Epsilon greedy policy.

        Takes an estimator and an epsilon greedy schedule to imbue an epsilon
        greedy policy.
        
        Args:
            estimator (nn.Module): Q-value estimator.
            action_space (int): No of actions.
            epsilon ([dict, iterator]): A dict with keys `name`, `start`, `end`,
                `steps`, `warmup_steps` or an iterator returned by
                `policy_evaluation.get_schedule`.
        """


        self.policy = DeterministicPolicy(estimator)
        self.action_space = action_space

        self.epsilon = epsilon
        try:
            epsilon = next(self.epsilon)
        except TypeError:
            self.epsilon = get_epsilon_schedule(**self.epsilon)
            epsilon = next(self.epsilon)

    def get_action(self, state):
        """ Selects an action based on an epsilon greedy strategy.

            Returns the Q-value and the epsilon greedy action.
        """
        pi = self.policy.get_action(state)
        epsilon = next(self.epsilon)
        if epsilon < random.uniform():
            return EpsilonGreedyOutput(
                action=pi.action, q_value=pi.q_value, full=pi.full
            )
        pi = EpsilonGreedyOutput(
            action=random.randint(0, self.action_space), q_value=0, full={}
        )
        return pi

    def get_estimator_state(self):
        return self.policy.get_estimator_state()

    def set_estimator_state(self, estimator_state):
        self.policy.set_estimator_state(estimator_state)

    def cuda(self):
        self.policy.cuda()

    def cpu(self):
        self.policy.cpu()

    def __call__(self, state):
        return self.get_action(state)

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.policy})"

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"
