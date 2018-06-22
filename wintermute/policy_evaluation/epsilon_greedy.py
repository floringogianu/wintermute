""" Epsilon Greedy. """

from typing import NamedTuple
from numpy import random
from gym.spaces import Discrete

from .deterministic import DeterministicPolicy
from .exploration_schedules import get_schedule as get_epsilon_schedule


class EpsilonGreedyOutput(NamedTuple):
    """ The output of the epsilon greedy policy. """
    action: int
    q_value: float
    full: object


class EpsilonGreedyPolicy(object):
    """ Epsilon greedy policy.

        Takes an estimator and an epsilon greedy schedule to imbue an epsilon
        greedy policy.
    """
    def __init__(self, estimator, action_space, epsilon):

        self.policy = DeterministicPolicy(estimator)

        self.action_space = action_space
        try:
            self.action_space.sample()
        except AttributeError:
            self.action_space = Discrete(self.action_space)

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
        if next(self.epsilon) < random.uniform():
            pi = EpsilonGreedyOutput(action=pi.action, q_value=pi.q_value,
                                     full=pi.full)
            return pi
        pi = EpsilonGreedyOutput(action=self.action_space.sample(), q_value=0,
                                 full={})
        return pi

    def get_estimator(self):
        return self.policy.get_estimator()

    def set_estimator(self, estimator):
        self.policy.set_estimator(estimator)

    def __call__(self, state):
        return self.get_action(state)

    def __str__(self):
        return f'{self.__class__.__name__}(id={self.policy})'

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f'{name} @ {obj_id}'
