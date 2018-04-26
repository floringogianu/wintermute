""" Deep Q-Learning policy improvement.
"""
from typing import NamedTuple
from copy import deepcopy
import torch
from torch import Tensor

from .td_error import get_td_error


class DQNLoss(NamedTuple):
    """ By-products of computing the DQN loss. """
    loss: Tensor
    q_values: Tensor
    q_targets: Tensor


class DQNPolicyImprovement(object):
    """ Object doing the Deep Q-Learning Policy Improvement. """
    def __init__(self, estimator, optimizer, gamma):
        self.estimator = estimator
        self.target_estimator = deepcopy(estimator)
        self.optimizer = optimizer
        self.gamma = gamma
        self.optimizer.zero_grad()

    def compute_loss(self, batch):
        """ Returns the DQN loss. """
        states, actions, rewards, next_states, mask = batch

        # Compute Q(s, a)
        q_values = self.estimator(states)
        qsa = q_values.gather(1, actions)

        # Compute Q(s_, a).
        with torch.no_grad():
            q_targets = self.target_estimator(next_states)

        # Bootstrap for non-terminal states
        qsa_target = torch.zeros_like(qsa)
        qsa_target[mask] = q_targets.max(1, keepdim=True)[0][mask]

        # Compute loss
        loss = get_td_error(qsa, qsa_target, rewards, self.gamma)

        return DQNLoss(loss=loss, q_values=q_values, q_targets=q_targets)

    def update_estimator(self):
        """ Do the estimator optimization step. """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_estimator(self):
        """ Update the target net with the parameters in the online model."""
        self.target_estimator.load_state_dict(self.estimator.state_dict())

    def get_estimator(self):
        """ Return a pointer to the estimator. """
        return self.estimator

    def __call__(self, batch):
        loss = self.compute_loss(batch).loss
        loss.backward()
        self.update_estimator()

    def __str__(self):
        lr = self.optimizer.param_groups[0]['lr']
        return (f'{self.__class__.__name__}' +
                f'(\u03B3={self.gamma}, \u03B1={lr})')

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f'{name} @ {obj_id}'
