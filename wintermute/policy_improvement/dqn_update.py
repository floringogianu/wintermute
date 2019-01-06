""" Deep Q-Learning policy improvement.
"""
from typing import NamedTuple
from copy import deepcopy
import torch
import torch.nn.functional as F


__all__ = ["DQNPolicyImprovement", "get_dqn_loss", "get_td_error", "DQNLoss"]


class DQNLoss(NamedTuple):
    """ By-products of computing the DQN loss. """

    loss: torch.Tensor
    q_values: torch.Tensor
    q_targets: torch.Tensor


def get_td_error(  # pylint: disable=bad-continuation
    q_values, q_target_values, rewards, gamma, reduction="elementwise_mean"
):
    """ Compute the temporal difference error:
        td_error = (r + gamma * max Q(s_,a)) - Q(s,a)

    Args:
        q_target_values (torch.Tensor): Target values.
        rewards (torch.Tensor): Rewards batch
        gamma (float): Discount factor γ.
        reduction (str, optional): Defaults to "elementwise_mean". Loss
            reduction method, see PyTorch docs.

    Returns:
        torch.Tensor: Either a single element or a batch size tensor, depending
            on the reduction method.
    """

    expected_q_values = (q_target_values * gamma) + rewards
    return F.smooth_l1_loss(q_values, expected_q_values, reduction=reduction)


def get_dqn_loss(  # pylint: disable=bad-continuation
    batch, estimator, gamma, target_estimator=None, is_double=False
):
    """ Computes the DQN loss or its Double-DQN variant.

    Args:
        estimator (nn.Module): The *online* estimator.
        gamma (float): Discount factor γ.
        target_estimator (nn.Module, optional): Defaults to None. The target
            estimator. If None the target is computed using the online
            estimator.
        is_double (bool, optional): Defaults to False. If True it computes
            the Double-DQN loss using the `target_estimator`.

    Returns:
        DQNLoss: A simple namespace containing the loss and its byproducts.
    """

    states, actions, rewards, next_states, mask = batch

    # Compute Q(s, a)
    q_values = estimator(states)
    qsa = q_values.gather(1, actions)

    # Compute Q(s_, a).
    if target_estimator is not None:
        with torch.no_grad():
            q_targets = target_estimator(next_states)
    else:
        with torch.no_grad():
            q_targets = estimator(next_states)

    # Bootstrap for non-terminal states
    qsa_target = torch.zeros_like(qsa)

    if is_double:
        with torch.no_grad():
            next_q_values = estimator(next_states)
            argmax_actions = next_q_values.max(1, keepdim=True)[1]
            qsa_target[mask] = q_targets.gather(1, argmax_actions)[mask]
    else:
        qsa_target[mask] = q_targets.max(1, keepdim=True)[0][mask]

    # Compute temporal difference error
    loss = get_td_error(qsa, qsa_target, rewards, gamma, reduction="none")

    return DQNLoss(loss=loss, q_values=q_values, q_targets=q_targets)


class DQNPolicyImprovement:
    """ Object doing the Deep Q-Learning Policy Improvement. """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        estimator,
        optimizer,
        gamma,
        target_estimator=None,
        is_double=False,
    ):
        # pylint: enable=bad-continuation
        self.estimator = estimator
        self.target_estimator = target_estimator
        if not target_estimator:
            self.target_estimator = deepcopy(estimator)
        self.optimizer = optimizer
        self.gamma = gamma
        self.is_double = is_double

        self.device = next(estimator.parameters()).device
        self.optimizer.zero_grad()

    def __call__(self, batch, cb=None):
        """ Performs a policy improvement step. Several things happen:
            1. Put the batch on the device the estimator is on,
            2. Computes DQN the loss,
            3. Calls the callback if available,
            4. Computes gradients and updates the estimator.

        Args:
            batch (list): A (s, a, r, s_, mask, (meta, optional)) list. States
                and States_ can also be lists of tensors for composed states
                (eg. frames + nlp_instructions).
            cb (function, optional): Defaults to None. A function performing
                some other operations with/on the `dqn_loss`. Examples
                include weighting the loss and updating priorities in
                prioritized experience replay.
        """

        if isinstance(batch[0], (list, tuple)):
            states, actions, rewards, next_states, mask = batch
            batch = [
                [el.to(self.device) for el in states],
                actions.to(self.device),
                rewards.to(self.device),
                [el.to(self.device) for el in next_states],
                mask.to(self.device),
            ]
        else:
            batch = [el.to(self.device) for el in batch]

        dqn_loss = get_dqn_loss(
            batch,
            self.estimator,
            self.gamma,
            target_estimator=self.target_estimator,
            is_double=self.is_double,
        )

        if cb:
            loss = cb(dqn_loss)
        else:
            loss = dqn_loss.loss.mean()

        loss.backward()
        self.update_estimator()

    def update_estimator(self):
        """ Do the estimator optimization step. """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_estimator(self):
        """ Update the target net with the parameters in the online model."""
        self.target_estimator.load_state_dict(self.estimator.state_dict())

    def get_estimator_state(self):
        """ Return a pointer to the estimator. """
        return self.estimator.state_dict()

    def __str__(self):
        lr = self.optimizer.param_groups[0]["lr"]
        name = self.__class__.__name__
        if self.is_double:
            name = f"Double{name}"
        return name + f"(\u03B3={self.gamma}, \u03B1={lr})"

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"
