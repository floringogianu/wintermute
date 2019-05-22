""" Deep Q-Learning policy improvement.
"""
from typing import NamedTuple
from copy import deepcopy
import torch


__all__ = ["DQNPolicyImprovement", "get_dqn_loss", "get_td_error", "DQNLoss"]


class DQNLoss(NamedTuple):
    r""" Object returned by :attr:`get_dqn_loss`. """

    loss: torch.Tensor
    q_values: torch.Tensor
    q_targets: torch.Tensor


def get_td_error(q_values, q_target_values, rewards, gamma, loss_fn):
    r""" Compute the temporal difference error using the Huber loss, called
    :class:`torch.nn.SmoothL1Loss`.

    .. math::
        \delta = r_t + \gamma * \text{max}_a \, Q(s_{t+1}, a) - Q(s_t,a_t) \\

        l_{i} =
        \begin{cases}
        0.5 \delta_i^2, & \text{if } |\delta_i| < 1 \\
        |\delta_i| - 0.5, & \text{otherwise }
        \end{cases}

    Args:
        q_values (torch.Tensor): Online Q-values batch.
        q_target_values (torch.Tensor): Target Q-values batch.
        rewards (torch.Tensor): Rewards batch.
        gamma (float): Discount factor :math:`\gamma`.
        loss_fn (torch.nn.Loss): Loss function.

    .. note::

        Return either a single element or a batch size tensor, depending on
        the reduction method.
    """

    expected_q_values = (q_target_values * gamma) + rewards
    return loss_fn(q_values, expected_q_values)


def get_dqn_loss(  # pylint: disable=bad-continuation
    batch,
    estimator,
    gamma,
    target_estimator=None,
    is_double=False,
    loss_fn=torch.nn.MSELoss(reduction="mean"),
):
    r""" Computes the DQN loss or its Double-DQN variant.

    Args:
        batch (list): The (states, actions, rewards, next_states, done_mask)
            batch.
        estimator (nn.Module): The *online* estimator.
        gamma (float): Discount factor γ.
        target_estimator (nn.Module, optional): Defaults to None. The target
            estimator. If None the target is computed using the online
            estimator.
        is_double (bool, optional): Defaults to False. If True it computes
            the Double-DQN loss using the `target_estimator`.
        loss_fn (torch.nn.Loss): Defaults to torch.nn.MSELoss. Custom loss
            function, eg.: torch.nn.SmoothL1Loss.

    Returns:
        DQNLoss: A simple namespace containing the loss and its byproducts.
    """

    states, actions, rewards, next_states, mask = batch

    # Compute Q(s, a)
    q_values = estimator(states)
    qsa = q_values.gather(1, actions)
    mask = mask.squeeze(1)

    if next_states.nelement() != 0:
        # Compute Q(s_, a).
        if target_estimator:
            with torch.no_grad():
                q_targets = target_estimator(next_states)
        else:
            with torch.no_grad():
                q_targets = estimator(next_states)
    else:
        q_targets = None

    # Bootstrap for non-terminal states
    qsa_target = torch.zeros_like(qsa)

    if q_targets is not None:
        if is_double:
            with torch.no_grad():
                next_q_values = estimator(next_states)
                argmax_actions = next_q_values.max(1, keepdim=True)[1]
                qsa_target[mask] = q_targets.gather(1, argmax_actions)
        else:
            qsa_target[mask] = q_targets.max(1, keepdim=True)[0]

    # Compute temporal difference error
    loss = get_td_error(qsa, qsa_target, rewards, gamma, loss_fn)

    return DQNLoss(loss=loss, q_values=q_values, q_targets=q_targets)


class DQNPolicyImprovement:
    r""" Object doing the Deep Q-Learning Policy Improvement step.

    As other objects in this library we override :attr:`__call__`. During a
    call as the one in the example below, several things happen:

        1. Put the batch on the same device as the estimator,
        2. Compute DQN the loss,
        3. Calls the callback if available (eg.: when doing prioritized
           experience replay),
        4. Computes gradients and updates the estimator.

    Example:

    .. code-block:: python

        # construction
        policy_improvement = DQNPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=0.25),
            gamma,
        )

        # usage
        for step in range(train_steps):
            # sample the env and push transitions in experience replay
            batch = experience_replay.sample()
            policy_improvement(batch, cb=None)

            if step % target_update_freq == 0:
                policy_improvement.update_target_estimator()

    Args:
        estimator (nn.Module): Q-Values estimator.
        optimizer (nn.Optim): PyTorch optimizer.
        gamma (float): Discount factor.
        target_estimator (nn.Module, optional): Defaults to None. This
            assumes we always want a target network, since it is a DQN
            update. Therefore if `None`, it will clone `estimator`. However
            if `False` the update rule will use the online network for
            computing targets.
        is_double (bool, optional): Defaults to `False`. Whether to use
            Double-DQN or not.
    """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        estimator,
        optimizer,
        gamma,
        target_estimator=None,
        is_double=False,
        loss_fn="MSELoss",
    ):
        # pylint: enable=bad-continuation
        self.estimator = estimator
        self.target_estimator = target_estimator
        if target_estimator in (None, True):
            self.target_estimator = deepcopy(estimator)
        self.optimizer = optimizer
        self.gamma = gamma
        self.is_double = is_double
        self.loss_fn = getattr(torch.nn, loss_fn)(reduction="none")

        self.device = next(estimator.parameters()).device
        self.optimizer.zero_grad()

    def __call__(self, batch, cb=None):
        r""" Performs a policy improvement step. Several things happen:
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
            loss_fn=self.loss_fn,
        )

        if cb:
            loss = cb(dqn_loss)
        else:
            loss = dqn_loss.loss.mean()

        loss.backward()
        self.update_estimator()

    def update_estimator(self):
        r""" Do the estimator optimization step. Usefull when computing
        gradients across several steps/batches and optimizing using the
        accumulated gradients.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_estimator(self):
        r""" Update the target net with the parameters in the online model."""
        self.target_estimator.load_state_dict(self.estimator.state_dict())

    def get_estimator_state(self):
        r""" Return a reference to the estimator. """
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
