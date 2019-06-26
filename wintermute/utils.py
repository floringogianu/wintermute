""" Utility functions used across the library.
"""
from typing import NamedTuple
import torch


class DQNLoss(NamedTuple):
    r""" Object returned by :attr:`get_dqn_loss`. """

    loss: torch.Tensor
    qsa: torch.Tensor
    qsa_targets: torch.Tensor
    q_values: torch.Tensor
    q_targets: torch.Tensor


def get_estimator_device(estimator):
    r""" Returns the estimator's device.
    """
    params = estimator.parameters()
    if isinstance(params, list):
        return next(params[0]["params"]).device
    return next(params).device


def to_device(data, device):
    r""" Moves the data on the specified device irrespective of
    data being a tensor or a tensor in a container.

    Usefull in several situations:
        1. move a `batch: [states, rewards, ...]`
        2. move a `state: [image, instructions]`
        3. move a batch of mixed types.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(el, device) for el in data]
    return data.to(device)
