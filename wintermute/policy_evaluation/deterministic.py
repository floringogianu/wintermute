from typing import NamedTuple
import torch


class DeterministicOutput(NamedTuple):
    """ The output of the deterministic policy. """

    action: int
    q_value: float
    full: object


class DeterministicPolicy(object):
    def __init__(self, estimator):
        self.estimator = estimator
        print(type(estimator))
        self.is_cuda = next(estimator.parameters()).is_cuda

    def get_action(self, state, is_train=False):
        """ Takes the best action based on estimated state-action values.

            Returns the best Q-value and its subsequent action.
        """
        if self.is_cuda:
            if isinstance(state, (list, tuple)):
                state = (el.cuda() for el in state)
            else:
                state = state.cuda()

        with torch.set_grad_enabled(is_train):
            qvals = self.estimator(state)
        q_val, argmax_a = qvals.max(1)

        return DeterministicOutput(
            action=argmax_a.squeeze().item(),
            q_value=q_val.squeeze().item(),
            full=qvals,
        )

    def get_estimator_state(self):
        return self.estimator.state_dict()

    def set_estimator_state(self, estimator_state):
        self.estimator.load_state_dict(estimator_state)

    def cuda(self):
        self.estimator.cuda()
        self.is_cuda = True

    def cpu(self):
        self.estimator.cpu()
        self.is_cuda = False

    def __call__(self, state):
        return self.get_action(state)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"
