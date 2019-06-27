r""" Module containing the Categorical Policy Improvement class.
"""
from copy import deepcopy
import torch
from wintermute.utils import get_estimator_device, to_device, DQNLoss


__all__ = [
    "CategoricalPolicyImprovement",
    "get_categorical_loss",
    "get_target_distribution",
]


def get_target_distribution(
    next_states, rewards, mask, gamma, target_estimator, support
):
    r""" Computes the target distribution TZ(x_,a).

    The size of `next_states` can be smaller than that of the actual
    batch size to save computation.
    """
    bsz = rewards.shape[0]
    bsz_ = next_states.shape[0]
    bin_no = support.shape[0]
    v_min, v_max = support[0].item(), support[-1].item()
    delta_z = (v_max - v_min) / (bin_no - 1)

    probs = target_estimator(next_states)
    qs = torch.mul(probs, support.expand_as(probs))
    argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
    action_mask = argmax_a.expand(bsz_, 1, bin_no)
    _qa_probs = probs.gather(1, action_mask).squeeze()

    # Next-states batch can be smaller so we scatter qa_probs in
    # a tensor the size of the full batch with each row summing to 1
    qa_probs = torch.eye(bsz, bin_no, device=_qa_probs.device)
    qa_probs.masked_scatter_(mask.expand_as(qa_probs), _qa_probs)

    # Mask gamma and reshape it torgether with rewards to fit p(x,a).
    rewards = rewards.expand_as(qa_probs)
    gamma = (mask.float() * gamma).expand_as(qa_probs)

    # Compute projection of the application of the Bellman operator.
    bellman_op = rewards + gamma * support.unsqueeze(0).expand_as(rewards)
    bellman_op = torch.clamp(bellman_op, v_min, v_max)

    # Compute categorical indices for distributing the probability
    m = torch.zeros(bsz, bin_no, device=qa_probs.device)
    b = (bellman_op - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # Fix disappearing probability mass when l = b = u (b is int)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (bin_no - 1)) * (l == u)] += 1

    # Distribute probability
    """
    for i in range(bsz):
        for j in range(self.bin_no):
            uidx = u[i][j]
            lidx = l[i][j]
            m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
            m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
    for i in range(bsz):
        m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
        m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))
    """
    # Optimized by https://github.com/tudor-berariu
    offset = (
        torch.linspace(0, ((bsz - 1) * bin_no), bsz, device=qa_probs.device)
        .long()
        .unsqueeze(1)
        .expand(bsz, bin_no)
    )

    m.view(-1).index_add_(
        0, (l + offset).view(-1), (qa_probs * (u.float() - b)).view(-1)
    )
    m.view(-1).index_add_(
        0, (u + offset).view(-1), (qa_probs * (b - l.float())).view(-1)
    )
    return m, probs


def get_categorical_loss(
    batch, estimator, gamma, support, target_estimator=None
):
    r""" Computes the Categorical KL loss phi(TZ(x_,a)) || Z(x,a).

        Largely analogue to the DQN loss routine.

        Args:
            batch (list): The (states, actions, rewards, next_states, done_mask)
                batch.
            estimator (nn.Module): The *online* estimator.
            gamma (float): Discount factor γ.
            support (torch.tensor): A vector of size `bin_no` that is the
                support of the categorical distribution.
            target_estimator (nn.Module, optional): Defaults to None. The target
                estimator. If None the target is computed using the online
                estimator.

        Returns:
            DQNLoss: A simple namespace containing the loss and its byproducts.
    """
    assert target_estimator is not None, "Online target not implemented yet."

    states, actions, rewards, next_states, mask = batch
    bsz = states.shape[0]
    bin_no = support.shape[0]

    # Compute probability distribution of Q(s, a)
    qs_probs = estimator(states)
    action_mask = actions.view(bsz, 1, 1).expand(bsz, 1, bin_no)
    qsa_probs = qs_probs.gather(1, action_mask).squeeze()

    # Compute probability distribution of Q(s_, a)
    with torch.no_grad():
        target_qsa_probs, target_qs_probs = get_target_distribution(
            next_states, rewards, mask, gamma, target_estimator, support
        )

    # Compute the cross-entropy of phi(TZ(x_,a)) || Z(x,a) for each
    # transition in the batch
    qsa_probs = qsa_probs.clamp(min=1e-7)  # Tudor's trick for avoiding nans
    loss = -(target_qsa_probs * torch.log(qsa_probs)).sum(dim=1)

    return DQNLoss(
        loss=loss,
        qsa=qsa_probs,
        qsa_targets=target_qsa_probs,
        q_values=(support, qs_probs),
        q_targets=(support, target_qs_probs),
    )


class CategoricalPolicyImprovement:
    r""" Categorical DQN.

    For more information see `A distributional perspective on RL
    <https://arxiv.org/pdf/1707.06887.pdf>`_.

    """

    def __init__(
        self, estimator, optimizer, gamma, support, target_estimator=True
    ):
        self.device = get_estimator_device(estimator)
        self.estimator = estimator
        self.optimizer = optimizer
        self.gamma = gamma
        self.support = torch.linspace(*support, device=self.device)
        self.v_min, self.v_max, self.bin_no = support
        if target_estimator is True:
            self.target_estimator = deepcopy(estimator)
        else:
            self.target_estimator = target_estimator

    def __call__(self, batch, cb=None):
        batch = to_device(batch, self.device)

        loss = get_categorical_loss(
            batch,
            self.estimator,
            self.gamma,
            self.support,
            target_estimator=self.target_estimator,
        )

        if cb:
            loss = cb(loss)
        else:
            loss = loss.loss.mean()

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
        name += f"(\u03B3={self.gamma}, \u03B1={lr}, "
        name += f"Ω=[{self.v_min, self.v_max}], bins={self.bin_no})"
        return name

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"


def main():
    from torch import nn
    from torch import optim

    action_no = 3
    bsz = 6
    bsz_ = 4
    support = (-1, 1, 7)

    class Net(nn.Module):
        def __init__(self, action_no, bin_no):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3 * 7 * 7, 24)
            self.fc2 = nn.Linear(24, action_no * bin_no)
            self.action_no, self.bin_no = action_no, bin_no

        def forward(self, x):
            y = self.fc2(torch.relu(self.fc1(x)))
            y = y.view(x.shape[0], self.action_no, self.bin_no)
            return torch.softmax(y, dim=2)

    x = torch.rand(bsz_, 3 * 7 * 7)
    rewards = torch.zeros(bsz, 1)
    mask = torch.ones(bsz, 1, dtype=torch.uint8)
    rewards[2, 0] = 0.33
    mask[2, 0] = 0
    mask[4, 0] = 0

    net = Net(action_no, support[2])
    y = net(x)

    print(
        "target p(Q(s_,a)):\n",
        get_target_distribution(x, rewards, mask, 0.92, net, support),
    )


if __name__ == "__main__":
    main()
