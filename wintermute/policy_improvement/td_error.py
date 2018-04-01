import torch.nn.functional as F


def get_td_error(q_values, q_target_values, rewards, gamma):
    """ Compute the temporal difference error.

        td_error = (r + gamma * max Q(s_,a)) - Q(s,a)
    """
    q_target_values.volatile = False  # So we don't mess the huber loss
    expected_q_values = (q_target_values * gamma) + rewards
    return F.smooth_l1_loss(q_values, expected_q_values)
