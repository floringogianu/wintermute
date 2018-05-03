""" Neural Network architecture for Atari games.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNet(nn.Module):
    """ Estimator used for ATARI games.
    """
    def __init__(self, input_channels, hist_len, out_size, hidden_size=256):
        super(AtariNet, self).__init__()

        input_depth = hist_len * input_channels

        self.is_categorical = False
        if isinstance(out_size, tuple):
            self.is_categorical = True
            self.action_no, atoms_no = out_size
            out_size = self.action_no * atoms_no

        self.conv1 = nn.Conv2d(input_depth, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.head = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        assert x.data.type().split('.')[-1] == 'ByteTensor', 'The model \
                expects states of type ByteTensor'
        x = x.float().div_(255)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        out = self.head(x.view(x.size(0), -1))
        if self.is_categorical:
            splits = out.chunk(self.action_no, 1)
            return torch.stack(list(map(lambda s: F.softmax(s), splits)), 1)
        return out
