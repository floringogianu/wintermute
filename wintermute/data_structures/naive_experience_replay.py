""" Naive Experience Replay.

    Stores in a circular buffer full transitions with no memory optimization
    as opposed to nTupleExperienceReplay.
"""
import torch


def _collate(samples):
    batch = list(zip(*samples))
    states = torch.cat(batch[0], 0)
    actions = torch.LongTensor(batch[1]).unsqueeze(1)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
    next_states = torch.cat(batch[3], 0)
    mask = 1 - torch.ByteTensor(batch[4]).unsqueeze(1)

    # if we train with full RGB information (three channels instead of one)
    if states.ndimension() == 5:
        n, hist, c, h, w = states.size()
        states = states.view(n, hist*c, h, w)
        next_states = next_states.view(n, hist*c, h, w)

    return [states, actions, rewards, next_states, mask]


class NaiveExperienceReplay(object):
    def __init__(self, capacity=100000, batch_size=32, collate=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0
        if collate:
            self._collate = collate
        else:
            self._collate = _collate

    def push(self, s, a, r, d):
        data = [s, a, r, d]
        if len(self.memory) < self.capacity:
            self.memory.append(data)
        else:
            self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        assert self.batch_size < len(self.memory)

        idxs = torch.LongTensor(self.batch_size).random_(0, len(self.memory) -1)
        samples = [[self.memory[idxs[i]][0],
                    self.memory[idxs[i]][1],
                    self.memory[idxs[i]][2],
                    self.memory[idxs[i] + 1][0],
                    self.memory[idxs[i]][3]] for i in range(self.batch_size)]

        return self._collate(samples)

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return (f'{self.__class__.__name__}' +
                f'(batch={self.batch_size}, sz={self.capacity})')

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f'{name} @ {obj_id}'
