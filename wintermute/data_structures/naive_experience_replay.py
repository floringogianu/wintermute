""" Naive Experience Replay.

    Stores in a circular buffer transitions containing observations formed by
    concatenating several (usually four, in DQN) frames as opposed to
    FlatExperienceReplay which stores transitions containing the current frame.

    This makes Naive Experience Replay faster at the expense of RAM. The only
    memory optimiation is that it can store either full transitions
    (_s, _a, r, s, d) or half transitions (_s, _a, r, d).
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
    def __init__(self, capacity=100000, batch_size=32, collate=None,
                 full_transition=False):
        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size

        # store (s, a, r_, s_, d_) if True and (s, a, r_, d_) if False
        self.full_transition = full_transition

        if self.full_transition:
            print("Experience Replay expects (s, a, r_, s_, d_) transitions.")
            self.sample = self._sample_full
        else:
            print("Experience Replay expects (s, a, r_, d_) transitions.")
            self.sample = self._sample

        self._collate = collate or _collate
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def _sample(self):
        idxs = torch.randint(0, len(self.memory)-1, (self.batch_size,))
        samples = [[self.memory[idxs[i]][0],
                    self.memory[idxs[i]][1],
                    self.memory[idxs[i]][2],
                    self.memory[idxs[i] + 1][0],
                    self.memory[idxs[i]][3]] for i in range(self.batch_size)]

        return self._collate(samples)

    def _sample_full(self):
        idxs = torch.randint(0, len(self.memory), (self.batch_size,))
        samples = [self.memory[idxs[i]] for i in range(self.batch_size)]
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
