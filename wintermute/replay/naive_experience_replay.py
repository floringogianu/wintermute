""" Naive Experience Replay and helper functions.

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
        states = states.view(n, hist * c, h, w)
        next_states = next_states.view(n, hist * c, h, w)

    return [states, actions, rewards, next_states, mask]


class NaiveExperienceReplay:
    """
    Stores in a circular buffer transitions containing observations formed by
    concatenating several (usually four, in DQN) frames as opposed to
    FlatExperienceReplay which stores transitions containing the current frame.

    This makes Naive Experience Replay faster at the expense of RAM. The only
    memory optimiation is that it can store either full transitions
    (_s, _a, r, s, d) or half transitions (_s, _a, r, d).
    """
    # pylint: disable=too-many-instance-attributes, bad-continuation
    # eight attrs is reasonable in this case.
    def __init__(
        self,
        capacity=100_000,
        batch_size=32,
        collate=None,
        full_transition=False,
    ):
        # pylint: enable=bad-continuation
        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.full_transition = full_transition

        self.sample = self._sample_full if full_transition else self._sample
        self._collate = collate or _collate
        self.position = 0
        self.__last_state = None

    def push(self, transition):
        """ Add a transition tuple to the buffer. Several things happen:
            1. Keep the last state for the corner case in which we sample the
            last transition in the buffer.
            2. If we don't store full transitions we strip the tuple
            3. Add to the cyclic buffer

        Args:
            transition (tuple): Contains an (_s, _a, r, [s], d) experience.
        """
        if not self.full_transition:
            self.__last_state = transition[3]
            transition = [el for i, el in enumerate(transition) if i != 3]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def _sample(self):
        samples = []
        idxs = torch.randint(0, len(self.memory), (self.batch_size,))

        for idx in idxs:
            if idx == self.position - 1:  # the most recent transition
                next_state = self.__last_state
            elif idx == self.capacity - 1:  # last in ER (not the most recent)
                next_state = self.memory[0][0]
            else:
                next_state = self.memory[idx + 1][0]

            samples.append(
                [
                    self.memory[idx][0],
                    self.memory[idx][1],
                    self.memory[idx][2],
                    next_state,
                    self.memory[idx][3],
                ]
            )

        return self._collate(samples)

    def _sample_full(self):
        idxs = torch.randint(0, len(self.memory), (self.batch_size,))
        samples = [self.memory[idxs[i]] for i in range(self.batch_size)]
        return self._collate(samples)

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            + f"(batch={self.batch_size}, sz={self.capacity})"
        )

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"
