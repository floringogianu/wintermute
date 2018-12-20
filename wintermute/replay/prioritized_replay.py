""" Prioritized Experience Replay implementations.

    1. ProportionalSampler implements the proportional-based prioritization
    using the SumTree in `data_structures.py`.

    2. RankSampler implements the rank-based prioritization using the
    PriorityQueue in `data_structures.py`.
"""

import torch
import numpy as np

from .data_structures import PriorityQueue, SumTree
from .naive_experience_replay import _collate


class ProportionalSampler:
    """ Implements the proportional-based sampling in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf).
    """

    # pylint: disable=too-many-instance-attributes, bad-continuation
    # nine attrs is reasonable in this case.
    def __init__(
        self,
        capacity,
        batch_size=32,
        collate=None,
        full_transition=False,
        optim_steps=49_980_000,
        **kwargs,
    ):
        # pylint: enable=bad-continuation
        self.__data = []
        self.__sumtree = SumTree(capacity=capacity)
        self.__capacity = capacity
        self.__batch_size = batch_size
        self.__collate = collate or _collate
        self.__alpha = kwargs["alpha"] if "alpha" in kwargs else 0.6
        self.__beta = kwargs["beta"] if "beta" in kwargs else 0.4
        self.__beta_step = (1 - self.__beta) / optim_steps
        self.__epsilon = (
            kwargs["epsilon"] if "epsilon" in kwargs else 0.000_000_1
        )

        self.__pos = 0
        self.__full_transition = full_transition
        if self.__full_transition:
            print("Experience Replay expects (s, a, r_, s_, d_) transitions.")
            self.__retrieve = self.__retrieve_full
        else:
            print("Experience Replay expects (s, a, r_, d_) transitions.")
            self.__retrieve = self.__retrieve_half
        self.__sampled_idxs = []
        self.__weights = []
        self.__max = 1

    def push(self, transition, priority=None):
        """ Push new transition to the experience replay. If priority not
        available then initialize with a large priority making sure every new
        transition is being sampled and updated.
        """
        priority = priority or (self.__epsilon ** self.__alpha + self.__max)
        self.__sumtree.update(self.__pos, priority)

        if len(self.__data) < self.__capacity:
            self.__data.append(transition)
        else:
            self.__data[self.__pos] = transition

        self.__pos = (self.__pos + 1) % self.__capacity

    def update(self, priorities):
        """ Updates the priorities of the last transitions sampled. """
        for priority, idx in zip(priorities, self.__sampled_idxs):
            priority = (priority + self.__epsilon) ** self.__alpha
            self.__sumtree.update(idx, priority)
            self.__max = max(priority, self.__max)

    def sample(self):
        # TODO: docstring

        self.__sampled_idxs = []
        probs = []  # keep the un-normalized probabilites
        mem_size = len(self)
        total_prob = self.__sumtree.get_sum()
        segment_sz = total_prob / self.__batch_size

        for i in range(self.__batch_size):
            a = i * segment_sz
            b = (i + 1) * segment_sz
            idx, prob = self.__sumtree.get(np.random.uniform(a, b))
            # hack, need to figure out this...
            is_valid = False
            while not is_valid:
                idx, prob = self.__sumtree.get(np.random.uniform(0, b))
                is_valid = idx not in (self.__pos - 2, mem_size - 1)
            self.__sampled_idxs.append(idx)
            probs.append(prob)

        # compute the importance sampling weights
        weights = torch.tensor(probs) / total_prob
        weights = (mem_size * weights) ** -self.__beta
        self.__weights = weights / weights.max()

        # anneal the beta
        self.__beta = min(self.__beta + self.__beta_step, 1)

        samples = self.__retrieve()
        return self.__collate(samples)

    @property
    def weights(self):
        """ Returns the importance sampling weights. """
        return self.__weights

    def __retrieve_full(self):
        return [self.__data[idx] for idx in self.__sampled_idxs]

    def __retrieve_half(self):
        return [
            [
                self.__data[idx][0],
                self.__data[idx][1],
                self.__data[idx][2],
                self.__data[idx + 1][0],
                self.__data[idx][3],
            ]
            for idx in self.__sampled_idxs
        ]

    def __len__(self):
        return len(self.__data)

    def __str__(self):
        props = f"size={len(self)}, α={self.__alpha}, batch={self.__batch_size}"
        return f"ProportionalSampler({props})"


class RankSampler:
    """ Implements the rank-based sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf).
    """

    def __init__(self, capacity, batch_size=32, collate=None, alpha=0.9):
        self.__pq = PriorityQueue()
        self.__capacity = capacity
        self.__batch_size = batch_size

        self.__collate = collate or _collate
        self.__position = 0

        self.__alpha = alpha
        self.__partitions = []
        self.__segments = []
        self.__segment_probs = []

    def push(self, transition, priority=None):
        """ Commit new transition to the PQ. If priority is not available then
        initialize with a large value making sure every new transition is being
        sampled and updated. Since our PQ is a Min-PQ, we use the negative of
        the priority.
        """
        priority = (self.__position + 1000) or priority
        self.__pq.push((-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity

        if self.__capacity == len(self):
            self.__compute_segments()

    def sample(self):
        # TODO: docstring
        segment_idxs = np.random.choice(
            len(self.__segments), size=self.__batch_size, p=self.__segment_probs
        )
        segments = [self.__segments[sid] for sid in segment_idxs]
        idxs = [np.random.randint(*segment) for segment in segments]

        # warning, atypical use of a priority queue
        # pylint: disable=protected-access
        samples = [(i, self.__pq._PriorityQueue__heap[i][1]) for i in idxs]
        # pylint: enable=protected-access

        return self.__collate(samples)

    def update(self, idx, priority):
        self.__pq.update(idx, -priority)

    def sort(self):
        for _ in range(len(self)):
            self.__pq.push(self.__pq.pop())

    def __compute_segments(self):
        N = len(self)
        self.__partitions = []
        self.__segments = []

        segment_sz = int(np.floor(N / self.__batch_size))
        for i in range(self.__batch_size):
            a = i * segment_sz
            b = (i + 1) * segment_sz if i != (self.__batch_size - 1) else N

            partition = [(1 / (idx + 1)) ** self.__alpha for idx in range(a, b)]

            self.__partitions.append(np.sum(partition))
            self.__segments.append((a, b))

        self.__segment_probs = [
            p / sum(self.__partitions) for p in self.__partitions
        ]

    def __len__(self):
        return len(self.__pq)

    def __repr__(self):
        props = f"size={len(self)}, α={self.__alpha}, batch={self.__batch_size}"
        return f"RankSampler({props})"
