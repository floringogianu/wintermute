""" Prioritized Experience Replay implementations.

    1. ProportionalSampler implements the proportional-based prioritization
    using the SumTree in `data_structures.py`.

    2. RankSampler implements the rank-based prioritization using the
    PriorityQueue in `data_structures.py`.
"""

import torch
import numpy as np

from .data_structures import SumTree
from .mem_efficient_experience_replay import MemoryEfficientExperienceReplay


class ProportionalSampler:
    """ Implements the proportional-based sampling in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf).
    """

    # pylint: disable=too-many-instance-attributes, bad-continuation
    # nine attrs is reasonable in this case.
    def __init__(  # pylint: disable=bad-continuation
        self,
        er,
        alpha=0.6,
        beta=None,
        async_memory: bool = True,
        optim_steps=None,
        epsilon=0.000_000_1,
        **kwargs,
    ) -> None:

        if not isinstance(er, MemoryEfficientExperienceReplay) or er.is_async:
            raise RuntimeError(
                "ER must be non-async MemoryEfficentExperienceReplay."
            )

        self._er = er
        self._sumtree = SumTree(capacity=self._er.capacity)
        self.__alpha = alpha
        self.__beta = beta
        if self.__beta is not None and optim_steps:
            print(self.__beta, optim_steps)
            self.__beta_step = (1 - self.__beta) / optim_steps
        else:
            self.__beta_step = None
        self.__epsilon = epsilon
        self.__max = 1

        if async_memory:
            import concurrent.futures

            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            )
            self.push = self._async_push
            self.sample = self._async_sample
            self.push_and_sample = self._async_push_and_sample

            self._sample_result = None
            self._push_result = None
        else:
            self.push = self._push
            self.sample = self._sample
            self.push_and_sample = self._push_and_sample

        self.__is_async = async_memory

    def __wait(self):
        if self._push_result is not None:
            self._push_result.result()
            self._push_result = None

        if self._sample_result is not None:
            self._sample_result.result()

    def _push(self, transition, priority=None):
        pos = self._er.push(transition)
        priority = priority or (self.__epsilon ** self.__alpha + self.__max)
        self._sumtree.update(pos, priority)

    def _async_push(self, transition, priority=None):
        self.__wait()
        self._push_result = self._executor.submit(
            self._push, transition, priority
        )

    def _sample(self):
        idxs = []
        batch_size = self.batch_size
        probs = []  # keep the un-normalized probabilites
        mem_size = len(self)
        total_prob = self._sumtree.get_sum()
        segment_sz = total_prob / batch_size

        for i in range(batch_size):
            start = i * segment_sz
            end = (i + 1) * segment_sz
            idx, prob = self._sumtree.get(np.random.uniform(start, end))
            idxs.append(idx)
            probs.append(prob)

        # compute the importance sampling weights
        if self.__beta is not None:
            weights = torch.tensor(probs) / total_prob  # pylint: disable=E1102
            weights = (mem_size * weights) ** -self.__beta
            weights /= weights.max()
        else:
            # we basically disable importance sampling
            weights = torch.tensor(probs).fill_(1)  # pylint: disable=E1102

        if self.__beta_step:
            # anneal the beta
            self.__beta = min(self.__beta + self.__beta_step, 1)

        return self._er.sample(gods_idxs=idxs), idxs, weights

    def _async_sample(self):
        self.__wait()
        if self._sample_result is None:
            batch = self._sample()
        else:
            batch = self._sample_result.result()

        self._sample_result = self._executor.submit(self._sample)
        return batch

    def _push_and_sample(self, transition: list):
        if isinstance(transition[0], list):
            for trans in transition:
                self._push(trans)
        else:
            self._push(transition)
        return self._sample()

    def _async_push_and_sample(self, transition):
        self.__wait()
        if self._sample_result is not None:
            batch = self._sample_result.result()
        else:
            batch = self._sample()

        self._sample_result = self._executor.submit(
            self._push_and_sample, transition
        )
        return batch

    def update(self, idxs, priorities):
        """ Updates the priorities of the last transitions sampled. """
        if self.__is_async:
            self.__wait()
        for priority, idx in zip(priorities, idxs):
            priority = (priority + self.__epsilon) ** self.__alpha
            self._sumtree.update(idx, priority)
            self.__max = max(priority, self.__max)

    @property
    def batch_size(self) -> int:
        """ Batch size, duh!
        """
        return self._er.batch_size

    def __len__(self):
        return len(self._er)

    def __str__(self):
        props = (
            "capacity={0}, size={1}, α={2}, β={3}, batch={4}, async={5}"
        ).format(
            self._er.capacity,
            len(self._er),
            self.__alpha,
            self.__beta,
            self.batch_size,
            self.__is_async,
        )

        return f"ProportionalExperienceReplay({props})"
