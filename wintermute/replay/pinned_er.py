""" This file contains a version of MemoryEfficientExperienceReplay that
    allocates from the start full-capacity tensors
"""

import numpy.random
import torch

from .mem_efficient_experience_replay import MemoryEfficientExperienceReplay


class PinnedExperienceReplay(MemoryEfficientExperienceReplay):
    """ docstring not found.
    """

    def __init__(  # pylint: disable=bad-continuation, super-init-not-called
        self,
        capacity: int = 100_000,
        batch_size: int = 32,
        hist_len: int = 4,
        async_memory: bool = True,
        scren_dtype=torch.uint8,
        mask_dtype=torch.uint8,
        screen_size: tuple = (84, 84),
        bootstrap_args=None,
        device="cuda",
    ) -> None:

        self.memory = (
            torch.empty(
                capacity + 1, 1, *screen_size, device=device, dtype=scren_dtype
            ),
            torch.empty(capacity, 1, device=device, dtype=torch.long),
            torch.empty(capacity, 1, device=device, dtype=torch.float),
            torch.empty(capacity, 1, device=device, dtype=mask_dtype),
        )
        self.capacity = capacity
        self.batch_size = batch_size
        self.histlen = hist_len
        self.__size = 0

        if bootstrap_args is not None:
            boot_no, boot_prob = bootstrap_args
            if boot_no < 1:
                raise ValueError(f"nheads should be positive (got {boot_no})")
            if not 0.0 <= boot_prob <= 1.0:
                raise ValueError(f"p should be a probability (got {boot_prob}")
            self.bootstrap_args = boot_no, boot_prob
            self._probs = torch.empty(boot_no, self.batch_size).fill_(boot_prob)

        self._sample = (
            self._simple_sample if bootstrap_args is None else self._boot_sample
        )

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

        self.position = 0
        self.__new_episode = True
        self.__mask_dtype = mask_dtype
        self.device = device

    def _push(self, transition: list) -> int:
        position = self.position
        self.memory[0][position].copy_(transition[0][0, -1:])
        self.memory[1][position] = transition[1]
        self.memory[2][position] = transition[2]
        self.memory[3][position] = 1 - bool(transition[4])
        self.memory[0][self.capacity].copy_(transition[3][0, -1:])
        self.position += 1
        self._size = max(self._size, self.position)
        self.position = self.position % self.capacity
        return position

    def _simple_sample(self, gods_idxs=None):
        obs_idxs = []
        next_obs_idxs = []
        idxs = []

        nmemory = self.__size
        capacity = self.capacity
        memory = self.memory
        batch_size = self.batch_size
        hist_len = self.histlen

        if gods_idxs is None:
            gods_idxs = numpy.random.randint(0, nmemory, (self.batch_size,))

        for idx in gods_idxs[::-1]:
            idxs.append(idx)
            obs_idxs.append(idx)
            if idx == self.position - 1:
                next_obs_idxs.append(capacity)
            else:
                next_obs_idxs.append((idx + 1) % capacity)
            bidx = idx
            last_idx = idx
            found_done = False
            for _ in range(hist_len - 1):
                bidx = (bidx - 1) % self.capacity
                next_obs_idxs.append(obs_idxs[-1])
                if not found_done:
                    if memory[3][bidx] == 0:
                        found_done = True
                    else:
                        last_idx = bidx
                obs_idxs.append(last_idx)

        # pylint: disable=E1102
        obs_idxs = torch.tensor(
            obs_idxs[::-1], dtype=torch.long, device=self.device
        )
        next_obs_idxs = torch.tensor(
            next_obs_idxs[::-1], dtype=torch.long, device=self.device
        )
        idxs = torch.tensor(idxs[::-1], dtype=torch.long, device=self.device)

        frame_size = memory[0].size()[2:]

        states = (
            memory[0]
            .index_select(0, obs_idxs)
            .view(batch_size, hist_len, *frame_size)
        )
        actions = memory[1].index_select(0, idxs)
        rewards = memory[2].index_select(0, idxs)
        notdone = memory[3].index_select(0, idxs)
        next_states = (
            memory[0]
            .index_select(0, next_obs_idxs)
            .view(batch_size, hist_len, *frame_size)
        )

        # if we train with full RGB information (three channels instead of one)
        if states.ndimension() == 5:
            bsz, hist, nch, height, width = states.size()
            states = states.view(bsz, hist * nch, height, width)
            next_states = next_states.view(bsz, hist * nch, height, width)

        return [states, actions, rewards, next_states, notdone]

    def __len__(self):
        return self.__size
