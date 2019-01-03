import numpy.random
import torch

from .mem_efficient_experience_replay import MemoryEfficientExperienceReplay


class PinnedExperienceReplay(MemoryEfficientExperienceReplay):
    """ docstring not found.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        capacity: int = 100_000,
        batch_size: int = 32,
        hist_len: int = 4,
        async_memory: bool = True,
        scren_dtype=torch.uint8,
        mask_dtype=torch.uint8,
        screen_size: tuple = (84, 84),
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

        if async_memory:
            import concurrent.futures

            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            )
            self.push = self._push
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

    def _push(self, transition: list):
        position = self.position
        self.memory[0][position].copy_(transition[0][0, -1:])
        self.memory[1][position] = transition[1]
        self.memory[2][position] = transition[2]
        self.memory[3][position] = 1 - bool(transition[4])
        self.memory[0][self.capacity].copy_(transition[3][0, -1:])
        self.position = (self.position + 1) % self.capacity
        self.__size = max(self.__size, self.position)

    def _sample(self):
        obs_idxs = []
        next_obs_idxs = []
        idxs = []

        nmemory = self.__size
        capacity = self.capacity
        memory = self.memory
        batch_size = self.batch_size
        hist_len = self.histlen

        for idx in numpy.random.randint(0, nmemory, (self.batch_size,)):
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
            bsz, hist, nch, h, w = states.size()
            states = states.view(bsz, hist * nch, h, w)
            next_states = next_states.view(bsz, hist * nch, h, w)

        return [states, actions, rewards, next_states, notdone]

    def __len__(self):
        return self.__size
