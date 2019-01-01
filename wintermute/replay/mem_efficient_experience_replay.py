import numpy.random
import torch


def _collate(batch, batch_size, histlen, mask_dtype=torch.uint8):
    device = batch[0][0].device
    frame_size = batch[0][0].size()[2:]
    states = torch.cat(batch[0], 0).view(batch_size, histlen, *frame_size)
    actions = torch.tensor(batch[1], device=device, dtype=torch.long).unsqueeze_(1)
    rewards = torch.tensor(batch[2], device=device, dtype=torch.float).unsqueeze_(1)
    next_states = torch.cat(batch[3], 0).view(batch_size, histlen, *frame_size)
    mask = 1 - torch.tensor(batch[4], device=device, dtype=mask_dtype).unsqueeze_(1)

    # if we train with full RGB information (three channels instead of one)
    if states.ndimension() == 5:
        n, hist, c, h, w = states.size()
        states = states.view(n, hist * c, h, w)
        next_states = next_states.view(n, hist * c, h, w)

    return [states, actions, rewards, next_states, mask]


class MemoryEfficientExperienceReplay:
    """Code is adapted from NaiveExperienceReplay. Same, same but different.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        capacity: int = 100_000,
        batch_size: int = 32,
        collate=None,
        hist_len: int = 4,
        async_memory: bool = True,
        mask_dtype=torch.uint8,
    ) -> None:

        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.histlen = hist_len

        if async_memory:
            import concurrent.futures

            self.__executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.push = self._push
            self.sample = self._async_sample
            self.push_and_sample = self._async_push_and_sample

            self.__sample_result = None
            self.__push_result = None
        else:
            self.push = self._push
            self.sample = self._sample
            self.push_and_sample = self._push_and_sample

        self._collate = collate or _collate
        self.position = 0
        self.__new_episode = True
        self.__last_state = None
        self.__mask_dtype = mask_dtype

    def __push_one(self, transition: list):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def _push(self, transition: list) -> None:
        with torch.no_grad():
            state = transition[0].split(1, 1)
        if self.__new_episode:
            for obs in state[:-1]:
                self.__push_one([obs.clone(), None, None, None])
        self.__new_episode = done = bool(transition[4])
        self.__push_one([state[-1].clone(), transition[1], transition[2], done])
        self.__last_state = transition[3][:, -1:]

    def _sample(self):
        batch = [], [], [], [], []
        ngood = 0
        nmemory = len(self.memory)

        while ngood < self.batch_size:
            idx = numpy.random.randint(0, nmemory)
            transition = self.memory[idx]
            if transition[1] is not None:
                batch[0].append(self.memory[idx - self.histlen + 1][0])
                for bidx in range(idx - self.histlen + 2, idx):
                    batch[0].append(self.memory[bidx][0])
                    batch[3].append(self.memory[bidx][0])
                batch[0].append(transition[0])
                batch[1].append(transition[1])
                batch[2].append(transition[2])
                batch[3].append(transition[0])
                batch[4].append(transition[3])

                if idx == self.position - 1:
                    batch[3].append(self.__last_state.to(batch[0][0].device))
                elif idx == self.capacity - 1:
                    batch[3].append(self.memory[0][0])
                else:
                    batch[3].append(self.memory[idx + 1][0])

                ngood += 1
        return self._collate(
            batch, self.batch_size, self.histlen, mask_dtype=self.__mask_dtype
        )

    def _push_and_sample(self, transition: list):
        self._push(transition)
        return self._sample()

    # -- Async versions

    def clear_ahead_results(self):
        if self.__sample_result is not None:
            self.__sample_result.cancel()
            self.__sample_result = None
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

    def _async_sample(self):
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

        if self.__sample_result is None:
            batch = self._sample()
        else:
            batch = self.__sample_result.result()

        self.__sample_result = self.__executor.submit(self._sample)
        return batch

    def _async_push(self, transition):
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

        if self.__sample_result is not None:
            self.__sample_result.result()

        self.__push_result = self.__executor.submit(self._push, transition)

    def _async_push_and_sample(self, transition):
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

        if self.__sample_result is not None:
            batch = self.__sample_result.result()
        else:
            batch = self._sample()

        self.__sample_result = self.__executor.submit(self._push_and_sample, transition)
        return batch

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
