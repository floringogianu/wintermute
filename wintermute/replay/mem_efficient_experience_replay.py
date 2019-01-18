""" This files implements a storage efficient Experience Replay.
"""

import numpy.random
import torch

class MemoryEfficientExperienceReplay:
    """ docstring not found.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        capacity: int = 100_000,
        batch_size: int = 32,
        hist_len: int = 4,
        async_memory: bool = True,
        mask_dtype=torch.uint8,
        bootstrap_args=None,
        serve_to_cuda=True
    ) -> None:

        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.histlen = hist_len

        self.bootstrap_args = bootstrap_args
        if bootstrap_args is not None:
            boot_no, boot_prob = bootstrap_args
            if boot_no < 1:
                raise ValueError(f"nheads should be positive (got {boot_no})")
            if not 0.0 <= boot_prob <= 1.0:
                raise ValueError(f"p should be a probability (got {boot_prob}")
            self._probs = torch.empty(boot_no).fill_(boot_prob)

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
        self._size = 0
        self.__last_state = None
        self.__mask_dtype = mask_dtype
        self.__is_async = bool(async_memory)
        self._serve_to_cuda = serve_to_cuda

    @property
    def is_async(self) -> bool:
        """ If memory uses threads. """
        return self.__is_async

    def _push(self, transition: list) -> int:
        with torch.no_grad():
            state = transition[0][:, -1:].clone()
        to_store = [state, transition[1].action, transition[2], bool(transition[4])]
        if self.bootstrap_args:
            self._probs = self._probs.to(transition[0].device)
            new_mask = torch.bernoulli(self._probs).byte()
            while new_mask.sum() == 0:
                new_mask = torch.bernoulli(self._probs).byte()
            to_store.append(new_mask)
        if len(self.memory) < self.capacity:
            self.memory.append(to_store)
        else:
            self.memory[self.position] = to_store
        self.__last_state = transition[3][:, -1:]
        pos = self.position
        self.position += 1
        self._size = max(self._size, self.position)
        self.position = self.position % self.capacity
        return pos

    def _sample(self, gods_idxs=None):
        if self.bootstrap_args is not None:
            masks = []
        batch = [], [], [], [], []

        memory = self.memory
        nmemory = len(self.memory)

        if gods_idxs is None:
            gods_idxs = numpy.random.randint(0, nmemory, (self.batch_size,))

        for idx in gods_idxs[::-1]:
            transition = memory[idx]
            batch[0].append(transition[0])
            batch[1].append(transition[1])
            batch[2].append(transition[2])
            if self.bootstrap_args is not None:
                masks.append(transition[4])
            is_final = transition[3]
            batch[4].append(is_final)
            if not is_final:
                if idx == self.position - 1:
                    batch[3].append(self.__last_state)
                else:
                    batch[3].append(self.memory[(idx + 1) % self.capacity][0])

            last_screen = transition[0]
            found_done = False
            bidx = idx
            for _ in range(self.histlen - 1):
                if not is_final:
                    batch[3].append(batch[0][-1])
                if not found_done:
                    bidx = (bidx - 1) % self.capacity
                    if bidx < self._size:
                        new_transition = memory[bidx]
                        if new_transition[3]:
                            found_done = True
                        else:
                            last_screen = new_transition[0]
                    else:
                        found_done = True
                batch[0].append(last_screen)

        transitions = self._collate(
            batch, self.batch_size, self.histlen, mask_dtype=self.__mask_dtype
        )
        if self.bootstrap_args is not None:
            masks = torch.stack(masks, dim=0)
            if self._serve_to_cuda:
                return ([x.cuda() for x in transitions], masks.cuda())
            return (transitions, masks)
        if self._serve_to_cuda:
            return [x.cuda() for x in transitions]
        return transitions

    def _collate(self, batch, batch_size, histlen, mask_dtype=torch.uint8):
        device = batch[0][0].device
        frame_size = batch[0][0].size()[2:]
        states = torch.cat(batch[0][::-1], 0).view(
            batch_size, histlen, *frame_size
        )
        actions = torch.tensor(  # pylint: disable=E1102
            batch[1][::-1], device=device, dtype=torch.long
        ).unsqueeze_(1)
        rewards = torch.tensor(  # pylint: disable=E1102
            batch[2][::-1], device=device, dtype=torch.float
        ).unsqueeze_(1)
        next_states = torch.cat(batch[3][::-1], 0).view(
            -1, histlen, *frame_size
        )
        mask = 1 - torch.tensor(  # pylint: disable=E1102
            batch[4][::-1], device=device, dtype=mask_dtype
        ).unsqueeze_(1)

        # if we train with full RGB information (three channels instead of one)
        if states.ndimension() == 5:
            bsz, hist, nch, height, width = states.size()
            states = states.view(bsz, hist * nch, height, width)
            next_states = next_states.view(bsz, hist * nch, height, width)

        return [states, actions, rewards, next_states, mask]

    def _push_and_sample(self, transition: list):
        if isinstance(transition[0], list):
            for trans in transition:
                self._push(trans)
        else:
            self._push(transition)
        return self._sample()

    # -- Async versions

    def clear_ahead_results(self):
        """ Waits for any asynchronous push and cancels any sample request.
        """
        if self._sample_result is not None:
            self._sample_result.cancel()
            self._sample_result = None
        if self._push_result is not None:
            self._push_result.result()
            self._push_result = None

    def _async_sample(self):
        if self._push_result is not None:
            self._push_result.result()
            self._push_result = None

        if self._sample_result is None:
            batch = self._sample()
        else:
            batch = self._sample_result.result()

        self._sample_result = self._executor.submit(self._sample)

        return batch

    def _async_push(self, transition):
        if self._push_result is not None:
            self._push_result.result()
            self._push_result = None

        if self._sample_result is not None:
            self._sample_result.result()

        self._push_result = self._executor.submit(self._push, transition)

    def _async_push_and_sample(self, transition):
        if self._push_result is not None:
            self._push_result.result()
            self._push_result = None

        if self._sample_result is not None:
            batch = self._sample_result.result()
        else:
            batch = self._sample()

        self._sample_result = self._executor.submit(
            self._push_and_sample, transition
        )
        return batch

    def __len__(self):
        return self._size

    def __str__(self):
        return (
            f"{self.__class__.__name__}"
            + f"(batch={self.batch_size}, sz={self.capacity})"
        )

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"
