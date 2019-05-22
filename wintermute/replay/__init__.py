""" Various versions of experience replays.
"""

from .naive_experience_replay import NaiveExperienceReplay
from .mem_efficient_experience_replay import MemoryEfficientExperienceReplay
from .pinned_er import PinnedExperienceReplay
from .prioritized_replay import ProportionalSampler


class ExperienceReplay:
    r""" Experience Replay Factory.

    Currently it supports:

        1. :class:`~wintermute.replay.MemoryEfficientExperienceReplay`.
        2. :class:`~wintermute.replay.prioritized_replay.ProportionalSampler`.

    Args:
        object (dict): Experience Replay arguments.

    .. note::
        Eventually it should support building any of the available
        implementations: :class:`~wintermute.replay.PinnedExperienceReplay`,
        :class:`~wintermute.replay.NaiveExperienceReplay`, etc.
    """

    def __init__(self, **kwargs):
        async_memory = kwargs["async_memory"]
        if "alpha" in kwargs:
            # ProportionalSampler requires a non_async ExperienceReplay
            kwargs["async_memory"] = False
            uniform_replay = MemoryEfficientExperienceReplay(**kwargs)
            kwargs["async_memory"] = async_memory
            self.__er = ProportionalSampler(uniform_replay, **kwargs)
        else:
            self.__er = MemoryEfficientExperienceReplay(**kwargs)

    def __call__(self):
        return self.__er
