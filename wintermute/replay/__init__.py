""" Various versions of experience replays.
"""

from .naive_experience_replay import NaiveExperienceReplay
from .mem_efficient_experience_replay import MemoryEfficientExperienceReplay
from .pinned_er import PinnedExperienceReplay
from .prioritized_replay import ProportionalSampler
