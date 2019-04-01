""" Transition containers.
"""

from typing import NamedTuple


class HalfTransition(NamedTuple):
    """ Members of a HalfTransition. """

    state: object
    action: object
    reward: object
    done: bool
    meta: dict


class FullTransition(NamedTuple):
    """ Members of a FullTransition. """

    state: object
    action: object
    reward: object
    next_state: object
    done: bool
    meta: dict


class ComparableTransition(NamedTuple):
    """ Attributes of a Comparable """

    priority: float
    transition: object

    def __eq__(self, other):
        return self.priority == other.priority

    def __ne__(self, other):
        return self.priority != other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __lt__(self, other):
        return self.priority < other.priority

