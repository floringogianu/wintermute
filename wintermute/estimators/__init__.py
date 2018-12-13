# Bitdefender, 2107
from .atari_net import AtariNet
from .catch_net import CatchNet

ESTIMATORS = {"atari": AtariNet, "catch": CatchNet}


def get_estimator(name, hist_len, action_no, in_ch=1, hidden_sz=128):
    return ESTIMATORS[name](in_ch, hist_len, action_no, hidden_sz)


__all__ = ["AtariNet", "CatchNet", "get_estimator"]
