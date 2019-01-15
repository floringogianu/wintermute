# Bitdefender, 2107

from .atari_ensembles import (
    AtariNet,
    FlatAtariEnsemble,
    FlatAtariEnsembleWithPriors,
)


def get_atari_estimator(actions_no: int = None, **kwargs):
    if actions_no is None:
        raise RuntimeError("Please specify number of actions for current game.")
    if kwargs.get("heads_no", 1) > 1:
        if kwargs.get("use_priors", False):
            return FlatAtariEnsembleWithPriors(actions_no=actions_no, **kwargs)
        return FlatAtariEnsemble(actions_no=actions_no, **kwargs)
    return AtariNet(**kwargs)


__all__ = [
    "AtariNet",
    "FlatAtariEnsemble",
    "FlatAtariEnsembleWithPriors",
    "get_atari_estimator",
]
