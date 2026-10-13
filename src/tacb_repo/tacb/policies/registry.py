"""Policy registry: lets experiment configs refer to policies by name (picklable, JSON-friendly)."""
from .ucb1 import UCB1
from .etc import ETC
from .greedy import Greedy
from .windowed import SWUCB, RAWUCB, SSWUCB
from .glr import GLRUCB

POLICIES = {"ucb1": UCB1, "etc": ETC, "greedy": Greedy, "swucb": SWUCB, "rawucb": RAWUCB,
            "sswucb": SSWUCB, "glrucb": GLRUCB}      # BTC = ETC(schedule="alternate")


def register(name):
    """Class decorator for user-defined policies: @register('mine') class Mine(Policy): ..."""
    def deco(cls):
        POLICIES[name] = cls
        return cls
    return deco


def make_policy(name, **params):
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; registered: {sorted(POLICIES)}")
    return POLICIES[name](**params)
