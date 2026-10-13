"""tacb: Two-Clock Sigmoid Bandits (independent arms, general K, alpha/beta in (0,1))."""
from .model import Instance, sigmoid
from .frontier import find_frontier
from .oracle import Oracle, OracleResult
from .env import Environment
from .simulator import Simulator, RunResult
from .policies import (Policy, Context, UCB1, ETC, Greedy, SWUCB, RAWUCB, SSWUCB, GLRUCB, FixedSequence,
                       OraclePolicy, POLICIES, make_policy)
from .generators import make_instance
from .analysis import fit_exponent, RegretCurve

__all__ = ["Instance", "sigmoid", "find_frontier", "Oracle", "OracleResult", "Environment",
           "Simulator", "RunResult", "Policy", "Context", "UCB1", "ETC", "Greedy", "SWUCB", "RAWUCB", "SSWUCB",
           "GLRUCB", "FixedSequence", "OraclePolicy", "POLICIES", "make_policy", "make_instance", "fit_exponent", "RegretCurve"]
