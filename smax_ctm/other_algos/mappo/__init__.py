"""MAPPO: Multi-Agent Proximal Policy Optimization.
JAX/Flax implementation ported from the MACA framework.
"""

from .config import get_default_mappo_config
from .actor import ActorRNN
from .critic import CriticRNN

__all__ = [
    "get_default_mappo_config",
    "ActorRNN",
    "CriticRNN",
]
