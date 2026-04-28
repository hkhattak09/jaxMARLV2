"""MAPPO-VD: Multi-Agent Proximal Policy Optimization with Value Decomposition.

This package contains the JAX/Flax implementation of MAPPO-VD,
ported from the MACA (Multi-Agent Credit Assignment) framework.

The key components are:
- GRU-based actor matching MACA's recurrent policy path
- Individual Q-networks per agent with shared parameters
- QMIX or VDN mixer for joint value estimation
"""

from .config import get_default_mappo_vd_config
from .actor import ActorRNN, ScannedRNN
from .critic import VDCriticRNN, QMixer, VDNMixer

__all__ = [
    "get_default_mappo_vd_config",
    "ActorRNN",
    "ScannedRNN",
    "VDCriticRNN",
    "QMixer",
    "VDNMixer",
]
