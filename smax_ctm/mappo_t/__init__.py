"""MAPPO-T: Multi-Agent Proximal Policy Optimization with Transformer.

This package contains the JAX/Flax implementation of MAPPO-T,
ported from the MACA (Multi-Agent Credit Assignment) framework.

The key components are:
- Transformer-based actor with multi-head self-attention
- Transformer-based critic with V, Q, and EQ value heads
- Multi-agent credit assignment using attention weights
"""

from .config import get_default_mappo_t_config
from .actor import ActorTrans, ScannedRNN
from .critic import TransVCritic

__all__ = [
    "get_default_mappo_t_config",
    "ActorTrans",
    "ScannedRNN",
    "TransVCritic",
]
