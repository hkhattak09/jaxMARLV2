"""MAPPO-T: Multi-Agent Proximal Policy Optimization with Transformer.

This package contains the JAX/Flax implementation of MAPPO-T,
ported from the MACA (Multi-Agent Credit Assignment) framework.

The key components are:
- MLP/recurrent actor matching MACA's StochasticPolicyTrans path
- Transformer-based critic with V, Q, and EQ value heads
- Multi-agent credit assignment using attention weights
"""

from .config import get_default_mappo_t_config
from .role_config import get_default_maca_role_config
from .actor import ActorTrans, ScannedRNN
from .critic import TransVCritic
from .lorasa_actor import (
    FrozenDense,
    LoRADense,
    LoRAExplicitGRUCell,
    LoRAScannedRNN,
    LoRASAActorTrans,
)
from .role_actor import RoleActorTrans
from .role_critic import RoleTransVCritic

__all__ = [
    "get_default_mappo_t_config",
    "get_default_maca_role_config",
    "ActorTrans",
    "ScannedRNN",
    "TransVCritic",
    "FrozenDense",
    "LoRADense",
    "LoRAExplicitGRUCell",
    "LoRAScannedRNN",
    "LoRASAActorTrans",
    "RoleActorTrans",
    "RoleTransVCritic",
]
