"""MAPPO Critic Network.

Follows the pattern from smax_ctm/train_mappo_gru.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Dict

from .actor import ScannedRNN


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        fc_dim = self.config.get("FC_DIM_SIZE", self.config["hidden_sizes"][0])
        gru_dim = self.config.get("GRU_HIDDEN_DIM", self.config["hidden_sizes"][1])

        embedding = nn.Dense(
            fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(gru_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)
