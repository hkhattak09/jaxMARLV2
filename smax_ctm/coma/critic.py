"""COMA Q-critic network.

Follows MACA's ComaQNet structure: a GRU encoder for the centralized state,
followed by a per-agent Q-head that takes the state encoding plus the
one-hot actions of all *other* agents and outputs Q-values for every action
of that agent.  The counterfactual baseline V(s, a_{-i}) is computed as the
expectation of Q under the current policy.
"""

import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ComaCriticRNN(nn.Module):
    action_dim: int
    num_agents: int
    config: Dict[str, Any]

    @nn.compact
    def __call__(self, hidden, x):
        world_state, actions_all, policy_probs_all, dones = x
        # world_state:       (time, batch_envs, world_state_dim)
        # actions_all:       (time, batch_envs, num_agents)
        # policy_probs_all:  (time, batch_envs, num_agents, action_dim)
        # dones:             (time, batch_envs)

        fc_dim = self.config.get("FC_DIM_SIZE", self.config["hidden_sizes"][0])
        gru_dim = self.config.get("GRU_HIDDEN_DIM", self.config["hidden_sizes"][-1])

        # --- state encoder ---
        embedding = nn.Dense(
            fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        # embedding: (time, batch_envs, gru_dim)

        # --- one-hot actions ---
        actions_onehot = jax.nn.one_hot(
            actions_all, self.action_dim, dtype=jnp.float32
        )
        # actions_onehot: (time, batch_envs, num_agents, action_dim)

        # Tile state encoding for each agent
        embedding_tiled = jnp.expand_dims(embedding, axis=-2)
        embedding_tiled = jnp.tile(embedding_tiled, (1, 1, self.num_agents, 1))
        # embedding_tiled: (time, batch_envs, num_agents, gru_dim)

        # Build a mask that zeros out each agent's *own* one-hot action vector.
        # agent_mask[i, j] = 0 iff i == j, else 1.
        agent_mask = 1.0 - jnp.eye(self.num_agents)
        agent_mask = agent_mask.reshape(1, 1, self.num_agents, self.num_agents, 1)
        # agent_mask: (1, 1, num_agents, num_agents, 1)

        actions_expanded = jnp.expand_dims(actions_onehot, axis=-3)
        # actions_expanded: (time, batch_envs, 1, num_agents, action_dim)

        masked_actions = actions_expanded * agent_mask
        # masked_actions: (time, batch_envs, num_agents, num_agents, action_dim)

        # Flatten other agents' actions for each agent
        prefix = masked_actions.shape[:-3]
        masked_actions = masked_actions.reshape(*prefix, self.num_agents, -1)
        # masked_actions: (time, batch_envs, num_agents, num_agents * action_dim)

        # Q-head MLP: [state_encoding, actions_of_others] -> Q(s, a_i, a_{-i})
        q_input = jnp.concatenate([embedding_tiled, masked_actions], axis=-1)
        # q_input: (time, batch_envs, num_agents, gru_dim + num_agents * action_dim)

        for idx, hidden_size in enumerate(self.config.get("hidden_sizes", [128, 128])):
            q_input = nn.Dense(
                hidden_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"q_head_{idx}",
            )(q_input)
            q_input = nn.relu(q_input)

        q_values = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="q_out",
        )(q_input)
        # q_values: (time, batch_envs, num_agents, action_dim)

        # Counterfactual baseline: V(s, a_{-i}) = sum_{a_i} pi(a_i) * Q(s, a_i, a_{-i})
        v_values = jnp.sum(policy_probs_all * q_values, axis=-1)
        # v_values: (time, batch_envs, num_agents)

        return hidden, (q_values, v_values)

    @classmethod
    def initialize_carry(cls, batch_size, hidden_size):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)
