"""MAPPO-VD critic networks.

Contains the value-decomposition critic components ported from MACA:
- IndividualQNet: per-agent GRU-based Q-network
- QMixer: hypernetwork-based QMIX mixer
- VDNMixer: simple additive VDN mixer
- VDCriticRNN: wrapper that combines individual Q-nets with a mixer
"""

import functools
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from .actor import ScannedRNN


class IndividualQNet(nn.Module):
    """GRU-based individual Q-network for a single agent.

    Outputs Q-values for all actions, analogous to ActorRNN but returning
    action-values instead of a policy distribution.
    """

    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        """Forward pass.

        Args:
            hidden: ``(batch, hidden_dim)`` recurrent state.
            x: tuple ``(obs, dones)`` where ``obs`` is
                ``(time, batch, obs_dim)`` and ``dones`` is ``(time, batch)``.

        Returns:
            (new_hidden, q_values) where q_values is ``(time, batch, action_dim)``.
        """
        obs, dones = x
        cfg = self.config

        embedding = nn.Dense(
            cfg["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_values = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(cfg.get("gain", 0.01)),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_values


class QMixer(nn.Module):
    """Hypernetwork-based QMIX mixer.

    Takes individual Q-values (for taken actions) and a world state,
    and produces a joint Q-value via monotonic mixing.
    """

    num_agents: int
    state_dim: int
    embed_dim: int
    hypernet_layers: int = 2

    @nn.compact
    def __call__(self, agent_qs, states):
        """Forward pass.

        Args:
            agent_qs: ``(batch, 1, num_agents)`` individual Q-values for taken actions.
            states: ``(batch, state_dim)`` world state.

        Returns:
            ``(batch, 1)`` joint Q-value.
        """
        batch_size = states.shape[0]

        # Hypernetwork for first-layer weights
        if self.hypernet_layers == 1:
            w1 = nn.Dense(
                self.embed_dim * self.num_agents,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="hyper_w_1",
            )(states)
            w_final = nn.Dense(
                self.embed_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="hyper_w_final",
            )(states)
        elif self.hypernet_layers == 2:
            hypernet_embed = self.embed_dim
            w1 = nn.Dense(
                hypernet_embed,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="hyper_w_1_0",
            )(states)
            w1 = nn.relu(w1)
            w1 = nn.Dense(
                self.embed_dim * self.num_agents,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="hyper_w_1_1",
            )(w1)

            w_final = nn.Dense(
                hypernet_embed,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="hyper_w_final_0",
            )(states)
            w_final = nn.relu(w_final)
            w_final = nn.Dense(
                self.embed_dim,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="hyper_w_final_1",
            )(w_final)
        else:
            raise ValueError("Only hypernet_layers=1 or 2 is supported.")

        # Ensure non-negative weights for monotonicity
        w1 = jnp.abs(w1)
        w_final = jnp.abs(w_final)

        w1 = w1.reshape(batch_size, self.num_agents, self.embed_dim)
        b1 = nn.Dense(
            self.embed_dim,
            kernel_init=constant(0.0),
            bias_init=constant(0.0),
            name="hyper_b_1",
        )(states).reshape(batch_size, 1, self.embed_dim)

        # First mixing layer
        hidden = jax.nn.elu(jnp.matmul(agent_qs, w1) + b1)  # (batch, 1, embed_dim)

        # Second mixing layer
        w_final = w_final.reshape(batch_size, self.embed_dim, 1)
        v = nn.Dense(
            self.embed_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="hyper_v_0",
        )(states)
        v = nn.relu(v)
        v = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="hyper_v_1",
        )(v).reshape(batch_size, 1, 1)

        y = jnp.matmul(hidden, w_final) + v  # (batch, 1, 1)
        return y.squeeze(-1)  # (batch, 1)


class VDNMixer(nn.Module):
    """Simple additive VDN mixer.

    Sums individual Q-values to produce a joint Q-value.
    """

    @nn.compact
    def __call__(self, agent_qs, states=None):
        """Forward pass.

        Args:
            agent_qs: ``(..., num_agents)`` individual Q-values.
            states: Unused, kept for API compatibility.

        Returns:
            ``(..., 1)`` joint Q-value.
        """
        del states
        return jnp.sum(agent_qs, axis=-1, keepdims=True)


class VDCriticRNN(nn.Module):
    """Value Decomposition critic wrapper.

    Combines individual GRU-based Q-networks with a mixer (QMIX or VDN)
    to produce joint Q-values, individual Q-values for taken actions,
    and individual V-values.
    """

    action_dim: int
    num_agents: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        """Forward pass.

        Args:
            hidden: ``(NUM_ACTORS, hidden_dim)`` recurrent state.
            x: tuple ``(local_obs_all, world_state, dones, actions, policy_probs)``
                where ``local_obs_all`` is ``(time, NUM_ACTORS, obs_dim)``,
                ``world_state`` is ``(time, NUM_ACTORS, state_dim)``,
                ``dones`` is ``(time, NUM_ACTORS)``,
                ``actions`` is ``(time, NUM_ACTORS)``,
                ``policy_probs`` is ``(time, NUM_ACTORS, action_dim)``.

        Returns:
            (new_hidden, (joint_q_value, individual_q_taken, individual_v_values))
            where each output is ``(time, NUM_ACTORS)``.
        """
        local_obs_all, world_state, dones, actions, policy_probs = x
        time_steps = local_obs_all.shape[0]
        num_envs = local_obs_all.shape[1] // self.num_agents
        cfg = self.config

        # Individual Q-values for all actions
        hidden, q_values = IndividualQNet(self.action_dim, cfg)(hidden, (local_obs_all, dones))
        # q_values: (time, NUM_ACTORS, action_dim)

        # Extract Q-values for taken actions
        ind_q_taken = jnp.take_along_axis(q_values, actions[..., None], axis=-1).squeeze(-1)
        # ind_q_taken: (time, NUM_ACTORS)

        # Compute V-values: V = sum_a pi(a) * Q(s, a)
        ind_v_values = jnp.sum(policy_probs * q_values, axis=-1)
        # ind_v_values: (time, NUM_ACTORS)

        # Reshape for mixer: separate agent and env axes
        # Current ordering in NUM_ACTORS: [agent0_env0..agent0_envN, agent1_env0..agent1_envN, ...]
        # Reshape to (time, num_agents, num_envs)
        ind_q_taken_mix = ind_q_taken.reshape(time_steps, self.num_agents, num_envs)
        ind_q_taken_mix = jnp.transpose(ind_q_taken_mix, (0, 2, 1))  # (time, num_envs, num_agents)
        ind_q_taken_mix = ind_q_taken_mix[..., None, :]  # (time, num_envs, 1, num_agents)

        # World state per environment (all agents share the same world state)
        world_state_env = world_state.reshape(time_steps, self.num_agents, num_envs, -1)[:, 0, :, :]
        # (time, num_envs, state_dim)

        # Flatten time and env dimensions for mixer (mixer expects 2D batch)
        batch_size = time_steps * num_envs
        ind_q_taken_mix_flat = ind_q_taken_mix.reshape(batch_size, 1, self.num_agents)
        world_state_env_flat = world_state_env.reshape(batch_size, -1)

        # Apply mixer
        mixer_type = cfg["valuedecomp"]["mixer"]
        if mixer_type == "qmix":
            embed_dim = cfg["hidden_sizes"][0]
            state_dim = world_state_env_flat.shape[-1]
            joint_q_value = QMixer(
                num_agents=self.num_agents,
                state_dim=state_dim,
                embed_dim=embed_dim,
                hypernet_layers=cfg["valuedecomp"]["hypernet_layers"],
                name="qmixer",
            )(ind_q_taken_mix_flat, world_state_env_flat)
        elif mixer_type == "vdn":
            joint_q_value = VDNMixer(name="vdnmixer")(
                ind_q_taken_mix_flat.squeeze(-2), world_state_env_flat
            )
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        # joint_q_value: (batch_size, 1) -> reshape back to (time_steps, num_envs, 1)
        joint_q_value = joint_q_value.reshape(time_steps, num_envs, 1)
        # Broadcast joint value to all actors
        joint_q_value = jnp.tile(joint_q_value, (1, 1, self.num_agents)).reshape(time_steps, -1)
        # (time, NUM_ACTORS)

        return hidden, (joint_q_value, ind_q_taken, ind_v_values)
