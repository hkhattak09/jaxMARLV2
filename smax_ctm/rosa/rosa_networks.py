import functools
from typing import Dict, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
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


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 5:
            obs, dones, avail_actions, role_id, adapter_id = x
        elif len(x) == 4:
            obs, dones, avail_actions, role_id = x
            adapter_id = role_id
        else:
            obs, dones, avail_actions = x
            role_id = jnp.zeros(obs.shape[:-1], dtype=jnp.int32)
            adapter_id = role_id
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        base_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        lora_delta = jnp.zeros_like(base_logits)
        if self.config["USE_ROLE_LORA"]:
            rank = self.config["ROLE_LORA_RANK"]
            lora_a = self.param(
                "role_lora_A",
                nn.initializers.normal(self.config["ROLE_LORA_A_INIT_STD"]),
                (self.config["LORA_NUM_ADAPTERS"], rank, self.config["GRU_HIDDEN_DIM"]),
            )
            lora_b = self.param(
                "role_lora_B",
                nn.initializers.zeros,
                (self.config["LORA_NUM_ADAPTERS"], self.action_dim, rank),
            )
            safe_adapter_id = jnp.clip(
                adapter_id.astype(jnp.int32),
                0,
                self.config["LORA_NUM_ADAPTERS"] - 1,
            )
            role_a = lora_a[safe_adapter_id]
            role_b = lora_b[safe_adapter_id]
            lora_hidden = jnp.einsum("...rh,...h->...r", role_a, actor_mean)
            lora_delta = jnp.einsum("...ar,...r->...a", role_b, lora_hidden)
        actor_mean = base_logits + self.config["ROLE_LORA_SCALE"] * lora_delta
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        aux = {
            "base_logits": base_logits,
            "lora_delta": lora_delta,
            "action_logits": action_logits,
            "unmasked_logits": actor_mean,
        }
        return hidden, pi, aux


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)
