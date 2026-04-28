"""MAPPO-T transformer critic."""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .transformer import Encoder


class TransVCritic(nn.Module):
    """Flax port of MACA's TransVCritic.

    Single-step inputs use shape ``(batch, n_agents, ...)``.  Sequence inputs
    use shape ``(time, batch, n_agents, ...)`` and are unrolled with shared
    transformer parameters so recurrent critic state is trained through time.
    """

    config: Dict[str, Any]
    share_obs_space: Any
    obs_space: Any
    act_space: Any
    num_agents: int
    state_type: str

    def _one_hot_actions(self, action):
        action = jnp.asarray(action)
        if (
            action.ndim >= 3
            and action.shape[-1] == getattr(self.act_space, "n", action.shape[-1])
            and jnp.issubdtype(action.dtype, jnp.floating)
        ):
            return action.astype(jnp.float32)
        if action.shape[-1] == 1 and action.ndim >= 3:
            action = jnp.squeeze(action, axis=-1)
        return jax.nn.one_hot(action.astype(jnp.int32), self.act_space.n)

    def _encoder_step(
        self,
        encoder: Encoder,
        carry: jnp.ndarray,
        inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        output_attentions: bool,
        deterministic: bool,
    ):
        obs_t, action_t, policy_t, reset_t = inputs
        out = encoder(
            obs_t,
            action_t,
            policy_t,
            carry,
            reset_t,
            output_attentions,
            deterministic,
        )
        return out[-1], out[:-1]

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def _scan_encoder_attn_det(self, carry, inputs):
        encoder = Encoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            name="transformer_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, True, True)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def _scan_encoder_no_attn_det(self, carry, inputs):
        encoder = Encoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            name="transformer_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, False, True)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False, "dropout": True},
    )
    @nn.compact
    def _scan_encoder_attn_train(self, carry, inputs):
        encoder = Encoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            name="transformer_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, True, False)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False, "dropout": True},
    )
    @nn.compact
    def _scan_encoder_no_attn_train(self, carry, inputs):
        encoder = Encoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            name="transformer_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, False, False)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        policy_prob: jnp.ndarray,
        rnn_states: jnp.ndarray,
        resets: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple:
        action_onehot = self._one_hot_actions(action)

        if obs.ndim == 4:
            if output_attentions:
                scan_fn = (
                    self._scan_encoder_attn_det
                    if deterministic
                    else self._scan_encoder_attn_train
                )
            else:
                scan_fn = (
                    self._scan_encoder_no_attn_det
                    if deterministic
                    else self._scan_encoder_no_attn_train
                )
            final_carry, outputs = scan_fn(
                rnn_states, (obs, action_onehot, policy_prob, resets)
            )
            (
                values,
                q_values,
                eq_values,
                vq_values,
                vq_coma_values,
                baseline_weights,
                attn_weights,
                zs_values,
                zsa_values,
            ) = outputs

            return (
                values,
                q_values,
                eq_values,
                vq_values,
                vq_coma_values,
                baseline_weights,
                attn_weights,
                zs_values,
                zsa_values,
                final_carry,
            )

        encoder = Encoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            name="transformer_encoder",
        )
        return encoder(
            obs,
            action_onehot,
            policy_prob,
            rnn_states,
            resets,
            output_attentions,
            deterministic,
        )
