"""MAPPO-T transformer critic."""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from .transformer import Encoder


class CriticTrainState(NamedTuple):
    apply_fn: Any
    params: Any
    tx: optax.GradientTransformation
    opt_state: Any

    def apply_gradients(self, *, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self._replace(params=new_params, opt_state=new_opt_state)


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
        encoder = Encoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            name="transformer_encoder",
        )
        action_onehot = self._one_hot_actions(action)

        if obs.ndim == 4:
            values = []
            q_values = []
            eq_values = []
            vq_values = []
            vq_coma_values = []
            baseline_weights = []
            attn_weights = []
            zs_values = []
            zsa_values = []
            carry = rnn_states
            for t in range(obs.shape[0]):
                out = encoder(
                    obs[t],
                    action_onehot[t],
                    policy_prob[t],
                    carry,
                    resets[t],
                    output_attentions,
                    deterministic,
                )
                (
                    value,
                    q_value,
                    eq_value,
                    vq_value,
                    vq_coma_value,
                    weights,
                    attn,
                    zs,
                    zsa,
                    carry,
                ) = out
                values.append(value)
                q_values.append(q_value)
                eq_values.append(eq_value)
                vq_values.append(vq_value)
                vq_coma_values.append(vq_coma_value)
                baseline_weights.append(weights)
                attn_weights.append(attn)
                zs_values.append(zs)
                zsa_values.append(zsa)

            return (
                jnp.stack(values),
                jnp.stack(q_values),
                jnp.stack(eq_values),
                jnp.stack(vq_values) if vq_values[0] is not None else None,
                jnp.stack(vq_coma_values) if vq_coma_values[0] is not None else None,
                jnp.stack(baseline_weights) if baseline_weights[0] is not None else None,
                jnp.stack(attn_weights) if attn_weights[0] is not None else None,
                jnp.stack(zs_values),
                jnp.stack(zsa_values),
                carry,
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


def create_critic_train_state(
    rng: jnp.ndarray,
    config: Dict[str, Any],
    share_obs_space: Any,
    obs_space: Any,
    act_space: Any,
    num_agents: int,
    state_type: str = "EP",
) -> CriticTrainState:
    critic = TransVCritic(
        config=config,
        share_obs_space=share_obs_space,
        obs_space=obs_space,
        act_space=act_space,
        num_agents=num_agents,
        state_type=state_type,
    )

    obs_dim = obs_space.shape[0]
    action_dim = act_space.n
    dummy_obs = jnp.zeros((1, num_agents, obs_dim), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, num_agents), dtype=jnp.int32)
    dummy_policy = jnp.ones((1, num_agents, action_dim), dtype=jnp.float32) / action_dim
    dummy_rnn = jnp.zeros((1, num_agents, config["transformer"]["n_embd"]), dtype=jnp.float32)
    dummy_resets = jnp.zeros((1, num_agents), dtype=bool)

    params = critic.init(
        rng,
        dummy_obs,
        dummy_action,
        dummy_policy,
        dummy_rnn,
        dummy_resets,
        True,
        True,
    )
    tx = optax.adam(config.get("CRITIC_LR", config.get("LR", 0.002)), eps=config.get("opti_eps", 1e-5))
    return CriticTrainState(
        apply_fn=critic.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
    )
