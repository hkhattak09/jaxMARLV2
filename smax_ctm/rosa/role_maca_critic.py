from typing import Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class RoleMACATransformerCritic(nn.Module):
    config: Dict

    def _activation(self, x):
        if self.config.get("ACTIVATION", "relu") == "gelu":
            return nn.gelu(x)
        return nn.relu(x)

    @nn.compact
    def __call__(self, obs_all, role_id, taken_action, policy_probs):
        num_agents = self.config["NUM_AGENTS"]
        action_dim = self.config["ACTION_DIM"]
        emb_dim = self.config["ROLE_MACA_EMBED_DIM"]
        z_dim = self.config["ROLE_MACA_Z_DIM"]

        role_onehot = jax.nn.one_hot(role_id.astype(jnp.int32), self.config["NUM_UNIT_TYPES"])
        agent_onehot = jnp.broadcast_to(
            jnp.eye(num_agents, dtype=obs_all.dtype),
            obs_all.shape[:-2] + (num_agents, num_agents),
        )
        token = jnp.concatenate((obs_all, role_onehot, agent_onehot), axis=-1)
        token = nn.Dense(emb_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=constant(0.0), name="token_dense")(token)
        token = self._activation(token)
        token = nn.LayerNorm(name="token_ln")(token)

        q = nn.Dense(emb_dim, use_bias=False, kernel_init=orthogonal(1.0), name="att_q")(token)
        k = nn.Dense(emb_dim, use_bias=False, kernel_init=orthogonal(1.0), name="att_k")(token)
        v = nn.Dense(emb_dim, use_bias=False, kernel_init=orthogonal(1.0), name="att_v")(token)
        att_logits = jnp.einsum("...ih,...jh->...ij", q, k) / jnp.sqrt(jnp.asarray(emb_dim, dtype=token.dtype))
        attention = jax.nn.softmax(att_logits, axis=-1)
        att_out = jnp.einsum("...ij,...jh->...ih", attention, v)
        att_out = nn.Dense(emb_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="att_out")(att_out)
        token = nn.LayerNorm(name="att_ln")(token + att_out)

        ffn = nn.Dense(emb_dim * 2, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=constant(0.0), name="ffn_in")(token)
        ffn = self._activation(ffn)
        ffn = nn.Dense(emb_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="ffn_out")(ffn)
        token = nn.LayerNorm(name="ffn_ln")(token + ffn)

        state_embed = token.reshape(token.shape[:-2] + (num_agents * emb_dim,))
        state_embed = nn.Dense(z_dim, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=constant(0.0), name="state_dense")(state_embed)
        state_embed = self._activation(state_embed)
        state_embed = nn.LayerNorm(name="state_ln")(state_embed)

        v_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="v_head")

        def linear_action_terms(name):
            h = nn.Dense(
                z_dim,
                kernel_init=orthogonal(jnp.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"{name}_dense",
            )(state_embed)
            h = self._activation(h)
            h = nn.LayerNorm(name=f"{name}_ln")(h)
            base = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name=f"{name}_base",
            )(h)
            coeff = nn.Dense(
                num_agents * action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name=f"{name}_action_coeff",
            )(h)
            coeff = coeff.reshape(coeff.shape[:-1] + (num_agents, action_dim))
            return jnp.squeeze(base, axis=-1), coeff

        q_base, q_coeff = linear_action_terms("q")
        eq_base, eq_coeff = linear_action_terms("eq")

        def action_linear_value(joint_action, base, coeff):
            # The action path is affine, so one-hot actions, policy probabilities,
            # and mixed action/probability tensors give exact expectations under
            # this critic instead of nonlinear "soft action" evaluations.
            action_coeff = coeff
            while action_coeff.ndim < joint_action.ndim:
                action_coeff = jnp.expand_dims(action_coeff, axis=-3)
            action_value = jnp.sum(joint_action * action_coeff, axis=(-2, -1))
            state_value = base
            while state_value.ndim < action_value.ndim:
                state_value = jnp.expand_dims(state_value, axis=-1)
            return state_value + action_value

        def q_value(joint_action):
            return action_linear_value(joint_action, q_base, q_coeff)

        def eq_value(joint_action):
            return action_linear_value(joint_action, eq_base, eq_coeff)

        v_value = jnp.squeeze(v_head(state_embed), axis=-1)
        q_taken = q_value(taken_action)
        q_policy = q_value(policy_probs)
        eq = eq_value(policy_probs)

        eye = jnp.eye(num_agents, dtype=bool)
        self_mask = jnp.broadcast_to(eye, attention.shape)
        coma_action = jnp.where(self_mask[..., None], policy_probs[..., None, :, :], taken_action[..., None, :, :])
        vq_coma = q_value(coma_action)

        sigma = self.config["ROLE_MACA_ATT_SIGMA"] / num_agents
        corr_mask = attention >= sigma
        corr_mask = jnp.logical_or(corr_mask, self_mask)
        group_action = jnp.where(corr_mask[..., None], policy_probs[..., None, :, :], taken_action[..., None, :, :])
        vq_group = q_value(group_action)

        self_att = jnp.diagonal(attention, axis1=-2, axis2=-1)
        self_w = self.config["ROLE_MACA_SELF_COEF"] * self_att
        group_raw = jnp.sum(jnp.where(corr_mask, attention, 0.0), axis=-1) - self_att
        group_w = self.config["ROLE_MACA_GROUP_COEF"] * group_raw
        joint_w = jnp.clip(1.0 - self_w - group_w, 0.0, 1.0)
        baseline = self_w * vq_coma + group_w * vq_group + joint_w * eq[..., None]
        baseline_weights = jnp.stack((self_w, group_w, joint_w), axis=-1)

        return {
            "v": v_value,
            "q_taken": q_taken,
            "q_policy": q_policy,
            "eq": eq,
            "vq_coma_i": vq_coma,
            "vq_group_i": vq_group,
            "attention": attention,
            "corr_mask": corr_mask,
            "baseline_weights": baseline_weights,
            "mixed_baseline_i": baseline,
        }


def actor_major_to_env_major(x: jnp.ndarray, num_agents: int, num_envs: int):
    return x.reshape(x.shape[0], num_agents, num_envs, *x.shape[2:]).transpose(0, 2, 1, *range(3, x.ndim + 1))


def env_major_to_actor_major(x: jnp.ndarray):
    return x.transpose(0, 2, 1, *range(3, x.ndim)).reshape(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])


def team_reward_from_actor_major(reward: jnp.ndarray, num_agents: int, num_envs: int):
    # SMAX is cooperative; if rewards are agent-shaped, use a consistent team mean per env.
    reward_env = reward.reshape(reward.shape[0], num_agents, num_envs).transpose(0, 2, 1)
    return reward_env.mean(axis=-1)


def done_from_actor_major(global_done: jnp.ndarray, num_agents: int, num_envs: int):
    return global_done.reshape(global_done.shape[0], num_agents, num_envs)[..., 0, :]


def td_lambda_returns(preds, last_pred, rewards, dones, gamma, gae_lambda):
    def _scan(carry, xs):
        gae, next_value = carry
        pred, reward, done = xs
        delta = reward + gamma * next_value * (1.0 - done) - pred
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, pred), gae + pred

    _, returns = jax.lax.scan(
        _scan,
        (jnp.zeros_like(last_pred), last_pred),
        (preds, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return returns
