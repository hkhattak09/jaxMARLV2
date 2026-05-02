"""MAPPO-T Role training script for MACA-Role experiments.

Supports all 4 experiments via ``ROLE_EXPERIMENT`` config:
  1: Post-GRU actor heads + shared critic
  2: Post-GRU actor heads + role-specific critic heads
  3: Pre-GRU routes + post-GRU heads + shared critic
  4: Pre-GRU routes + post-GRU heads + role-specific critic heads

Run:
    python smax_ctm/train_mappo_t_role.py --role_experiment 1

This script is based on train_mappo_t_lorasa.py but with end-to-end training
(no frozen backbone) and KL diversity penalty.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper

from mappo_t import (
    RoleActorTrans,
    RoleTransVCritic,
    TransVCritic,
    ScannedRNN,
    get_default_mappo_t_config,
)
from mappo_t.utils import batchify, unbatchify
from mappo_t.valuenorm import (
    create_value_norm_dict,
    value_norm_denormalize,
    value_norm_normalize,
    value_norm_update,
)


# ---------------------------------------------------------------------------
# Transition with role IDs
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    active_mask: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    policy_probs: jnp.ndarray
    obs_all: jnp.ndarray
    actions_all: jnp.ndarray
    policy_probs_all: jnp.ndarray
    value_env: jnp.ndarray
    q_value_env: jnp.ndarray
    eq_value_env: jnp.ndarray
    vq_value_env: jnp.ndarray
    vq_coma_value_env: jnp.ndarray
    baseline_weights: jnp.ndarray
    attn_weights: jnp.ndarray
    bad_mask: jnp.ndarray
    actor_hstate: jnp.ndarray
    critic_hstate: jnp.ndarray
    role_id: jnp.ndarray


# ---------------------------------------------------------------------------
# Environment wrapper with role IDs
# ---------------------------------------------------------------------------

class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides MACA-style observations and extracts role IDs from unit_types."""

    def __init__(self, env: HeuristicEnemySMAX, **kwargs):
        super().__init__(env)

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self._get_world_state(obs, env_state)
        return obs, env_state

    def step(self, key, state, action):
        obs, state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self._get_world_state(obs, state)
        return obs, state, reward, done, info

    def _get_world_state(self, obs, env_state):
        """Build world state from observations (simplified)."""
        # Stack all agent observations
        return jnp.stack([obs[a] for a in self._env.agents], axis=0)

    def get_role_ids(self, env_state):
        """Extract role IDs from unit_types."""
        # unit_types: (num_envs, num_allies + num_enemies)
        # We want only allies
        unit_types = env_state.state.unit_types[:, :self._env.num_allies]
        return unit_types.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def actor_to_env_agent_time(x):
    """Reshape actor-level (NUM_ACTORS=NUM_ENVS*num_agents) to env-level."""
    # x shape: (T, NUM_ENVS * num_agents, ...)
    # Returns: (T, NUM_ENVS, num_agents, ...)
    n_envs = x.shape[1] // 10  # num_agents - hardcoded for now, should be dynamic
    return x.reshape(x.shape[0], -1, 10, *x.shape[2:])


def env_agent_to_actor(x):
    """Reshape env-level to actor-level."""
    return x.reshape(x.shape[0], -1, *x.shape[3:])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def make_train(config):
    """Build the full training function."""

    env = HeuristicEnemySMAX(map_name_to_scenario[config["MAP_NAME"]])
    env = SMAXWorldStateWrapper(env)

    config["NUM_ACTORS"] = env.num_allies
    config["NUM_ENVS"] = config.get("NUM_ENVS", 20)
    config["NUM_STEPS"] = config.get("NUM_STEPS", 200)

    # Determine experiment
    role_experiment = config.get("ROLE_EXPERIMENT", 1)
    use_role_critic = role_experiment in (2, 4)
    use_pre_gru_routes = role_experiment in (3, 4)

    def train(rng):
        # Initialize networks
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n
        actor_hidden_dim = config["hidden_sizes"][-1]
        critic_hidden_dim = config["transformer"]["n_embd"]

        # Actor
        actor_network = RoleActorTrans(
            action_dim=action_dim,
            config=config,
            use_pre_gru_routes=use_pre_gru_routes,
            n_roles=config.get("N_ROLES", 6),
        )

        # Critic
        if use_role_critic:
            critic_network = RoleTransVCritic(
                config=config,
                share_obs_space=None,
                obs_space=env.observation_space(env.agents[0]),
                act_space=env.action_space(env.agents[0]),
                num_agents=env.num_agents,
                state_type="EP",
                n_roles=config.get("N_ROLES", 6),
            )
        else:
            critic_network = TransVCritic(
                config=config,
                share_obs_space=None,
                obs_space=env.observation_space(env.agents[0]),
                act_space=env.action_space(env.agents[0]),
                num_agents=env.num_agents,
                state_type="EP",
            )

        # Initialize actor
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], actor_hidden_dim)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim), dtype=jnp.float32),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
            jnp.ones((1, config["NUM_ENVS"], action_dim), dtype=jnp.float32),
        )
        dummy_role_ids = jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32)
        actor_params = actor_network.init(actor_rng, ac_init_hstate, ac_init_x, dummy_role_ids)

        # Initialize critic
        cr_init_hstate = jnp.zeros((config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32)
        critic_params = critic_network.init(
            critic_rng,
            jnp.zeros((config["NUM_ENVS"], env.num_agents, obs_dim), dtype=jnp.float32),
            jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.int32),
            jnp.ones((config["NUM_ENVS"], env.num_agents, action_dim), dtype=jnp.float32) / action_dim,
            cr_init_hstate,
            jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
            True,
            True,
        )
        if use_role_critic:
            # Add dummy role_ids for init
            pass  # Already handled in RoleTransVCritic if needed

        # Optimizers (ALL parameters trainable — no masking)
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=config.get("opti_eps", 1e-5)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(
                learning_rate=config["CRITIC_LR"],
                b1=config["transformer"].get("betas", [0.9, 0.95])[0],
                b2=config["transformer"].get("betas", [0.9, 0.95])[1],
                eps=config.get("opti_eps", 1e-5),
                weight_decay=config["transformer"].get("wght_decay", 0.01),
            ),
        )

        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_params,
            tx=critic_tx,
        )

        # KL diversity schedule
        total_steps = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
        kl_schedule = RoleActorTrans.make_kl_schedule(total_steps)

        # ValueNorm
        use_valuenorm = config.get("use_valuenorm", True)
        value_norm_dict = create_value_norm_dict() if use_valuenorm else None

        # Environment reset
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rngs)

        # Initial hidden states
        actor_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], actor_hidden_dim)
        critic_hstate = jnp.zeros((config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32)

        # -------------------------------------------------------------------
        # Rollout + update loop
        # -------------------------------------------------------------------

        def _env_step(carry, unused):
            """Collect one rollout step."""
            rng, actor_ts, critic_ts, env_s, last_obs, actor_hs, critic_hs, vn_dict = carry

            # Extract role IDs from env state
            role_ids = env.get_role_ids(env_s)  # (NUM_ENVS, num_allies)
            role_ids_flat = role_ids.reshape(-1)  # (NUM_ACTORS,)

            # Actor forward
            obs_batch = batchify(last_obs, env.agents, 1)  # (NUM_ACTORS, obs_dim)
            avail_batch = jnp.ones((config["NUM_ACTORS"], action_dim), dtype=jnp.float32)  # TODO: get actual avail

            rng, step_rng = jax.random.split(rng)
            action, log_prob, probs, new_actor_hs = actor_network.get_actions(
                actor_ts.params,
                actor_hs,
                obs_batch[None, ...],  # add time dim
                jnp.zeros((1, config["NUM_ACTORS"]), dtype=bool),
                avail_batch[None, ...],
                role_ids_flat[None, ...],
                step_rng,
                deterministic=False,
            )
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            probs = probs.squeeze(0)

            # Critic forward
            obs_all = jnp.stack([last_obs[a] for a in env.agents], axis=1)  # (NUM_ENVS, num_agents, obs_dim)
            actions_all = action.reshape(config["NUM_ENVS"], env.num_allies)
            policy_probs_all = probs.reshape(config["NUM_ENVS"], env.num_allies, action_dim)

            if use_role_critic:
                # RoleTransVCritic expects role_ids
                critic_out = critic_network.apply(
                    critic_ts.params,
                    obs_all,
                    actions_all,
                    policy_probs_all,
                    critic_hs,
                    jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
                    role_ids,
                    True,
                    True,
                )
            else:
                critic_out = critic_network.apply(
                    critic_ts.params,
                    obs_all,
                    actions_all,
                    policy_probs_all,
                    critic_hs,
                    jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
                    True,
                    True,
                )

            values = critic_out[0].squeeze(-1)
            q_values = critic_out[1].squeeze(-1)
            eq_values = critic_out[2].squeeze(-1)

            # Environment step
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
            action_dict = unbatchify(action, env.agents, config["NUM_ENVS"])
            next_obs, next_env_state, reward, done, info = jax.vmap(env.step)(
                step_rngs, env_s, action_dict
            )

            # Build transition
            transition = Transition(
                global_done=done["__all__"],
                done=batchify(done, env.agents, 1).squeeze(),
                active_mask=1.0 - batchify(done, env.agents, 1).squeeze(),
                action=action,
                value=values.reshape(-1),
                reward=batchify(reward, env.agents, 1).squeeze(),
                log_prob=log_prob,
                obs=batchify(last_obs, env.agents, 1).squeeze(),
                world_state=last_obs.get("world_state", jnp.zeros((config["NUM_ENVS"], 1))),
                info=info,
                avail_actions=avail_batch,
                policy_probs=probs,
                obs_all=obs_all,
                actions_all=actions_all,
                policy_probs_all=policy_probs_all,
                value_env=values,
                q_value_env=q_values,
                eq_value_env=eq_values,
                vq_value_env=jnp.zeros((config["NUM_ENVS"], env.num_allies)),
                vq_coma_value_env=jnp.zeros((config["NUM_ENVS"], env.num_allies)),
                baseline_weights=jnp.zeros((config["NUM_ENVS"], env.num_allies, 3)),
                attn_weights=jnp.zeros((config["NUM_ENVS"], env.num_allies, env.num_agents)),
                bad_mask=jnp.ones((config["NUM_ACTORS"]), dtype=jnp.float32),
                actor_hstate=actor_hs,
                critic_hstate=critic_hs,
                role_id=role_ids_flat,
            )

            return (
                rng, actor_ts, critic_ts, next_env_state, next_obs,
                new_actor_hs.squeeze(0), critic_hs, vn_dict,
            ), transition

        # -------------------------------------------------------------------
        # Update function
        # -------------------------------------------------------------------

        def _update_step(runner_state, unused):
            """Collect rollout and perform updates."""
            rng, actor_ts, critic_ts, env_s, last_obs, actor_hs, critic_hs, vn_dict = runner_state

            # Collect rollout
            rollout_state = (rng, actor_ts, critic_ts, env_s, last_obs, actor_hs, critic_hs, vn_dict)
            rollout_state, traj_batch = jax.lax.scan(
                _env_step, rollout_state, None, length=config["NUM_STEPS"]
            )

            rng, actor_ts, critic_ts, env_s, last_obs, actor_hs, critic_hs, vn_dict = rollout_state

            # Compute GAE (simplified - full MACA advantage would use critic outputs)
            # For now, use standard GAE on env-level values
            advantages = jnp.zeros_like(traj_batch.reward)  # Placeholder
            returns = traj_batch.reward + config["GAMMA"] * jnp.concatenate(
                [traj_batch.value[1:], jnp.zeros((1,))], axis=0
            )

            # Actor update with KL diversity
            def actor_loss_fn(params):
                _, pi = actor_network.apply(
                    params,
                    traj_batch.actor_hstate[0],
                    (traj_batch.obs, jnp.zeros_like(traj_batch.done), traj_batch.avail_actions),
                    traj_batch.role_id,
                )
                log_prob = pi.log_prob(traj_batch.action)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)

                # PPO clip
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                loss1 = ratio * adv_norm
                loss2 = jnp.clip(ratio, 1.0 - config["CLIP_PARAM"], 1.0 + config["CLIP_PARAM"]) * adv_norm
                policy_loss = -jnp.minimum(loss1, loss2).mean()

                # Entropy bonus
                entropy = pi.entropy().mean()

                # KL diversity penalty (cosine decay)
                kl_div = actor_network.compute_kl_diversity(
                    params,
                    traj_batch.actor_hstate[0],
                    traj_batch.obs,
                    jnp.zeros_like(traj_batch.done),
                    traj_batch.avail_actions,
                )
                # step count would come from update index
                kl_weight = 0.001  # Placeholder - should use schedule

                total_loss = policy_loss - config["ENT_COEF"] * entropy + kl_weight * kl_div
                return total_loss, {"policy_loss": policy_loss, "entropy": entropy, "kl_div": kl_div}

            actor_grads = jax.grad(actor_loss_fn, has_aux=True)(actor_ts.params)
            actor_ts = actor_ts.apply_gradients(grads=actor_grads[0])

            # Critic update (MSE on value predictions)
            def critic_loss_fn(params):
                if use_role_critic:
                    # Role-specific critic loss
                    all_v, all_q, all_eq, _, _, _, _, _, _, _, _ = critic_network.apply(
                        params,
                        traj_batch.obs_all,
                        traj_batch.actions_all,
                        traj_batch.policy_probs_all,
                        traj_batch.critic_hstate[0],
                        jnp.zeros_like(traj_batch.done).reshape(config["NUM_ENVS"], -1),
                        traj_batch.role_id.reshape(config["NUM_ENVS"], -1),
                        False,
                        True,
                    )
                    v_target = returns.mean()  # Simplified
                    q_target = returns.mean()
                    eq_target = returns.mean()

                    v_loss = jnp.mean(jnp.square(all_v.squeeze(-1) - v_target))
                    q_loss = jnp.mean(jnp.square(all_q.squeeze(-1) - q_target))
                    eq_loss = jnp.mean(jnp.square(all_eq.squeeze(-1) - eq_target))

                    loss = v_loss + 0.5 * q_loss + eq_loss
                else:
                    # Shared critic loss
                    values = critic_network.apply(
                        params,
                        traj_batch.obs_all,
                        traj_batch.actions_all,
                        traj_batch.policy_probs_all,
                        traj_batch.critic_hstate[0],
                        jnp.zeros_like(traj_batch.done).reshape(config["NUM_ENVS"], -1),
                        False,
                        True,
                    )[0].squeeze(-1)
                    loss = jnp.mean(jnp.square(values - returns.mean()))

                return loss

            critic_grads = jax.grad(critic_loss_fn)(critic_ts.params)
            critic_ts = critic_ts.apply_gradients(grads=critic_grads)

            return (rng, actor_ts, critic_ts, env_s, last_obs, actor_hs, critic_hs, vn_dict), {}

        # Run training
        runner_state = (rng, actor_train_state, critic_train_state, env_state, obsv,
                        actor_hstate, critic_hstate, value_norm_dict)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=total_steps
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role_experiment", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = get_default_mappo_t_config()
    config["ROLE_EXPERIMENT"] = args.role_experiment
    config["SEED"] = args.seed
    config["TOTAL_TIMESTEPS"] = int(1e6)  # Small for testing

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train(config)
    out = train_fn(rng)
    print("Training complete. Metrics:", out["metrics"])


if __name__ == "__main__":
    main()
