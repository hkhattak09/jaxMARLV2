"""
MAPPO-T training script for SMAX.

This is a JAX/Flax port of MACA's MAPPO-T actor plus transformer critic, wired
to the same JaxMARL SMAX wrapper style used by ``train_mappo_gru.py``.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper

from mappo_t import ActorTrans, ScannedRNN, TransVCritic, get_default_mappo_t_config
from mappo_t.utils import batchify, unbatchify
from mappo_t.valuenorm import (
    init_value_norm,
    value_norm_update,
    value_norm_normalize,
    value_norm_denormalize,
    create_value_norm_dict,
    update_value_norm_dict,
    normalize_targets,
    denormalize_predictions,
)


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


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides a per-agent ``world_state`` observation for centralized critics."""

    def __init__(self, env: HeuristicEnemySMAX, obs_with_agent_id=True):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        del state
        world_state = obs["world_state"]
        return world_state[None].repeat(self._env.num_allies, axis=0)

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        del state
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):
        return self._world_state_size


def make_train(config):
    """Create a JIT-able MAPPO-T training function."""

    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["PPO_EPOCH"] = config.get("PPO_EPOCH", config.get("UPDATE_EPOCHS", 10))
    config["UPDATE_EPOCHS"] = config.get("UPDATE_EPOCHS", config["PPO_EPOCH"])
    config["ACTOR_NUM_MINI_BATCH"] = config.get(
        "ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1)
    )
    config["NUM_MINIBATCHES"] = config.get("NUM_MINIBATCHES", config["ACTOR_NUM_MINI_BATCH"])
    config["CRITIC_EPOCH"] = config.get("CRITIC_EPOCH", 10)
    config["CRITIC_NUM_MINI_BATCH"] = config.get("CRITIC_NUM_MINI_BATCH", 1)
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["ACTOR_NUM_MINI_BATCH"]
    )
    config["transformer"]["n_block"] = env.num_agents
    if config.get("SCALE_CLIP_EPS", False):
        config["CLIP_PARAM"] = config["CLIP_PARAM"] / env.num_agents

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    actor_hidden_dim = config["hidden_sizes"][-1]
    critic_hidden_dim = config["transformer"]["n_embd"]

    def actor_to_env_agent(x):
        return x.reshape((env.num_agents, config["NUM_ENVS"]) + x.shape[1:]).swapaxes(0, 1)

    def env_agent_to_actor(x):
        return x.swapaxes(0, 1).reshape((config["NUM_ACTORS"],) + x.shape[2:])

    def actor_to_env_agent_time(x):
        return x.reshape(
            (config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"]) + x.shape[2:]
        ).swapaxes(1, 2)

    def env_agent_to_actor_time(x):
        return x.swapaxes(1, 2).reshape(
            (config["NUM_STEPS"], config["NUM_ACTORS"]) + x.shape[3:]
        )

    def env_value_to_actor(values):
        values = jnp.broadcast_to(values[:, None, :], (config["NUM_ENVS"], env.num_agents, values.shape[-1]))
        return env_agent_to_actor(values).squeeze(-1)

    def env_value_to_actor_time(values):
        values = jnp.broadcast_to(
            values[:, :, None, :],
            (values.shape[0], config["NUM_ENVS"], env.num_agents, values.shape[-1]),
        )
        return values.swapaxes(1, 2).reshape((config["NUM_STEPS"], config["NUM_ACTORS"]))

    def linear_schedule(base_lr, steps_per_update):
        """Linear schedule with proper minibatch step counting.
        
        Args:
            base_lr: Base learning rate.
            steps_per_update: Number of gradient steps per update (epochs * minibatches).
        """
        def schedule(count):
            # count is the total number of gradient steps taken
            # We want to decay based on update number, not step number
            update_num = count // steps_per_update
            frac = 1.0 - update_num / config["NUM_UPDATES"]
            return base_lr * jnp.maximum(frac, 0.0)

        return schedule

    def critic_cosine_schedule():
        """Cosine schedule for critic with proper minibatch step counting."""
        base_lr = config["CRITIC_LR"]
        min_lr = config["transformer"].get("min_lr", 0.1 * base_lr)
        warmup_epochs = config["transformer"].get("warmup_epochs", 10)
        critic_steps_per_update = config.get("CRITIC_EPOCH", 10) * config.get("CRITIC_NUM_MINI_BATCH", 1)

        def schedule(count):
            # count is the total number of gradient steps taken.
            # Decay by rollout update, not by critic minibatch or critic epoch.
            epoch = count // critic_steps_per_update + 1
            epoch = jnp.asarray(epoch, dtype=jnp.float32)
            warmup = jnp.asarray(warmup_epochs, dtype=jnp.float32)
            total = jnp.asarray(config["NUM_UPDATES"], dtype=jnp.float32)
            warmup_lr = base_lr * epoch / warmup
            decay_ratio = (epoch - warmup) / jnp.maximum(total - warmup, 1.0)
            decay_ratio = jnp.clip(decay_ratio, 0.0, 1.0)
            coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
            decay_lr = min_lr + coeff * (base_lr - min_lr)
            return jnp.where(
                epoch < warmup,
                warmup_lr,
                jnp.where(epoch > total, min_lr, decay_lr),
            )

        return schedule

    def train(rng):
        # Validate minibatch configuration
        from mappo_t.config import validate_mappo_t_config
        validate_mappo_t_config(config, env.num_agents)
        
        actor_network = ActorTrans(action_dim=action_dim, config=config)
        critic_network = TransVCritic(
            config=config,
            share_obs_space=None,
            obs_space=env.observation_space(env.agents[0]),
            act_space=env.action_space(env.agents[0]),
            num_agents=env.num_agents,
            state_type="EP",
        )

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        ac_init_hstate_small = ScannedRNN.initialize_carry(config["NUM_ENVS"], actor_hidden_dim)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim), dtype=jnp.float32),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
            jnp.ones((1, config["NUM_ENVS"], action_dim), dtype=jnp.float32),
        )
        actor_params = actor_network.init(actor_rng, ac_init_hstate_small, ac_init_x)

        cr_init_hstate_small = jnp.zeros(
            (config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32
        )
        critic_params = critic_network.init(
            critic_rng,
            jnp.zeros((config["NUM_ENVS"], env.num_agents, obs_dim), dtype=jnp.float32),
            jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.int32),
            jnp.ones((config["NUM_ENVS"], env.num_agents, action_dim), dtype=jnp.float32) / action_dim,
            cr_init_hstate_small,
            jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
            True,
            True,
        )
        
        # Apply T-Fixup initialization scaling if configured
        if config["transformer"].get("weight_init") == "tfixup":
            from mappo_t.transformer import apply_tfixup_scaling
            critic_params = apply_tfixup_scaling(critic_params, config["transformer"])

        # Calculate steps per update for learning rate schedules
        actor_steps_per_update = config.get("PPO_EPOCH", config.get("UPDATE_EPOCHS", 10)) * config.get("ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1))
        critic_steps_per_update = config.get("CRITIC_EPOCH", 10) * config.get("CRITIC_NUM_MINI_BATCH", 1)

        if config["ANNEAL_LR"]:
            actor_lr = linear_schedule(config["LR"], actor_steps_per_update)
        else:
            actor_lr = config["LR"]

        if config.get("USE_CRITIC_LR_DECAY", False):
            critic_lr = critic_cosine_schedule()
        elif config["ANNEAL_LR"]:
            critic_lr = linear_schedule(config["CRITIC_LR"], critic_steps_per_update)
        else:
            critic_lr = config["CRITIC_LR"]

        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(actor_lr, eps=config.get("opti_eps", 1e-5)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(critic_lr, eps=config.get("opti_eps", 1e-5)),
        )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply, params=actor_params, tx=actor_tx
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply, params=critic_params, tx=critic_tx
        )

        # Initialize ValueNorm for v, q, eq if enabled
        use_valuenorm = config.get("use_valuenorm", True)
        value_norm_dict = create_value_norm_dict(
            use_valuenorm=use_valuenorm,
            v_shape=(1,),
            q_shape=(1,),
            eq_shape=(1,),
        )

        rng, reset_rng = jax.random.split(rng)
        reset_rng = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], actor_hidden_dim)
        cr_init_hstate = jnp.zeros(
            (config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32
        )

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_env_done,
                    last_agent_done,
                    hstates,
                    value_norm_dict,
                    rng,
                ) = runner_state
                actor_train_state, critic_train_state = train_states
                ac_hstate, cr_hstate = hstates

                rng, action_rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                ac_in = (
                    obs_batch[None, :],
                    last_env_done[None, :],
                    avail_actions[None, :],
                )
                ac_hstate, pi = actor_network.apply(actor_train_state.params, ac_hstate, ac_in)
                action = pi.sample(seed=action_rng).squeeze(0)
                log_prob = pi.log_prob(action[None, :]).squeeze(0)
                policy_probs = pi.probs.squeeze(0)

                obs_all = jnp.stack([last_obs[a] for a in env.agents], axis=1)
                actions_all = actor_to_env_agent(action)
                policy_probs_all = actor_to_env_agent(policy_probs)
                critic_resets = actor_to_env_agent(last_env_done)
                (
                    values,
                    q_values,
                    eq_values,
                    vq_values,
                    vq_coma_values,
                    baseline_weights,
                    attn_weights,
                    _,
                    _,
                    cr_hstate,
                ) = critic_network.apply(
                    critic_train_state.params,
                    obs_all,
                    actions_all,
                    policy_probs_all,
                    cr_hstate,
                    critic_resets,
                    True,
                    True,
                )

                rng, step_rng = jax.random.split(rng)
                step_rng = jax.random.split(step_rng, config["NUM_ENVS"])
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
                env_act = {k: v.squeeze(-1) for k, v in env_act.items()}
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    step_rng, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                env_done_batch = jnp.tile(done["__all__"], env.num_agents)
                agent_done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                active_mask = jnp.where(
                    last_env_done,
                    jnp.ones_like(last_agent_done, dtype=jnp.float32),
                    1.0 - last_agent_done.astype(jnp.float32),
                )

                transition = Transition(
                    env_done_batch,
                    last_env_done,
                    active_mask,
                    action,
                    env_value_to_actor(values),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    last_obs["world_state"].swapaxes(0, 1).reshape((config["NUM_ACTORS"], -1)),
                    info,
                    avail_actions,
                    policy_probs,
                    obs_all,
                    actions_all,
                    policy_probs_all,
                    values.squeeze(-1),
                    q_values.squeeze(-1),
                    eq_values.squeeze(-1),
                    vq_values.squeeze(-1),
                    vq_coma_values.squeeze(-1),
                    baseline_weights,
                    attn_weights,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    env_done_batch,
                    agent_done_batch,
                    (ac_hstate, cr_hstate),
                    value_norm_dict,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                train_states,
                env_state,
                last_obs,
                last_env_done,
                last_agent_done,
                hstates,
                value_norm_dict,
                rng,
            ) = runner_state
            actor_train_state, critic_train_state = train_states
            _, cr_hstate = hstates

            last_obs_all = jnp.stack([last_obs[a] for a in env.agents], axis=1)
            (
                last_values,
                last_q_values,
                last_eq_values,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = critic_network.apply(
                critic_train_state.params,
                last_obs_all,
                traj_batch.actions_all[-1],
                traj_batch.policy_probs_all[-1],
                cr_hstate,
                actor_to_env_agent(last_env_done),
                True,
                True,
            )
            last_val = env_value_to_actor(last_values)

            # Denormalize predictions for GAE calculation if ValueNorm is enabled
            if use_valuenorm:
                last_val_denorm = value_norm_denormalize(value_norm_dict["v"], last_val)
                last_q_denorm = value_norm_denormalize(value_norm_dict["q"], env_value_to_actor(last_q_values))
                last_eq_denorm = value_norm_denormalize(value_norm_dict["eq"], env_value_to_actor(last_eq_values))
            else:
                last_val_denorm = last_val
                last_q_denorm = env_value_to_actor(last_q_values)
                last_eq_denorm = env_value_to_actor(last_eq_values)

            def _calculate_gae(preds, rewards, dones, last_pred):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition
                    delta = (
                        reward
                        + config["GAMMA"] * next_value * (1 - done)
                        - value
                    )
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_pred), last_pred),
                    (dones, preds, rewards),
                    reverse=True,
                    unroll=16,
                )
                return advantages + preds

            # Use denormalized values for GAE targets
            value_targets = _calculate_gae(
                value_norm_denormalize(value_norm_dict["v"], traj_batch.value) if use_valuenorm else traj_batch.value,
                traj_batch.reward,
                traj_batch.global_done,
                last_val_denorm,
            )
            q_targets = _calculate_gae(
                env_value_to_actor_time(value_norm_denormalize(value_norm_dict["q"], traj_batch.q_value_env[..., None])) if use_valuenorm else env_value_to_actor_time(traj_batch.q_value_env[..., None]),
                traj_batch.reward,
                traj_batch.global_done,
                last_q_denorm,
            )
            eq_targets = _calculate_gae(
                env_value_to_actor_time(value_norm_denormalize(value_norm_dict["eq"], traj_batch.eq_value_env[..., None])) if use_valuenorm else env_value_to_actor_time(traj_batch.eq_value_env[..., None]),
                traj_batch.reward,
                traj_batch.global_done,
                last_eq_denorm,
            )
            eq_returns_env = actor_to_env_agent_time(eq_targets)
            eq_value_env_for_baseline = (
                value_norm_denormalize(
                    value_norm_dict["eq"], traj_batch.eq_value_env[..., None]
                ).squeeze(-1)
                if use_valuenorm
                else traj_batch.eq_value_env
            )
            eq_values_env = jnp.broadcast_to(
                eq_value_env_for_baseline[:, :, None],
                (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents),
            )
            baselines = (
                traj_batch.baseline_weights[..., 0] * traj_batch.vq_coma_value_env
                + traj_batch.baseline_weights[..., 1] * traj_batch.vq_value_env
                + traj_batch.baseline_weights[..., 2] * eq_values_env
            )
            advantages = env_agent_to_actor_time(eq_returns_env - baselines)
            active_masks = traj_batch.active_mask.astype(jnp.float32)
            active_count = jnp.sum(active_masks) + 1e-8
            mean_adv = jnp.sum(advantages * active_masks) / active_count
            var_adv = jnp.sum(jnp.square(advantages - mean_adv) * active_masks) / active_count
            norm_advantages = (advantages - mean_adv) / jnp.sqrt(var_adv + 1e-8)
            
            # === Prepare actor minibatch data ===
            # Flatten: (T, NUM_ACTORS, ...) -> (T * NUM_ACTORS, ...)
            actor_batch_size = config["NUM_STEPS"] * config["NUM_ACTORS"]
            actor_num_mini_batch = config.get("ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1))
            actor_mini_batch_size = actor_batch_size // actor_num_mini_batch
            
            # Flatten actor tensors
            actor_obs = traj_batch.obs.reshape(actor_batch_size, obs_dim)  # (T*NA, obs_dim)
            actor_done = traj_batch.done.reshape(actor_batch_size)  # (T*NA,)
            actor_avail = traj_batch.avail_actions.reshape(actor_batch_size, action_dim)  # (T*NA, action_dim)
            actor_action = traj_batch.action.reshape(actor_batch_size)  # (T*NA,)
            actor_log_prob = traj_batch.log_prob.reshape(actor_batch_size)  # (T*NA,)
            actor_active_mask = traj_batch.active_mask.reshape(actor_batch_size).astype(jnp.float32)  # (T*NA,)
            actor_norm_adv = norm_advantages.reshape(actor_batch_size)  # (T*NA,)
            
            # === Prepare critic minibatch data ===
            # Flatten: (T, NUM_ENVS, ...) -> (T * NUM_ENVS, ...) but preserve agent axis
            critic_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
            critic_num_mini_batch = config.get("CRITIC_NUM_MINI_BATCH", 1)
            critic_mini_batch_size = critic_batch_size // critic_num_mini_batch
            
            # Critic tensors: reshape to (T*NUM_ENVS, num_agents, ...)
            critic_obs_all = traj_batch.obs_all.reshape(critic_batch_size, env.num_agents, obs_dim)  # (T*NE, agents, obs_dim)
            critic_actions_all = traj_batch.actions_all.reshape(critic_batch_size, env.num_agents)  # (T*NE, agents)
            critic_policy_probs_all = traj_batch.policy_probs_all.reshape(critic_batch_size, env.num_agents, action_dim)  # (T*NE, agents, action_dim)
            critic_done = actor_to_env_agent_time(traj_batch.done).reshape(critic_batch_size, env.num_agents)  # (T*NE, agents)
            
            # Critic targets are env-level. Returns were calculated in actor order
            # for PPO advantages, so convert back to (T, env, agent) and average
            # the broadcast agent dimension.
            critic_value_targets = actor_to_env_agent_time(value_targets).mean(axis=-1).reshape(critic_batch_size)
            critic_q_targets = actor_to_env_agent_time(q_targets).mean(axis=-1).reshape(critic_batch_size)
            critic_eq_targets = actor_to_env_agent_time(eq_targets).mean(axis=-1).reshape(critic_batch_size)
            
            # Old predictions for clipping: (T*NUM_ENVS,)
            critic_value_old = traj_batch.value_env.reshape(critic_batch_size)
            critic_q_old = traj_batch.q_value_env.reshape(critic_batch_size)
            critic_eq_old = traj_batch.eq_value_env.reshape(critic_batch_size)
            
            # === Minibatch update functions ===
            def _actor_minibatch_update(actor_state, minibatch_idx):
                """Update actor on a single minibatch."""
                actor_train_state = actor_state
                
                # Gather minibatch data using indices
                mb_obs = jnp.take(actor_obs, minibatch_idx, axis=0)
                mb_done = jnp.take(actor_done, minibatch_idx, axis=0)
                mb_avail = jnp.take(actor_avail, minibatch_idx, axis=0)
                mb_action = jnp.take(actor_action, minibatch_idx, axis=0)
                mb_log_prob = jnp.take(actor_log_prob, minibatch_idx, axis=0)
                mb_active = jnp.take(actor_active_mask, minibatch_idx, axis=0)
                mb_adv = jnp.take(actor_norm_adv, minibatch_idx, axis=0)
                
                # Initial hidden state for minibatch (non-recurrent case)
                ac_init_hstate_mb = ScannedRNN.initialize_carry(
                    minibatch_idx.shape[0], actor_hidden_dim
                )
                
                def _actor_loss_fn(actor_params):
                    _, pi = actor_network.apply(
                        actor_params,
                        ac_init_hstate_mb,
                        (mb_obs, mb_done, mb_avail),
                    )
                    log_prob = pi.log_prob(mb_action)
                    logratio = log_prob - mb_log_prob
                    ratio = jnp.exp(logratio)
                    loss_actor1 = ratio * mb_adv
                    loss_actor2 = (
                        jnp.clip(ratio, 1.0 - config["CLIP_PARAM"], 1.0 + config["CLIP_PARAM"])
                        * mb_adv
                    )
                    mb_active_count = jnp.sum(mb_active) + 1e-8
                    policy_loss = (
                        -jnp.sum(jnp.minimum(loss_actor1, loss_actor2) * mb_active)
                        / mb_active_count
                    )
                    entropy = jnp.sum(pi.entropy() * mb_active) / mb_active_count
                    approx_kl = jnp.sum(((ratio - 1) - logratio) * mb_active) / mb_active_count
                    clip_frac = (
                        jnp.sum((jnp.abs(ratio - 1) > config["CLIP_PARAM"]) * mb_active)
                        / mb_active_count
                    )
                    actor_loss = policy_loss - config["ENT_COEF"] * entropy
                    return actor_loss, (policy_loss, entropy, approx_kl, clip_frac, mb_active_count)
                
                (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                    _actor_loss_fn, has_aux=True
                )(actor_train_state.params)
                actor_grad_norm = optax.global_norm(actor_grads)
                actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                
                actor_info = {
                    "actor_loss": actor_aux[0],
                    "entropy": actor_aux[1],
                    "approx_kl": actor_aux[2],
                    "clip_frac": actor_aux[3],
                    "actor_grad_norm": actor_grad_norm,
                    "mb_active_count": actor_aux[4],
                }
                return actor_train_state, actor_info
            
            def _critic_minibatch_update(critic_state, minibatch_idx):
                """Update critic on a single minibatch."""
                critic_train_state, value_norm_dict = critic_state
                
                # Gather minibatch data preserving agent axis
                mb_obs_all = jnp.take(critic_obs_all, minibatch_idx, axis=0)  # (mb, agents, obs_dim)
                mb_actions_all = jnp.take(critic_actions_all, minibatch_idx, axis=0)  # (mb, agents)
                mb_policy_probs_all = jnp.take(critic_policy_probs_all, minibatch_idx, axis=0)  # (mb, agents, action_dim)
                mb_done = jnp.take(critic_done, minibatch_idx, axis=0)  # (mb, agents)
                
                # Targets and old predictions (no agent axis for these)
                mb_value_targets = jnp.take(critic_value_targets, minibatch_idx, axis=0)
                mb_q_targets = jnp.take(critic_q_targets, minibatch_idx, axis=0)
                mb_eq_targets = jnp.take(critic_eq_targets, minibatch_idx, axis=0)
                mb_value_old = jnp.take(critic_value_old, minibatch_idx, axis=0)
                mb_q_old = jnp.take(critic_q_old, minibatch_idx, axis=0)
                mb_eq_old = jnp.take(critic_eq_old, minibatch_idx, axis=0)
                
                # Initial hidden state for minibatch
                cr_init_hstate_mb = jnp.zeros(
                    (minibatch_idx.shape[0], env.num_agents, critic_hidden_dim), dtype=jnp.float32
                )
                
                def _critic_loss_fn(critic_params, norm_dict):
                    (
                        values,
                        q_values,
                        eq_values,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) = critic_network.apply(
                        critic_params,
                        mb_obs_all,
                        mb_actions_all,
                        mb_policy_probs_all,
                        cr_init_hstate_mb,
                        mb_done,
                        True,
                        True,
                    )
                    value_pred = values.squeeze(-1)  # (mb,)
                    q_pred = q_values.squeeze(-1)  # (mb,)
                    eq_pred = eq_values.squeeze(-1)  # (mb,)
                    
                    # Update ValueNorm with raw targets before computing loss
                    if norm_dict is not None and use_valuenorm:
                        new_v_norm = value_norm_update(norm_dict["v"], mb_value_targets)
                        new_q_norm = value_norm_update(norm_dict["q"], mb_q_targets)
                        new_eq_norm = value_norm_update(norm_dict["eq"], mb_eq_targets)
                        norm_dict = {
                            "v": new_v_norm,
                            "q": new_q_norm,
                            "eq": new_eq_norm,
                        }
                        v_targets_norm = value_norm_normalize(norm_dict["v"], mb_value_targets)
                        q_targets_norm = value_norm_normalize(norm_dict["q"], mb_q_targets)
                        eq_targets_norm = value_norm_normalize(norm_dict["eq"], mb_eq_targets)
                    else:
                        v_targets_norm = mb_value_targets
                        q_targets_norm = mb_q_targets
                        eq_targets_norm = mb_eq_targets
                    
                    def _element_loss(error):
                        if config.get("use_huber_loss", False):
                            delta = config.get("huber_delta", 10.0)
                            abs_error = jnp.abs(error)
                            return jnp.where(
                                abs_error <= delta,
                                0.5 * jnp.square(error),
                                delta * (abs_error - 0.5 * delta),
                            )
                        return 0.5 * jnp.square(error)
                    
                    def _value_loss(pred, old_pred, target):
                        clipped = old_pred + jnp.clip(
                            pred - old_pred,
                            -config["CLIP_PARAM"],
                            config["CLIP_PARAM"],
                        )
                        original_loss = _element_loss(target - pred)
                        clipped_loss = _element_loss(target - clipped)
                        if config.get("use_clipped_value_loss", True):
                            return jnp.maximum(original_loss, clipped_loss).mean()
                        return original_loss.mean()
                    
                    value_loss = _value_loss(value_pred, mb_value_old, v_targets_norm)
                    q_value_loss = _value_loss(q_pred, mb_q_old, q_targets_norm)
                    eq_value_loss = _value_loss(eq_pred, mb_eq_old, eq_targets_norm)
                    critic_loss = (
                        config["VALUE_LOSS_COEF"] * value_loss
                        + config["transformer"]["q_value_loss_coef"] * q_value_loss
                        + config["transformer"]["eq_value_loss_coef"] * eq_value_loss
                    )
                    return critic_loss, (value_loss, q_value_loss, eq_value_loss, norm_dict)
                
                (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(critic_train_state.params, value_norm_dict)
                critic_grad_norm = optax.global_norm(critic_grads)
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                
                # Update value_norm_dict from critic loss output
                value_norm_dict = critic_aux[3]
                
                critic_info = {
                    "value_loss": critic_aux[0],
                    "q_value_loss": critic_aux[1],
                    "eq_value_loss": critic_aux[2],
                    "critic_grad_norm": critic_grad_norm,
                }
                return (critic_train_state, value_norm_dict), critic_info
            
            # === Run minibatch updates with nested scans ===
            def _run_actor_epoch(actor_state, rng_epoch):
                """Run one PPO epoch with minibatches."""
                # Generate random permutation for this epoch
                rng_perm, rng_epoch = jax.random.split(rng_epoch)
                perm = jax.random.permutation(rng_perm, actor_batch_size)
                # Reshape to (num_mini_batches, mini_batch_size)
                minibatch_idx = perm.reshape(actor_num_mini_batch, actor_mini_batch_size)
                
                # Scan over minibatches
                def scan_actor_minibatch(astate, mb_idx):
                    new_astate, info = _actor_minibatch_update(astate, mb_idx)
                    return new_astate, info
                
                final_actor_state, actor_infos = jax.lax.scan(
                    scan_actor_minibatch, actor_state, minibatch_idx
                )
                # Average info over minibatches
                actor_avg_info = jax.tree.map(lambda x: x.mean(), actor_infos)
                return final_actor_state, (rng_epoch, actor_avg_info)
            
            def _run_critic_epoch(critic_state, rng_epoch):
                """Run one critic epoch with minibatches."""
                critic_train_state, value_norm_dict = critic_state
                # Generate random permutation for this epoch
                rng_perm, rng_epoch = jax.random.split(rng_epoch)
                perm = jax.random.permutation(rng_perm, critic_batch_size)
                # Reshape to (num_mini_batches, mini_batch_size)
                minibatch_idx = perm.reshape(critic_num_mini_batch, critic_mini_batch_size)
                
                # Scan over minibatches
                def scan_critic_minibatch(cstate, mb_idx):
                    new_cstate, info = _critic_minibatch_update(cstate, mb_idx)
                    return new_cstate, info
                
                final_critic_state, critic_infos = jax.lax.scan(
                    scan_critic_minibatch, (critic_train_state, value_norm_dict), minibatch_idx
                )
                # Average info over minibatches
                critic_avg_info = jax.tree.map(lambda x: x.mean(), critic_infos)
                return final_critic_state, (rng_epoch, critic_avg_info)
            
            # Run PPO epochs for actor
            rng, actor_rng = jax.random.split(rng)
            def _scan_actor_epoch(carry, _):
                actor_state, epoch_rng = carry
                actor_state, (epoch_rng, info) = _run_actor_epoch(actor_state, epoch_rng)
                return (actor_state, epoch_rng), info

            (actor_train_state, actor_rng), actor_epoch_infos = jax.lax.scan(
                _scan_actor_epoch,
                (actor_train_state, actor_rng),
                None,
                config.get("PPO_EPOCH", config.get("UPDATE_EPOCHS", 10)),
            )
            
            # Run epochs for critic
            rng, critic_rng = jax.random.split(rng)
            def _scan_critic_epoch(carry, _):
                critic_state, epoch_rng = carry
                critic_state, (epoch_rng, info) = _run_critic_epoch(critic_state, epoch_rng)
                return (critic_state, epoch_rng), info

            ((critic_train_state, value_norm_dict), critic_rng), critic_epoch_infos = jax.lax.scan(
                _scan_critic_epoch,
                ((critic_train_state, value_norm_dict), critic_rng),
                None,
                config.get("CRITIC_EPOCH", 10),
            )
            actor_epoch_infos = jax.tree.map(lambda x: x.mean(), actor_epoch_infos)
            critic_epoch_infos = jax.tree.map(lambda x: x.mean(), critic_epoch_infos)
            
            # Combine loss info
            loss_info = {
                "actor_loss": actor_epoch_infos["actor_loss"],
                "value_loss": critic_epoch_infos["value_loss"],
                "q_value_loss": critic_epoch_infos["q_value_loss"],
                "eq_value_loss": critic_epoch_infos["eq_value_loss"],
                "entropy": actor_epoch_infos["entropy"],
                "approx_kl": actor_epoch_infos["approx_kl"],
                "clip_frac": actor_epoch_infos["clip_frac"],
                "actor_grad_norm": actor_epoch_infos["actor_grad_norm"],
                "critic_grad_norm": critic_epoch_infos["critic_grad_norm"],
            }
            loss_info["total_loss"] = (
                loss_info["actor_loss"]
                + config["VALUE_LOSS_COEF"] * loss_info["value_loss"]
                + config["transformer"]["q_value_loss_coef"] * loss_info["q_value_loss"]
                + config["transformer"]["eq_value_loss_coef"] * loss_info["eq_value_loss"]
            )

            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            mask = metric["returned_episode"][:, :, 0]
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count

            def log_callback(r, w, s, tl, ent, agn, cgn):
                print(
                    f"Step {s:8d} | Return: {r:10.2f} | Win Rate: {w:5.2f} "
                    f"| Loss: {tl:10.4f} | Ent: {ent:8.4f} "
                    f"| GradN(actor/critic): {agn:8.4f}/{cgn:8.4f}"
                )

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.experimental.io_callback(
                log_callback,
                None,
                returns,
                win_rate,
                step_count,
                loss_info["total_loss"],
                loss_info["entropy"],
                loss_info["actor_grad_norm"],
                loss_info["critic_grad_norm"],
            )

            runner_state = (
                (actor_train_state, critic_train_state),
                env_state,
                last_obs,
                last_env_done,
                last_agent_done,
                hstates,
                value_norm_dict,
                rng,
            )
            return (runner_state, update_steps + 1), metric

        rng, run_rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"],), dtype=bool),
            jnp.zeros((config["NUM_ACTORS"],), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            value_norm_dict,
            run_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    config = get_default_mappo_t_config()

    print(f"Starting MAPPO-T training on {config['MAP_NAME']}...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smax_mappo_t_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_actor_state = final_runner_state[0][0]
    checkpoint = {
        "model_type": "transformer",
        "config": config,
        "actor_params": jax.device_get(final_actor_state.params),
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved Transformer actor checkpoint to {model_path}")
