"""
MAPPO-T training script for SMAX.

This is a JAX/Flax port of MACA's MAPPO-T actor plus transformer critic, wired
to the same JaxMARL SMAX wrapper style used by ``train_mappo_gru.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime
from functools import partial
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from jaxmarl.environments.spaces import Box
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper

from mappo_t import ActorTrans, ScannedRNN, TransVCritic, get_default_mappo_t_config
from mappo_t.utils import batchify, unbatchify
from mappo_t.valuenorm import (
    create_value_norm_dict,
    value_norm_denormalize,
    value_norm_normalize,
    value_norm_update,
)


def _debug_print(condition, fmt, *args, **kwargs):
    """Conditional JAX debug print inside JIT."""
    jax.lax.cond(
        condition,
        lambda: jax.debug.print(fmt, *args, **kwargs),
        lambda: None,
    )


def _check_finite(x, name, update_steps, step_idx=-1):
    """Warn if tensor contains NaN or Inf."""
    all_finite = jnp.all(jnp.isfinite(x))
    jax.lax.cond(
        jnp.logical_not(all_finite),
        lambda: jax.debug.print(
            "!!! NON-FINITE in {name} at update {upd} step {step} | finite={fc}/{tot} !!!",
            name=name,
            upd=update_steps,
            step=step_idx,
            fc=jnp.sum(jnp.isfinite(x)),
            tot=x.size,
        ),
        lambda: None,
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
    bad_mask: jnp.ndarray
    actor_hstate: jnp.ndarray
    critic_hstate: jnp.ndarray


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides MACA-style local observations and per-agent world_state."""

    def __init__(
        self,
        env: HeuristicEnemySMAX,
        obs_with_agent_id=True,
        local_obs_with_agent_id=True,
    ):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        self.local_obs_with_agent_id = local_obs_with_agent_id
        base_obs_space = self._env.observation_space(self._env.agents[0])
        self._obs_size = base_obs_space.shape[0] + (
            self._env.num_allies if self.local_obs_with_agent_id else 0
        )
        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    def _battle_terminal(self, raw_state):
        ally_alive = raw_state.unit_alive[: self._env.num_allies]
        enemy_alive = raw_state.unit_alive[self._env.num_allies :]
        if self._env.medivac_type_idx is not None:
            ally_alive = ally_alive & (
                raw_state.unit_types[: self._env.num_allies] != self._env.medivac_type_idx
            )
            enemy_alive = enemy_alive & (
                raw_state.unit_types[self._env.num_allies :] != self._env.medivac_type_idx
            )
        battle_done = jnp.all(~ally_alive) | jnp.all(~enemy_alive)
        won_battle = jnp.all(~enemy_alive) & jnp.any(ally_alive)
        return battle_done, won_battle

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        obs = self.local_obs_fn(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        # Use step_env to access pre-auto-reset state for timeout detection.
        step_key, reset_key = jax.random.split(key)
        obs_st, state_st, reward, done, info = self._env.step_env(step_key, state, action)

        # Compute bad_transition before losing stepped state.
        raw_step_state = state_st.state
        battle_done, won_battle = self._battle_terminal(raw_step_state)
        timeout_done = raw_step_state.time >= self._env.max_steps
        bad_transition = done["__all__"] & timeout_done & ~battle_done
        info["bad_transition"] = jnp.full((self._env.num_allies,), bad_transition)
        info["battle_won"] = jnp.full((self._env.num_allies,), won_battle)

        # Manual auto-reset matching MultiAgentEnv.step.
        obs_reset, state_reset = self._env.reset(reset_key)
        env_state = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), state_reset, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), obs_reset, obs_st
        )

        obs["world_state"] = self.world_state_fn(obs, env_state)
        obs = self.local_obs_fn(obs)
        return obs, env_state, reward, done, info

    def local_obs_fn(self, obs):
        if not self.local_obs_with_agent_id:
            return obs
        one_hot = jnp.eye(self._env.num_allies, dtype=jnp.float32)
        obs = dict(obs)
        for idx, agent in enumerate(self._env.agents):
            obs[agent] = jnp.concatenate((obs[agent], one_hot[idx]), axis=-1)
        return obs

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, env_state):
        del env_state
        world_state = obs["world_state"]
        return world_state[None].repeat(self._env.num_allies, axis=0)

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, env_state):
        del env_state
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):
        return self._world_state_size

    def observation_space(self, agent):
        base = self._env.observation_space(agent)
        return Box(low=base.low, high=base.high, shape=(self._obs_size,), dtype=base.dtype)


def make_train(config):
    """Create a JIT-able MAPPO-T training function."""

    config["ENV_KWARGS"] = dict(config.get("ENV_KWARGS", {}))
    config["ENV_KWARGS"].setdefault("max_steps", config["NUM_STEPS"])

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

    env = SMAXWorldStateWrapper(
        env,
        obs_with_agent_id=config["OBS_WITH_AGENT_ID"],
        local_obs_with_agent_id=config.get(
            "LOCAL_OBS_WITH_AGENT_ID", config["OBS_WITH_AGENT_ID"]
        ),
    )
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
        critic_steps_per_update = config["CRITIC_EPOCH"] * config["CRITIC_NUM_MINI_BATCH"]

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
        # Validate configuration
        from mappo_t.config import validate_mappo_t_config
        validate_mappo_t_config(config, env.num_agents)

        # Guard unsupported features
        if config["transformer"].get("next_s_pred_loss_coef", 0.0) > 0.0:
            raise NotImplementedError(
                "next_s_pred_loss_coef > 0 is not implemented in this JAX MAPPO-T trainer."
            )
        if not config.get("share_param", True):
            raise NotImplementedError(
                "This JAX MAPPO-T trainer currently implements MACA's shared-parameter actor path only."
            )
        if not config.get("use_gae", True):
            raise NotImplementedError(
                "use_gae=False is not implemented in this JAX MAPPO-T trainer."
            )
        if config["transformer"].get("dropout", 0.0) != 0.0:
            raise NotImplementedError(
                "Transformer dropout > 0 requires training-time dropout RNG plumbing, which is not implemented."
            )
        action_space = env.action_space(env.agents[0])
        if not hasattr(action_space, "n"):
            raise ValueError(
                f"This port currently supports Discrete action spaces only. Got {type(action_space).__name__}."
            )
        if config.get("action_aggregation", "single") not in ("single", "prod", "mean"):
            raise ValueError(
                f"Unsupported action_aggregation: {config.get('action_aggregation', 'single')}"
            )
        
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
        actor_steps_per_update = config["PPO_EPOCH"] * config["ACTOR_NUM_MINI_BATCH"]
        critic_steps_per_update = config["CRITIC_EPOCH"] * config["CRITIC_NUM_MINI_BATCH"]

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
        betas = config["transformer"].get("betas", [0.9, 0.95])
        weight_decay = config["transformer"].get("wght_decay", 0.01)

        def decay_mask(params):
            return jax.tree.map(lambda p: p.ndim >= 2, params)

        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(
                learning_rate=critic_lr,
                b1=betas[0],
                b2=betas[1],
                eps=config.get("opti_eps", 1e-5),
                weight_decay=weight_decay,
                mask=decay_mask(critic_params),
            ),
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

        def _run_eval(eval_rng, actor_ts):
            eval_num_envs = config.get("EVAL_NUM_ENVS", config["NUM_ENVS"])
            eval_steps = config.get("EVAL_STEPS", config["NUM_STEPS"])
            eval_rng, reset_rng = jax.random.split(eval_rng)
            reset_rng = jax.random.split(reset_rng, eval_num_envs)
            eval_obsv, eval_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            eval_ac_hstate = ScannedRNN.initialize_carry(
                eval_num_envs * env.num_agents, actor_hidden_dim
            )
            eval_last_done = jnp.zeros(
                (eval_num_envs * env.num_agents,), dtype=bool
            )

            def _eval_env_step(carry, _):
                actor_train_state, env_s, last_obs, last_done, ac_hstate, rng = carry
                rng, _ = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_s.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, eval_num_envs * env.num_agents)
                )
                obs_batch = batchify(last_obs, env.agents, eval_num_envs * env.num_agents)
                ac_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                    avail_actions[None, :],
                )
                ac_hstate, pi = actor_network.apply(
                    actor_train_state.params, ac_hstate, ac_in
                )
                action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
                env_act = unbatchify(
                    action, env.agents, eval_num_envs, eval_num_envs * env.num_agents
                )
                env_act = {k: v.squeeze(-1) for k, v in env_act.items()}
                rng, step_rng = jax.random.split(rng)
                step_rng = jax.random.split(step_rng, eval_num_envs)
                obsv, env_s, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_s, env_act)
                reward_batch = batchify(
                    reward, env.agents, eval_num_envs * env.num_agents
                ).squeeze()
                env_done_batch = jnp.tile(done["__all__"], env.num_agents)
                return (
                    actor_train_state,
                    env_s,
                    obsv,
                    env_done_batch,
                    ac_hstate,
                    rng,
                ), reward_batch

            _, eval_rewards = jax.lax.scan(
                _eval_env_step,
                (
                    actor_ts,
                    eval_env_state,
                    eval_obsv,
                    eval_last_done,
                    eval_ac_hstate,
                    eval_rng,
                ),
                None,
                eval_steps,
            )
            return jnp.mean(eval_rewards)

        # === Checkpoint / logging setup ===
        save_interval = config.get("SAVE_INTERVAL", 1000000)
        print_interval = max(1, save_interval // 5)

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        saved_models_dir = os.path.join(_REPO_ROOT, "saved_models")
        os.makedirs(saved_models_dir, exist_ok=True)
        run_dir = os.path.join(saved_models_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)

        params_path = os.path.join(run_dir, "run_params.json")
        with open(params_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        csv_path = os.path.join(run_dir, "progress.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "update", "return", "win_rate", "win_rate_std",
                "ep_len", "timeout_rate",
                "value_loss", "entropy", "clip_frac", "approx_kl",
                "actor_grad_norm", "critic_grad_norm",
            ])

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, step_idx):
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
                actor_hstate_in = ac_hstate
                critic_hstate_in = cr_hstate

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
                bad_mask_env = 1.0 - info["bad_transition"][:, 0].astype(jnp.float32)
                bad_mask = jnp.tile(bad_mask_env, env.num_agents)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                env_done_batch = jnp.tile(done["__all__"], env.num_agents)
                agent_done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                active_mask = jnp.where(
                    last_env_done,
                    jnp.ones_like(last_agent_done, dtype=jnp.float32),
                    1.0 - last_agent_done.astype(jnp.float32),
                )
                reward_batch = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()

                # --- Comprehensive Environment Step Debugging ---
                _check_finite(obs_batch, "obs_batch", update_steps, step_idx)
                _check_finite(avail_actions, "avail_actions", update_steps, step_idx)
                _check_finite(pi.logits, "pi.logits", update_steps, step_idx)
                _check_finite(action, "action", update_steps, step_idx)
                _check_finite(values, "critic_values", update_steps, step_idx)
                _check_finite(reward_batch, "reward", update_steps, step_idx)
                _check_finite(ac_hstate, "ac_hstate", update_steps, step_idx)
                _check_finite(cr_hstate, "cr_hstate", update_steps, step_idx)

                zero_avail = jnp.any(jnp.sum(avail_actions, axis=-1) == 0)
                jax.lax.cond(
                    zero_avail,
                    lambda: jax.debug.print(
                        "!!! ZERO AVAIL ACTIONS at upd {upd} step {step} !!!",
                        upd=update_steps,
                        step=step_idx,
                    ),
                    lambda: None,
                )

                action_out_of_bounds = jnp.any(action < 0) | jnp.any(action >= action_dim)
                jax.lax.cond(
                    action_out_of_bounds,
                    lambda: jax.debug.print(
                        "!!! ACTION OUT OF BOUNDS at upd {upd} step {step} | min={amin} max={amax} dim={adim} !!!",
                        upd=update_steps,
                        step=step_idx,
                        amin=jnp.min(action),
                        amax=jnp.max(action),
                        adim=action_dim,
                    ),
                    lambda: None,
                )

                debug_env = (update_steps < 2) & (step_idx < 3)
                _debug_print(
                    debug_env,
                    "[ENV] upd={upd} step={step} | "
                    "obs(min/max/mean)={omin:.3f}/{omax:.3f}/{omean:.3f} | "
                    "avail_sum={asum:.1f} | "
                    "logits(min/max)={lmin:.3f}/{lmax:.3f} | "
                    "actions(min/max)={amin}/{amax} | "
                    "rew(sum/mean)={rsum:.3f}/{rmean:.3f} | "
                    "done_all={dsum} | won={wsum} | "
                    "values(min/max/mean)={vmin:.3f}/{vmax:.3f}/{vmean:.3f} | "
                    "ac_hstate_norm={hn:.3f} | cr_hstate_norm={crn:.3f}",
                    upd=update_steps,
                    step=step_idx,
                    omin=jnp.min(obs_batch),
                    omax=jnp.max(obs_batch),
                    omean=jnp.mean(obs_batch),
                    asum=jnp.sum(avail_actions),
                    lmin=jnp.min(pi.logits),
                    lmax=jnp.max(pi.logits),
                    amin=jnp.min(action),
                    amax=jnp.max(action),
                    rsum=jnp.sum(reward_batch),
                    rmean=jnp.mean(reward_batch),
                    dsum=jnp.sum(done["__all__"].astype(jnp.int32)),
                    wsum=jnp.sum(info["battle_won"].astype(jnp.int32)),
                    vmin=jnp.min(env_value_to_actor(values)),
                    vmax=jnp.max(env_value_to_actor(values)),
                    vmean=jnp.mean(env_value_to_actor(values)),
                    hn=jnp.linalg.norm(ac_hstate),
                    crn=jnp.linalg.norm(cr_hstate),
                )
                # --- End Environment Step Debugging ---

                transition = Transition(
                    env_done_batch,
                    last_env_done,
                    active_mask,
                    action,
                    env_value_to_actor(values),
                    reward_batch,
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
                    bad_mask,
                    actor_hstate_in,
                    critic_hstate_in,
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
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
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

            def _denorm_if_needed(norm_key, x):
                if use_valuenorm:
                    return value_norm_denormalize(value_norm_dict[norm_key], x[..., None]).squeeze(-1)
                return x

            # GAE helper computes returns from denormalized raw-scale value
            # predictions when ValueNorm is enabled (matching MACA).
            # MACA computes GAE with raw values, stores raw returns, and only
            # normalizes targets inside the critic loss via ValueNorm.
            def _calculate_gae(preds, rewards, dones, bad_masks, last_pred):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, bad_mask, value, reward = transition
                    mask = 1.0 - done
                    delta = reward + config["GAMMA"] * next_value * mask - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * mask * gae
                    if config.get("use_proper_time_limits", True):
                        gae = bad_mask * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_pred), last_pred),
                    (dones, bad_masks, preds, rewards),
                    reverse=True,
                    unroll=16,
                )
                return advantages + preds

            # Compute GAE at env-level (T, NUM_ENVS) with denormalized raw-scale
            # values when ValueNorm is enabled.
            reward_env = actor_to_env_agent_time(traj_batch.reward)[..., 0]  # (T, NUM_ENVS)
            done_env = actor_to_env_agent_time(traj_batch.global_done)[..., 0]  # (T, NUM_ENVS)
            bad_mask_env = actor_to_env_agent_time(traj_batch.bad_mask)[..., 0]  # (T, NUM_ENVS)

            value_preds_for_gae = _denorm_if_needed("v", traj_batch.value_env)
            q_preds_for_gae = _denorm_if_needed("q", traj_batch.q_value_env)
            eq_preds_for_gae = _denorm_if_needed("eq", traj_batch.eq_value_env)

            last_values_for_gae = _denorm_if_needed("v", last_values.squeeze(-1))
            last_q_values_for_gae = _denorm_if_needed("q", last_q_values.squeeze(-1))
            last_eq_values_for_gae = _denorm_if_needed("eq", last_eq_values.squeeze(-1))

            value_targets = _calculate_gae(
                value_preds_for_gae,
                reward_env,
                done_env,
                bad_mask_env,
                last_values_for_gae,
            )  # (T, NUM_ENVS) - env-level value returns
            q_targets = _calculate_gae(
                q_preds_for_gae,
                reward_env,
                done_env,
                bad_mask_env,
                last_q_values_for_gae,
            )  # (T, NUM_ENVS)
            eq_targets = _calculate_gae(
                eq_preds_for_gae,
                reward_env,
                done_env,
                bad_mask_env,
                last_eq_values_for_gae,
            )  # (T, NUM_ENVS)

            # Broadcast eq returns to (T, NUM_ENVS, num_agents) for advantage
            # computation. Each agent in an env sees the same env-level return.
            eq_returns_env = jnp.broadcast_to(
                eq_targets[..., None],
                (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents),
            )  # (T, NUM_ENVS, num_agents)

            # Denormalize eq predictions for the baseline (matching MACA).
            # VQ and VQ_COMA stay raw; ValueNorm is not used for vq pred.
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

            # --- Trajectory / GAE Debugging ---
            _check_finite(value_targets, "value_targets", update_steps)
            _check_finite(q_targets, "q_targets", update_steps)
            _check_finite(eq_targets, "eq_targets", update_steps)
            _check_finite(advantages, "advantages", update_steps)
            _check_finite(norm_advantages, "norm_advantages", update_steps)

            debug_gae = update_steps < 2
            _debug_print(
                debug_gae,
                "[GAE] upd={upd} | "
                "val_targ(min/max/mean)={vtmin:.3f}/{vtmax:.3f}/{vtmean:.3f} | "
                "q_targ(min/max/mean)={qtmin:.3f}/{qtmax:.3f}/{qtmean:.3f} | "
                "eq_targ(min/max/mean)={etmin:.3f}/{etmax:.3f}/{etmean:.3f} | "
                "adv(min/max/mean/std)={amin:.3f}/{amax:.3f}/{amean:.3f}/{astd:.3f} | "
                "norm_adv(min/max)={namin:.3f}/{namax:.3f} | "
                "active_count={ac}",
                upd=update_steps,
                vtmin=jnp.min(value_targets),
                vtmax=jnp.max(value_targets),
                vtmean=jnp.mean(value_targets),
                qtmin=jnp.min(q_targets),
                qtmax=jnp.max(q_targets),
                qtmean=jnp.mean(q_targets),
                etmin=jnp.min(eq_targets),
                etmax=jnp.max(eq_targets),
                etmean=jnp.mean(eq_targets),
                amin=jnp.min(advantages),
                amax=jnp.max(advantages),
                amean=jnp.mean(advantages),
                astd=jnp.std(advantages),
                namin=jnp.min(norm_advantages),
                namax=jnp.max(norm_advantages),
                ac=jnp.sum(active_masks),
            )
            # --- End Trajectory / GAE Debugging ---

            # === Prepare minibatch data ===
            use_recurrent = config.get("use_recurrent_policy", False)
            data_chunk_length = config.get("DATA_CHUNK_LENGTH", config["NUM_STEPS"])
            chunks_per_rollout = config["NUM_STEPS"] // data_chunk_length
            actor_num_mini_batch = config["ACTOR_NUM_MINI_BATCH"]
            critic_num_mini_batch = config["CRITIC_NUM_MINI_BATCH"]

            def _actor_chunks(x):
                x = x.swapaxes(0, 1)  # (NUM_ACTORS, T, ...)
                return x.reshape(
                    (config["NUM_ACTORS"] * chunks_per_rollout, data_chunk_length)
                    + x.shape[2:]
                )

            def _critic_chunks(x):
                x = x.swapaxes(0, 1)  # (NUM_ENVS, T, ...)
                return x.reshape(
                    (config["NUM_ENVS"] * chunks_per_rollout, data_chunk_length)
                    + x.shape[2:]
                )

            if use_recurrent:
                actor_sample_count = config["NUM_ACTORS"] * chunks_per_rollout
                critic_sample_count = config["NUM_ENVS"] * chunks_per_rollout
                actor_mini_batch_size = actor_sample_count // actor_num_mini_batch
                critic_mini_batch_size = critic_sample_count // critic_num_mini_batch

                actor_obs = _actor_chunks(traj_batch.obs)
                actor_done = _actor_chunks(traj_batch.done)
                actor_avail = _actor_chunks(traj_batch.avail_actions)
                actor_action = _actor_chunks(traj_batch.action)
                actor_log_prob = _actor_chunks(traj_batch.log_prob)
                actor_active_mask = _actor_chunks(
                    traj_batch.active_mask.astype(jnp.float32)
                )
                actor_norm_adv = _actor_chunks(norm_advantages)
                actor_init_hstate = _actor_chunks(traj_batch.actor_hstate)[:, 0]

                critic_obs_all = _critic_chunks(traj_batch.obs_all)
                critic_actions_all = _critic_chunks(traj_batch.actions_all)
                critic_policy_probs_all = _critic_chunks(traj_batch.policy_probs_all)
                critic_done = _critic_chunks(actor_to_env_agent_time(traj_batch.done))
                critic_init_hstate = _critic_chunks(traj_batch.critic_hstate)[:, 0]
                critic_value_targets = _critic_chunks(value_targets)
                critic_q_targets = _critic_chunks(q_targets)
                critic_eq_targets = _critic_chunks(eq_targets)
                critic_value_old = _critic_chunks(traj_batch.value_env)
                critic_q_old = _critic_chunks(traj_batch.q_value_env)
                critic_eq_old = _critic_chunks(traj_batch.eq_value_env)
            else:
                actor_sample_count = config["NUM_STEPS"] * config["NUM_ACTORS"]
                critic_sample_count = config["NUM_STEPS"] * config["NUM_ENVS"]
                actor_mini_batch_size = actor_sample_count // actor_num_mini_batch
                critic_mini_batch_size = critic_sample_count // critic_num_mini_batch

                actor_obs = traj_batch.obs.reshape(actor_sample_count, obs_dim)
                actor_done = traj_batch.done.reshape(actor_sample_count)
                actor_avail = traj_batch.avail_actions.reshape(
                    actor_sample_count, action_dim
                )
                actor_action = traj_batch.action.reshape(actor_sample_count)
                actor_log_prob = traj_batch.log_prob.reshape(actor_sample_count)
                actor_active_mask = traj_batch.active_mask.reshape(
                    actor_sample_count
                ).astype(jnp.float32)
                actor_norm_adv = norm_advantages.reshape(actor_sample_count)

                critic_obs_all = traj_batch.obs_all.reshape(
                    critic_sample_count, env.num_agents, obs_dim
                )
                critic_actions_all = traj_batch.actions_all.reshape(
                    critic_sample_count, env.num_agents
                )
                critic_policy_probs_all = traj_batch.policy_probs_all.reshape(
                    critic_sample_count, env.num_agents, action_dim
                )
                critic_done = actor_to_env_agent_time(traj_batch.done).reshape(
                    critic_sample_count, env.num_agents
                )
                critic_value_targets = value_targets.reshape(critic_sample_count)
                critic_q_targets = q_targets.reshape(critic_sample_count)
                critic_eq_targets = eq_targets.reshape(critic_sample_count)
                critic_value_old = traj_batch.value_env.reshape(critic_sample_count)
                critic_q_old = traj_batch.q_value_env.reshape(critic_sample_count)
                critic_eq_old = traj_batch.eq_value_env.reshape(critic_sample_count)
            
            # === Minibatch update functions ===
            def _actor_minibatch_update(actor_state, minibatch_idx):
                """Update actor on a single minibatch."""
                actor_train_state = actor_state

                if use_recurrent:
                    # Gather whole sequence chunks: (chunks, L, ...) -> (L, chunks, ...).
                    mb_obs = jnp.take(actor_obs, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_done = jnp.take(actor_done, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_avail = jnp.take(actor_avail, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_action = jnp.take(actor_action, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_log_prob = jnp.take(actor_log_prob, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_active = jnp.take(actor_active_mask, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_adv = jnp.take(actor_norm_adv, minibatch_idx, axis=0).swapaxes(0, 1)
                    ac_init_hstate_mb = jnp.take(actor_init_hstate, minibatch_idx, axis=0)
                else:
                    mb_obs = jnp.take(actor_obs, minibatch_idx, axis=0)
                    mb_done = jnp.take(actor_done, minibatch_idx, axis=0)
                    mb_avail = jnp.take(actor_avail, minibatch_idx, axis=0)
                    mb_action = jnp.take(actor_action, minibatch_idx, axis=0)
                    mb_log_prob = jnp.take(actor_log_prob, minibatch_idx, axis=0)
                    mb_active = jnp.take(actor_active_mask, minibatch_idx, axis=0)
                    mb_adv = jnp.take(actor_norm_adv, minibatch_idx, axis=0)
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

                    def _aggregate_ratio_and_logratio(lp, old_lp):
                        logratio_parts = lp - old_lp
                        ratio_parts = jnp.exp(logratio_parts)
                        if ratio_parts.ndim > mb_adv.ndim:
                            agg = config.get("action_aggregation", "single")
                            if agg == "prod":
                                return (
                                    jnp.prod(ratio_parts, axis=-1),
                                    jnp.sum(logratio_parts, axis=-1),
                                )
                            if agg == "mean":
                                ratio = jnp.mean(ratio_parts, axis=-1)
                                return ratio, jnp.log(jnp.maximum(ratio, 1e-8))
                            raise ValueError(f"Unsupported action_aggregation: {agg}")
                        return ratio_parts, logratio_parts

                    ratio, logratio = _aggregate_ratio_and_logratio(log_prob, mb_log_prob)
                    ratio = ratio.reshape(-1)
                    logratio = logratio.reshape(-1)
                    mb_adv_flat = mb_adv.reshape(-1)
                    mb_active_flat = mb_active.reshape(-1)
                    entropy = pi.entropy().reshape(-1)
                    loss_actor1 = ratio * mb_adv_flat
                    loss_actor2 = (
                        jnp.clip(ratio, 1.0 - config["CLIP_PARAM"], 1.0 + config["CLIP_PARAM"])
                        * mb_adv_flat
                    )
                    mb_active_count = jnp.sum(mb_active_flat) + 1e-8
                    policy_loss = (
                        -jnp.sum(jnp.minimum(loss_actor1, loss_actor2) * mb_active_flat)
                        / mb_active_count
                    )
                    entropy = jnp.sum(entropy * mb_active_flat) / mb_active_count
                    approx_kl = jnp.sum(((ratio - 1) - logratio) * mb_active_flat) / mb_active_count
                    clip_frac = (
                        jnp.sum((jnp.abs(ratio - 1) > config["CLIP_PARAM"]) * mb_active_flat)
                        / mb_active_count
                    )
                    actor_loss = policy_loss - config["ENT_COEF"] * entropy

                    _debug_print(
                        update_steps < 2,
                        "[ACTOR] upd={upd} | loss={loss:.4f} | pl={pl:.4f} | ent={ent:.4f} | "
                        "kl={kl:.6f} | clip={cf:.4f} | ratio(min/max/mean)={rmin:.4f}/{rmax:.4f}/{rmean:.4f}",
                        upd=update_steps,
                        loss=actor_loss,
                        pl=policy_loss,
                        ent=entropy,
                        kl=approx_kl,
                        cf=clip_frac,
                        rmin=jnp.min(ratio),
                        rmax=jnp.max(ratio),
                        rmean=jnp.mean(ratio),
                    )
                    _check_finite(actor_loss, "actor_loss", update_steps)
                    _check_finite(ratio, "actor_ratio", update_steps)

                    return actor_loss, (policy_loss, entropy, approx_kl, clip_frac, mb_active_count)
                
                (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                    _actor_loss_fn, has_aux=True
                )(actor_train_state.params)
                actor_grad_norm = optax.global_norm(actor_grads)
                _check_finite(actor_grad_norm, "actor_grad_norm", update_steps)
                actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                
                actor_grad_var = 0.0
                if config.get("log_actor_grad_var", False):
                    flat_grads = jax.flatten_util.ravel_pytree(actor_grads)[0]
                    actor_grad_var = jnp.mean(jnp.square(flat_grads)) - jnp.square(jnp.mean(flat_grads))
                    actor_grad_var = jnp.linalg.norm(actor_grad_var)

                actor_info = {
                    "actor_loss": actor_aux[0],
                    "entropy": actor_aux[1],
                    "approx_kl": actor_aux[2],
                    "clip_frac": actor_aux[3],
                    "actor_grad_norm": actor_grad_norm,
                    "mb_active_count": actor_aux[4],
                    "actor_grad_var": actor_grad_var,
                }
                return actor_train_state, actor_info
            
            def _critic_minibatch_update(critic_state, minibatch_idx):
                """Update critic on a single minibatch."""
                critic_train_state, value_norm_dict = critic_state

                if use_recurrent:
                    mb_obs_all = jnp.take(critic_obs_all, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_actions_all = jnp.take(critic_actions_all, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_policy_probs_all = jnp.take(
                        critic_policy_probs_all, minibatch_idx, axis=0
                    ).swapaxes(0, 1)
                    mb_done = jnp.take(critic_done, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_value_targets = jnp.take(critic_value_targets, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_q_targets = jnp.take(critic_q_targets, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_eq_targets = jnp.take(critic_eq_targets, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_value_old = jnp.take(critic_value_old, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_q_old = jnp.take(critic_q_old, minibatch_idx, axis=0).swapaxes(0, 1)
                    mb_eq_old = jnp.take(critic_eq_old, minibatch_idx, axis=0).swapaxes(0, 1)
                    cr_init_hstate_mb = jnp.take(critic_init_hstate, minibatch_idx, axis=0)
                else:
                    mb_obs_all = jnp.take(critic_obs_all, minibatch_idx, axis=0)
                    mb_actions_all = jnp.take(critic_actions_all, minibatch_idx, axis=0)
                    mb_policy_probs_all = jnp.take(critic_policy_probs_all, minibatch_idx, axis=0)
                    mb_done = jnp.take(critic_done, minibatch_idx, axis=0)
                    mb_value_targets = jnp.take(critic_value_targets, minibatch_idx, axis=0)
                    mb_q_targets = jnp.take(critic_q_targets, minibatch_idx, axis=0)
                    mb_eq_targets = jnp.take(critic_eq_targets, minibatch_idx, axis=0)
                    mb_value_old = jnp.take(critic_value_old, minibatch_idx, axis=0)
                    mb_q_old = jnp.take(critic_q_old, minibatch_idx, axis=0)
                    mb_eq_old = jnp.take(critic_eq_old, minibatch_idx, axis=0)
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
                    value_pred = values.squeeze(-1).reshape(-1)
                    q_pred = q_values.squeeze(-1).reshape(-1)
                    eq_pred = eq_values.squeeze(-1).reshape(-1)
                    mb_value_targets_flat = mb_value_targets.reshape(-1)
                    mb_q_targets_flat = mb_q_targets.reshape(-1)
                    mb_eq_targets_flat = mb_eq_targets.reshape(-1)
                    mb_value_old_flat = mb_value_old.reshape(-1)
                    mb_q_old_flat = mb_q_old.reshape(-1)
                    mb_eq_old_flat = mb_eq_old.reshape(-1)
                    
                    # Update ValueNorm with raw targets before computing loss
                    if norm_dict is not None and use_valuenorm:
                        mb_value_targets_vn = mb_value_targets_flat[..., None]
                        mb_q_targets_vn = mb_q_targets_flat[..., None]
                        mb_eq_targets_vn = mb_eq_targets_flat[..., None]
                        new_v_norm = value_norm_update(norm_dict["v"], mb_value_targets_vn)
                        new_q_norm = value_norm_update(norm_dict["q"], mb_q_targets_vn)
                        new_eq_norm = value_norm_update(norm_dict["eq"], mb_eq_targets_vn)
                        norm_dict = {
                            "v": new_v_norm,
                            "q": new_q_norm,
                            "eq": new_eq_norm,
                        }
                        v_targets_norm = value_norm_normalize(
                            norm_dict["v"], mb_value_targets_vn
                        ).squeeze(-1)
                        q_targets_norm = value_norm_normalize(
                            norm_dict["q"], mb_q_targets_vn
                        ).squeeze(-1)
                        eq_targets_norm = value_norm_normalize(
                            norm_dict["eq"], mb_eq_targets_vn
                        ).squeeze(-1)
                    else:
                        v_targets_norm = mb_value_targets_flat
                        q_targets_norm = mb_q_targets_flat
                        eq_targets_norm = mb_eq_targets_flat
                    
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
                    
                    value_loss = _value_loss(value_pred, mb_value_old_flat, v_targets_norm)
                    q_value_loss = _value_loss(q_pred, mb_q_old_flat, q_targets_norm)
                    eq_value_loss = _value_loss(eq_pred, mb_eq_old_flat, eq_targets_norm)
                    critic_loss = (
                        config["VALUE_LOSS_COEF"] * value_loss
                        + config["transformer"]["q_value_loss_coef"] * q_value_loss
                        + config["transformer"]["eq_value_loss_coef"] * eq_value_loss
                    )

                    _debug_print(
                        update_steps < 2,
                        "[CRITIC] upd={upd} | loss={loss:.4f} | v={vl:.4f} | q={ql:.4f} | eq={eql:.4f}",
                        upd=update_steps,
                        loss=critic_loss,
                        vl=value_loss,
                        ql=q_value_loss,
                        eql=eq_value_loss,
                    )
                    _check_finite(critic_loss, "critic_loss", update_steps)

                    return critic_loss, (value_loss, q_value_loss, eq_value_loss, norm_dict)
                
                (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(critic_train_state.params, value_norm_dict)
                critic_grad_norm = optax.global_norm(critic_grads)
                _check_finite(critic_grad_norm, "critic_grad_norm", update_steps)
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                
                # Update value_norm_dict from critic loss output
                value_norm_dict = critic_aux[3]
                
                critic_info = {
                    "value_loss": critic_aux[0],
                    "q_value_loss": critic_aux[1],
                    "eq_value_loss": critic_aux[2],
                    "critic_grad_norm": critic_grad_norm,
                    "next_s_pred_loss": 0.0,
                }
                return (critic_train_state, value_norm_dict), critic_info
            
            # === Run minibatch updates with nested scans ===
            def _run_actor_epoch(actor_state, rng_epoch):
                """Run one PPO epoch with minibatches."""
                # Generate random permutation for this epoch
                rng_perm, rng_epoch = jax.random.split(rng_epoch)
                perm = jax.random.permutation(rng_perm, actor_sample_count)
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
                perm = jax.random.permutation(rng_perm, critic_sample_count)
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
                config["PPO_EPOCH"],
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
                config["CRITIC_EPOCH"],
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

            rng, eval_rng = jax.random.split(rng)
            do_eval = config.get("USE_EVAL", False) & (
                update_steps % config.get("EVAL_INTERVAL", 100) == 0
            )
            eval_return = jax.lax.cond(
                do_eval,
                lambda r: _run_eval(r, actor_train_state),
                lambda _: jnp.array(0.0),
                eval_rng,
            )
            loss_info["eval_return"] = eval_return

            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            mask = metric["returned_episode"][:, :, 0]
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count
            ep_len = jnp.sum(metric["returned_episode_lengths"][:, :, 0] * mask) / ep_count
            timeout_rate = (
                jnp.sum(metric["bad_transition"][:, :, 0].astype(jnp.float32) * mask)
                / ep_count
            )
            env_ep_count = jnp.sum(mask, axis=0)
            env_wins = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask, axis=0)
            env_win_rates = env_wins / (env_ep_count + 1e-8)
            win_rate_std = jnp.std(env_win_rates, ddof=1)

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]

            def _print_and_csv(r, w, ws, el, tr, s, u, vl, ent, cf, akl, agn, cgn):
                s_int = int(s)
                if s_int > 0 and s_int % print_interval == 0:
                    msg = (
                        f"Step {s:8d} | Update {u:5d} | Return: {r:10.2f} | "
                        f"Win: {w:5.2f}+-{ws:5.2f} | Len: {el:5.1f} | "
                        f"TO: {tr:5.2f} | VLoss: {vl:8.4f} | "
                        f"Ent: {ent:6.4f} | Clip: {cf:5.3f} | KL: {akl:6.5f} | "
                        f"GradN(A/C): {agn:6.3f}/{cgn:6.3f}"
                    )
                    print(msg)
                    with open(csv_path, "a", newline="") as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow([
                            s_int, int(u), float(r), float(w), float(ws),
                            float(el), float(tr),
                            float(vl), float(ent), float(cf), float(akl),
                            float(agn), float(cgn),
                        ])

            jax.experimental.io_callback(
                _print_and_csv, None,
                returns, win_rate, win_rate_std, ep_len, timeout_rate,
                step_count, update_steps,
                loss_info["value_loss"], loss_info["entropy"],
                loss_info["clip_frac"], loss_info["approx_kl"],
                loss_info["actor_grad_norm"], loss_info["critic_grad_norm"],
            )

            def _checkpoint(step, actor_params, critic_params, r, w, ws):
                s_int = int(step)
                ckpt_dir = os.path.join(run_dir, f"step_{s_int}")
                os.makedirs(ckpt_dir, exist_ok=True)

                ap = jax.device_get(actor_params)
                cp = jax.device_get(critic_params)

                with open(os.path.join(ckpt_dir, f"actor_{s_int}.pkl"), "wb") as fa:
                    pickle.dump(ap, fa, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(ckpt_dir, f"critic_{s_int}.pkl"), "wb") as fc:
                    pickle.dump(cp, fc, protocol=pickle.HIGHEST_PROTOCOL)

                csv_data = np.genfromtxt(
                    csv_path, delimiter=",", names=True, dtype=None,
                )
                steps = csv_data["step"].astype(float)
                win_rates = csv_data["win_rate"].astype(float)
                win_stds = csv_data["win_rate_std"].astype(float)

                plt.figure(figsize=(10, 6))
                plt.plot(steps, win_rates, "b-", linewidth=1.5, label="Win Rate")
                plt.fill_between(
                    steps,
                    win_rates - win_stds,
                    win_rates + win_stds,
                    alpha=0.2, color="b", label=r"$\pm$1 std",
                )
                plt.xlabel("Timesteps")
                plt.ylabel("Win Rate")
                plt.xlim(0, config["TOTAL_TIMESTEPS"])
                plt.title(f"MAPPO-T on {config['MAP_NAME']}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(run_dir, f"win_rate_{s_int}.png")
                plt.savefig(plot_path, dpi=100, bbox_inches="tight")
                plt.close()

                print(f"Checkpoint saved to {ckpt_dir}/")
                print(f"Plot saved to {plot_path}")

            should_save = (step_count > 0) & (step_count % save_interval == 0)
            jax.lax.cond(
                should_save,
                lambda: jax.experimental.io_callback(
                    _checkpoint, None,
                    step_count,
                    actor_train_state.params,
                    critic_train_state.params,
                    returns, win_rate, win_rate_std,
                ),
                lambda: None,
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


def _override_config_from_cli(config):
    """Override config values with command-line arguments."""
    parser = argparse.ArgumentParser(description="MAPPO-T training for SMAX")

    # Environment & training
    parser.add_argument("--map_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)

    # Learning rates & scheduling
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_critic_lr_decay", action=argparse.BooleanOptionalAction, default=None)

    # Epochs & minibatches
    parser.add_argument("--ppo_epoch", type=int, default=None)
    parser.add_argument("--critic_epoch", type=int, default=None)
    parser.add_argument("--actor_num_mini_batch", type=int, default=None)
    parser.add_argument("--critic_num_mini_batch", type=int, default=None)

    # Algorithm hyperparameters
    parser.add_argument("--clip_param", type=float, default=None)
    parser.add_argument("--ent_coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--value_loss_coef", type=float, default=None)

    # Architecture & features
    parser.add_argument("--use_recurrent_policy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--data_chunk_length", type=int, default=None)
    parser.add_argument("--use_valuenorm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local_obs_with_agent_id", action=argparse.BooleanOptionalAction, default=None)

    # Transformer-specific
    parser.add_argument("--transformer_n_embd", type=int, default=None)
    parser.add_argument("--transformer_n_head", type=int, default=None)
    parser.add_argument("--transformer_n_encode_layer", type=int, default=None)
    parser.add_argument("--transformer_q_value_loss_coef", type=float, default=None)
    parser.add_argument("--transformer_eq_value_loss_coef", type=float, default=None)
    parser.add_argument("--transformer_weight_init", type=str, default=None)

    args = parser.parse_args()

    # Top-level overrides
    if args.map_name is not None:
        config["MAP_NAME"] = args.map_name
    if args.num_envs is not None:
        config["NUM_ENVS"] = args.num_envs
    if args.num_steps is not None:
        config["NUM_STEPS"] = args.num_steps
    if args.total_timesteps is not None:
        config["TOTAL_TIMESTEPS"] = args.total_timesteps
    if args.seed is not None:
        config["SEED"] = args.seed
    if args.save_interval is not None:
        config["SAVE_INTERVAL"] = args.save_interval
    if args.max_steps is not None:
        config.setdefault("ENV_KWARGS", {})["max_steps"] = args.max_steps
    if args.lr is not None:
        config["LR"] = args.lr
    if args.critic_lr is not None:
        config["CRITIC_LR"] = args.critic_lr
    if args.anneal_lr is not None:
        config["ANNEAL_LR"] = args.anneal_lr
    if args.use_critic_lr_decay is not None:
        config["USE_CRITIC_LR_DECAY"] = args.use_critic_lr_decay
    if args.ppo_epoch is not None:
        config["PPO_EPOCH"] = args.ppo_epoch
        config["UPDATE_EPOCHS"] = args.ppo_epoch
    if args.critic_epoch is not None:
        config["CRITIC_EPOCH"] = args.critic_epoch
    if args.actor_num_mini_batch is not None:
        config["ACTOR_NUM_MINI_BATCH"] = args.actor_num_mini_batch
        config["NUM_MINIBATCHES"] = args.actor_num_mini_batch
    if args.critic_num_mini_batch is not None:
        config["CRITIC_NUM_MINI_BATCH"] = args.critic_num_mini_batch
    if args.clip_param is not None:
        config["CLIP_PARAM"] = args.clip_param
    if args.ent_coef is not None:
        config["ENT_COEF"] = args.ent_coef
    if args.gamma is not None:
        config["GAMMA"] = args.gamma
    if args.gae_lambda is not None:
        config["GAE_LAMBDA"] = args.gae_lambda
    if args.max_grad_norm is not None:
        config["MAX_GRAD_NORM"] = args.max_grad_norm
    if args.value_loss_coef is not None:
        config["VALUE_LOSS_COEF"] = args.value_loss_coef
    if args.use_recurrent_policy is not None:
        config["use_recurrent_policy"] = args.use_recurrent_policy
    if args.data_chunk_length is not None:
        config["DATA_CHUNK_LENGTH"] = args.data_chunk_length
    if args.use_valuenorm is not None:
        config["use_valuenorm"] = args.use_valuenorm
    if args.local_obs_with_agent_id is not None:
        config["LOCAL_OBS_WITH_AGENT_ID"] = args.local_obs_with_agent_id

    # Transformer overrides
    if args.transformer_n_embd is not None:
        config["transformer"]["n_embd"] = args.transformer_n_embd
    if args.transformer_n_head is not None:
        config["transformer"]["n_head"] = args.transformer_n_head
    if args.transformer_n_encode_layer is not None:
        config["transformer"]["n_encode_layer"] = args.transformer_n_encode_layer
    if args.transformer_q_value_loss_coef is not None:
        config["transformer"]["q_value_loss_coef"] = args.transformer_q_value_loss_coef
    if args.transformer_eq_value_loss_coef is not None:
        config["transformer"]["eq_value_loss_coef"] = args.transformer_eq_value_loss_coef
    if args.transformer_weight_init is not None:
        config["transformer"]["weight_init"] = args.transformer_weight_init

    return config


if __name__ == "__main__":
    config = get_default_mappo_t_config()
    config = _override_config_from_cli(config)

    print(f"Starting MAPPO-T training on {config['MAP_NAME']}...")
    print(f"SMAX env max_steps={config['ENV_KWARGS'].get('max_steps', config['NUM_STEPS'])}")
    print("Comprehensive debugging enabled (verbose for first 2 updates).")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "saved_model")
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
