"""MAPPO-T Role training script for MACA-Role experiments.

Supports all 4 experiments via ``ROLE_EXPERIMENT`` config:
  1: Post-GRU actor heads + shared critic
  2: Post-GRU actor heads + role-specific critic heads
  3: Pre-GRU routes + post-GRU heads + shared critic
  4: Pre-GRU routes + post-GRU heads + role-specific critic heads

Run:
    python smax_ctm/train_mappo_t_role.py --role_experiment 1
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
from flax.traverse_util import flatten_dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.environments.spaces import Box
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper

from mappo_t import (
    ActorTrans,
    RoleActorTrans,
    RoleTransVCritic,
    ScannedRNN,
    TransVCritic,
    get_default_maca_role_config,
)
from mappo_t.role_config import MACA_ROLE_EXPERIMENTS
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
    role_ids_env: jnp.ndarray   # (NUM_ENVS, num_agents)
    role_ids_actor: jnp.ndarray  # (NUM_ACTORS,)


# ---------------------------------------------------------------------------
# Environment wrapper with role IDs
# ---------------------------------------------------------------------------

class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides MACA-style observations and extracts role IDs from unit_types."""

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
        step_key, reset_key = jax.random.split(key)
        obs_st, state_st, reward, done, info = self._env.step_env(step_key, state, action)

        raw_step_state = state_st.state
        battle_done, won_battle = self._battle_terminal(raw_step_state)
        timeout_done = raw_step_state.time >= self._env.max_steps
        bad_transition = done["__all__"] & timeout_done & ~battle_done
        info["bad_transition"] = jnp.full((self._env.num_allies,), bad_transition)
        info["battle_won"] = jnp.full((self._env.num_allies,), won_battle)

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

    @partial(jax.jit, static_argnums=0)
    def get_role_ids(self, env_state):
        """Extract role IDs from unit_types (allies only)."""
        return env_state.state.unit_types[:, :self._env.num_allies].astype(jnp.int32)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def make_train(config):
    """Create a JIT-able MAPPO-T Role training function."""

    config["ENV_KWARGS"] = dict(config.get("ENV_KWARGS", {}))
    config["ENV_KWARGS"].setdefault("max_steps", config["NUM_STEPS"])

    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])

    # Compute N_ROLES from scenario (ally unit types only)
    if len(scenario.unit_type_indices) > 0:
        # SMACv2 weighted distribution scenario
        n_roles = int(len(scenario.unit_type_indices))
    else:
        # Fixed unit type scenario - count unique ally types
        import numpy as np
        ally_types = np.array(scenario.unit_types[:scenario.num_allies])
        n_roles = int(len(np.unique(ally_types)))
    config["N_ROLES"] = n_roles

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

    role_experiment = config.get("ROLE_EXPERIMENT", 1)
    exp_cfg = MACA_ROLE_EXPERIMENTS[role_experiment]
    use_role_critic = exp_cfg["use_role_critic"]
    use_pre_gru_routes = exp_cfg["use_pre_gru_routes"]
    n_roles = config["N_ROLES"]

    # Auto-enable critic diversity for role-specific critics unless explicitly disabled
    if use_role_critic and "USE_CRITIC_DIVERSITY" not in config:
        config["USE_CRITIC_DIVERSITY"] = True

    # KL diversity schedule config
    kl_initial_weight = config.get("KL_DIVERSITY_WEIGHT", 0.001)
    kl_decay_fraction = config.get("KL_DECAY_FRACTION", 0.3)
    use_kl_diversity = config.get("USE_KL_DIVERSITY", True)
    use_critic_diversity = config.get("USE_CRITIC_DIVERSITY", False) and use_role_critic
    critic_diversity_coef = config.get("CRITIC_DIVERSITY_COEF", 1e-4)

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
        def schedule(count):
            update_num = count // steps_per_update
            frac = 1.0 - update_num / config["NUM_UPDATES"]
            return base_lr * jnp.maximum(frac, 0.0)
        return schedule

    def critic_cosine_schedule():
        base_lr = config["CRITIC_LR"]
        min_lr = config["transformer"].get("min_lr", 0.1 * base_lr)
        warmup_epochs = config["transformer"].get("warmup_epochs", 10)
        critic_steps_per_update = config["CRITIC_EPOCH"] * config["CRITIC_NUM_MINI_BATCH"]

        def schedule(count):
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

    def kl_schedule_fn(update_step):
        """Cosine decay from initial_weight to 0 over first decay_fraction of updates."""
        decay_end = int(config["NUM_UPDATES"] * kl_decay_fraction)
        t = jnp.minimum(update_step, decay_end)
        decay = 0.5 * (1.0 + jnp.cos(jnp.pi * t / jnp.maximum(decay_end, 1)))
        return kl_initial_weight * decay

    def train(rng):
        from mappo_t.role_config import validate_maca_role_config
        validate_maca_role_config(config, env.num_agents)

        if config["transformer"].get("dropout", 0.0) != 0.0:
            raise NotImplementedError("Transformer dropout > 0 not implemented.")
        if not config.get("share_param", True):
            raise NotImplementedError("Shared-parameter actor only.")

        # Networks
        actor_network = RoleActorTrans(
            action_dim=action_dim,
            config=config,
            use_pre_gru_routes=use_pre_gru_routes,
            n_roles=n_roles,
        )

        if use_role_critic:
            critic_network = RoleTransVCritic(
                config=config,
                share_obs_space=None,
                obs_space=env.observation_space(env.agents[0]),
                act_space=env.action_space(env.agents[0]),
                num_agents=env.num_agents,
                state_type="EP",
                n_roles=n_roles,
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

        # Init actor
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        ac_init_hstate_small = ScannedRNN.initialize_carry(config["NUM_ENVS"], actor_hidden_dim)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim), dtype=jnp.float32),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
            jnp.ones((1, config["NUM_ENVS"], action_dim), dtype=jnp.float32),
        )
        dummy_role_ids_actor = jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32)
        actor_params = actor_network.init(actor_rng, ac_init_hstate_small, ac_init_x, dummy_role_ids_actor)

        # Init critic
        cr_init_hstate_small = jnp.zeros(
            (config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32
        )
        critic_init_args = [
            critic_rng,
            jnp.zeros((config["NUM_ENVS"], env.num_agents, obs_dim), dtype=jnp.float32),
            jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.int32),
            jnp.ones((config["NUM_ENVS"], env.num_agents, action_dim), dtype=jnp.float32) / action_dim,
            cr_init_hstate_small,
            jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
            True,
            True,
        ]
        if use_role_critic:
            # Insert role_ids before output_attentions
            critic_init_args.insert(6, jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.int32))
        critic_params = critic_network.init(*critic_init_args)

        if config["transformer"].get("weight_init") == "tfixup":
            from mappo_t.transformer import apply_tfixup_scaling
            critic_params = apply_tfixup_scaling(critic_params, config["transformer"])

        # Optimizers
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

        # ValueNorm
        use_valuenorm = config.get("use_valuenorm", True)
        value_norm_dict = create_value_norm_dict(
            use_valuenorm=use_valuenorm,
            v_shape=(1,),
            q_shape=(1,),
            eq_shape=(1,),
        )

        # Env reset
        rng, reset_rng = jax.random.split(rng)
        reset_rng = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], actor_hidden_dim)
        cr_init_hstate = jnp.zeros(
            (config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32
        )

        # === Eval function (greedy, compiled separately) ===
        def _run_eval(eval_rng, actor_params):
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
                env_s, last_obs, last_done, ac_hstate, rng = carry
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
                # Dummy role IDs for eval (uniform zeros)
                eval_role_ids = jnp.zeros((1, eval_num_envs * env.num_agents), dtype=jnp.int32)
                ac_hstate, pi = actor_network.apply(
                    actor_params, ac_hstate, ac_in, eval_role_ids
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
                    env_s,
                    obsv,
                    env_done_batch,
                    ac_hstate,
                    rng,
                ), reward_batch

            _, eval_rewards = jax.lax.scan(
                _eval_env_step,
                (
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

        _run_eval_jit = jax.jit(_run_eval)

        # === Logging setup ===
        save_interval = config.get("SAVE_INTERVAL", 1000000)
        print_interval = max(1, save_interval // 20)

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        saved_models_dir = os.path.join(_REPO_ROOT, "saved_models")
        os.makedirs(saved_models_dir, exist_ok=True)
        run_dir = os.path.join(saved_models_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)

        params_path = os.path.join(run_dir, "run_params.json")
        with open(params_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        csv_path = os.path.join(run_dir, "progress.csv")
        progress_header = [
            "step", "update", "return", "win_rate", "win_rate_std",
            "ep_len", "timeout_rate",
            "value_loss", "entropy", "clip_frac", "approx_kl",
            "kl_div", "q_value_loss", "eq_value_loss", "eval_return",
            "actor_grad_norm", "critic_grad_norm",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(progress_header)

        # === Update step ===
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

                # Role IDs from env state
                role_ids_env = env.get_role_ids(env_state.env_state)  # (NUM_ENVS, num_agents)
                role_ids_actor = env_agent_to_actor(role_ids_env)  # (NUM_ACTORS,)

                ac_in = (
                    obs_batch[None, :],
                    last_env_done[None, :],
                    avail_actions[None, :],
                )
                ac_hstate, pi = actor_network.apply(
                    actor_train_state.params, ac_hstate, ac_in, role_ids_actor[None, :]
                )
                action = pi.sample(seed=action_rng).squeeze(0)
                log_prob = pi.log_prob(action[None, :]).squeeze(0)
                policy_probs = pi.probs.squeeze(0)

                obs_all = jnp.stack([last_obs[a] for a in env.agents], axis=1)
                actions_all = actor_to_env_agent(action)
                policy_probs_all = actor_to_env_agent(policy_probs)
                critic_resets = actor_to_env_agent(last_env_done)

                # Critic forward
                critic_apply_args = [
                    critic_train_state.params,
                    obs_all,
                    actions_all,
                    policy_probs_all,
                    cr_hstate,
                    critic_resets,
                ]
                if use_role_critic:
                    critic_apply_args.append(role_ids_env)
                critic_apply_args.extend([True, True])

                critic_out = critic_network.apply(*critic_apply_args)
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
                ) = critic_out[:10]

                # For role-specific critic, mean-pool across roles for env-level values
                if use_role_critic:
                    values_env = jnp.mean(values, axis=0)    # (batch, 1)
                    q_values_env = jnp.mean(q_values, axis=0)  # (batch, 1)
                    eq_values_env = jnp.mean(eq_values, axis=0)  # (batch, 1)
                else:
                    values_env = values
                    q_values_env = q_values
                    eq_values_env = eq_values

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

                transition = Transition(
                    env_done_batch,
                    last_env_done,
                    active_mask,
                    action,
                    env_value_to_actor(values_env),
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
                    values_env.squeeze(-1),
                    q_values_env.squeeze(-1),
                    eq_values_env.squeeze(-1),
                    vq_values.squeeze(-1) if vq_values is not None else jnp.zeros((config["NUM_ENVS"], env.num_agents)),
                    vq_coma_values.squeeze(-1) if vq_coma_values is not None else jnp.zeros((config["NUM_ENVS"], env.num_agents)),
                    baseline_weights if baseline_weights is not None else jnp.zeros((config["NUM_ENVS"], env.num_agents, 3)),
                    attn_weights if attn_weights is not None else jnp.zeros((config["NUM_ENVS"], env.num_agents, env.num_agents)),
                    bad_mask,
                    actor_hstate_in,
                    critic_hstate_in,
                    role_ids_env,
                    role_ids_actor,
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

            # Bootstrap values
            last_obs_all = jnp.stack([last_obs[a] for a in env.agents], axis=1)
            last_role_ids_env = env.get_role_ids(env_state.env_state)

            last_critic_args = [
                critic_train_state.params,
                last_obs_all,
                traj_batch.actions_all[-1],
                traj_batch.policy_probs_all[-1],
                cr_hstate,
                actor_to_env_agent(last_env_done),
            ]
            if use_role_critic:
                last_critic_args.append(last_role_ids_env)
            last_critic_args.extend([True, True])

            last_critic_out = critic_network.apply(*last_critic_args)
            last_values = last_critic_out[0]
            last_q_values = last_critic_out[1]
            last_eq_values = last_critic_out[2]

            if use_role_critic:
                last_values = jnp.mean(last_values, axis=0)
                last_q_values = jnp.mean(last_q_values, axis=0)
                last_eq_values = jnp.mean(last_eq_values, axis=0)

            def _denorm_if_needed(norm_key, x):
                if use_valuenorm:
                    return value_norm_denormalize(value_norm_dict[norm_key], x[..., None]).squeeze(-1)
                return x

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

            # Compute GAE at env-level
            reward_env = actor_to_env_agent_time(traj_batch.reward)[..., 0]
            done_env = actor_to_env_agent_time(traj_batch.global_done)[..., 0]
            bad_mask_env = actor_to_env_agent_time(traj_batch.bad_mask)[..., 0]

            preds_stack = jnp.stack([
                _denorm_if_needed("v", traj_batch.value_env),
                _denorm_if_needed("q", traj_batch.q_value_env),
                _denorm_if_needed("eq", traj_batch.eq_value_env),
            ], axis=0).swapaxes(0, 1)

            last_preds = jnp.stack([
                _denorm_if_needed("v", last_values.squeeze(-1)),
                _denorm_if_needed("q", last_q_values.squeeze(-1)),
                _denorm_if_needed("eq", last_eq_values.squeeze(-1)),
            ], axis=0)

            targets_stack = _calculate_gae(
                preds_stack, reward_env, done_env, bad_mask_env, last_preds
            )

            value_targets = targets_stack[:, 0]
            q_targets = targets_stack[:, 1]
            eq_targets = targets_stack[:, 2]

            # Baseline computation
            eq_returns_env = jnp.broadcast_to(
                eq_targets[..., None],
                (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents),
            )

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

            # Prepare minibatch data
            use_recurrent = config.get("use_recurrent_policy", False)
            data_chunk_length = config.get("DATA_CHUNK_LENGTH", config["NUM_STEPS"])
            chunks_per_rollout = config["NUM_STEPS"] // data_chunk_length
            actor_num_mini_batch = config["ACTOR_NUM_MINI_BATCH"]
            critic_num_mini_batch = config["CRITIC_NUM_MINI_BATCH"]

            def _actor_chunks(x):
                x = x.swapaxes(0, 1)
                return x.reshape(
                    (config["NUM_ACTORS"] * chunks_per_rollout, data_chunk_length) + x.shape[2:]
                )

            def _critic_chunks(x):
                x = x.swapaxes(0, 1)
                return x.reshape(
                    (config["NUM_ENVS"] * chunks_per_rollout, data_chunk_length) + x.shape[2:]
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
                actor_active_mask = _actor_chunks(traj_batch.active_mask.astype(jnp.float32))
                actor_norm_adv = _actor_chunks(norm_advantages)
                actor_init_hstate = _actor_chunks(traj_batch.actor_hstate)[:, 0]
                actor_role_ids = _actor_chunks(traj_batch.role_ids_actor)[:, 0]

                critic_obs_all = _critic_chunks(traj_batch.obs_all)
                critic_actions_all = _critic_chunks(traj_batch.actions_all)
                critic_policy_probs_all = _critic_chunks(traj_batch.policy_probs_all)
                critic_done = _critic_chunks(actor_to_env_agent_time(traj_batch.done))
                critic_init_hstate = _critic_chunks(traj_batch.critic_hstate)[:, 0]
                critic_role_ids = _critic_chunks(traj_batch.role_ids_env)[:, 0]
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
                actor_avail = traj_batch.avail_actions.reshape(actor_sample_count, action_dim)
                actor_action = traj_batch.action.reshape(actor_sample_count)
                actor_log_prob = traj_batch.log_prob.reshape(actor_sample_count)
                actor_active_mask = traj_batch.active_mask.reshape(actor_sample_count).astype(jnp.float32)
                actor_norm_adv = norm_advantages.reshape(actor_sample_count)
                actor_role_ids = traj_batch.role_ids_actor.reshape(actor_sample_count)

                critic_obs_all = traj_batch.obs_all.reshape(critic_sample_count, env.num_agents, obs_dim)
                critic_actions_all = traj_batch.actions_all.reshape(critic_sample_count, env.num_agents)
                critic_policy_probs_all = traj_batch.policy_probs_all.reshape(critic_sample_count, env.num_agents, action_dim)
                critic_done = actor_to_env_agent_time(traj_batch.done).reshape(critic_sample_count, env.num_agents)
                critic_value_targets = value_targets.reshape(critic_sample_count)
                critic_q_targets = q_targets.reshape(critic_sample_count)
                critic_eq_targets = eq_targets.reshape(critic_sample_count)
                critic_value_old = traj_batch.value_env.reshape(critic_sample_count)
                critic_q_old = traj_batch.q_value_env.reshape(critic_sample_count)
                critic_eq_old = traj_batch.eq_value_env.reshape(critic_sample_count)
                critic_role_ids = traj_batch.role_ids_env.reshape(critic_sample_count, env.num_agents)

            # === Actor minibatch update ===
            @partial(jax.jit, donate_argnums=(0,))
            def _actor_minibatch_update(actor_state, mb_data):
                actor_train_state = actor_state
                (
                    mb_obs,
                    mb_done,
                    mb_avail,
                    mb_action,
                    mb_log_prob,
                    mb_active,
                    mb_adv,
                    ac_init_hstate_mb,
                    mb_role_ids,
                ) = mb_data

                def _actor_loss_fn(actor_params):
                    _, pi = actor_network.apply(
                        actor_params,
                        ac_init_hstate_mb,
                        (mb_obs, mb_done, mb_avail),
                        mb_role_ids,
                    )
                    log_prob = pi.log_prob(mb_action)

                    ratio = jnp.exp(log_prob - mb_log_prob)
                    ratio = ratio.reshape(-1)
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
                    approx_kl = jnp.sum(((ratio - 1) - jnp.log(ratio + 1e-8)) * mb_active_flat) / mb_active_count
                    clip_frac = (
                        jnp.sum((jnp.abs(ratio - 1) > config["CLIP_PARAM"]) * mb_active_flat)
                        / mb_active_count
                    )
                    actor_loss = policy_loss - config["ENT_COEF"] * entropy

                    # KL diversity penalty
                    kl_div = 0.0
                    if use_kl_diversity:
                        kl_div = actor_network.compute_kl_diversity(
                            actor_params,
                            ac_init_hstate_mb,
                            mb_obs,
                            mb_done,
                            mb_avail,
                        )
                        kl_weight = kl_schedule_fn(update_steps)
                        actor_loss = actor_loss + kl_weight * kl_div

                    return actor_loss, (policy_loss, entropy, approx_kl, clip_frac, mb_active_count, kl_div)

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
                    "kl_div": actor_aux[5],
                }
                return actor_train_state, actor_info

            # === Critic minibatch update ===
            @partial(jax.jit, donate_argnums=(0,))
            def _critic_minibatch_update(critic_state, mb_data):
                critic_train_state, value_norm_dict = critic_state
                (
                    mb_obs_all,
                    mb_actions_all,
                    mb_policy_probs_all,
                    mb_done,
                    mb_value_targets,
                    mb_q_targets,
                    mb_eq_targets,
                    mb_value_old,
                    mb_q_old,
                    mb_eq_old,
                    cr_init_hstate_mb,
                    mb_role_ids,
                ) = mb_data

                mb_value_targets_flat = mb_value_targets.reshape(-1)
                mb_q_targets_flat = mb_q_targets.reshape(-1)
                mb_eq_targets_flat = mb_eq_targets.reshape(-1)

                if use_valuenorm and value_norm_dict is not None:
                    new_norm_dict = {
                        "v": value_norm_update(value_norm_dict["v"], mb_value_targets_flat[..., None]),
                        "q": value_norm_update(value_norm_dict["q"], mb_q_targets_flat[..., None]),
                        "eq": value_norm_update(value_norm_dict["eq"], mb_eq_targets_flat[..., None]),
                    }
                else:
                    new_norm_dict = None

                def _critic_loss_fn(critic_params, norm_dict):
                    critic_apply_args = [
                        critic_params,
                        mb_obs_all,
                        mb_actions_all,
                        mb_policy_probs_all,
                        cr_init_hstate_mb,
                        mb_done,
                    ]
                    if use_role_critic:
                        critic_apply_args.append(mb_role_ids)
                    critic_apply_args.extend([True, True])

                    critic_out = critic_network.apply(*critic_apply_args)
                    values = critic_out[0]
                    q_values = critic_out[1]
                    eq_values = critic_out[2]

                    # Handle role-specific critic: tile targets across roles
                    if use_role_critic:
                        n_r = values.shape[0]
                        value_pred = values.squeeze(-1).reshape(-1)            # (n_roles * batch)
                        q_pred = q_values.squeeze(-1).reshape(-1)
                        eq_pred = eq_values.squeeze(-1).reshape(-1)
                        mb_value_old_flat = jnp.tile(mb_value_old.reshape(-1), n_r)
                        mb_q_old_flat = jnp.tile(mb_q_old.reshape(-1), n_r)
                        mb_eq_old_flat = jnp.tile(mb_eq_old.reshape(-1), n_r)
                        v_t_flat = jnp.tile(mb_value_targets_flat, n_r)
                        q_t_flat = jnp.tile(mb_q_targets_flat, n_r)
                        eq_t_flat = jnp.tile(mb_eq_targets_flat, n_r)
                    else:
                        value_pred = values.squeeze(-1).reshape(-1)
                        q_pred = q_values.squeeze(-1).reshape(-1)
                        eq_pred = eq_values.squeeze(-1).reshape(-1)
                        mb_value_old_flat = mb_value_old.reshape(-1)
                        mb_q_old_flat = mb_q_old.reshape(-1)
                        mb_eq_old_flat = mb_eq_old.reshape(-1)
                        v_t_flat = mb_value_targets_flat
                        q_t_flat = mb_q_targets_flat
                        eq_t_flat = mb_eq_targets_flat

                    if norm_dict is not None and use_valuenorm:
                        v_targets_norm = value_norm_normalize(
                            norm_dict["v"], v_t_flat[..., None]
                        ).squeeze(-1)
                        q_targets_norm = value_norm_normalize(
                            norm_dict["q"], q_t_flat[..., None]
                        ).squeeze(-1)
                        eq_targets_norm = value_norm_normalize(
                            norm_dict["eq"], eq_t_flat[..., None]
                        ).squeeze(-1)
                    else:
                        v_targets_norm = v_t_flat
                        q_targets_norm = q_t_flat
                        eq_targets_norm = eq_t_flat

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

                    # Critic diversity penalty (role-specific heads only)
                    if use_critic_diversity:
                        div_penalty = critic_network.compute_diversity_penalty(
                            critic_params,
                            mb_obs_all,
                            mb_actions_all,
                            mb_policy_probs_all,
                            cr_init_hstate_mb,
                            mb_done,
                        )
                        critic_loss = critic_loss + critic_diversity_coef * div_penalty

                    return critic_loss, (value_loss, q_value_loss, eq_value_loss)

                (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(critic_train_state.params, new_norm_dict)
                critic_grad_norm = optax.global_norm(critic_grads)
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                value_norm_dict = new_norm_dict

                critic_info = {
                    "value_loss": critic_aux[0],
                    "q_value_loss": critic_aux[1],
                    "eq_value_loss": critic_aux[2],
                    "critic_grad_norm": critic_grad_norm,
                }
                return (critic_train_state, value_norm_dict), critic_info

            # === Run epochs ===
            def _run_actor_epoch(actor_state, rng_epoch):
                rng_perm, rng_epoch = jax.random.split(rng_epoch)
                perm = jax.random.permutation(rng_perm, actor_sample_count)

                def _shuffle_reshape(x):
                    x = jnp.take(x, perm, axis=0)
                    return x.reshape(actor_num_mini_batch, actor_mini_batch_size, *x.shape[1:])

                shuffled_obs = _shuffle_reshape(actor_obs)
                shuffled_done = _shuffle_reshape(actor_done)
                shuffled_avail = _shuffle_reshape(actor_avail)
                shuffled_action = _shuffle_reshape(actor_action)
                shuffled_log_prob = _shuffle_reshape(actor_log_prob)
                shuffled_active_mask = _shuffle_reshape(actor_active_mask)
                shuffled_norm_adv = _shuffle_reshape(actor_norm_adv)
                shuffled_role_ids = _shuffle_reshape(actor_role_ids)
                if use_recurrent:
                    shuffled_init_hstate = jnp.take(actor_init_hstate, perm, axis=0)
                    shuffled_init_hstate = shuffled_init_hstate.reshape(
                        actor_num_mini_batch, actor_mini_batch_size, *shuffled_init_hstate.shape[1:]
                    )

                def scan_actor_minibatch(astate, i):
                    if use_recurrent:
                        mb_data = (
                            shuffled_obs[i].swapaxes(0, 1),
                            shuffled_done[i].swapaxes(0, 1),
                            shuffled_avail[i].swapaxes(0, 1),
                            shuffled_action[i].swapaxes(0, 1),
                            shuffled_log_prob[i].swapaxes(0, 1),
                            shuffled_active_mask[i].swapaxes(0, 1),
                            shuffled_norm_adv[i].swapaxes(0, 1),
                            shuffled_init_hstate[i],
                            shuffled_role_ids[i].swapaxes(0, 1),
                        )
                    else:
                        mb_data = (
                            shuffled_obs[i],
                            shuffled_done[i],
                            shuffled_avail[i],
                            shuffled_action[i],
                            shuffled_log_prob[i],
                            shuffled_active_mask[i],
                            shuffled_norm_adv[i],
                            ScannedRNN.initialize_carry(actor_mini_batch_size, actor_hidden_dim),
                            shuffled_role_ids[i],
                        )
                    new_astate, info = _actor_minibatch_update(astate, mb_data)
                    return new_astate, info

                final_actor_state, actor_infos = jax.lax.scan(
                    scan_actor_minibatch, actor_state, jnp.arange(actor_num_mini_batch)
                )
                actor_avg_info = jax.tree.map(lambda x: x.mean(), actor_infos)
                return final_actor_state, (rng_epoch, actor_avg_info)

            def _run_critic_epoch(critic_state, rng_epoch):
                critic_train_state, value_norm_dict = critic_state
                rng_perm, rng_epoch = jax.random.split(rng_epoch)
                perm = jax.random.permutation(rng_perm, critic_sample_count)

                def _shuffle_reshape(x):
                    x = jnp.take(x, perm, axis=0)
                    return x.reshape(critic_num_mini_batch, critic_mini_batch_size, *x.shape[1:])

                shuffled_obs_all = _shuffle_reshape(critic_obs_all)
                shuffled_actions_all = _shuffle_reshape(critic_actions_all)
                shuffled_policy_probs_all = _shuffle_reshape(critic_policy_probs_all)
                shuffled_done = _shuffle_reshape(critic_done)
                shuffled_value_targets = _shuffle_reshape(critic_value_targets)
                shuffled_q_targets = _shuffle_reshape(critic_q_targets)
                shuffled_eq_targets = _shuffle_reshape(critic_eq_targets)
                shuffled_value_old = _shuffle_reshape(critic_value_old)
                shuffled_q_old = _shuffle_reshape(critic_q_old)
                shuffled_eq_old = _shuffle_reshape(critic_eq_old)
                shuffled_role_ids = _shuffle_reshape(critic_role_ids)
                if use_recurrent:
                    shuffled_init_hstate = jnp.take(critic_init_hstate, perm, axis=0)
                    shuffled_init_hstate = shuffled_init_hstate.reshape(
                        critic_num_mini_batch, critic_mini_batch_size, *shuffled_init_hstate.shape[1:]
                    )

                def scan_critic_minibatch(cstate, i):
                    if use_recurrent:
                        mb_data = (
                            shuffled_obs_all[i].swapaxes(0, 1),
                            shuffled_actions_all[i].swapaxes(0, 1),
                            shuffled_policy_probs_all[i].swapaxes(0, 1),
                            shuffled_done[i].swapaxes(0, 1),
                            shuffled_value_targets[i].swapaxes(0, 1),
                            shuffled_q_targets[i].swapaxes(0, 1),
                            shuffled_eq_targets[i].swapaxes(0, 1),
                            shuffled_value_old[i].swapaxes(0, 1),
                            shuffled_q_old[i].swapaxes(0, 1),
                            shuffled_eq_old[i].swapaxes(0, 1),
                            shuffled_init_hstate[i],
                            shuffled_role_ids[i].swapaxes(0, 1),
                        )
                    else:
                        mb_data = (
                            shuffled_obs_all[i],
                            shuffled_actions_all[i],
                            shuffled_policy_probs_all[i],
                            shuffled_done[i],
                            shuffled_value_targets[i],
                            shuffled_q_targets[i],
                            shuffled_eq_targets[i],
                            shuffled_value_old[i],
                            shuffled_q_old[i],
                            shuffled_eq_old[i],
                            jnp.zeros(
                                (critic_mini_batch_size, env.num_agents, critic_hidden_dim),
                                dtype=jnp.float32,
                            ),
                            shuffled_role_ids[i],
                        )
                    new_cstate, info = _critic_minibatch_update(cstate, mb_data)
                    return new_cstate, info

                final_critic_state, critic_infos = jax.lax.scan(
                    scan_critic_minibatch, (critic_train_state, value_norm_dict), jnp.arange(critic_num_mini_batch)
                )
                critic_avg_info = jax.tree.map(lambda x: x.mean(), critic_infos)
                return final_critic_state, (rng_epoch, critic_avg_info)

            # Run actor epochs
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

            # Run critic epochs
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

            # Metrics
            metrics = {
                "actor_loss": actor_epoch_infos["actor_loss"].mean(),
                "entropy": actor_epoch_infos["entropy"].mean(),
                "approx_kl": actor_epoch_infos["approx_kl"].mean(),
                "clip_frac": actor_epoch_infos["clip_frac"].mean(),
                "actor_grad_norm": actor_epoch_infos["actor_grad_norm"].mean(),
                "kl_div": actor_epoch_infos["kl_div"].mean(),
                "value_loss": critic_epoch_infos["value_loss"].mean(),
                "q_value_loss": critic_epoch_infos["q_value_loss"].mean(),
                "eq_value_loss": critic_epoch_infos["eq_value_loss"].mean(),
                "critic_grad_norm": critic_epoch_infos["critic_grad_norm"].mean(),
                "returned_episode_returns": traj_batch.info["returned_episode_returns"].mean(),
                "returned_episode_lengths": traj_batch.info["returned_episode_lengths"].mean(),
                "battle_won": traj_batch.info["battle_won"].mean(),
            }

            # Per-env stats for logging
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                traj_batch.info,
            )
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

            # Eval callback (same interval as checkpoint, in timesteps)
            rng, eval_rng = jax.random.split(rng)
            do_eval = config.get("USE_EVAL", False) & (
                (step_count > 0) & (step_count % save_interval == 0)
            )
            eval_return = jax.lax.cond(
                do_eval,
                lambda er: jax.experimental.io_callback(
                    lambda er2, p: np.array(float(_run_eval_jit(er2, p)), dtype=np.float32),
                    jax.ShapeDtypeStruct((), jnp.float32),
                    er,
                    actor_train_state.params,
                    ordered=True,
                ),
                lambda _: jnp.array(0.0),
                eval_rng,
            )
            metrics["eval_return"] = eval_return

            # Print + CSV callback
            def _print_and_csv(
                r, w, ws, el, tr, s, u,
                vl, ent, cf, akl, kld, qvl, eqvl, evr, agn, cgn
            ):
                s_int = int(s)
                if s_int > 0 and s_int % print_interval == 0:
                    msg = (
                        f"Step {s:8d} | Update {u:5d} | Return: {r:10.2f} | "
                        f"Win: {w:5.2f}+-{ws:5.2f} | Len: {el:5.1f} | "
                        f"TO: {tr:5.2f} | VLoss: {vl:8.4f} | "
                        f"Ent: {ent:6.4f} | Clip: {cf:5.3f} | KL: {akl:6.5f} | "
                        f"KLDiv: {kld:6.5f} | QVL: {qvl:8.4f} | EQVL: {eqvl:8.4f} | "
                        f"Eval: {evr:8.4f} | GradN(A/C): {agn:6.3f}/{cgn:6.3f}"
                    )
                    print(msg)
                    with open(csv_path, "a", newline="") as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow([
                            s_int, int(u), float(r), float(w), float(ws),
                            float(el), float(tr),
                            float(vl), float(ent), float(cf), float(akl),
                            float(kld), float(qvl), float(eqvl), float(evr),
                            float(agn), float(cgn),
                        ])

            jax.experimental.io_callback(
                _print_and_csv, None,
                returns, win_rate, win_rate_std, ep_len, timeout_rate,
                step_count, update_steps,
                metrics["value_loss"], metrics["entropy"],
                metrics["clip_frac"], metrics["approx_kl"],
                metrics["kl_div"], metrics["q_value_loss"], metrics["eq_value_loss"],
                eval_return,
                metrics["actor_grad_norm"], metrics["critic_grad_norm"],
            )

            # Checkpoint callback
            def _checkpoint(
                step,
                update,
                actor_params,
                critic_params,
                value_norm_state,
                actor_opt_state,
                critic_opt_state,
                actor_step,
                critic_step,
                r,
                w,
                ws,
            ):
                s_int = int(step)
                ckpt_path = os.path.join(run_dir, f"checkpoint_{s_int}.pkl")

                ap = jax.device_get(actor_params)
                cp = jax.device_get(critic_params)
                vn = jax.device_get(value_norm_state)
                aos = jax.device_get(actor_opt_state)
                cos = jax.device_get(critic_opt_state)
                actor_step_int = int(jax.device_get(actor_step))
                critic_step_int = int(jax.device_get(critic_step))
                update_int = int(jax.device_get(update))

                checkpoint = {
                    "model_type": "maca_role",
                    "format_version": 1,
                    "checkpoint_kind": "periodic",
                    "step": s_int,
                    "update": update_int,
                    "config": config,
                    "actor_params": ap,
                    "critic_params": cp,
                    "value_norm_dict": vn,
                    "actor_opt_state": aos,
                    "critic_opt_state": cos,
                    "actor_step": actor_step_int,
                    "critic_step": critic_step_int,
                    "metrics": {
                        "return": float(r),
                        "win_rate": float(w),
                        "win_rate_std": float(ws),
                    },
                }
                with open(ckpt_path, "wb") as fckpt:
                    pickle.dump(checkpoint, fckpt, protocol=pickle.HIGHEST_PROTOCOL)

                # Win rate plot from CSV
                try:
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
                    plt.title(f"MACA-Role Exp {config['ROLE_EXPERIMENT']} on {config['MAP_NAME']}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plot_path = os.path.join(run_dir, f"win_rate_{s_int}.png")
                    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
                    plt.close()

                    print(f"Checkpoint saved to {ckpt_path}")
                    print(f"Plot saved to {plot_path}")
                except Exception as e:
                    print(f"Checkpoint saved to {ckpt_path} (plot failed: {e})")

            should_save = (step_count > 0) & (step_count % save_interval == 0)
            jax.lax.cond(
                should_save,
                lambda: jax.experimental.io_callback(
                    _checkpoint, None,
                    step_count,
                    update_steps,
                    actor_train_state.params,
                    critic_train_state.params,
                    value_norm_dict,
                    actor_train_state.opt_state,
                    critic_train_state.opt_state,
                    actor_train_state.step,
                    critic_train_state.step,
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
                (ac_init_hstate, cr_init_hstate),
                value_norm_dict,
                rng,
            )
            return (runner_state, update_steps + 1), metrics

        # === Main loop ===
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            value_norm_dict,
            rng,
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )

        # Final checkpoint via io_callback (same run_dir as periodic checkpoints)
        def _final_checkpoint(
            update,
            actor_params,
            critic_params,
            value_norm_state,
            actor_opt_state,
            critic_opt_state,
            actor_step,
            critic_step,
        ):
            update_int = int(jax.device_get(update))
            s_int = update_int * config["NUM_ENVS"] * config["NUM_STEPS"]
            ckpt_path = os.path.join(run_dir, "checkpoint_final.pkl")
            checkpoint = {
                "model_type": "maca_role",
                "format_version": 1,
                "checkpoint_kind": "final",
                "step": s_int,
                "update": update_int,
                "config": config,
                "actor_params": jax.device_get(actor_params),
                "critic_params": jax.device_get(critic_params),
                "value_norm_dict": jax.device_get(value_norm_state),
                "actor_opt_state": jax.device_get(actor_opt_state),
                "critic_opt_state": jax.device_get(critic_opt_state),
                "actor_step": int(jax.device_get(actor_step)),
                "critic_step": int(jax.device_get(critic_step)),
            }
            with open(ckpt_path, "wb") as fckpt:
                pickle.dump(checkpoint, fckpt, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Final checkpoint saved to {ckpt_path}")

        final_runner_state, final_update_steps = runner_state
        final_actor_state, final_critic_state = final_runner_state[0]
        final_value_norm_dict = final_runner_state[6]
        jax.experimental.io_callback(
            _final_checkpoint,
            None,
            final_update_steps,
            final_actor_state.params,
            final_critic_state.params,
            final_value_norm_dict,
            final_actor_state.opt_state,
            final_critic_state.opt_state,
            final_actor_state.step,
            final_critic_state.step,
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
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--map_name", type=str, default="protoss_10_vs_10")
    parser.add_argument("--save_interval", type=int, default=1000000)
    parser.add_argument("--eval_num_envs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--n_roles", type=int, default=6)
    parser.add_argument("--use_kl_diversity", action="store_true", default=True)
    parser.add_argument("--use_critic_diversity", action="store_true", default=True)
    parser.add_argument("--kl_diversity_weight", type=float, default=0.001)
    parser.add_argument("--critic_diversity_coef", type=float, default=1e-4)
    args = parser.parse_args()

    config = get_default_maca_role_config()
    config["ROLE_EXPERIMENT"] = args.role_experiment
    config["SEED"] = args.seed
    config["TOTAL_TIMESTEPS"] = args.total_timesteps
    config["NUM_ENVS"] = args.num_envs
    config["NUM_STEPS"] = args.num_steps
    config["MAP_NAME"] = args.map_name
    config["SAVE_INTERVAL"] = args.save_interval
    config["EVAL_NUM_ENVS"] = args.eval_num_envs
    config["EVAL_STEPS"] = args.eval_steps
    config["USE_EVAL"] = True
    config["N_ROLES"] = args.n_roles
    config["USE_KL_DIVERSITY"] = args.use_kl_diversity
    config["USE_CRITIC_DIVERSITY"] = args.use_critic_diversity
    config["KL_DIVERSITY_WEIGHT"] = args.kl_diversity_weight
    config["CRITIC_DIVERSITY_COEF"] = args.critic_diversity_coef

    print(f"Starting MACA-Role Experiment {args.role_experiment} on {config['MAP_NAME']}...")
    print(f"Config: NUM_ENVS={config['NUM_ENVS']}, NUM_STEPS={config['NUM_STEPS']}, TOTAL_TIMESTEPS={config['TOTAL_TIMESTEPS']}")

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out)
    elapsed = (time.time() - start_time) / 60.0

    metrics = out["metrics"]
    print(f"\nTraining completed in {elapsed:.1f} minutes.")
    print(f"Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {float(v[-1]):.4f}")


if __name__ == "__main__":
    main()
