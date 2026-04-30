"""
MAPPO-VD (Value Decomposition) Baseline for SMAX
Colab-ready, dependency-light version (no Hydra/wandb).

Based on train_mappo_gru.py with value-decomposition critic:
- Individual GRU-based Q-networks per agent (shared parameters)
- QMIX or VDN mixer for joint value estimation
- Joint Q-value used as centralized value for GAE
"""
import csv
import os
import sys
import pickle
import argparse
from datetime import datetime
# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict, Tuple
import functools
from flax.training.train_state import TrainState
import distrax
from functools import partial
import time

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

from mappo_vd.actor import ActorRNN, ScannedRNN
from mappo_vd.critic import VDCriticRNN
from mappo_vd.config import get_default_mappo_vd_config


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides a 'world_state' observation for the centralised critic."""
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
    def ws_just_env_state(self, obs, env_state):
        del env_state
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, env_state):
        del env_state
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):
        return self._world_state_size


@struct.dataclass
class ValueNormState:
    """Simple running statistics for value normalization."""
    running_mean: jnp.ndarray
    running_mean_sq: jnp.ndarray
    debiasing_term: jnp.ndarray
    beta: float = struct.field(pytree_node=False, default=0.99999)
    epsilon: float = struct.field(pytree_node=False, default=1e-5)
    var_clamp_min: float = struct.field(pytree_node=False, default=1e-2)


def init_value_norm(shape: Tuple[int, ...] = (1,)) -> ValueNormState:
    return ValueNormState(
        running_mean=jnp.zeros(shape, dtype=jnp.float32),
        running_mean_sq=jnp.zeros(shape, dtype=jnp.float32),
        debiasing_term=jnp.zeros((), dtype=jnp.float32),
    )


def value_norm_stats(state: ValueNormState):
    debiased_mean = state.running_mean / jnp.maximum(state.debiasing_term, state.epsilon)
    debiased_mean_sq = state.running_mean_sq / jnp.maximum(state.debiasing_term, state.epsilon)
    debiased_var = jnp.maximum(debiased_mean_sq - debiased_mean ** 2, state.var_clamp_min)
    return debiased_mean, debiased_var


def value_norm_normalize(state: ValueNormState, x: jnp.ndarray) -> jnp.ndarray:
    mean, var = value_norm_stats(state)
    return (x - mean) / jnp.sqrt(var)


def value_norm_denormalize(state: ValueNormState, x: jnp.ndarray) -> jnp.ndarray:
    mean, var = value_norm_stats(state)
    return x * jnp.sqrt(var) + mean


def value_norm_update(state: ValueNormState, x: jnp.ndarray) -> ValueNormState:
    axes = tuple(range(x.ndim - len(state.running_mean.shape)))
    batch_mean = jnp.mean(x, axis=axes)
    batch_sq_mean = jnp.mean(x ** 2, axis=axes)
    new_running_mean = state.beta * state.running_mean + (1.0 - state.beta) * batch_mean
    new_running_mean_sq = state.beta * state.running_mean_sq + (1.0 - state.beta) * batch_sq_mean
    new_debiasing_term = state.beta * state.debiasing_term + (1.0 - state.beta) * 1.0
    return ValueNormState(
        running_mean=new_running_mean,
        running_mean_sq=new_running_mean_sq,
        debiasing_term=new_debiasing_term,
        beta=state.beta,
        epsilon=state.epsilon,
        var_clamp_min=state.var_clamp_min,
    )


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    policy_probs: jnp.ndarray
    actor_hstate: jnp.ndarray
    critic_hstate: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_PARAM"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_PARAM"]

    config["ACTOR_NUM_MINI_BATCH"] = config.get("ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1))
    config["CRITIC_NUM_MINI_BATCH"] = config.get("CRITIC_NUM_MINI_BATCH", 1)

    use_recurrent_policy = config.get("use_recurrent_policy", True)
    if not use_recurrent_policy and config["NUM_MINIBATCHES"] != 1:
        raise NotImplementedError(
            "MAPPO-VD currently requires NUM_MINIBATCHES=1 (and CRITIC_NUM_MINI_BATCH=1) "
            "when use_recurrent_policy=False, because the QMIX/VDN mixer needs complete environment groups."
        )

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    use_valuenorm = config.get("use_valuenorm", False)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # === Logging setup ===
    save_interval = config.get("SAVE_INTERVAL", 1000000)
    print_interval = max(1, save_interval // 20)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(_REPO_ROOT, "saved_models", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "progress.csv")
    progress_header = [
        "step", "update", "return", "win_rate", "win_rate_std",
        "ep_len",
        "value_loss", "entropy", "clip_frac", "approx_kl",
        "actor_grad_norm", "critic_grad_norm",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(progress_header)

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = VDCriticRNN(
            action_dim=env.action_space(env.agents[0]).n,
            num_agents=env.num_agents,
            config=config,
        )
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ACTORS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ACTORS"])),
            jnp.zeros((1, config["NUM_ACTORS"]), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ACTORS"], env.action_space(env.agents[0]).n)),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["CRITIC_LR"], eps=1e-5),
            )

        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply, params=actor_network_params, tx=actor_tx
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply, params=critic_network_params, tx=critic_tx
        )

        # INIT ValueNorm
        value_norm_state = init_value_norm((1,)) if use_valuenorm else None

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, value_norm_state, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)

                ac_hstate_in = hstates[0]
                cr_hstate_in = hstates[1]

                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                policy_probs = pi.probs

                env_act = unbatchify(action.squeeze(), env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE (VD Critic)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
                cr_in = (
                    obs_batch[np.newaxis, :],
                    world_state[np.newaxis, :],
                    last_done[np.newaxis, :],
                    action,
                    policy_probs,
                )
                cr_hstate, (joint_q_value, ind_q_taken, ind_v_values) = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )
                joint_q_value = joint_q_value.squeeze()

                # Optional value normalization
                if use_valuenorm:
                    value_to_store = value_norm_normalize(value_norm_state, joint_q_value[..., None]).squeeze(-1)
                else:
                    value_to_store = joint_q_value

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value_to_store,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                    policy_probs.squeeze(),
                    ac_hstate_in,
                    cr_hstate_in,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), value_norm_state, rng)
                return runner_state, transition

            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            train_states, env_state, last_obs, last_done, hstates, value_norm_state, rng = runner_state

            # Compute last value for GAE
            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

            # We need dummy actions and policy_probs for the last state critic eval
            # In practice, for QMIX/VDN the mixer only needs the taken-action Q and state.
            # For bootstrapping we can use a zero-action Q or sample from the current policy.
            rng, _rng = jax.random.split(rng)
            last_avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_actions = jax.lax.stop_gradient(
                batchify(last_avail_actions, env.agents, config["NUM_ACTORS"])
            )
            last_ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], last_avail_actions)
            _, last_pi = actor_network.apply(train_states[0].params, hstates[0], last_ac_in)
            last_action = last_pi.sample(seed=_rng)
            last_policy_probs = last_pi.probs

            last_cr_in = (
                last_obs_batch[np.newaxis, :],
                last_world_state[np.newaxis, :],
                last_done[np.newaxis, :],
                last_action,
                last_policy_probs,
            )
            _, (last_joint_q_value, _, _) = critic_network.apply(
                train_states[1].params, hstates[1], last_cr_in
            )
            last_val = last_joint_q_value.squeeze()

            if use_valuenorm:
                last_val = value_norm_denormalize(value_norm_state, last_val[..., None]).squeeze(-1)
                traj_values = value_norm_denormalize(value_norm_state, traj_batch.value[..., None]).squeeze(-1)
            else:
                traj_values = traj_batch.value

            def _calculate_gae(traj_values, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.global_done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch._replace(value=traj_values),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_values

            advantages, targets = _calculate_gae(traj_values, last_val)

            # Update ValueNorm with returns if enabled
            if use_valuenorm and value_norm_state is not None:
                value_norm_state = value_norm_update(value_norm_state, targets[..., None])

            use_recurrent = config.get("use_recurrent_policy", True)
            data_chunk_length = config.get("DATA_CHUNK_LENGTH", config["NUM_STEPS"])
            chunks_per_rollout = config["NUM_STEPS"] // data_chunk_length
            actor_num_mini_batch = config["ACTOR_NUM_MINI_BATCH"]
            critic_num_mini_batch = config["CRITIC_NUM_MINI_BATCH"]

            def _actor_chunks(x):
                x = x.swapaxes(0, 1)
                return x.reshape((config["NUM_ACTORS"] * chunks_per_rollout, data_chunk_length) + x.shape[2:])

            def _critic_chunks(x):
                x = x.swapaxes(0, 1)
                return x.reshape((config["NUM_ACTORS"] * chunks_per_rollout, data_chunk_length) + x.shape[2:])

            norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            if use_recurrent:
                actor_sample_count = config["NUM_ACTORS"] * chunks_per_rollout
                critic_sample_count = config["NUM_ACTORS"] * chunks_per_rollout
                actor_mini_batch_size = actor_sample_count // actor_num_mini_batch
                critic_mini_batch_size = critic_sample_count // critic_num_mini_batch

                actor_obs = _actor_chunks(traj_batch.obs)
                actor_done = _actor_chunks(traj_batch.done)
                actor_avail = _actor_chunks(traj_batch.avail_actions)
                actor_action = _actor_chunks(traj_batch.action)
                actor_log_prob = _actor_chunks(traj_batch.log_prob)
                actor_norm_adv = _actor_chunks(norm_advantages)
                actor_init_hstate = _actor_chunks(traj_batch.actor_hstate)[:, 0]

                critic_obs = _critic_chunks(traj_batch.obs)
                critic_world_state = _critic_chunks(traj_batch.world_state)
                critic_done = _critic_chunks(traj_batch.done)
                critic_action = _critic_chunks(traj_batch.action)
                critic_policy_probs = _critic_chunks(traj_batch.policy_probs)
                critic_targets = _critic_chunks(targets)
                critic_value_old = _critic_chunks(traj_batch.value)
                critic_init_hstate = _critic_chunks(traj_batch.critic_hstate)[:, 0]

            def _actor_update_epoch(update_state, unused):
                def _actor_update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states

                    if use_recurrent:
                        mb_obs, mb_done, mb_avail, mb_action, mb_log_prob, mb_adv, ac_init_hstate_mb = batch_info
                    else:
                        ac_init_hstate, cr_init_hstate, traj_batch_mb, advantages_mb, targets_mb = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_obs, traj_done, traj_avail, traj_action, traj_log_prob, gae):
                        _, pi = actor_network.apply(
                            actor_params, init_hstate, (traj_obs, traj_done, traj_avail)
                        )
                        log_prob = pi.log_prob(traj_action)
                        if use_recurrent:
                            logratio = (log_prob - traj_log_prob).reshape(-1)
                            ratio = jnp.exp(logratio)
                            gae_flat = gae.reshape(-1)
                            loss_actor1 = ratio * gae_flat
                            loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_flat
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().reshape(-1).mean()
                        else:
                            logratio = log_prob - traj_log_prob
                            ratio = jnp.exp(logratio)
                            gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae_norm
                            loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    if use_recurrent:
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate_mb, mb_obs, mb_done, mb_avail, mb_action, mb_log_prob, mb_adv)
                    else:
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch_mb.obs, traj_batch_mb.done, traj_batch_mb.avail_actions, traj_batch_mb.action, traj_batch_mb.log_prob, advantages_mb)

                    actor_grad_norm = optax.global_norm(actor_grads)

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)

                    actor_loss_info = {
                        "actor_loss": actor_loss[0],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "actor_grad_norm": actor_grad_norm,
                    }
                    return (actor_train_state, critic_train_state), actor_loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                if use_recurrent:
                    perm = jax.random.permutation(_rng, actor_sample_count)
                    def _shuffle_reshape(x):
                        x = jnp.take(x, perm, axis=0)
                        return x.reshape(actor_num_mini_batch, actor_mini_batch_size, *x.shape[1:])

                    shuffled_obs = _shuffle_reshape(actor_obs)
                    shuffled_done = _shuffle_reshape(actor_done)
                    shuffled_avail = _shuffle_reshape(actor_avail)
                    shuffled_action = _shuffle_reshape(actor_action)
                    shuffled_log_prob = _shuffle_reshape(actor_log_prob)
                    shuffled_norm_adv = _shuffle_reshape(actor_norm_adv)
                    shuffled_init_hstate = jnp.take(actor_init_hstate, perm, axis=0)
                    shuffled_init_hstate = shuffled_init_hstate.reshape(actor_num_mini_batch, actor_mini_batch_size, *shuffled_init_hstate.shape[1:])

                    def scan_actor_minibatch(train_states, i):
                        mb_data = (
                            shuffled_obs[i].swapaxes(0, 1),
                            shuffled_done[i].swapaxes(0, 1),
                            shuffled_avail[i].swapaxes(0, 1),
                            shuffled_action[i].swapaxes(0, 1),
                            shuffled_log_prob[i].swapaxes(0, 1),
                            shuffled_norm_adv[i].swapaxes(0, 1),
                            shuffled_init_hstate[i],
                        )
                        train_states, info = _actor_update_minbatch(train_states, mb_data)
                        return train_states, info

                    train_states, actor_loss_info = jax.lax.scan(scan_actor_minibatch, train_states, jnp.arange(actor_num_mini_batch))
                else:
                    init_hstates_reshaped = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)
                    batch = (init_hstates_reshaped[0], init_hstates_reshaped[1], traj_batch, advantages.squeeze(), targets.squeeze())
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                    shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                            1, 0
                        ),
                        shuffled_batch,
                    )
                    train_states, actor_loss_info = jax.lax.scan(_actor_update_minbatch, train_states, minibatches)
                    init_hstates = jax.tree.map(lambda x: x.squeeze(), init_hstates_reshaped)

                update_state = (train_states, init_hstates, traj_batch, advantages, targets, rng)
                return update_state, actor_loss_info

            def _critic_update_epoch(update_state, unused):
                def _critic_update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states

                    if use_recurrent:
                        mb_obs, mb_world_state, mb_done, mb_action, mb_policy_probs, mb_targets, mb_value_old, cr_init_hstate_mb = batch_info
                    else:
                        ac_init_hstate, cr_init_hstate, traj_batch_mb, advantages_mb, targets_mb = batch_info

                    def _critic_loss_fn(critic_params, init_hstate, cr_in_obs, cr_in_world_state, cr_in_done, cr_in_action, cr_in_policy_probs, targets, value_old):
                        cr_in = (cr_in_obs, cr_in_world_state, cr_in_done, cr_in_action, cr_in_policy_probs)
                        _, (joint_q_value, _, _) = critic_network.apply(
                            critic_params, init_hstate, cr_in
                        )
                        joint_q_value = joint_q_value.squeeze()

                        if use_valuenorm and value_norm_state is not None:
                            targets_norm = value_norm_normalize(value_norm_state, targets[..., None]).squeeze(-1)
                        else:
                            targets_norm = targets

                        if config.get("use_clipped_value_loss", True):
                            value_pred_clipped = value_old + (
                                joint_q_value - value_old
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            if config.get("use_huber_loss", False):
                                delta = config.get("huber_delta", 10.0)
                                def huber(err):
                                    abs_err = jnp.abs(err)
                                    return jnp.where(abs_err <= delta, 0.5 * abs_err ** 2, delta * (abs_err - 0.5 * delta))
                                value_losses = huber(joint_q_value - targets_norm)
                                value_losses_clipped = huber(value_pred_clipped - targets_norm)
                            else:
                                value_losses = jnp.square(joint_q_value - targets_norm)
                                value_losses_clipped = jnp.square(value_pred_clipped - targets_norm)
                            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
                        else:
                            if config.get("use_huber_loss", False):
                                delta = config.get("huber_delta", 10.0)
                                def huber(err):
                                    abs_err = jnp.abs(err)
                                    return jnp.where(abs_err <= delta, 0.5 * abs_err ** 2, delta * (abs_err - 0.5 * delta))
                                value_loss = huber(joint_q_value - targets_norm).mean()
                            else:
                                value_loss = jnp.square(joint_q_value - targets_norm).mean()

                        critic_loss = config["VALUE_LOSS_COEF"] * value_loss
                        return critic_loss, (value_loss,)

                    if use_recurrent:
                        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                        critic_loss, critic_grads = critic_grad_fn(
                            critic_train_state.params, cr_init_hstate_mb,
                            mb_obs, mb_world_state, mb_done, mb_action, mb_policy_probs,
                            mb_targets, mb_value_old
                        )
                    else:
                        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                        critic_loss, critic_grads = critic_grad_fn(
                            critic_train_state.params, cr_init_hstate.squeeze(),
                            traj_batch_mb.obs, traj_batch_mb.world_state, traj_batch_mb.done,
                            traj_batch_mb.action, traj_batch_mb.policy_probs,
                            targets_mb, traj_batch_mb.value
                        )

                    critic_grad_norm = optax.global_norm(critic_grads)

                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    critic_loss_info = {
                        "value_loss": critic_loss[1][0],
                        "critic_grad_norm": critic_grad_norm,
                    }
                    return (actor_train_state, critic_train_state), critic_loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                if use_recurrent:
                    perm = jax.random.permutation(_rng, critic_sample_count)
                    def _shuffle_reshape(x):
                        x = jnp.take(x, perm, axis=0)
                        return x.reshape(critic_num_mini_batch, critic_mini_batch_size, *x.shape[1:])

                    shuffled_obs = _shuffle_reshape(critic_obs)
                    shuffled_world_state = _shuffle_reshape(critic_world_state)
                    shuffled_done = _shuffle_reshape(critic_done)
                    shuffled_action = _shuffle_reshape(critic_action)
                    shuffled_policy_probs = _shuffle_reshape(critic_policy_probs)
                    shuffled_targets = _shuffle_reshape(critic_targets)
                    shuffled_value_old = _shuffle_reshape(critic_value_old)
                    shuffled_init_hstate = jnp.take(critic_init_hstate, perm, axis=0)
                    shuffled_init_hstate = shuffled_init_hstate.reshape(critic_num_mini_batch, critic_mini_batch_size, *shuffled_init_hstate.shape[1:])

                    def scan_critic_minibatch(train_states, i):
                        mb_data = (
                            shuffled_obs[i].swapaxes(0, 1),
                            shuffled_world_state[i].swapaxes(0, 1),
                            shuffled_done[i].swapaxes(0, 1),
                            shuffled_action[i].swapaxes(0, 1),
                            shuffled_policy_probs[i].swapaxes(0, 1),
                            shuffled_targets[i].swapaxes(0, 1),
                            shuffled_value_old[i].swapaxes(0, 1),
                            shuffled_init_hstate[i],
                        )
                        train_states, info = _critic_update_minbatch(train_states, mb_data)
                        return train_states, info

                    train_states, critic_loss_info = jax.lax.scan(scan_critic_minibatch, train_states, jnp.arange(critic_num_mini_batch))
                else:
                    init_hstates = jax.tree.map(
                        lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates
                    )
                    batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                    shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                            1, 0
                        ),
                        shuffled_batch,
                    )

                    train_states, critic_loss_info = jax.lax.scan(_critic_update_minbatch, train_states, minibatches)
                    init_hstates = jax.tree.map(lambda x: x.squeeze(), init_hstates)

                update_state = (train_states, init_hstates, traj_batch, advantages, targets, rng)
                return update_state, critic_loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, actor_loss_info = jax.lax.scan(_actor_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            update_state, critic_loss_info = jax.lax.scan(_critic_update_epoch, update_state, None, config["CRITIC_EPOCH"])
            actor_loss_info = jax.tree.map(lambda x: x.mean(), actor_loss_info)
            critic_loss_info = jax.tree.map(lambda x: x.mean(), critic_loss_info)
            loss_info = {
                "total_loss": actor_loss_info["actor_loss"] + critic_loss_info["value_loss"],
                "actor_loss": actor_loss_info["actor_loss"],
                "value_loss": critic_loss_info["value_loss"],
                "entropy": actor_loss_info["entropy"],
                "approx_kl": actor_loss_info["approx_kl"],
                "clip_frac": actor_loss_info["clip_frac"],
                "actor_grad_norm": actor_loss_info["actor_grad_norm"],
                "critic_grad_norm": critic_loss_info["critic_grad_norm"],
            }

            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info

            mask = metric["returned_episode"][:, :, 0]
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count

            env_win_rates = jnp.sum(
                metric["returned_won_episode"][:, :, :] * mask[..., None], axis=0
            ) / (jnp.sum(mask, axis=0)[..., None] + 1e-8)
            win_rate_std = jnp.std(env_win_rates, ddof=1)
            ep_len = jnp.sum(metric["returned_episode_lengths"][:, :, 0] * mask) / ep_count

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]

            def _print_and_csv(r, w, ws, el, s, u, vl, ent, cf, akl, agn, cgn):
                s_int = int(s)
                if s_int > 0 and s_int % print_interval == 0:
                    msg = (
                        f"Step {s:8d} | Update {u:5d} | Return: {r:10.2f} | "
                        f"Win: {w:5.2f}+-{ws:5.2f} | Len: {el:5.1f} | "
                        f"VLoss: {vl:8.4f} | "
                        f"Ent: {ent:6.4f} | Clip: {cf:5.3f} | KL: {akl:6.5f} | "
                        f"GradN(A/C): {agn:6.3f}/{cgn:6.3f}"
                    )
                    print(msg)
                    with open(csv_path, "a", newline="") as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow([
                            s_int, int(u), float(r), float(w), float(ws),
                            float(el),
                            float(vl), float(ent), float(cf), float(akl),
                            float(agn), float(cgn),
                        ])

            jax.experimental.io_callback(
                _print_and_csv, None,
                returns, win_rate, win_rate_std, ep_len,
                step_count, update_steps,
                loss_info.get("value_loss", 0.0), loss_info.get("entropy", 0.0),
                loss_info.get("clip_frac", 0.0), loss_info.get("approx_kl", 0.0),
                loss_info.get("actor_grad_norm", 0.0), loss_info.get("critic_grad_norm", 0.0),
            )

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, value_norm_state, update_state[-1])
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            value_norm_state,
            _rng,
        )
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train


def _override_config_from_cli(config):
    """Override config values with command-line arguments."""
    parser = argparse.ArgumentParser(description="MAPPO-VD training for SMAX")
    parser.add_argument("--map_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--ppo_epoch", type=int, default=None)
    parser.add_argument("--critic_epoch", type=int, default=None)
    parser.add_argument("--actor_num_mini_batch", type=int, default=None)
    parser.add_argument("--critic_num_mini_batch", type=int, default=None)
    parser.add_argument("--clip_param", type=float, default=None)
    parser.add_argument("--ent_coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--value_loss_coef", type=float, default=None)
    parser.add_argument("--local_obs_with_agent_id", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_recurrent_policy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_valuenorm", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

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
        config["CLIP_EPS"] = args.clip_param
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
        config["VF_COEF"] = args.value_loss_coef
    if args.local_obs_with_agent_id is not None:
        config["LOCAL_OBS_WITH_AGENT_ID"] = args.local_obs_with_agent_id
    if args.use_recurrent_policy is not None:
        config["use_recurrent_policy"] = args.use_recurrent_policy
    if args.use_valuenorm is not None:
        config["use_valuenorm"] = args.use_valuenorm
    return config


if __name__ == "__main__":
    config = get_default_mappo_vd_config()
    config = _override_config_from_cli(config)

    print(f"Starting {config['MAP_NAME']} MAPPO-VD Baseline...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smax_mappo_vd_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "mappo_vd",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved MAPPO-VD actor checkpoint to {model_path}")
