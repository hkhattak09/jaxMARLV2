"""
COMA (Counterfactual Multi-Agent Policy Gradients) for SMAX.
JAX/Flax implementation based on MACA's COMA and train_mappo_gru.py.
"""
import csv
import os
import sys
import pickle
import time
import argparse
from datetime import datetime
from functools import partial
from typing import NamedTuple, Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

from coma import ActorRNN, ComaCriticRNN, get_default_coma_config


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


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    q_value: jnp.ndarray
    v_value: jnp.ndarray
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
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if config.get("SCALE_CLIP_EPS", False):
        config["CLIP_EPS"] = config["CLIP_PARAM"] / env.num_agents
    else:
        config["CLIP_EPS"] = config["CLIP_PARAM"]

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    action_dim = env.action_space(env.agents[0]).n
    world_state_dim = env.world_state_size()

    fc_dim = config.get("FC_DIM_SIZE", config["hidden_sizes"][0])
    gru_dim = config.get("GRU_HIDDEN_DIM", config["hidden_sizes"][-1])
    config["FC_DIM_SIZE"] = fc_dim
    config["GRU_HIDDEN_DIM"] = gru_dim

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(action_dim=action_dim, config=config)
        critic_network = ComaCriticRNN(
            action_dim=action_dim,
            num_agents=env.num_agents,
            config=config,
        )
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], action_dim)),
        )
        ac_init_hstate = ActorRNN.initialize_carry(config["NUM_ENVS"], gru_dim)
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], world_state_dim)),
            jnp.zeros((1, config["NUM_ENVS"], env.num_agents), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ENVS"], env.num_agents, action_dim)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ComaCriticRNN.initialize_carry(config["NUM_ENVS"], gru_dim)
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config.get("ANNEAL_LR", False):
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ActorRNN.initialize_carry(config["NUM_ACTORS"], gru_dim)
        cr_init_hstate = ComaCriticRNN.initialize_carry(config["NUM_ENVS"], gru_dim)

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

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    last_global_done,
                    hstates,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )

                ac_hstate_in = hstates[0]
                cr_hstate_in = hstates[1]
                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                policy_probs = pi.probs.squeeze(0)  # (NUM_ACTORS, action_dim)

                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # CRITIC: per-env inputs
                world_state_env = last_obs["world_state"][
                    :, 0, :
                ]  # (NUM_ENVS, world_state_dim)
                actions_all = action.reshape(config["NUM_ENVS"], env.num_agents)
                policy_probs_all = policy_probs.reshape(
                    config["NUM_ENVS"], env.num_agents, action_dim
                )
                # Critic RNN resets when the env was globally done on the previous step
                cr_in = (
                    world_state_env[None, :],
                    actions_all[None, :],
                    policy_probs_all[None, :],
                    last_global_done[None, :],
                )
                cr_hstate, (q_values, v_values) = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )
                # q_values: (1, NUM_ENVS, num_agents, action_dim)
                # v_values: (1, NUM_ENVS, num_agents)
                q_values_actor = q_values.squeeze(0).reshape(
                    config["NUM_ACTORS"], action_dim
                )
                v_values_actor = v_values.squeeze(0).reshape(config["NUM_ACTORS"])

                q_taken = jnp.take_along_axis(
                    q_values_actor, action.squeeze()[:, None], axis=-1
                ).squeeze(-1)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                global_done_batch = jnp.tile(done["__all__"], env.num_agents)

                transition = Transition(
                    global_done_batch,
                    last_done,
                    action.squeeze(),
                    q_taken.squeeze(),
                    v_values_actor.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    last_obs["world_state"].swapaxes(0, 1).reshape((config["NUM_ACTORS"], -1)),
                    info,
                    avail_actions,
                    policy_probs,
                    ac_hstate_in,
                    cr_hstate_in,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    done["__all__"],  # new global done for next step
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[5]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                train_states,
                env_state,
                last_obs,
                last_done,
                last_global_done,
                hstates,
                rng,
            ) = runner_state

            # LAST CRITIC VALUE for GAE bootstrap
            last_world_state_env = last_obs["world_state"][:, 0, :]
            last_actions_all = traj_batch.action[-1].reshape(
                config["NUM_ENVS"], env.num_agents
            )
            last_policy_probs_all = traj_batch.policy_probs[-1].reshape(
                config["NUM_ENVS"], env.num_agents, action_dim
            )
            cr_in = (
                last_world_state_env[None, :],
                last_actions_all[None, :],
                last_policy_probs_all[None, :],
                last_global_done[None, :],
            )
            _, (_, last_v) = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_v.squeeze(0).reshape(config["NUM_ACTORS"])

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.v_value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.v_value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            use_recurrent = config.get("use_recurrent_policy", True)
            data_chunk_length = config.get("DATA_CHUNK_LENGTH", config["NUM_STEPS"])
            chunks_per_rollout = config["NUM_STEPS"] // data_chunk_length
            actor_num_mini_batch = config["NUM_MINIBATCHES"]
            critic_num_mini_batch = config.get("CRITIC_NUM_MINI_BATCH", config["NUM_MINIBATCHES"])

            def _actor_chunks(x):
                x = x.swapaxes(0, 1)
                return x.reshape((config["NUM_ACTORS"] * chunks_per_rollout, data_chunk_length) + x.shape[2:])

            def _critic_chunks(x):
                x = x.swapaxes(0, 1)
                return x.reshape((config["NUM_ENVS"] * chunks_per_rollout, data_chunk_length) + x.shape[2:])

            norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                actor_q_value = _actor_chunks(traj_batch.q_value)
                actor_v_value = _actor_chunks(traj_batch.v_value)
                actor_norm_adv = _actor_chunks(norm_advantages)
                actor_init_hstate = _actor_chunks(traj_batch.actor_hstate)[:, 0]

                critic_world_state = _critic_chunks(
                    traj_batch.world_state.reshape(config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents, -1)[:, :, 0, :]
                )
                critic_actions = _critic_chunks(
                    traj_batch.action.reshape(config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                )
                critic_policy_probs = _critic_chunks(
                    traj_batch.policy_probs.reshape(config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents, action_dim)
                )
                critic_done = _critic_chunks(
                    traj_batch.global_done.reshape(config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)[:, :, 0]
                )
                critic_targets = _critic_chunks(
                    targets.reshape(config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                )
                critic_init_hstate = _critic_chunks(traj_batch.critic_hstate)[:, 0]

            def _actor_update_epoch(update_state, unused):
                def _actor_update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states

                    if use_recurrent:
                        mb_obs, mb_done, mb_avail, mb_action, mb_log_prob, mb_q_value, mb_v_value, mb_adv, ac_init_hstate_mb = batch_info
                    else:
                        ac_init_hstate, cr_init_hstate, traj_batch_mb, advantages_mb, targets_mb = batch_info

                    if use_recurrent:
                        coma_advantages = mb_q_value - mb_v_value
                        coma_advantages = (coma_advantages - coma_advantages.mean()) / (coma_advantages.std() + 1e-8)
                    else:
                        coma_advantages = traj_batch_mb.q_value - traj_batch_mb.v_value
                        coma_advantages = (coma_advantages - coma_advantages.mean()) / (coma_advantages.std() + 1e-8)

                    def _actor_loss_fn(actor_params, init_hstate, traj_obs, traj_done, traj_avail, traj_action, traj_log_prob, gae):
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate,
                            (traj_obs, traj_done, traj_avail),
                        )
                        log_prob = pi.log_prob(traj_action)
                        logratio = log_prob - traj_log_prob
                        ratio = jnp.exp(logratio)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    if use_recurrent:
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params,
                            ac_init_hstate_mb,
                            mb_obs, mb_done, mb_avail, mb_action, mb_log_prob, mb_adv,
                        )
                    else:
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params,
                            ac_init_hstate.squeeze(0),
                            traj_batch_mb.obs, traj_batch_mb.done, traj_batch_mb.avail_actions,
                            traj_batch_mb.action, traj_batch_mb.log_prob, coma_advantages,
                        )

                    actor_grad_norm = optax.global_norm(actor_grads)
                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )

                    actor_loss_info = {
                        "actor_loss": actor_loss[0],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "actor_grad_norm": actor_grad_norm,
                    }
                    return (actor_train_state, critic_train_state), actor_loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
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
                    shuffled_q_value = _shuffle_reshape(actor_q_value)
                    shuffled_v_value = _shuffle_reshape(actor_v_value)
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
                            shuffled_q_value[i].swapaxes(0, 1),
                            shuffled_v_value[i].swapaxes(0, 1),
                            shuffled_norm_adv[i].swapaxes(0, 1),
                            shuffled_init_hstate[i],
                        )
                        train_states, info = _actor_update_minbatch(train_states, mb_data)
                        return train_states, info

                    train_states, actor_loss_info = jax.lax.scan(
                        scan_actor_minibatch, train_states, jnp.arange(actor_num_mini_batch)
                    )
                else:
                    ac_init_hstate = jnp.reshape(
                        init_hstates[0], (1, config["NUM_ACTORS"], -1)
                    )
                    cr_init_hstate = jnp.reshape(
                        init_hstates[1], (1, config["NUM_ENVS"], -1)
                    )

                    batch = (
                        ac_init_hstate,
                        cr_init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )

                    if config["NUM_MINIBATCHES"] == 1:
                        minibatches = jax.tree.map(
                            lambda x: jnp.reshape(x, (1,) + x.shape),
                            batch,
                        )
                    else:
                        permutation = jax.random.permutation(_rng, config["NUM_STEPS"])
                        shuffled_batch = jax.tree.map(
                            lambda x: jnp.take(x, permutation, axis=0), batch
                        )
                        minibatches = jax.tree.map(
                            lambda x: jnp.swapaxes(
                                jnp.reshape(
                                    x,
                                    [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                    + list(x.shape[2:]),
                                ),
                                1,
                                0,
                            ),
                            shuffled_batch,
                        )

                    train_states, actor_loss_info = jax.lax.scan(
                        _actor_update_minbatch, train_states, minibatches
                    )

                update_state = (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, actor_loss_info

            def _critic_update_epoch(update_state, unused):
                def _critic_update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states

                    if use_recurrent:
                        mb_world_state, mb_actions, mb_policy_probs, mb_done, mb_targets, cr_init_hstate_mb = batch_info
                    else:
                        ac_init_hstate, cr_init_hstate, traj_batch_mb, advantages_mb, targets_mb = batch_info

                    def _critic_loss_fn(critic_params, init_hstate, cr_world_state, cr_actions, cr_policy_probs, cr_done, cr_targets):
                        _, (q_values, v_values) = critic_network.apply(
                            critic_params,
                            init_hstate,
                            (cr_world_state, cr_actions, cr_policy_probs, cr_done),
                        )
                        q_taken = jnp.take_along_axis(
                            q_values, cr_actions[..., None], axis=-1
                        ).squeeze(-1)

                        if use_recurrent:
                            q_taken_flat = q_taken.reshape(-1)
                            targets_flat = cr_targets.reshape(-1)
                        else:
                            q_taken_flat = q_taken.reshape(-1)
                            targets_flat = cr_targets.reshape(-1)

                        error = q_taken_flat - targets_flat
                        if config.get("use_huber_loss", True):
                            delta = config.get("huber_delta", 10.0)
                            value_loss = jnp.where(
                                jnp.abs(error) <= delta,
                                0.5 * error ** 2,
                                delta * (jnp.abs(error) - 0.5 * delta),
                            ).mean()
                        else:
                            value_loss = jnp.square(error).mean()

                        critic_loss = config["VALUE_LOSS_COEF"] * value_loss
                        return critic_loss, value_loss

                    if use_recurrent:
                        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                        critic_loss, critic_grads = critic_grad_fn(
                            critic_train_state.params,
                            cr_init_hstate_mb,
                            mb_world_state, mb_actions, mb_policy_probs, mb_done, mb_targets,
                        )
                    else:
                        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                        critic_loss, critic_grads = critic_grad_fn(
                            critic_train_state.params,
                            cr_init_hstate.squeeze(0),
                            traj_batch_mb.world_state.reshape(
                                config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents, -1
                            )[:, :, 0, :],
                            traj_batch_mb.action.reshape(
                                config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents
                            ),
                            traj_batch_mb.policy_probs.reshape(
                                config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents, action_dim
                            ),
                            traj_batch_mb.global_done.reshape(
                                config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents
                            )[:, :, 0],
                            targets_mb,
                        )

                    critic_grad_norm = optax.global_norm(critic_grads)
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    critic_loss_info = {
                        "value_loss": critic_loss[0],
                        "critic_grad_norm": critic_grad_norm,
                    }
                    return (actor_train_state, critic_train_state), critic_loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                if use_recurrent:
                    perm = jax.random.permutation(_rng, critic_sample_count)
                    def _shuffle_reshape(x):
                        x = jnp.take(x, perm, axis=0)
                        return x.reshape(critic_num_mini_batch, critic_mini_batch_size, *x.shape[1:])

                    shuffled_world_state = _shuffle_reshape(critic_world_state)
                    shuffled_actions = _shuffle_reshape(critic_actions)
                    shuffled_policy_probs = _shuffle_reshape(critic_policy_probs)
                    shuffled_done = _shuffle_reshape(critic_done)
                    shuffled_targets = _shuffle_reshape(critic_targets)
                    shuffled_init_hstate = jnp.take(critic_init_hstate, perm, axis=0)
                    shuffled_init_hstate = shuffled_init_hstate.reshape(critic_num_mini_batch, critic_mini_batch_size, *shuffled_init_hstate.shape[1:])

                    def scan_critic_minibatch(train_states, i):
                        mb_data = (
                            shuffled_world_state[i].swapaxes(0, 1),
                            shuffled_actions[i].swapaxes(0, 1),
                            shuffled_policy_probs[i].swapaxes(0, 1),
                            shuffled_done[i].swapaxes(0, 1),
                            shuffled_targets[i].swapaxes(0, 1),
                            shuffled_init_hstate[i],
                        )
                        train_states, info = _critic_update_minbatch(train_states, mb_data)
                        return train_states, info

                    train_states, critic_loss_info = jax.lax.scan(
                        scan_critic_minibatch, train_states, jnp.arange(critic_num_mini_batch)
                    )
                else:
                    ac_init_hstate = jnp.reshape(
                        init_hstates[0], (1, config["NUM_ACTORS"], -1)
                    )
                    cr_init_hstate = jnp.reshape(
                        init_hstates[1], (1, config["NUM_ENVS"], -1)
                    )

                    batch = (
                        ac_init_hstate,
                        cr_init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )

                    if config["NUM_MINIBATCHES"] == 1:
                        minibatches = jax.tree.map(
                            lambda x: jnp.reshape(x, (1,) + x.shape),
                            batch,
                        )
                    else:
                        permutation = jax.random.permutation(_rng, config["NUM_STEPS"])
                        shuffled_batch = jax.tree.map(
                            lambda x: jnp.take(x, permutation, axis=0), batch
                        )
                        minibatches = jax.tree.map(
                            lambda x: jnp.swapaxes(
                                jnp.reshape(
                                    x,
                                    [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                    + list(x.shape[2:]),
                                ),
                                1,
                                0,
                            ),
                            shuffled_batch,
                        )

                    train_states, critic_loss_info = jax.lax.scan(
                        _critic_update_minbatch, train_states, minibatches
                    )

                update_state = (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, critic_loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, actor_loss_info = jax.lax.scan(
                _actor_update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            update_state, critic_loss_info = jax.lax.scan(
                _critic_update_epoch, update_state, None, config["CRITIC_EPOCH"]
            )
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
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info

            mask = metric["returned_episode"][:, :, 0]
            ep_count = jnp.sum(mask) + 1e-8
            returns = (
                jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            )
            win_rate = (
                jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count
            )

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
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                last_global_done,
                update_state[1],  # updated hstates
                update_state[-1],  # rng
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def _override_config_from_cli(config):
    """Override config values with command-line arguments."""
    parser = argparse.ArgumentParser(description="COMA training for SMAX")
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
    if args.use_recurrent_policy is not None:
        config["use_recurrent_policy"] = args.use_recurrent_policy
    if args.use_valuenorm is not None:
        config["use_valuenorm"] = args.use_valuenorm
    return config


if __name__ == "__main__":
    config = get_default_coma_config()
    config = _override_config_from_cli(config)
    # Ensure aliases are set for backward compatibility with the GRU baseline style
    config.setdefault("UPDATE_EPOCHS", config["PPO_EPOCH"])
    config.setdefault("CRITIC_EPOCH", config["UPDATE_EPOCHS"])
    config.setdefault("NUM_MINIBATCHES", config["ACTOR_NUM_MINI_BATCH"])
    config.setdefault("CLIP_EPS", config["CLIP_PARAM"])
    config.setdefault("ENT_COEF", config["ENT_COEF"])
    config.setdefault("VF_COEF", config["VALUE_LOSS_COEF"])
    config.setdefault("FC_DIM_SIZE", config["hidden_sizes"][0])
    config.setdefault("GRU_HIDDEN_DIM", config["hidden_sizes"][-1])

    print(f"Starting {config['MAP_NAME']} COMA Baseline...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smax_coma_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "coma_gru",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved COMA actor checkpoint to {model_path}")
