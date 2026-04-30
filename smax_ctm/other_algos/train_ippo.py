"""
IPPO GRU Baseline for SMAX
Independent PPO with per-agent critics using local observations.
Based on train_mappo_gru.py but with decentralized critic inputs.
"""
import os
import sys
import pickle
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
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
import functools
from flax.training.train_state import TrainState
import distrax
from functools import partial
import time

from jaxmarl.wrappers.baselines import SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

from ippo.config import get_default_ippo_config
from ippo.actor import ActorRNN, ScannedRNN
from ippo.critic import CriticRNN


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def huber_loss(error, delta):
    return jnp.where(
        jnp.abs(error) <= delta,
        0.5 * jnp.square(error),
        delta * (jnp.abs(error) - 0.5 * delta),
    )


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        opti_eps = config.get("opti_eps", 1e-5)
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=opti_eps),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=opti_eps),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=opti_eps),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["CRITIC_LR"], eps=opti_eps),
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
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)

                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE (IPPO: local obs instead of world_state)
                cr_in = (obs_batch[None, :], last_done[np.newaxis, :])
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

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
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            # IPPO: last value from local obs
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            cr_in = (last_obs_batch[None, :], last_done[np.newaxis, :])
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.global_done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )

                        if config.get("USE_CLIPPED_VALUE_LOSS", True):
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            if config.get("USE_HUBER_LOSS", False):
                                value_losses = huber_loss(value - targets, config["HUBER_DELTA"])
                                value_losses_clipped = huber_loss(
                                    value_pred_clipped - targets, config["HUBER_DELTA"]
                                )
                            else:
                                value_losses = jnp.square(value - targets)
                                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
                        else:
                            if config.get("USE_HUBER_LOSS", False):
                                value_loss = huber_loss(value - targets, config["HUBER_DELTA"]).mean()
                            else:
                                value_loss = 0.5 * jnp.square(value - targets).mean()

                        critic_loss = config["VALUE_LOSS_COEF"] * value_loss
                        return critic_loss, (value_loss,)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_grad_norm = optax.global_norm(actor_grads)
                    critic_grad_norm = optax.global_norm(critic_grads)

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][3],
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates
                )

                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info

            mask = metric["returned_episode"][:, :, 0]  # (steps, envs) bool
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count

            total_loss = loss_info["total_loss"]
            entropy = loss_info["entropy"]
            actor_grad_norm = loss_info["actor_grad_norm"]
            critic_grad_norm = loss_info["critic_grad_norm"]

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
                total_loss,
                entropy,
                actor_grad_norm,
                critic_grad_norm,
            )

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, update_state[-1])
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    config = get_default_ippo_config()

    print(f"Starting {config['MAP_NAME']} IPPO Baseline...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smax_ippo_gru_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "ippo_gru",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved IPPO actor checkpoint to {model_path}")
