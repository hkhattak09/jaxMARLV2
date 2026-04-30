"""
HAPPO GRU Baseline for SMAX
Colab-ready, dependency-light version (no Hydra/wandb).
"""
import csv
import os
import sys
import pickle
import time
import argparse
from datetime import datetime

# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Sequence, NamedTuple, Dict
import functools
from flax.training.train_state import TrainState
import distrax
from functools import partial

from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

from happo import ActorRNN, CriticRNN, get_default_happo_config
from happo.actor import ScannedRNN
from happo.config import validate_happo_config


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
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


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
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_PARAM"] / env.num_agents if config.get("SCALE_CLIP_EPS", False) else config["CLIP_PARAM"]

    validate_happo_config(config, env.num_agents)
    assert config["ACTOR_NUM_MINI_BATCH"] == 1, "HAPPO baseline only supports ACTOR_NUM_MINI_BATCH=1"
    assert config["CRITIC_NUM_MINI_BATCH"] == 1, "HAPPO baseline only supports CRITIC_NUM_MINI_BATCH=1"

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        actor_steps_per_update = env.num_agents * config["PPO_EPOCH"] * config["ACTOR_NUM_MINI_BATCH"]
        update_num = count // actor_steps_per_update
        frac = 1.0 - update_num / config["NUM_UPDATES"]
        return config["LR"] * jnp.maximum(frac, 0.0)

    # === Logging setup ===
    save_interval = config.get("SAVE_INTERVAL", 1000000)
    print_interval = max(1, save_interval // 20)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(_REPO_ROOT, "saved_models", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "progress.csv")
    progress_header = [
        "step", "update", "return", "win_rate", "win_rate_std",
        "ep_len", "timeout_rate",
        "value_loss", "entropy", "clip_frac", "approx_kl",
        "actor_grad_norm", "critic_grad_norm",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(progress_header)

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
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            actor_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["CRITIC_LR"], eps=1e-5))
            
        actor_train_state = TrainState.create(apply_fn=actor_network.apply, params=actor_network_params, tx=actor_tx)
        critic_train_state = TrainState.create(apply_fn=critic_network.apply, params=critic_network_params, tx=critic_tx)

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
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
                
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                world_state = last_obs["world_state"].swapaxes(0,1)  
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
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
                    world_state,
                    info,
                    avail_actions,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state
            
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
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

            # --- HAPPO Sequential Actor Update ---
            actor_train_state, critic_train_state = train_states
            num_agents = env.num_agents

            # Compute old log probs for all agents using the behaviour (old) policy
            _, pi_old = actor_network.apply(
                actor_train_state.params,
                initial_hstates[0],
                (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
            )
            old_log_probs_all = pi_old.log_prob(traj_batch.action)

            # Reshape to (T, num_envs, num_agents)
            old_log_probs_all = old_log_probs_all.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents)
            advantages_r = advantages.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents)
            targets_r = targets.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents)

            ac_init_hstate_r = initial_hstates[0].reshape(config["NUM_ENVS"], num_agents, config["GRU_HIDDEN_DIM"])

            traj_obs_r = traj_batch.obs.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents, -1)
            traj_done_r = traj_batch.done.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents)
            traj_avail_r = traj_batch.avail_actions.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents, -1)
            traj_action_r = traj_batch.action.reshape(config["NUM_STEPS"], config["NUM_ENVS"], num_agents)

            # Determine agent update order
            rng, _rng_order = jax.random.split(rng)
            if config.get("FIXED_ORDER", False):
                agent_order = jnp.arange(num_agents)
            else:
                agent_order = jax.random.permutation(_rng_order, num_agents)

            factor = jnp.ones((config["NUM_STEPS"], config["NUM_ENVS"]))

            actor_loss_sum = 0.0
            entropy_sum = 0.0
            approx_kl_sum = 0.0
            actor_grad_norm_sum = 0.0

            for i in range(num_agents):
                agent_id = agent_order[i]

                obs_i = traj_obs_r[:, :, agent_id, :]
                dones_i = traj_done_r[:, :, agent_id]
                avail_i = traj_avail_r[:, :, agent_id, :]
                actions_i = traj_action_r[:, :, agent_id]
                old_log_prob_i = old_log_probs_all[:, :, agent_id]
                adv_i = advantages_r[:, :, agent_id]
                hstate_i = ac_init_hstate_r[:, agent_id, :]

                # Run PPO epochs for this agent
                for _ in range(config["PPO_EPOCH"]):
                    def _actor_loss_fn(actor_params, init_hstate, obs, dones, avail, actions, old_lp, adv, fac):
                        _, pi = actor_network.apply(actor_params, init_hstate, (obs, dones, avail))
                        log_prob = pi.log_prob(actions)
                        logratio = log_prob - old_lp
                        ratio = jnp.exp(logratio)

                        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
                        weighted_adv = fac * adv_norm

                        loss_actor1 = ratio * weighted_adv
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * weighted_adv
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy, approx_kl, clip_frac, ratio)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    (loss, aux), grads = actor_grad_fn(
                        actor_train_state.params, hstate_i, obs_i, dones_i, avail_i, actions_i, old_log_prob_i, adv_i, factor
                    )
                    actor_train_state = actor_train_state.apply_gradients(grads=grads)

                # Compute new log prob after finishing all epochs for this agent
                _, pi_new = actor_network.apply(actor_train_state.params, hstate_i, (obs_i, dones_i, avail_i))
                new_log_prob_i = pi_new.log_prob(actions_i)

                # Update factor for the next agent
                log_prob_diff = new_log_prob_i - old_log_prob_i
                if config.get("action_aggregation", "prod") == "prod":
                    imp_ratio = jnp.exp(log_prob_diff)
                else:
                    imp_ratio = jnp.exp(log_prob_diff)
                factor = factor * imp_ratio

                # Accumulate metrics from the last epoch
                actor_loss_sum += aux[0]
                entropy_sum += aux[1]
                approx_kl_sum += aux[2]
                actor_grad_norm_sum += optax.global_norm(grads)

            actor_loss_avg = actor_loss_sum / num_agents
            entropy_avg = entropy_sum / num_agents
            approx_kl_avg = approx_kl_sum / num_agents
            actor_grad_norm_avg = actor_grad_norm_sum / num_agents

            # --- Critic Update ---
            cr_init_hstate_flat = initial_hstates[1]
            critic_loss_sum = 0.0
            critic_grad_norm_sum = 0.0

            for _ in range(config["CRITIC_EPOCH"]):
                def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                    _, value = critic_network.apply(critic_params, init_hstate, (traj_batch.world_state, traj_batch.done))

                    if config.get("use_huber_loss", False):
                        delta = config.get("huber_delta", 10.0)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        error = value - targets
                        error_clipped = value_pred_clipped - targets
                        def huber_fn(e):
                            abs_e = jnp.abs(e)
                            return jnp.where(abs_e <= delta, 0.5 * jnp.square(abs_e), delta * (abs_e - 0.5 * delta))
                        value_losses = huber_fn(error)
                        value_losses_clipped = huber_fn(error_clipped)
                        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
                    else:
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    critic_loss = config["VALUE_LOSS_COEF"] * value_loss
                    return critic_loss, (value_loss,)

                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                (critic_loss, c_aux), critic_grads = critic_grad_fn(
                    critic_train_state.params, cr_init_hstate_flat, traj_batch, targets
                )
                critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                critic_loss_sum += c_aux[0]
                critic_grad_norm_sum += optax.global_norm(critic_grads)

            critic_loss_avg = critic_loss_sum / config["CRITIC_EPOCH"]
            critic_grad_norm_avg = critic_grad_norm_sum / config["CRITIC_EPOCH"]

            train_states = (actor_train_state, critic_train_state)

            # LOGGING
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)), traj_batch.info
            )
            
            mask = metric["returned_episode"][:, :, 0]  # (steps, envs) bool
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count

            env_win_rates = jnp.sum(
                metric["returned_won_episode"][:, :, :] * mask[..., None], axis=0
            ) / (jnp.sum(mask, axis=0)[..., None] + 1e-8)
            win_rate_std = jnp.std(env_win_rates, ddof=1)
            ep_len = jnp.sum(metric["returned_episode_lengths"][:, :, 0] * mask) / ep_count
            timeout_rate = jnp.sum(metric["returned_timed_out"][:, :, 0] * mask) / ep_count

            loss_info = {
                "value_loss": critic_loss_avg,
                "entropy": entropy_avg,
                "approx_kl": approx_kl_avg,
                "actor_grad_norm": actor_grad_norm_avg,
                "critic_grad_norm": critic_grad_norm_avg,
            }

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
                loss_info.get("value_loss", 0.0), loss_info.get("entropy", 0.0),
                loss_info.get("clip_frac", 0.0), loss_info.get("approx_kl", 0.0),
                loss_info.get("actor_grad_norm", 0.0), loss_info.get("critic_grad_norm", 0.0),
            )
            
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
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
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train


def _override_config_from_cli(config):
    """Override config values with command-line arguments."""
    parser = argparse.ArgumentParser(description="HAPPO training for SMAX")
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
    config = get_default_happo_config()
    config = _override_config_from_cli(config)

    print(f"Starting {config['MAP_NAME']} HAPPO Baseline...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    
    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smax_happo_gru_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "gru",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved GRU actor checkpoint to {model_path}")
