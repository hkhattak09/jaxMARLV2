"""
MAPPO CTM for Hanabi.
Colab-ready, dependency-light version (no Hydra/wandb).
"""
import os
import sys
import pickle

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TEST_LOGGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_and_logger")
if os.path.isdir(_TEST_LOGGER_DIR) and _TEST_LOGGER_DIR not in sys.path:
    sys.path.insert(0, _TEST_LOGGER_DIR)

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

from ctm_jax import ScannedCTM, CTMCell
from jaxmarl.wrappers.baselines import LogWrapper, JaxMARLWrapper
from jaxmarl.environments.hanabi import Hanabi as HanabiEnv


class HanabiWorldStateWrapper(JaxMARLWrapper):
    """Global state for the centralised critic = concatenation of all agent observations."""

    def __init__(self, env: HanabiEnv):
        super().__init__(env)
        self._world_state_size = self._env.num_agents * self._env.obs_size

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self._get_world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self._get_world_state(obs)
        return obs, env_state, reward, done, info

    def _get_world_state(self, obs):
        # Single-env call (inside vmap): obs[a] shape (obs_size,)
        global_state = jnp.concatenate([obs[a] for a in self._env.agents])  # (num_agents * obs_size,)
        return jnp.tile(global_state[None, :], (self._env.num_agents, 1))   # (num_agents, world_state_size)

    def get_legal_moves(self, state):
        return self._env.get_legal_moves(state)

    def world_state_size(self):
        return self._world_state_size


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCTM(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        num_consensus_iterations = int(self.config.get("NUM_CONSENSUS_ITERATIONS", 0))
        if num_consensus_iterations < 0:
            raise ValueError(
                f"NUM_CONSENSUS_ITERATIONS must be >= 0, got {num_consensus_iterations}."
            )
        if num_consensus_iterations != 0:
            raise NotImplementedError(
                "Stage 1 only supports NUM_CONSENSUS_ITERATIONS == 0. "
                "Consensus pooling will be added in the next stage."
            )

        ctm_in = (obs, dones, avail_actions)
        hidden, synch = ScannedCTM(self.config)(hidden, ctm_in)

        x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(synch)
        x_head = nn.relu(x_head)
        x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(x_head)
        x_head = nn.relu(x_head)
        x_head = nn.Dense(self.action_dim)(x_head)

        unavail_actions = 1 - avail_actions
        action_logits = x_head - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, jnp.squeeze(critic, axis=-1)


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
    config.setdefault("NUM_CONSENSUS_ITERATIONS", 0)
    if config.get("CTM_NEURON_SELECT", "first-last") != "first-last":
        raise ValueError(
            f"Unsupported CTM_NEURON_SELECT={config.get('CTM_NEURON_SELECT')}. "
            "This RL port currently supports only 'first-last'."
        )
    if config["CTM_N_SYNCH_OUT"] > config["CTM_D_MODEL"]:
        raise ValueError(
            f"CTM_N_SYNCH_OUT ({config['CTM_N_SYNCH_OUT']}) must be <= CTM_D_MODEL ({config['CTM_D_MODEL']})."
        )

    env = HanabiEnv(num_agents=config["NUM_AGENTS"], **config.get("ENV_KWARGS", {}))
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = HanabiWorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorCTM(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n

        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], action_dim)),
        )
        ac_init_hstate = CTMCell.initialize_carry(config["NUM_ENVS"], config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            actor_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        actor_train_state = TrainState.create(apply_fn=actor_network.apply, params=actor_network_params, tx=actor_tx)
        critic_train_state = TrainState.create(apply_fn=critic_network.apply, params=critic_network_params, tx=critic_tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = CTMCell.initialize_carry(config["NUM_ACTORS"], config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # HanabiEnv.get_legal_moves returns a dict {agent: (num_moves,)};
                # vmapping over envs yields {agent: (num_envs, num_moves)}.
                avail_actions_dict = jax.vmap(env._env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions_dict, env.agents, config["NUM_ACTORS"]))

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # CTM ScannedCTM expects time axis on all three inputs
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])

                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
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

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))
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

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        _, pi = actor_network.apply(
                            actor_params, jax.tree.map(lambda x: x[0], init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(critic_params, jax.tree.map(lambda x: x[0], init_hstate), (traj_batch.world_state, traj_batch.done))
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch, advantages)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, cr_init_hstate, traj_batch, targets)

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
                init_hstates = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), init_hstates)

                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                    ), shuffled_batch
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, jax.tree.map(lambda x: x[0], init_hstates), traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)), traj_batch.info
            )
            metric["loss"] = loss_info

            # Episode return logging (Hanabi score = cumulative reward, max 25 for 2-player)
            mask = metric["returned_episode"][:, :, 0]
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count

            total_loss = loss_info["total_loss"]
            entropy = loss_info["entropy"]
            actor_grad_norm = loss_info["actor_grad_norm"]
            critic_grad_norm = loss_info["critic_grad_norm"]
            has_nan = (
                jnp.isnan(total_loss)
                | jnp.isnan(entropy)
                | jnp.isnan(actor_grad_norm)
                | jnp.isnan(critic_grad_norm)
                | jnp.isnan(returns)
            )
            entropy_low = entropy < 1e-3

            def log_callback(r, s, tl, ent, agn, cgn, nan_flag, ent_low):
                line = (
                    f"Step {s:8d} | Score: {r:6.2f}/25 | Loss: {tl:10.4f} | Ent: {ent:8.4f} "
                    f"| GradN(actor/critic): {agn:8.4f}/{cgn:8.4f}"
                )
                if nan_flag:
                    line += " | ALERT: NaN detected"
                if ent_low:
                    line += " | ALERT: entropy collapse risk"
                print(line)

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.debug.callback(
                log_callback, returns, step_count, total_loss, entropy,
                actor_grad_norm, critic_grad_norm, has_nan, entropy_low,
            )

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, update_state[-1])
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            # Force reset-on-done once at rollout start so CTM uses learned start traces.
            jnp.ones((config["NUM_ACTORS"],), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 128,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": int(5e7),
        "FC_DIM_SIZE": 512,
        "GRU_HIDDEN_DIM": 512,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "SCALE_CLIP_EPS": False,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "NUM_AGENTS": 2,
        "CTM_D_MODEL": 128,
        "CTM_D_INPUT": 64,
        "CTM_ITERATIONS": 1,
        "CTM_N_SYNCH_OUT": 32,
        "CTM_MEMORY_LENGTH": 5,
        "CTM_DEEP_NLMS": True,
        "CTM_NLM_HIDDEN_DIM": 2,
        "CTM_DO_LAYERNORM_NLM": False,
        "CTM_USE_SYNC": True,
        "CTM_NEURON_SELECT": "first-last",
        "CTM_ACTOR_HEAD_DIM": 64,
        "NUM_CONSENSUS_ITERATIONS": 0,
        "SEED": 42,
        "ENV_KWARGS": {},  # passed to HanabiEnv (num_colors, num_ranks, hand_size, etc.)
        "ANNEAL_LR": True,
    }

    print(f"Starting Hanabi MAPPO CTM (num_agents={config['NUM_AGENTS']})...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))

    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    suffix = "_nosync" if not config.get("CTM_USE_SYNC", True) else ""
    model_path = os.path.join(model_dir, f"hanabi_mappo_ctm_actor{suffix}.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "ctm",
        "env": "hanabi",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved CTM actor checkpoint to {model_path}")
