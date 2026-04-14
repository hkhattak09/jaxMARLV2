"""
MAPPO GRU Baseline for SMAX
Colab-ready, dependency-light version (no Hydra/wandb).
"""
import os
import sys
import pickle
# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
import functools
from flax.training.train_state import TrainState
import distrax
from functools import partial
import time

# You may need to adapt imports based on where this is running relative to JaxMARL
from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

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
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

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


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    def setup(self):
        agent_embed_dim = self.config["AGENT_EMBED_DIM"]
        couple_hidden_dim = self.config["COUPLE_HIDDEN_DIM"]
        critic_hidden_dim = self.config["CRITIC_HIDDEN_DIM"]

        self.embed1 = nn.Dense(
            agent_embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.embed2 = nn.Dense(
            agent_embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.rnn = ScannedRNN()
        self.couple_h = nn.Dense(
            couple_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.couple_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        self.update_h = nn.Dense(
            agent_embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.update_out = nn.Dense(
            agent_embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.value_h1 = nn.Dense(
            critic_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.value_h2 = nn.Dense(
            critic_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.value_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, hidden, x):
        obs, dones = x

        if obs.ndim != 3:
            raise ValueError(f"Critic expects obs shape (T, B, obs_dim), got {obs.shape}")
        if dones.ndim != 2:
            raise ValueError(f"Critic expects done shape (T, B), got {dones.shape}")

        timesteps, num_actors, _ = obs.shape

        num_agents = self.config["NUM_AGENTS"]
        if num_actors % num_agents != 0:
            raise ValueError(
                f"NUM_ACTORS={num_actors} must be divisible by NUM_AGENTS={num_agents}"
            )
        num_envs = num_actors // num_agents

        coupling_iterations = int(self.config["COUPLING_ITERATIONS"])
        if coupling_iterations < 1:
            raise ValueError(
                f"COUPLING_ITERATIONS must be >= 1, got {coupling_iterations}"
            )

        if dones.shape != (timesteps, num_actors):
            raise ValueError(
                f"Critic done shape mismatch: expected {(timesteps, num_actors)}, got {dones.shape}"
            )

        agent_embed = self.embed1(obs)
        agent_embed = nn.relu(agent_embed)
        agent_embed = self.embed2(agent_embed)
        agent_embed = nn.relu(agent_embed)

        rnn_in = (agent_embed, dones)
        hidden, rnn_out = self.rnn(hidden, rnn_in)

        # env-major flat -> (T, E, A, D)
        D = self.config["AGENT_EMBED_DIM"]
        e = rnn_out.reshape((timesteps, num_envs, num_agents, D))

        # Explicitly mask dead agents to prevent dead-state signal propagation.
        alive_mask = (1.0 - dones.astype(jnp.float32)).reshape((timesteps, num_envs, num_agents))
        e = e * alive_mask[..., None]

        identity_mask = 1.0 - jnp.eye(num_agents, dtype=jnp.float32)
        pre_norm_sum = jnp.asarray(0.0, dtype=e.dtype)
        post_norm_sum = jnp.asarray(0.0, dtype=e.dtype)
        for _ in range(coupling_iterations):
            pre_norm_sum = pre_norm_sum + jnp.mean(jnp.linalg.norm(e, axis=-1))

            h_i = jnp.broadcast_to(e[:, :, :, None, :], (timesteps, num_envs, num_agents, num_agents, D))
            h_j = jnp.broadcast_to(e[:, :, None, :, :], (timesteps, num_envs, num_agents, num_agents, D))
            pair = jnp.concatenate([h_i, h_j], axis=-1)  # (T, E, A, A, 2D)

            C = self.couple_h(pair)
            C = nn.relu(C)
            C = self.couple_out(C)
            C = nn.sigmoid(C).squeeze(-1)  # (T, E, A, A)
            C = C * alive_mask[:, :, None, :] * identity_mask[None, None, :, :]

            context = jnp.einsum("teij,tejd->teid", C, e)  # (T, E, A, D)

            update_in = jnp.concatenate([e, context], axis=-1)  # (T, E, A, 2D)
            delta = self.update_h(update_in)
            delta = nn.relu(delta)
            delta = self.update_out(delta)
            delta = nn.relu(delta)
            e = e + delta
            e = e * alive_mask[..., None]

            post_norm_sum = post_norm_sum + jnp.mean(jnp.linalg.norm(e, axis=-1))

        mean_pre_norm = pre_norm_sum / coupling_iterations
        mean_post_norm = post_norm_sum / coupling_iterations
        self.sow("intermediates", "residual_pre_norm", mean_pre_norm)
        self.sow("intermediates", "residual_post_norm", mean_post_norm)

        critic = self.value_h1(e)
        critic = nn.relu(critic)
        critic = self.value_h2(critic)
        critic = nn.relu(critic)
        critic = self.value_out(critic)

        values = jnp.squeeze(critic, axis=-1)  # (T, E, A)
        values = values.reshape((timesteps, num_actors))  # env-major flat
        return hidden, values


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    critic_obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    x = jnp.swapaxes(x, 0, 1)  # (num_envs, num_agents, ...)
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_envs, num_agents, -1))
    x = jnp.swapaxes(x, 0, 1)  # (num_agents, num_envs, ...)
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_AGENTS"] = env.num_agents
    config["ACTION_DIM"] = env.action_space(env.agents[0]).n
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]
    if config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            f"NUM_ENVS ({config['NUM_ENVS']}) must be divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']}) "
            "to preserve full teams per minibatch"
        )

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
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
            jnp.zeros((1, config["NUM_ACTORS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["AGENT_EMBED_DIM"])
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
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["AGENT_EMBED_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state
                ac_hstate, cr_hstate = hstates

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])
                
                ac_hstate, pi = actor_network.apply(train_states[0].params, ac_hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                world_state = last_obs["world_state"].reshape((config["NUM_ACTORS"], -1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                cr_hstate, value = critic_network.apply(train_states[1].params, cr_hstate, cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                
                transition = Transition(
                    jnp.repeat(done["__all__"], env.num_agents),
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
            ac_hstate, cr_hstate = hstates
            
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_actions = jax.lax.stop_gradient(batchify(last_avail_actions, env.agents, config["NUM_ACTORS"]))
            rng, _rng_action = jax.random.split(rng)
            _, bootstrap_pi = actor_network.apply(
                train_states[0].params,
                ac_hstate,
                (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], last_avail_actions[np.newaxis, :]),
            )
            bootstrap_action = bootstrap_pi.sample(seed=_rng_action)
            last_world_state = last_obs["world_state"].reshape((config["NUM_ACTORS"], -1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            _, last_val = critic_network.apply(train_states[1].params, cr_hstate, cr_in)
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
                            actor_params, init_hstate.squeeze(), (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
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
                        (_, value), critic_debug = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.critic_obs, traj_batch.done),
                            mutable=["intermediates"],
                        )
                        residual_pre_norm = critic_debug["intermediates"]["residual_pre_norm"][0]
                        residual_post_norm = critic_debug["intermediates"]["residual_post_norm"][0]
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss, residual_pre_norm, residual_post_norm)

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
                        "residual_pre_norm": critic_loss[1][1],
                        "residual_post_norm": critic_loss[1][2],
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _ = jax.random.split(rng)
                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)
                
                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                shuffled_batch = batch
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                    ), shuffled_batch
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, jax.tree.map(lambda x: x.squeeze(), init_hstates), traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            
            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)), traj_batch.info
            )
            metric["loss"] = loss_info
            
            # JAX 0.7.x compatible logging: boolean masked indexing not allowed
            # inside jit/scan. Use weighted sum instead.
            mask = metric["returned_episode"][:, :, 0]  # (steps, envs) bool
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count

            total_loss = loss_info["total_loss"]
            entropy = loss_info["entropy"]
            actor_grad_norm = loss_info["actor_grad_norm"]
            critic_grad_norm = loss_info["critic_grad_norm"]
            residual_pre_norm = loss_info["residual_pre_norm"]
            residual_post_norm = loss_info["residual_post_norm"]

            def log_callback(r, w, s, tl, ent, agn, cgn, rpn, rpon):
                print(
                    f"Step {s:8d} | Return: {r:10.2f} | Win Rate: {w:5.2f} "
                    f"| Loss: {tl:10.4f} | Ent: {ent:8.4f} "
                    f"| GradN(actor/critic): {agn:8.4f}/{cgn:8.4f} "
                    f"| e_norm(pre/post): {rpn:8.4f}/{rpon:8.4f}"
                )

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.experimental.io_callback(
                log_callback, None,
                returns, win_rate, step_count,
                total_loss, entropy, actor_grad_norm, critic_grad_norm,
                residual_pre_norm, residual_post_norm,
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
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    config = {
        "LR": 0.002,
        "NUM_ENVS": 128,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": int(3e6),  # Train for 3M steps to see convergence
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "AGENT_EMBED_DIM": 128,
        "COUPLE_HIDDEN_DIM": 128,
        "COUPLING_ITERATIONS": 2,
        "CRITIC_HIDDEN_DIM": 128,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "SCALE_CLIP_EPS": False,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "OBS_WITH_AGENT_ID": True,
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "3m",    # We start with 3m
        "SEED": 42,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest"
        },
        "ANNEAL_LR": True
    }

    print(f"Starting {config['MAP_NAME']} MAPPO Iterative Coupling Stage 3...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    
    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "smax_mappo_ic_stage3_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "gru_actor_ic_stage3_iterative_critic",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved Stage 3 actor checkpoint to {model_path}")
    
    # Can optionally save metrics
    # jnp.save("gru_baseline_metrics.npy", out["metric"])
