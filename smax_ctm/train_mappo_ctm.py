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
_TEST_LOGGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_and_logger")
if os.path.isdir(_TEST_LOGGER_DIR) and _TEST_LOGGER_DIR not in sys.path:
    sys.path.insert(0, _TEST_LOGGER_DIR)

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
from ctm_jax import ScannedCTM, CTMCell
from step9_report import print_step9_summary

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



class ActorCTM(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x, deterministic: bool = True):
        obs, dones, avail_actions = x
        # `deterministic` is a Python bool and must stay one all the way down to
        # nn.Dropout, which branches on it with a Python `if`. Propagating it as a
        # scanned JAX array would raise ConcretizationTypeError under JIT once
        # INC_CONSENSUS_DROPOUT > 0. Pass it as a static module attribute instead.
        hidden, synch = ScannedCTM(self.config, deterministic=deterministic)(
            hidden, (obs, dones, avail_actions)
        )
        
        x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(synch)
        x_head = nn.relu(x_head)
        x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(x_head)
        x_head = nn.relu(x_head)
        x_head = nn.Dense(self.action_dim)(x_head)
        
        unavail_actions = 1 - avail_actions
        action_logits = x_head - (unavail_actions * 1e10)
        
        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi, synch

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
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

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


def compute_focus_fire_mask(
    actions: jnp.ndarray,
    num_agents: int,
    num_envs: int,
    num_movement_actions: int,
    num_enemies: int,
) -> jnp.ndarray:
    """Return (T, E) mask where at least two allies target the same enemy."""
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, A*E), got {actions.shape}.")
    expected = num_agents * num_envs
    if actions.shape[1] != expected:
        raise ValueError(
            f"actions second dim must be num_agents*num_envs={expected}, got {actions.shape[1]}."
        )
    if num_enemies <= 0:
        raise ValueError(f"num_enemies must be > 0, got {num_enemies}.")

    act = actions.reshape(actions.shape[0], num_agents, num_envs)
    min_attack_idx = num_movement_actions
    max_attack_idx = num_movement_actions + num_enemies - 1
    is_attack = (act >= min_attack_idx) & (act <= max_attack_idx)
    target_idx = act - min_attack_idx

    enemy_ids = jnp.arange(num_enemies)[None, None, None, :]
    hits_per_enemy = jnp.sum(
        is_attack[..., None] & (target_idx[..., None] == enemy_ids),
        axis=1,
    )
    return jnp.any(hits_per_enemy >= 2, axis=-1)


def compute_saal_alignment_terms(
    pair_cos_ff: jnp.ndarray,
    pair_cos_nff: jnp.ndarray,
    align_enabled: bool,
    align_alpha: float,
    align_beta: float,
):
    """Compute Stage 5 SAAL terms from pre-computed masked cosine means.

    Args:
        pair_cos_ff:  Scalar mean of pair_cos over focus-fire steps (already computed
                      for Stage 4 logging — pass it directly, do not recompute).
        pair_cos_nff: Scalar mean of pair_cos over non-focus-fire steps (same).
        align_enabled: Master switch; returns exact zeros when False.
        align_alpha:  Pull weight on coordinated events.
        align_beta:   Push weight on non-coordinated steps.
    """
    if not align_enabled:
        zero = jnp.asarray(0.0, dtype=pair_cos_ff.dtype)
        return zero, zero, zero

    align_pos = -align_alpha * pair_cos_ff
    align_neg = align_beta * pair_cos_nff
    l_align = align_pos + align_neg
    return l_align, align_pos, align_neg


def shuffle_and_split_actor_batch_env_grouped(
    batch_tree,
    env_permutation: jnp.ndarray,
    num_agents: int,
    num_envs: int,
    num_minibatches: int,
):
    """Shuffle envs and build minibatches without mixing agents across env groups."""
    if num_envs % num_minibatches != 0:
        raise ValueError(
            f"NUM_ENVS ({num_envs}) must be divisible by NUM_MINIBATCHES ({num_minibatches})."
        )
    if env_permutation.shape != (num_envs,):
        raise ValueError(
            f"env_permutation must have shape ({num_envs},), got {env_permutation.shape}."
        )
    envs_per_mb = num_envs // num_minibatches

    def _transform(x):
        if x.ndim < 2:
            raise ValueError(f"Expected rank>=2 tensor in batch tree, got shape {x.shape}.")
        expected_actor_dim = num_agents * num_envs
        if x.shape[1] != expected_actor_dim:
            raise ValueError(
                f"Actor axis mismatch: expected {expected_actor_dim}, got {x.shape[1]}."
            )

        grouped = jnp.reshape(x, (x.shape[0], num_agents, num_envs) + x.shape[2:])
        shuffled = jnp.take(grouped, env_permutation, axis=2)
        mb_grouped = jnp.reshape(
            shuffled,
            (x.shape[0], num_agents, num_minibatches, envs_per_mb) + x.shape[2:],
        )
        rest_axes = tuple(range(4, mb_grouped.ndim))
        mb_first = jnp.transpose(mb_grouped, (2, 0, 1, 3) + rest_axes)
        return jnp.reshape(
            mb_first,
            (num_minibatches, x.shape[0], num_agents * envs_per_mb) + x.shape[2:],
        )

    return jax.tree.map(_transform, batch_tree)

def make_train(config):
    config.setdefault("NUM_CONSENSUS_ITERATIONS", 0)
    config.setdefault("INC_ENABLED", False)
    config.setdefault("INC_POOLING", "mean")
    config.setdefault("INC_CONSENSUS_DROPOUT", 0.0)
    config.setdefault("INC_DEBUG_SHAPES", False)
    config.setdefault("INC_USE_ALIVE_MASK_FROM_DONES", True)
    # Stage 2.1 disambiguation flags.
    config.setdefault("CTM_ITER_DROPOUT", 0.0)
    config.setdefault("INC_FORCE_ZERO_CONSENSUS", False)
    config.setdefault("ALIGN_ENABLED", False)
    config.setdefault("ALIGN_ALPHA", 0.0)
    config.setdefault("ALIGN_BETA", 0.0)
    if config["INC_CONSENSUS_DROPOUT"] < 0.0 or config["INC_CONSENSUS_DROPOUT"] >= 1.0:
        raise ValueError(
            f"INC_CONSENSUS_DROPOUT must be in [0.0, 1.0), got {config['INC_CONSENSUS_DROPOUT']}."
        )
    if config["CTM_ITER_DROPOUT"] < 0.0 or config["CTM_ITER_DROPOUT"] >= 1.0:
        raise ValueError(
            f"CTM_ITER_DROPOUT must be in [0.0, 1.0), got {config['CTM_ITER_DROPOUT']}."
        )
    if config["ALIGN_ALPHA"] < 0.0:
        raise ValueError(f"ALIGN_ALPHA must be >= 0.0, got {config['ALIGN_ALPHA']}.")
    if config["ALIGN_BETA"] < 0.0:
        raise ValueError(f"ALIGN_BETA must be >= 0.0, got {config['ALIGN_BETA']}.")
    if config.get("CTM_NEURON_SELECT", "first-last") != "first-last":
        raise ValueError(
            f"Unsupported CTM_NEURON_SELECT={config.get('CTM_NEURON_SELECT')}. "
            "This RL port currently supports only 'first-last'."
        )
    if config["CTM_N_SYNCH_OUT"] > config["CTM_D_MODEL"]:
        raise ValueError(
            f"CTM_N_SYNCH_OUT ({config['CTM_N_SYNCH_OUT']}) must be <= CTM_D_MODEL ({config['CTM_D_MODEL']})."
        )

    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["INC_NUM_AGENTS"] = env.num_agents
    config["NUM_MOVEMENT_ACTIONS"] = int(env.num_movement_actions)
    config["NUM_ENEMIES"] = int(env.num_enemies)
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

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
            jnp.zeros((1, config["NUM_ACTORS"], obs_dim)),
            jnp.zeros((1, config["NUM_ACTORS"])),
            jnp.zeros((1, config["NUM_ACTORS"], action_dim)),
        )
        ac_init_hstate = CTMCell.initialize_carry(config["NUM_ACTORS"], config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x, deterministic=True)
        
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
                # Dropout RNG is needed if INC consensus dropout OR the
                # Stage 2.1 iteration-loop dropout control is on.
                inc_dropout_active = (
                    (config["INC_ENABLED"] and (config["INC_CONSENSUS_DROPOUT"] > 0.0))
                    or (config["CTM_ITER_DROPOUT"] > 0.0)
                )

                # SELECT ACTION
                rng, _rng_policy, _rng_dropout = jax.random.split(rng, 3)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # ScannedCTM expects a time axis on all sequence inputs.
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])

                if inc_dropout_active:
                    ac_hstate, pi, _ = actor_network.apply(
                        train_states[0].params,
                        hstates[0],
                        ac_in,
                        deterministic=False,
                        rngs={"dropout": _rng_dropout},
                    )
                else:
                    ac_hstate, pi, _ = actor_network.apply(
                        train_states[0].params,
                        hstates[0],
                        ac_in,
                        deterministic=False,
                    )
                action = pi.sample(seed=_rng_policy)
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

            if config["INC_DEBUG_SHAPES"]:
                def _shape_cb(t_dim, b_dim, n_agents):
                    print(
                        f"INC shape check | traj obs shape=(T={int(t_dim)}, B={int(b_dim)}, ...) "
                        f"| INC_NUM_AGENTS={int(n_agents)} | inferred num_envs={int(b_dim // n_agents)}"
                    )

                jax.lax.cond(
                    update_steps == 0,
                    lambda _: jax.debug.callback(
                        _shape_cb,
                        jnp.asarray(traj_batch.obs.shape[0]),
                        jnp.asarray(traj_batch.obs.shape[1]),
                        jnp.asarray(config["INC_NUM_AGENTS"]),
                    ),
                    lambda _: None,
                    operand=None,
                )
            
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

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    minibatch_tensors, dropout_key = batch_info
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = minibatch_tensors
                    inc_dropout_active = (
                        (config["INC_ENABLED"] and (config["INC_CONSENSUS_DROPOUT"] > 0.0))
                        or (config["CTM_ITER_DROPOUT"] > 0.0)
                    )

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae, rng_dropout):
                        if inc_dropout_active:
                            _, pi, synch = actor_network.apply(
                                actor_params,
                                jax.tree.map(lambda x: x[0], init_hstate),
                                (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                                deterministic=False,
                                rngs={"dropout": rng_dropout},
                            )
                        else:
                            _, pi, synch = actor_network.apply(
                                actor_params,
                                jax.tree.map(lambda x: x[0], init_hstate),
                                (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                                deterministic=False,
                            )

                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        if synch.ndim != 3:
                            raise ValueError(f"Expected synch shape (T, B, S), got {synch.shape}.")
                        t_steps = synch.shape[0]
                        num_agents = int(config["INC_NUM_AGENTS"])
                        if synch.shape[1] % num_agents != 0:
                            raise ValueError(
                                f"synch actor axis {synch.shape[1]} not divisible by num_agents {num_agents}."
                            )
                        num_envs_mb = synch.shape[1] // num_agents
                        synch_am = synch.reshape(t_steps, num_agents, num_envs_mb, synch.shape[-1])
                        s_norm = synch_am / (jnp.linalg.norm(synch_am, axis=-1, keepdims=True) + 1e-8)
                        cos_mat = jnp.einsum("taec,tbec->teab", s_norm, s_norm)
                        iu, ju = jnp.triu_indices(num_agents, k=1)
                        pair_cos = cos_mat[..., iu, ju].mean(axis=-1)

                        ff_mask = compute_focus_fire_mask(
                            traj_batch.action,
                            num_agents,
                            num_envs_mb,
                            int(config["NUM_MOVEMENT_ACTIONS"]),
                            int(config["NUM_ENEMIES"]),
                        )
                        ff_mask_f = ff_mask.astype(pair_cos.dtype)
                        nff_mask_f = (~ff_mask).astype(pair_cos.dtype)

                        pair_cos_all = pair_cos.mean()
                        pair_cos_ff = jnp.sum(pair_cos * ff_mask_f) / (jnp.sum(ff_mask_f) + 1e-8)
                        pair_cos_nff = jnp.sum(pair_cos * nff_mask_f) / (jnp.sum(nff_mask_f) + 1e-8)
                        ff_frac = ff_mask_f.mean()

                        if config["ALIGN_ENABLED"]:
                            l_align, align_pos, align_neg = compute_saal_alignment_terms(
                                pair_cos_ff,
                                pair_cos_nff,
                                True,
                                float(config["ALIGN_ALPHA"]),
                                float(config["ALIGN_BETA"]),
                            )
                        else:
                            zero = jnp.asarray(0.0, dtype=pair_cos.dtype)
                            l_align, align_pos, align_neg = zero, zero, zero

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        if config["ALIGN_ENABLED"]:
                            actor_loss = actor_loss + l_align
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            pair_cos_all,
                            pair_cos_ff,
                            pair_cos_nff,
                            ff_frac,
                            l_align,
                            align_pos,
                            align_neg,
                        )

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(
                            critic_params,
                            jax.tree.map(lambda x: x[0], init_hstate),
                            (traj_batch.world_state, traj_batch.done),
                        )
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        ac_init_hstate,
                        traj_batch,
                        advantages,
                        dropout_key,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        cr_init_hstate,
                        traj_batch,
                        targets,
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
                        "pair_cos_all": actor_loss[1][5],
                        "pair_cos_ff": actor_loss[1][6],
                        "pair_cos_nff": actor_loss[1][7],
                        "ff_frac": actor_loss[1][8],
                        "L_align": actor_loss[1][9],
                        "align_pos": actor_loss[1][10],
                        "align_neg": actor_loss[1][11],
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng_perm, _rng_dropout = jax.random.split(rng, 3)
                init_hstates = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), init_hstates)

                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                env_permutation = jax.random.permutation(_rng_perm, config["NUM_ENVS"])
                minibatches = shuffle_and_split_actor_batch_env_grouped(
                    batch,
                    env_permutation,
                    num_agents=env.num_agents,
                    num_envs=config["NUM_ENVS"],
                    num_minibatches=config["NUM_MINIBATCHES"],
                )
                mb_dropout_rngs = jax.random.split(_rng_dropout, config["NUM_MINIBATCHES"])

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch,
                    train_states,
                    (minibatches, mb_dropout_rngs),
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x[0], init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

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

            total_loss = loss_info["total_loss"]
            entropy = loss_info["entropy"]
            actor_grad_norm = loss_info["actor_grad_norm"]
            critic_grad_norm = loss_info["critic_grad_norm"]
            pair_cos_all = loss_info["pair_cos_all"]
            pair_cos_ff = loss_info["pair_cos_ff"]
            pair_cos_nff = loss_info["pair_cos_nff"]
            ff_frac = loss_info["ff_frac"]
            l_align = loss_info["L_align"]
            align_pos = loss_info["align_pos"]
            align_neg = loss_info["align_neg"]
            has_nan = (
                jnp.isnan(total_loss)
                | jnp.isnan(entropy)
                | jnp.isnan(actor_grad_norm)
                | jnp.isnan(critic_grad_norm)
                | jnp.isnan(returns)
                | jnp.isnan(win_rate)
                | jnp.isnan(pair_cos_all)
                | jnp.isnan(pair_cos_ff)
                | jnp.isnan(pair_cos_nff)
                | jnp.isnan(ff_frac)
                | jnp.isnan(l_align)
                | jnp.isnan(align_pos)
                | jnp.isnan(align_neg)
            )
            entropy_low = entropy < 1e-3

            def log_callback(
                r,
                w,
                s,
                tl,
                ent,
                agn,
                cgn,
                p_all,
                p_ff,
                p_nff,
                ff,
                l_align_cb,
                align_pos_cb,
                align_neg_cb,
                nan_flag,
                ent_low,
            ):
                line = (
                    f"Step {s:8d} | Return: {r:10.2f} | Win Rate: {w:5.2f} "
                    f"| Loss: {tl:10.4f} | Ent: {ent:8.4f} "
                    f"| GradN(actor/critic): {agn:8.4f}/{cgn:8.4f} "
                    f"| pair_cos(all/ff/nff): {p_all:7.4f}/{p_ff:7.4f}/{p_nff:7.4f} "
                    f"| ff_frac: {ff:6.4f} "
                    f"| align(L/pos/neg): {l_align_cb:8.5f}/{align_pos_cb:8.5f}/{align_neg_cb:8.5f}"
                )
                if nan_flag:
                    line += " | ALERT: NaN detected"
                if ent_low:
                    line += " | ALERT: entropy collapse risk"
                print(line)

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.debug.callback(
                log_callback,
                returns,
                win_rate,
                step_count,
                total_loss,
                entropy,
                actor_grad_norm,
                critic_grad_norm,
                pair_cos_all,
                pair_cos_ff,
                pair_cos_nff,
                ff_frac,
                l_align,
                align_pos,
                align_neg,
                has_nan,
                entropy_low,
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
            jnp.ones((config["NUM_ACTORS"]), dtype=bool),
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
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "SCALE_CLIP_EPS": False,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "OBS_WITH_AGENT_ID": True,
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
        "INC_ENABLED": False,
        "INC_POOLING": "mean",
        "INC_CONSENSUS_DROPOUT": 0.0,
        "INC_DEBUG_SHAPES": False,
        "INC_USE_ALIVE_MASK_FROM_DONES": True,
        # Stage 2.1 disambiguation defaults.
        "CTM_ITER_DROPOUT": 0.0,
        "INC_FORCE_ZERO_CONSENSUS": False,
        # Stage 5 SAAL defaults.
        "ALIGN_ENABLED": False,
        "ALIGN_ALPHA": 0.0,
        "ALIGN_BETA": 0.0,
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

    print(f"Starting {config['MAP_NAME']} MAPPO Baseline...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    
    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")
    print_step9_summary(out["metric"])

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    suffix = "_nosync" if not config.get("CTM_USE_SYNC", True) else ""
    model_path = os.path.join(model_dir, f"smax_mappo_ctm_actor{suffix}.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "ctm",
        "config": config,
        "actor_params": actor_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved CTM actor checkpoint to {model_path}")
    
    # Can optionally save metrics
    # jnp.save("gru_baseline_metrics.npy", out["metric"])
