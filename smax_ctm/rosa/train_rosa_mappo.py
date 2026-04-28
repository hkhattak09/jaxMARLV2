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
import numpy as np
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
import time

# You may need to adapt imports based on where this is running relative to JaxMARL
from jaxmarl.wrappers.baselines import SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from smax_ctm.rosa_config import (
    apply_cli_overrides,
    default_config,
    parse_args,
    print_resolved_config,
)
from smax_ctm.rosa_networks import ActorRNN, CriticRNN, ScannedRNN
from smax_ctm.rosa_utils import (
    adapter_id_from_mode,
    batchify,
    extract_role_id,
    role_lora_param_norms,
    role_mean_metric,
    role_std_metric,
    unbatchify,
)
from smax_ctm.smax_wrappers import SMAXWorldStateWrapper
from smax_ctm.role_maca_critic import (
    RoleMACATransformerCritic,
    actor_major_to_env_major,
    done_from_actor_major,
    env_major_to_actor_major,
    td_lambda_returns,
    team_reward_from_actor_major,
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
    role_id: jnp.ndarray
    adapter_id: jnp.ndarray


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    if obs_dim < config["NUM_UNIT_TYPES"]:
        raise ValueError(
            f"Cannot extract role IDs: obs_dim={obs_dim}, "
            f"NUM_UNIT_TYPES={config['NUM_UNIT_TYPES']}. Expected own unit-type bits "
            "at the end of the SMAX local observation."
        )
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_AGENTS"] = env.num_agents
    config["ACTION_DIM"] = env.action_space(env.agents[0]).n
    if config["ADAPTER_MODE"] == "global_lora":
        config["LORA_NUM_ADAPTERS"] = 1
    elif config["ADAPTER_MODE"] == "agent_lora":
        config["LORA_NUM_ADAPTERS"] = env.num_agents
    else:
        config["LORA_NUM_ADAPTERS"] = config["NUM_UNIT_TYPES"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    if config["NUM_ACTORS"] * config["NUM_STEPS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "Invalid minibatch setup: NUM_ACTORS * NUM_STEPS must be divisible by "
            f"NUM_MINIBATCHES. Got NUM_ACTORS={config['NUM_ACTORS']}, "
            f"NUM_STEPS={config['NUM_STEPS']}, NUM_MINIBATCHES={config['NUM_MINIBATCHES']}."
        )
    if config["NUM_UPDATES"] <= 0:
        raise ValueError(
            "Invalid training horizon: TOTAL_TIMESTEPS must be at least NUM_ENVS * NUM_STEPS. "
            f"Got TOTAL_TIMESTEPS={config['TOTAL_TIMESTEPS']}, NUM_ENVS={config['NUM_ENVS']}, "
            f"NUM_STEPS={config['NUM_STEPS']}."
        )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        if config["USE_ROLE_MACA"]:
            maca_critic_network = RoleMACATransformerCritic(config=config)
            rng, _rng_actor, _rng_critic, _rng_maca = jax.random.split(rng, 4)
        else:
            rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
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
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            
        actor_train_state = TrainState.create(apply_fn=actor_network.apply, params=actor_network_params, tx=actor_tx)
        critic_train_state = TrainState.create(apply_fn=critic_network.apply, params=critic_network_params, tx=critic_tx)
        if config["USE_ROLE_MACA"]:
            maca_critic_params = maca_critic_network.init(
                _rng_maca,
                jnp.zeros((1, config["NUM_ENVS"], config["NUM_AGENTS"], obs_dim)),
                jnp.zeros((1, config["NUM_ENVS"], config["NUM_AGENTS"]), dtype=jnp.int32),
                jnp.zeros((1, config["NUM_ENVS"], config["NUM_AGENTS"], config["ACTION_DIM"])),
                jnp.zeros((1, config["NUM_ENVS"], config["NUM_AGENTS"], config["ACTION_DIM"])),
            )
            if config["ANNEAL_LR"]:
                maca_critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
            else:
                maca_critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            maca_critic_train_state = TrainState.create(
                apply_fn=maca_critic_network.apply,
                params=maca_critic_params,
                tx=maca_critic_tx,
            )
            train_states = (actor_train_state, critic_train_state, maca_critic_train_state)
        else:
            train_states = (actor_train_state, critic_train_state)

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
                role_id = extract_role_id(obs_batch, env_state, env, config)
                agent_id = jnp.repeat(jnp.arange(env.num_agents, dtype=jnp.int32), config["NUM_ENVS"])
                adapter_id = adapter_id_from_mode(role_id, agent_id, config)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                    role_id[np.newaxis, :],
                    adapter_id[np.newaxis, :],
                )
                
                ac_hstate, pi, _ = actor_network.apply(train_states[0].params, hstates[0], ac_in)
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
                    role_id,
                    adapter_id,
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
            global_step = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]

            def _empty_maca_info():
                zeros_by_role = jnp.zeros((config["NUM_UNIT_TYPES"],))
                zeros_env = jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"]))
                return {
                    "advantages": advantages,
                    "obs_all": jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS"], obs_dim)),
                    "role_all": jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS"]), dtype=jnp.int32),
                    "taken_all": jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS"], config["ACTION_DIM"])),
                    "policy_all": jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS"], config["ACTION_DIM"])),
                    "return_v": zeros_env,
                    "return_q": zeros_env,
                    "return_eq": zeros_env,
                    "q_taken_mean_by_role": zeros_by_role,
                    "eq_mean_by_role": zeros_by_role,
                    "v_mean_by_role": zeros_by_role,
                    "return_eq_mean_by_role": zeros_by_role,
                    "return_eq_std_by_role": zeros_by_role,
                    "baseline_mean_by_role": zeros_by_role,
                    "baseline_std_by_role": zeros_by_role,
                    "baseline_self_w_by_role": zeros_by_role,
                    "baseline_group_w_by_role": zeros_by_role,
                    "baseline_joint_w_by_role": zeros_by_role,
                    "adv_maca_mean_by_role": zeros_by_role,
                    "adv_maca_std_by_role": zeros_by_role,
                    "attention_entropy": jnp.array(0.0),
                    "corrset_size": jnp.array(0.0),
                }

            def _calculate_role_maca_info():
                _, _, actor_aux = actor_network.apply(
                    train_states[0].params,
                    initial_hstates[0],
                    (
                        traj_batch.obs,
                        traj_batch.done,
                        traj_batch.avail_actions,
                        traj_batch.role_id,
                        traj_batch.adapter_id,
                    ),
                )
                policy_probs = jax.lax.stop_gradient(
                    jax.nn.softmax(actor_aux["action_logits"], axis=-1)
                )
                taken_probs = jax.nn.one_hot(traj_batch.action, config["ACTION_DIM"])

                obs_all = actor_major_to_env_major(traj_batch.obs, config["NUM_AGENTS"], config["NUM_ENVS"])
                role_all = actor_major_to_env_major(traj_batch.role_id, config["NUM_AGENTS"], config["NUM_ENVS"])
                taken_all = actor_major_to_env_major(taken_probs, config["NUM_AGENTS"], config["NUM_ENVS"])
                policy_all = actor_major_to_env_major(policy_probs, config["NUM_AGENTS"], config["NUM_ENVS"])

                maca_params = train_states[2].params
                maca_out = maca_critic_network.apply(maca_params, obs_all, role_all, taken_all, policy_all)

                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                last_role_id = extract_role_id(last_obs_batch, env_state, env, config)
                agent_id = jnp.repeat(jnp.arange(env.num_agents, dtype=jnp.int32), config["NUM_ENVS"])
                last_adapter_id = adapter_id_from_mode(last_role_id, agent_id, config)
                last_avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                last_avail_actions = jax.lax.stop_gradient(batchify(last_avail_actions, env.agents, config["NUM_ACTORS"]))
                _, last_pi, last_actor_aux = actor_network.apply(
                    train_states[0].params,
                    hstates[0],
                    (
                        last_obs_batch[np.newaxis, :],
                        last_done[np.newaxis, :],
                        last_avail_actions,
                        last_role_id[np.newaxis, :],
                        last_adapter_id[np.newaxis, :],
                    ),
                )
                del last_pi
                last_policy_probs = jax.lax.stop_gradient(
                    jax.nn.softmax(last_actor_aux["action_logits"], axis=-1)
                )
                last_obs_all = actor_major_to_env_major(last_obs_batch[np.newaxis, :], config["NUM_AGENTS"], config["NUM_ENVS"])
                last_role_all = actor_major_to_env_major(last_role_id[np.newaxis, :], config["NUM_AGENTS"], config["NUM_ENVS"])
                last_policy_all = actor_major_to_env_major(last_policy_probs, config["NUM_AGENTS"], config["NUM_ENVS"])
                last_maca_out = maca_critic_network.apply(
                    maca_params,
                    last_obs_all,
                    last_role_all,
                    last_policy_all,
                    last_policy_all,
                )
                team_reward = team_reward_from_actor_major(
                    traj_batch.reward,
                    config["NUM_AGENTS"],
                    config["NUM_ENVS"],
                )
                env_done = done_from_actor_major(
                    traj_batch.global_done,
                    config["NUM_AGENTS"],
                    config["NUM_ENVS"],
                )
                return_v = td_lambda_returns(
                    maca_out["v"],
                    last_maca_out["v"].squeeze(axis=0),
                    team_reward,
                    env_done,
                    config["GAMMA"],
                    config["GAE_LAMBDA"],
                )
                return_q = td_lambda_returns(
                    maca_out["q_taken"],
                    last_maca_out["q_policy"].squeeze(axis=0),
                    team_reward,
                    env_done,
                    config["GAMMA"],
                    config["GAE_LAMBDA"],
                )
                return_eq = td_lambda_returns(
                    maca_out["eq"],
                    last_maca_out["eq"].squeeze(axis=0),
                    team_reward,
                    env_done,
                    config["GAMMA"],
                    config["GAE_LAMBDA"],
                )
                baseline = env_major_to_actor_major(maca_out["mixed_baseline_i"])
                return_eq_by_actor = env_major_to_actor_major(
                    jnp.broadcast_to(return_eq[..., None], role_all.shape)
                )
                maca_advantages = jax.lax.stop_gradient(return_eq_by_actor - baseline)
                q_taken_by_actor = env_major_to_actor_major(
                    jnp.broadcast_to(maca_out["q_taken"][..., None], role_all.shape)
                )
                eq_by_actor = env_major_to_actor_major(
                    jnp.broadcast_to(maca_out["eq"][..., None], role_all.shape)
                )
                v_by_actor = env_major_to_actor_major(
                    jnp.broadcast_to(maca_out["v"][..., None], role_all.shape)
                )
                weights_by_actor = env_major_to_actor_major(maca_out["baseline_weights"])
                att = maca_out["attention"]
                attention_entropy = -(att * jnp.log(att + 1e-8)).sum(axis=-1).mean()
                corrset_size = maca_out["corr_mask"].sum(axis=-1).mean()
                return {
                    "advantages": maca_advantages,
                    "obs_all": obs_all,
                    "role_all": role_all,
                    "taken_all": taken_all,
                    "policy_all": policy_all,
                    "return_v": return_v,
                    "return_q": return_q,
                    "return_eq": return_eq,
                    "q_taken_mean_by_role": role_mean_metric(
                        q_taken_by_actor,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "eq_mean_by_role": role_mean_metric(
                        eq_by_actor,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "v_mean_by_role": role_mean_metric(
                        v_by_actor,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "return_eq_mean_by_role": role_mean_metric(
                        return_eq_by_actor,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "return_eq_std_by_role": role_std_metric(
                        return_eq_by_actor,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_mean_by_role": role_mean_metric(
                        baseline,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_std_by_role": role_std_metric(
                        baseline,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_self_w_by_role": role_mean_metric(
                        weights_by_actor[..., 0],
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_group_w_by_role": role_mean_metric(
                        weights_by_actor[..., 1],
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_joint_w_by_role": role_mean_metric(
                        weights_by_actor[..., 2],
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "adv_maca_mean_by_role": role_mean_metric(
                        maca_advantages,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "adv_maca_std_by_role": role_std_metric(
                        maca_advantages,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "attention_entropy": attention_entropy,
                    "corrset_size": corrset_size,
                }

            maca_info = _calculate_role_maca_info() if config["USE_ROLE_MACA"] else _empty_maca_info()
            if config["USE_ROLE_MACA"]:
                if config["ROLE_MACA_BLEND_SCHEDULE"] == "constant":
                    maca_alpha = jnp.asarray(config["ROLE_MACA_BLEND_ALPHA"], dtype=advantages.dtype)
                elif config["ROLE_MACA_BLEND_SCHEDULE"] == "linear_warmup":
                    progress = jnp.clip(
                        (
                            global_step - config["ROLE_MACA_BLEND_WARMUP_STEPS"]
                        )
                        / config["ROLE_MACA_BLEND_RAMP_STEPS"],
                        0.0,
                        1.0,
                    )
                    maca_alpha = config["ROLE_MACA_BLEND_ALPHA"] * progress
                else:
                    raise ValueError(
                        f"Unsupported ROLE_MACA_BLEND_SCHEDULE={config['ROLE_MACA_BLEND_SCHEDULE']!r}"
                    )
                gae_component = advantages
                maca_component = maca_info["advantages"]
                if config["ROLE_MACA_BLEND_NORMALIZE"]:
                    gae_component = (gae_component - gae_component.mean()) / (gae_component.std() + 1e-8)
                    maca_component = (maca_component - maca_component.mean()) / (maca_component.std() + 1e-8)
                actor_advantages = (1.0 - maca_alpha) * gae_component + maca_alpha * maca_component
            else:
                actor_advantages = advantages

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    if config["USE_ROLE_MACA"]:
                        actor_train_state, critic_train_state, maca_critic_train_state = train_states
                        ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info
                    else:
                        actor_train_state, critic_train_state = train_states
                        ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        _, pi, actor_aux = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (
                                traj_batch.obs,
                                traj_batch.done,
                                traj_batch.avail_actions,
                                traj_batch.role_id,
                                traj_batch.adapter_id,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        role_adv_mean = role_mean_metric(gae, traj_batch.role_id, config["NUM_UNIT_TYPES"])
                        role_adv_std = role_std_metric(gae, traj_batch.role_id, config["NUM_UNIT_TYPES"])
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        ppo_loss_per_sample = -jnp.minimum(loss_actor1, loss_actor2)
                        entropy_per_sample = pi.entropy()
                        entropy = entropy_per_sample.mean()
                        
                        approx_kl_per_sample = (ratio - 1) - logratio
                        clip_frac_per_sample = (jnp.abs(ratio - 1) > config["CLIP_EPS"]).astype(jnp.float32)
                        approx_kl = approx_kl_per_sample.mean()
                        clip_frac = clip_frac_per_sample.mean()
                        role_entropy = role_mean_metric(
                            entropy_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_approx_kl = role_mean_metric(
                            approx_kl_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_clip_frac = role_mean_metric(
                            clip_frac_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_ppo_loss = role_mean_metric(
                            ppo_loss_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        lora_delta_norm_by_role = role_mean_metric(
                            jnp.linalg.norm(actor_aux["lora_delta"], axis=-1),
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        policy_loss = ppo_loss_per_sample.mean()
                        loss_actor = policy_loss - config["ENT_COEF"] * entropy
                        return loss_actor, {
                            "policy_loss": policy_loss,
                            "entropy": entropy,
                            "approx_kl": approx_kl,
                            "clip_frac": clip_frac,
                            "lora_delta_norm_by_role": lora_delta_norm_by_role,
                            "role_adv_mean": role_adv_mean,
                            "role_adv_std": role_adv_std,
                            "role_entropy": role_entropy,
                            "role_approx_kl": role_approx_kl,
                            "role_clip_frac": role_clip_frac,
                            "role_ppo_loss": role_ppo_loss,
                        }
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch, advantages)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, cr_init_hstate, traj_batch, targets)
                    
                    actor_grad_norm = optax.global_norm(actor_grads)
                    critic_grad_norm = optax.global_norm(critic_grads)
                    lora_grad_norm_by_role = role_lora_param_norms(actor_grads, config)
                    lora_param_norm_by_role = role_lora_param_norms(actor_train_state.params, config)

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    if config["USE_ROLE_MACA"]:
                        next_train_states = (actor_train_state, critic_train_state, maca_critic_train_state)
                    else:
                        next_train_states = (actor_train_state, critic_train_state)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "mappo_value_loss": critic_loss[1],
                        "v_loss": jnp.array(0.0),
                        "q_loss": jnp.array(0.0),
                        "eq_loss": jnp.array(0.0),
                        "entropy": actor_loss[1]["entropy"],
                        "approx_kl": actor_loss[1]["approx_kl"],
                        "clip_frac": actor_loss[1]["clip_frac"],
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "maca_critic_grad_norm": jnp.array(0.0),
                        "lora_delta_norm_by_role": actor_loss[1]["lora_delta_norm_by_role"],
                        "role_adv_mean": actor_loss[1]["role_adv_mean"],
                        "role_adv_std": actor_loss[1]["role_adv_std"],
                        "role_entropy": actor_loss[1]["role_entropy"],
                        "role_approx_kl": actor_loss[1]["role_approx_kl"],
                        "role_clip_frac": actor_loss[1]["role_clip_frac"],
                        "role_ppo_loss": actor_loss[1]["role_ppo_loss"],
                        "lora_grad_norm_by_role": lora_grad_norm_by_role,
                        "lora_param_norm_by_role": lora_param_norm_by_role,
                    }
                    return next_train_states, loss_info

                if config["USE_ROLE_MACA"]:
                    train_states, init_hstates, traj_batch, advantages, targets, rng, maca_info = update_state
                else:
                    train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)

                if config["USE_ROLE_MACA"]:
                    actor_train_state, critic_train_state, maca_critic_train_state = train_states

                    def _maca_critic_loss_fn(maca_params, maca_info):
                        maca_out = maca_critic_network.apply(
                            maca_params,
                            maca_info["obs_all"],
                            maca_info["role_all"],
                            maca_info["taken_all"],
                            maca_info["policy_all"],
                        )
                        v_loss = 0.5 * jnp.square(maca_out["v"] - maca_info["return_v"]).mean()
                        q_loss = 0.5 * jnp.square(maca_out["q_taken"] - maca_info["return_q"]).mean()
                        eq_loss = 0.5 * jnp.square(maca_out["eq"] - maca_info["return_eq"]).mean()
                        maca_critic_loss = (
                            config["ROLE_MACA_VALUE_LOSS_COEF"] * v_loss
                            + config["ROLE_MACA_Q_LOSS_COEF"] * q_loss
                            + config["ROLE_MACA_EQ_LOSS_COEF"] * eq_loss
                        )
                        return maca_critic_loss, {
                            "v_loss": v_loss,
                            "q_loss": q_loss,
                            "eq_loss": eq_loss,
                        }

                    maca_grad_fn = jax.value_and_grad(_maca_critic_loss_fn, has_aux=True)
                    maca_loss, maca_grads = maca_grad_fn(maca_critic_train_state.params, maca_info)
                    maca_critic_grad_norm = optax.global_norm(maca_grads)
                    maca_critic_train_state = maca_critic_train_state.apply_gradients(grads=maca_grads)
                    train_states = (actor_train_state, critic_train_state, maca_critic_train_state)
                else:
                    maca_loss = (jnp.array(0.0), {"v_loss": jnp.array(0.0), "q_loss": jnp.array(0.0), "eq_loss": jnp.array(0.0)})
                    maca_critic_grad_norm = jnp.array(0.0)
                
                if config["USE_ROLE_MACA"]:
                    batch = (
                        init_hstates[0],
                        init_hstates[1],
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )
                else:
                    batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                    ), shuffled_batch
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                if config["USE_ROLE_MACA"]:
                    loss_info["total_loss"] = loss_info["total_loss"] + maca_loss[0]
                    loss_info["value_loss"] = loss_info["value_loss"] + maca_loss[0]
                    loss_info["v_loss"] = jnp.broadcast_to(maca_loss[1]["v_loss"], loss_info["v_loss"].shape)
                    loss_info["q_loss"] = jnp.broadcast_to(maca_loss[1]["q_loss"], loss_info["q_loss"].shape)
                    loss_info["eq_loss"] = jnp.broadcast_to(maca_loss[1]["eq_loss"], loss_info["eq_loss"].shape)
                    loss_info["maca_critic_grad_norm"] = jnp.broadcast_to(
                        maca_critic_grad_norm,
                        loss_info["maca_critic_grad_norm"].shape,
                    )
                if config["USE_ROLE_MACA"]:
                    update_state = (
                        train_states,
                        jax.tree.map(lambda x: x.squeeze(), init_hstates),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        maca_info,
                    )
                else:
                    update_state = (
                        train_states,
                        jax.tree.map(lambda x: x.squeeze(), init_hstates),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                return update_state, loss_info

            if config["USE_ROLE_MACA"]:
                update_state = (
                    train_states,
                    initial_hstates,
                    traj_batch,
                    actor_advantages,
                    targets,
                    rng,
                    maca_info,
                )
            else:
                update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            def reduce_loss_metric(x):
                if x.ndim <= 2:
                    return x.mean()
                return x.mean(axis=tuple(range(x.ndim - 1)))
            loss_info = jax.tree.map(reduce_loss_metric, loss_info)
            train_states = update_state[0]
            rng_after_update = update_state[5] if config["USE_ROLE_MACA"] else update_state[-1]

            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)), traj_batch.info
            )
            metric["loss"] = loss_info
            metric["role_maca_alpha"] = maca_alpha if config["USE_ROLE_MACA"] else jnp.array(0.0)
            
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
            q_loss_value = loss_info["q_loss"]
            eq_loss_value = loss_info["eq_loss"]
            v_loss_value = loss_info["v_loss"]
            maca_critic_grad_norm = loss_info["maca_critic_grad_norm"]
            lora_delta_norm_by_role = loss_info["lora_delta_norm_by_role"]
            lora_grad_norm_by_role = loss_info["lora_grad_norm_by_role"]
            lora_param_norm_by_role = loss_info["lora_param_norm_by_role"]
            role_adv_mean = loss_info["role_adv_mean"]
            role_adv_std = loss_info["role_adv_std"]
            role_entropy = loss_info["role_entropy"]
            role_approx_kl = loss_info["role_approx_kl"]
            role_clip_frac = loss_info["role_clip_frac"]
            role_ppo_loss = loss_info["role_ppo_loss"]
            role_counts = jnp.bincount(
                traj_batch.role_id.reshape(-1),
                length=config["NUM_UNIT_TYPES"],
            )
            role_present_mask = role_counts > 0
            role_bits = traj_batch.obs[:, :, -config["NUM_UNIT_TYPES"]:]
            role_id_min = traj_batch.role_id.min()
            role_id_max = traj_batch.role_id.max()
            role_bit_sums = role_bits.sum(axis=-1)
            role_bit_sum_min = role_bit_sums.min()
            role_bit_sum_max = role_bit_sums.max()
            nonzero_role_bits = role_bit_sums > 0
            obs_role_id = jnp.argmax(role_bits, axis=-1).astype(jnp.int32)
            obs_role_mismatch_count = jnp.sum(
                nonzero_role_bits & (obs_role_id != traj_batch.role_id)
            )
            zero_obs_role_bits = jnp.sum(role_bit_sums == 0)

            def log_callback(
                r, w, s, tl, ent, agn, cgn, ql, qgn, rc, rpm, rbmin, rbmax, mismatch,
                zero_bits, ldn, lgn, lpn, ram, ras, rent, rkl, rcf, rpl,
                eq_loss, v_loss, qtm, eqm, vm, rem, res, bm, bs, bsw, bgw, bjw,
                mam, mas, att_entropy, corrset_size, role_min, role_max,
                role_maca_alpha,
            ):
                if role_min < 0 or role_max >= config["NUM_UNIT_TYPES"]:
                    raise ValueError(
                        "Role extraction failed: role ids are outside the configured range. "
                        f"role_id_min={int(role_min)}, role_id_max={int(role_max)}, "
                        f"NUM_UNIT_TYPES={config['NUM_UNIT_TYPES']}, "
                        f"ROLE_ID_SOURCE={config['ROLE_ID_SOURCE']!r}"
                    )
                if rbmax > 1.0 or mismatch > 0:
                    raise ValueError(
                        "Role extraction failed: obs unit-type bits disagree with env-state roles. "
                        f"role_bit_sum_min={rbmin}, role_bit_sum_max={rbmax}, "
                        f"obs_role_mismatch_count={int(mismatch)}, "
                        f"zero_obs_role_bits={int(zero_bits)}, "
                        f"NUM_UNIT_TYPES={config['NUM_UNIT_TYPES']}, "
                        f"ROLE_ID_SOURCE={config['ROLE_ID_SOURCE']!r}"
                    )
                role_counts_str = " ".join(
                    f"role_count_{idx}: {int(count)}" for idx, count in enumerate(np.asarray(rc))
                )
                role_mask_str = "[" + " ".join(str(int(x)) for x in np.asarray(rpm)) + "]"
                msg = (
                    f"Step {s:8d} | Return: {r:10.2f} | Win Rate: {w:5.2f} "
                    f"| Loss: {tl:10.4f} | Ent: {ent:8.4f} "
                    f"| GradN(actor/critic): {agn:8.4f}/{cgn:8.4f} "
                    f"| {role_counts_str} | role_present_mask: {role_mask_str} "
                    f"| zero_obs_role_bits: {int(zero_bits)}"
                )
                def fmt(xs):
                    return "[" + " ".join(
                        f"{float(x):.4e}" if present else "NA"
                        for x, present in zip(np.asarray(xs), np.asarray(rpm))
                    ) + "]"
                def fmt_all(xs):
                    return "[" + " ".join(f"{float(x):.4e}" for x in np.asarray(xs)) + "]"

                if config["USE_ROLE_LORA"]:
                    msg += (
                        f" | lora_delta_norm_by_role: {fmt(ldn)}"
                        f" | lora_grad_norm_by_adapter: {fmt_all(lgn)}"
                        f" | lora_param_norm_by_adapter: {fmt_all(lpn)}"
                    )
                if config["LOG_ROLE_DIAGNOSTICS"]:
                    msg += (
                        f" | role_adv_mean: {fmt(ram)}"
                        f" | role_adv_std: {fmt(ras)}"
                        f" | role_entropy: {fmt(rent)}"
                        f" | role_approx_kl: {fmt(rkl)}"
                        f" | role_clip_frac: {fmt(rcf)}"
                        f" | role_ppo_loss: {fmt(rpl)}"
                    )
                if config["USE_ROLE_MACA"]:
                    msg += (
                        f" | role_maca_alpha: {float(role_maca_alpha):.4f}"
                        f" | role_maca_blend_schedule: {config['ROLE_MACA_BLEND_SCHEDULE']}"
                        f" | v_loss: {float(v_loss):.4e}"
                        f" | q_loss: {float(ql):.4e}"
                        f" | eq_loss: {float(eq_loss):.4e}"
                        f" | maca_critic_grad_norm: {float(qgn):.4e}"
                        f" | q_taken_mean_by_role: {fmt(qtm)}"
                        f" | eq_mean_by_role: {fmt(eqm)}"
                        f" | v_mean_by_role: {fmt(vm)}"
                        f" | return_eq_mean_by_role: {fmt(rem)}"
                        f" | return_eq_std_by_role: {fmt(res)}"
                        f" | baseline_mean_by_role: {fmt(bm)}"
                        f" | baseline_std_by_role: {fmt(bs)}"
                        f" | baseline_self_w_by_role: {fmt(bsw)}"
                        f" | baseline_group_w_by_role: {fmt(bgw)}"
                        f" | baseline_joint_w_by_role: {fmt(bjw)}"
                        f" | adv_maca_mean_by_role: {fmt(mam)}"
                        f" | adv_maca_std_by_role: {fmt(mas)}"
                        f" | attention_entropy: {float(att_entropy):.4e}"
                        f" | mean_corrset_size: {float(corrset_size):.4e}"
                    )
                print(msg)
                if config["LOG_ROLE_DIAGNOSTIC_TABLE"]:
                    print("role | count | adv_mean | adv_std | entropy | approx_kl | clip_frac | ppo_loss | lora_grad_norm")
                    lgn_arr = np.asarray(lgn)
                    for idx in range(len(np.asarray(rc))):
                        if not bool(np.asarray(rpm)[idx]):
                            print(f"{idx:4d} | {0:5d} | NA | NA | NA | NA | NA | NA | NA")
                            continue
                        adapter_grad_norm = float(lgn_arr[idx]) if idx < len(lgn_arr) else float("nan")
                        print(
                            f"{idx:4d} | {int(np.asarray(rc)[idx]):5d} "
                            f"| {float(np.asarray(ram)[idx]): .4e} "
                            f"| {float(np.asarray(ras)[idx]): .4e} "
                            f"| {float(np.asarray(rent)[idx]): .4e} "
                            f"| {float(np.asarray(rkl)[idx]): .4e} "
                            f"| {float(np.asarray(rcf)[idx]): .4e} "
                            f"| {float(np.asarray(rpl)[idx]): .4e} "
                            f"| {adapter_grad_norm: .4e}"
                        )

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.experimental.io_callback(
                log_callback, None,
                returns, win_rate, step_count,
                total_loss, entropy, actor_grad_norm, critic_grad_norm, q_loss_value, maca_critic_grad_norm,
                role_counts, role_present_mask, role_bit_sum_min, role_bit_sum_max,
                obs_role_mismatch_count, zero_obs_role_bits,
                lora_delta_norm_by_role, lora_grad_norm_by_role, lora_param_norm_by_role,
                role_adv_mean, role_adv_std, role_entropy, role_approx_kl, role_clip_frac,
                role_ppo_loss,
                eq_loss_value,
                v_loss_value,
                maca_info["q_taken_mean_by_role"],
                maca_info["eq_mean_by_role"],
                maca_info["v_mean_by_role"],
                maca_info["return_eq_mean_by_role"],
                maca_info["return_eq_std_by_role"],
                maca_info["baseline_mean_by_role"],
                maca_info["baseline_std_by_role"],
                maca_info["baseline_self_w_by_role"],
                maca_info["baseline_group_w_by_role"],
                maca_info["baseline_joint_w_by_role"],
                maca_info["adv_maca_mean_by_role"],
                maca_info["adv_maca_std_by_role"],
                maca_info["attention_entropy"],
                maca_info["corrset_size"],
                role_id_min, role_id_max,
                metric["role_maca_alpha"],
            )
            
            update_steps = update_steps + 1
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                rng_after_update,
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_states,
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
    config = default_config()
    args = parse_args()
    config = apply_cli_overrides(config, args)
    print_resolved_config(config)
    print(f"Starting {config['MAP_NAME']} ROSA-MAPPO experiment ({config['RUN_NAME']})...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    
    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    safe_run_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in config["RUN_NAME"])
    model_path = os.path.join(model_dir, f"{safe_run_name}_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "gru",
        "config": config,
        "actor_params": actor_params,
    }
    if config["USE_ROLE_MACA"]:
        final_maca_critic_state = final_train_states[2]
        checkpoint["role_maca_critic_params"] = jax.device_get(final_maca_critic_state.params)
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved ROSA-MAPPO actor checkpoint to {model_path}")
    
    # Can optionally save metrics
    # jnp.save("gru_baseline_metrics.npy", out["metric"])
