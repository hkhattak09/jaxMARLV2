from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper
from smax_ctm.ctm_jax import CTMCell

from smax_ctm.analysis.policy_head import choose_actions, policy_from_synch


def _stack_agent_array(values_by_agent: Dict[str, jnp.ndarray], agents: List[str]) -> jnp.ndarray:
    missing = [a for a in agents if a not in values_by_agent]
    if missing:
        raise KeyError(f"Missing agent keys in dict: {missing}")
    return jnp.stack([values_by_agent[a] for a in agents])


def _get_avail_actions(env: Any, env_state: Any) -> jnp.ndarray:
    avail_actions = env.get_avail_actions(env_state)
    if isinstance(avail_actions, dict):
        return _stack_agent_array(avail_actions, env.agents)
    return avail_actions


def _extract_raw_smax_state(log_state: Any) -> Any:
    if not hasattr(log_state, "env_state"):
        raise AttributeError("Expected log_state with env_state field (SMAXLogWrapper state)")

    env_state = log_state.env_state
    if hasattr(env_state, "state"):
        return env_state.state
    if hasattr(env_state, "unit_health") and hasattr(env_state, "unit_alive"):
        return env_state

    raise AttributeError(
        "Could not find raw SMAX state. Expected env_state.state or env_state with unit fields."
    )


def _extract_terminal_win_flag(infos: Dict[str, Any], agents: List[str]) -> Optional[bool]:
    # Different wrappers may expose slightly different won keys.
    candidate_keys = (
        "won",
        "won_episode",
        "returned_won_episode",
        "battle_won",
    )
    for agent in agents:
        info = infos.get(agent, None)
        if not isinstance(info, dict):
            continue
        for key in candidate_keys:
            if key in info:
                value = np.asarray(jax.device_get(info[key]))
                if value.size != 1:
                    raise ValueError(
                        f"Expected scalar win flag for key '{key}', got shape {value.shape}"
                    )
                return bool(value.reshape(()).item())
    return None


def _pairwise_distances(xy: np.ndarray) -> np.ndarray:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected positions with shape (N, 2), got {xy.shape}")
    deltas = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=-1))


def _is_focus_fire(
    prev_attack_actions: np.ndarray,
    num_allies: int,
    num_enemies: int,
    num_movement_actions: int,
) -> bool:
    # prev_attack_actions[i] is 0 if ally i was moving, else the full action index
    # (>= num_movement_actions). Attack actions for enemies are indexed as
    # num_movement_actions + k for enemy k, NOT num_allies + k.
    ally_targets = prev_attack_actions[:num_allies]
    min_attack_idx = num_movement_actions
    max_attack_idx = num_movement_actions + num_enemies - 1
    valid_targets = ally_targets[(ally_targets >= min_attack_idx) & (ally_targets <= max_attack_idx)]
    if valid_targets.size < 2:
        return False
    _, counts = np.unique(valid_targets, return_counts=True)
    return bool(np.any(counts >= 2))


def _is_grouping(ally_positions: np.ndarray, ally_alive: np.ndarray, grouping_radius: float) -> bool:
    alive_positions = ally_positions[ally_alive.astype(bool)]
    if alive_positions.shape[0] < 2:
        return False
    d = _pairwise_distances(alive_positions)
    triu = d[np.triu_indices(alive_positions.shape[0], k=1)]
    if triu.size == 0:
        return False
    return bool(np.mean(triu) <= grouping_radius)


def build_eval_env(config: Dict[str, Any], map_name: Optional[str]) -> Any:
    selected_map = map_name or config.get("MAP_NAME")
    if not selected_map:
        raise ValueError("Could not determine map name from args or checkpoint config")

    scenario = map_name_to_scenario(selected_map)
    env_kwargs = config.get("ENV_KWARGS", {})
    env = HeuristicEnemySMAX(scenario=scenario, **env_kwargs)
    env = SMAXLogWrapper(env)
    return env


def collect_episodes(
    config: Dict[str, Any],
    ctm_params: Dict[str, Any],
    head_params: Dict[str, Dict[str, jnp.ndarray]],
    num_episodes: int,
    seed: int,
    map_name: Optional[str],
    max_steps: Optional[int],
    stochastic: bool,
    grouping_radius: float,
) -> Dict[str, Any]:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be > 0, got {num_episodes}")
    if grouping_radius <= 0.0:
        raise ValueError(f"grouping_radius must be > 0, got {grouping_radius}")

    env = build_eval_env(config, map_name)
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n
    num_movement_actions = int(env.num_movement_actions)
    ctm_cell = CTMCell(
        d_model=config["CTM_D_MODEL"],
        d_input=config["CTM_D_INPUT"],
        memory_length=config["CTM_MEMORY_LENGTH"],
        n_synch_out=config["CTM_N_SYNCH_OUT"],
        iterations=config["CTM_ITERATIONS"],
        deep_nlms=config["CTM_DEEP_NLMS"],
        memory_hidden_dims=config["CTM_NLM_HIDDEN_DIM"],
        obs_dim=obs_dim,
        neuron_select_type=config.get("CTM_NEURON_SELECT", "first-last"),
        do_layernorm_nlm=config.get("CTM_DO_LAYERNORM_NLM", False),
    )

    synch_size = config["CTM_N_SYNCH_OUT"] * (config["CTM_N_SYNCH_OUT"] + 1) // 2
    if "Dense_2" not in head_params or "kernel" not in head_params["Dense_2"]:
        raise KeyError("head_params missing Dense_2/kernel required for action_dim check")
    dense2_out = int(head_params["Dense_2"]["kernel"].shape[1])
    if dense2_out != action_dim:
        raise ValueError(f"Head output dim ({dense2_out}) must match action dim ({action_dim})")

    rng = jax.random.PRNGKey(seed)
    episodes: List[Dict[str, Any]] = []
    max_rollout_steps = int(max_steps) if max_steps is not None else int(env.max_steps)

    for ep_idx in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)

        carry = CTMCell.initialize_carry(
            batch_size=env.num_agents,
            d_model=config["CTM_D_MODEL"],
            memory_length=config["CTM_MEMORY_LENGTH"],
        )

        done_batch = jnp.ones((env.num_agents,), dtype=bool)
        prev_enemy_alive: Optional[np.ndarray] = None
        episode_won: Optional[bool] = None
        episode_return = np.zeros((env.num_agents,), dtype=np.float64)
        step_records: List[Dict[str, Any]] = []

        for t in range(max_rollout_steps):
            obs_batch = _stack_agent_array(obs, env.agents)
            avail_batch = _get_avail_actions(env, state.env_state)

            if obs_batch.shape != (env.num_agents, obs_dim):
                raise ValueError(f"Unexpected obs shape at step {t}: {obs_batch.shape}")
            if avail_batch.shape != (env.num_agents, action_dim):
                raise ValueError(f"Unexpected avail_actions shape at step {t}: {avail_batch.shape}")

            carry, synch = ctm_cell.apply(ctm_params, carry, (obs_batch, done_batch, avail_batch))
            if synch.shape != (env.num_agents, synch_size):
                raise ValueError(f"Unexpected synch shape at step {t}: {synch.shape}")

            rng, act_rng = jax.random.split(rng)
            pi = policy_from_synch(synch, avail_batch, head_params)
            actions, action_log_probs = choose_actions(pi, act_rng, stochastic=stochastic)

            action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}

            rng, step_rng = jax.random.split(rng)
            next_obs, next_state, rewards, dones, infos = env.step(step_rng, state, action_dict)
            reward_batch = _stack_agent_array(rewards, env.agents)
            episode_return += np.asarray(jax.device_get(reward_batch), dtype=np.float64)

            raw_state = _extract_raw_smax_state(next_state)
            all_health = np.asarray(jax.device_get(raw_state.unit_health), dtype=np.float64)
            all_alive = np.asarray(jax.device_get(raw_state.unit_alive), dtype=bool)
            all_positions = np.asarray(jax.device_get(raw_state.unit_positions), dtype=np.float64)
            all_prev_attack = np.asarray(jax.device_get(raw_state.prev_attack_actions), dtype=np.int32)

            num_allies = env.num_agents
            num_enemies = env.num_enemies
            ally_health = all_health[:num_allies]
            ally_alive = all_alive[:num_allies]
            ally_positions = all_positions[:num_allies]
            enemy_alive = all_alive[num_allies : num_allies + num_enemies]

            enemy_kill = False
            if prev_enemy_alive is not None:
                enemy_kill = bool(np.any(prev_enemy_alive & (~enemy_alive)))
            prev_enemy_alive = enemy_alive

            focus_fire = _is_focus_fire(
                all_prev_attack,
                num_allies=num_allies,
                num_enemies=num_enemies,
                num_movement_actions=num_movement_actions,
            )
            grouping = _is_grouping(ally_positions, ally_alive, grouping_radius=grouping_radius)

            step_records.append(
                {
                    "t": int(t),
                    "synch": np.asarray(jax.device_get(synch)),
                    "state_trace": np.asarray(jax.device_get(carry[0])),
                    "activated_trace": np.asarray(jax.device_get(carry[1])),
                    "obs": np.asarray(jax.device_get(obs_batch)),
                    "avail_actions": np.asarray(jax.device_get(avail_batch)),
                    "actions": np.asarray(jax.device_get(actions)),
                    "action_log_prob": np.asarray(jax.device_get(action_log_probs)),
                    "rewards": np.asarray(jax.device_get(reward_batch)),
                    "ally_health": ally_health,
                    "ally_alive": ally_alive,
                    "ally_positions": ally_positions,
                    "events": {
                        "focus_fire": focus_fire,
                        "grouping": grouping,
                        "enemy_kill": enemy_kill,
                        "episode_done": bool(dones["__all__"]),
                    },
                }
            )

            done_batch = jnp.array([dones[a] for a in env.agents], dtype=bool)
            obs, state = next_obs, next_state
            if dones["__all__"]:
                episode_won = _extract_terminal_win_flag(infos, env.agents)
                break

        episodes.append(
            {
                "episode_index": ep_idx,
                "num_steps": len(step_records),
                "episode_return_per_agent": episode_return,
                "episode_return_mean": float(np.mean(episode_return)),
                "episode_won": episode_won,
                "steps": step_records,
            }
        )

    return {
        "metadata": {
            "map_name": map_name or config.get("MAP_NAME", "unknown"),
            "num_episodes": num_episodes,
            "num_agents": int(env.num_agents),
            "num_enemies": int(env.num_enemies),
            "num_movement_actions": num_movement_actions,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "synch_size": int(synch_size),
            "grouping_radius": float(grouping_radius),
            "stochastic": bool(stochastic),
            "seed": int(seed),
            "max_rollout_steps": int(max_rollout_steps),
        },
        "episodes": episodes,
    }
