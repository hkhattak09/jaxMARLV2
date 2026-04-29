#!/usr/bin/env python3
"""Standalone deterministic evaluator for MAPPO-T and MAPPO-T LoRASA checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
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

from mappo_t import ActorTrans, LoRASAActorTrans, ScannedRNN
from mappo_t.utils import batchify, unbatchify


SUPPORTED_MODEL_TYPES = ("mappo_t", "mappo_t_backbone", "mappo_t_lorasa")
SUPPORTED_ABLATION_MODES = ("full", "no_recurrent_lora", "no_gru_lora", "mlp_only_lora")


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides MACA-style local observations and per-agent world_state.

    Duplicated from training scripts to keep the evaluator standalone.
    """

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


def load_checkpoint(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict (got {type(ckpt).__name__})")
    return ckpt


def validate_checkpoint(ckpt: dict, path: str) -> None:
    for key in ("model_type", "config", "actor_params"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint {path} is missing required key: {key!r}")

    model_type = ckpt["model_type"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Checkpoint {path} has unsupported model_type {model_type!r}. "
            f"Supported: {SUPPORTED_MODEL_TYPES}"
        )

    if model_type == "mappo_t_lorasa":
        lorasa = ckpt.get("lorasa")
        if lorasa is None:
            raise ValueError(
                f"LoRASA checkpoint {path} is missing 'lorasa' metadata"
            )
        for subkey in ("rank", "num_adapter_slots", "ablation_mode"):
            if subkey not in lorasa:
                raise ValueError(
                    f"LoRASA checkpoint {path} is missing lorasa.{subkey}"
                )
        if lorasa["ablation_mode"] not in SUPPORTED_ABLATION_MODES:
            raise ValueError(
                f"LoRASA checkpoint {path} has unsupported ablation_mode "
                f"{lorasa['ablation_mode']!r}. Supported: {SUPPORTED_ABLATION_MODES}"
            )


def build_env(map_name: str, config: dict, max_steps: Optional[int] = None):
    try:
        scenario = map_name_to_scenario(map_name)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid map_name: {map_name!r}") from e

    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps

    env = HeuristicEnemySMAX(scenario=scenario, **env_kwargs)

    env = SMAXWorldStateWrapper(
        env,
        obs_with_agent_id=config.get("OBS_WITH_AGENT_ID", True),
        local_obs_with_agent_id=config.get(
            "LOCAL_OBS_WITH_AGENT_ID", config.get("OBS_WITH_AGENT_ID", True)
        ),
    )
    env = SMAXLogWrapper(env)
    return env


def validate_actor_params(init_params, loaded_params, label: str) -> None:
    init_flat = flatten_dict(init_params)
    loaded_flat = flatten_dict(loaded_params)

    missing_in_loaded = set(init_flat.keys()) - set(loaded_flat.keys())
    extra_in_loaded = set(loaded_flat.keys()) - set(init_flat.keys())

    if missing_in_loaded:
        raise ValueError(
            f"{label}: loaded params missing keys present in initialised params: "
            f"{missing_in_loaded}"
        )
    if extra_in_loaded:
        raise ValueError(
            f"{label}: loaded params have extra keys not in initialised params: "
            f"{extra_in_loaded}"
        )
    for key, value in loaded_flat.items():
        if value.shape != init_flat[key].shape:
            raise ValueError(
                f"{label}: param {key} shape {value.shape} != initialised "
                f"shape {init_flat[key].shape}"
            )


def make_eval_fn(env, actor_network, actor_hidden_dim, eval_steps, num_envs, num_loops, is_lorasa):
    """Create a JIT-compiled eval function for the given actor type."""

    num_agents = env.num_agents

    if is_lorasa:

        def _actor_apply(actor_params, ac_hstate, ac_in, env_state):
            unit_types_env = env_state.env_state.state.unit_types[:, :num_agents]
            adapter_ids = unit_types_env.T.reshape(num_envs * num_agents)
            adapter_ids = adapter_ids.astype(jnp.int32)
            ac_hstate, pi = actor_network.apply(
                actor_params, ac_hstate, ac_in, adapter_ids[None, :]
            )
            return ac_hstate, pi

    else:

        def _actor_apply(actor_params, ac_hstate, ac_in, env_state):
            ac_hstate, pi = actor_network.apply(actor_params, ac_hstate, ac_in)
            return ac_hstate, pi

    def eval_fn(rng, actor_params):
        def _outer_loop(rng, _):
            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

            ac_hstate = ScannedRNN.initialize_carry(
                num_envs * num_agents, actor_hidden_dim
            )
            last_done = jnp.zeros((num_envs * num_agents,), dtype=bool)
            already_recorded = jnp.zeros((num_envs,), dtype=bool)
            episode_won = jnp.zeros((num_envs,), dtype=jnp.float32)

            def _step(carry, _):
                (
                    env_state,
                    obsv,
                    ac_hstate,
                    last_done,
                    already_recorded,
                    episode_won,
                    rng,
                ) = carry
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_envs * num_agents)
                )
                obs_batch = batchify(obsv, env.agents, num_envs * num_agents)
                ac_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                    avail_actions[None, :],
                )

                ac_hstate, pi = _actor_apply(actor_params, ac_hstate, ac_in, env_state)

                action = jnp.argmax(pi.logits, axis=-1).squeeze(0)

                env_act = unbatchify(
                    action, env.agents, num_envs, num_envs * num_agents
                )
                env_act = {k: v.squeeze(-1) for k, v in env_act.items()}

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_act)

                done_env = done["__all__"]
                won_now = info["returned_won_episode"][:, 0]
                new_record = done_env & ~already_recorded
                episode_won = jnp.where(new_record, won_now, episode_won)
                already_recorded = already_recorded | new_record

                last_done = jnp.tile(done["__all__"], num_agents)

                return (
                    env_state,
                    obsv,
                    ac_hstate,
                    last_done,
                    already_recorded,
                    episode_won,
                    rng,
                ), None

            init_carry = (
                env_state,
                obsv,
                ac_hstate,
                last_done,
                already_recorded,
                episode_won,
                rng,
            )
            final_carry, _ = jax.lax.scan(_step, init_carry, None, eval_steps)
            (
                _,
                _,
                _,
                _,
                already_recorded,
                episode_won,
                rng,
            ) = final_carry
            episode_won = jnp.where(already_recorded, episode_won, 0.0)
            return rng, episode_won

        _, wins_by_loop = jax.lax.scan(_outer_loop, rng, None, num_loops)
        return wins_by_loop

    return jax.jit(eval_fn)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone deterministic evaluator for MAPPO-T / MAPPO-T LoRASA checkpoints."
    )
    parser.add_argument(
        "--map_name", required=True, help="SMAX map name, e.g. 'protoss_10_vs_10'"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint .pkl files",
    )
    parser.add_argument(
        "--num_envs", type=int, default=32, help="Parallel envs per loop (default: 32)"
    )
    parser.add_argument(
        "--num_loops", type=int, default=20, help="Number of eval batches (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (default: 0)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override env max_steps / eval horizon",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display names for checkpoints (same length as --checkpoints)",
    )
    parser.add_argument(
        "--show_loops",
        action="store_true",
        help="Print per-loop stats for each checkpoint",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV summary path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_envs < 1:
        raise ValueError(f"--num_envs must be >= 1, got {args.num_envs}")
    if args.num_loops < 1:
        raise ValueError(f"--num_loops must be >= 1, got {args.num_loops}")

    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        raise ValueError(
            f"--labels has {len(args.labels)} entries but --checkpoints has "
            f"{len(args.checkpoints)} entries; they must be the same length."
        )

    checkpoints = []
    for path in args.checkpoints:
        ckpt = load_checkpoint(path)
        validate_checkpoint(ckpt, path)
        checkpoints.append((path, ckpt))

    all_results = []

    for idx, (path, ckpt) in enumerate(checkpoints):
        model_type = ckpt["model_type"]
        config = ckpt["config"]
        actor_params = ckpt["actor_params"]

        env = build_env(args.map_name, config, max_steps=args.max_steps)

        eval_steps = (
            args.max_steps
            if args.max_steps is not None
            else config.get("ENV_KWARGS", {}).get("max_steps", config.get("NUM_STEPS", 200))
        )

        action_dim = env.action_space(env.agents[0]).n
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        actor_hidden_dim = config["hidden_sizes"][-1]

        if "transformer" not in config:
            config["transformer"] = {}
        config["transformer"]["n_block"] = env.num_agents

        is_lorasa = model_type == "mappo_t_lorasa"

        if is_lorasa:
            lorasa = ckpt["lorasa"]
            rank = lorasa["rank"]
            num_adapter_slots = lorasa["num_adapter_slots"]
            ablation_mode = lorasa["ablation_mode"]
            init_scale = lorasa.get("init_scale", config.get("LORASA_INIT_SCALE", 0.01))

            actor_network = LoRASAActorTrans(
                action_dim=action_dim,
                config=config,
                num_adapter_slots=num_adapter_slots,
                rank=rank,
                init_scale=init_scale,
                ablation_mode=ablation_mode,
            )
        else:
            actor_network = ActorTrans(action_dim=action_dim, config=config)

        rng = jax.random.PRNGKey(0)
        init_ac_hstate = ScannedRNN.initialize_carry(1, actor_hidden_dim)
        init_ac_in = (
            jnp.zeros((1, 1, obs_dim), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=bool),
            jnp.ones((1, 1, action_dim), dtype=jnp.float32),
        )

        if is_lorasa:
            init_adapter_ids = jnp.zeros((1, 1), dtype=jnp.int32)
            init_params = actor_network.init(rng, init_ac_hstate, init_ac_in, init_adapter_ids)
        else:
            init_params = actor_network.init(rng, init_ac_hstate, init_ac_in)

        label = os.path.basename(path) if args.labels is None else args.labels[idx]
        validate_actor_params(init_params, actor_params, label)

        eval_fn = make_eval_fn(
            env, actor_network, actor_hidden_dim, eval_steps, args.num_envs, args.num_loops, is_lorasa
        )

        rng = jax.random.PRNGKey(args.seed)
        print(f"\nEvaluating {label} ({model_type}) on {args.map_name} ...")
        wins_by_loop = np.array(eval_fn(rng, actor_params))

        loop_wr = wins_by_loop.mean(axis=1)
        loop_std = (
            wins_by_loop.std(axis=1, ddof=1) if args.num_envs > 1
            else np.zeros(args.num_loops)
        )

        all_wins = wins_by_loop.reshape(-1)
        episodes = int(args.num_loops * args.num_envs)
        mean_wr = float(all_wins.mean())
        std = float(all_wins.std(ddof=1)) if episodes > 1 else 0.0
        sem = std / math.sqrt(episodes) if episodes > 1 else 0.0

        if args.show_loops:
            print(f"\ncheckpoint: {path}")
            print(f"{'loop':<8}{'win_rate':<12}{'std':<10}")
            for i in range(args.num_loops):
                print(f"{i + 1:<8}{loop_wr[i]:<12.4f}{loop_std[i]:<10.4f}")

        all_results.append({
            "label": label,
            "path": path,
            "mean_wr": mean_wr,
            "std": std,
            "sem": sem,
            "episodes": episodes,
            "model_type": model_type,
        })

    print(f"\n{'model':<22}{'mean_wr':<10}{'std':<9}{'sem':<9}{'episodes':<10}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['label']:<22}{r['mean_wr']:<10.4f}{r['std']:<9.4f}"
            f"{r['sem']:<9.4f}{r['episodes']:<10}"
        )

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label", "checkpoint", "map_name", "num_envs", "num_loops",
                "episodes", "mean_wr", "std", "sem",
            ])
            for r in all_results:
                writer.writerow([
                    r["label"],
                    r["path"],
                    args.map_name,
                    args.num_envs,
                    args.num_loops,
                    r["episodes"],
                    f"{r['mean_wr']:.6f}",
                    f"{r['std']:.6f}",
                    f"{r['sem']:.6f}",
                ])
        print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
