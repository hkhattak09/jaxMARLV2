""" Wrappers for use with jaxmarl baselines. """
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
from functools import partial

from typing import Dict, Optional, List, Tuple, Union
from jaxmarl_mpe_robotswarm.environments.spaces import Box, Discrete, MultiDiscrete
from jaxmarl_mpe_robotswarm.environments.multi_agent_env import MultiAgentEnv, State

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


@chex.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self._env.num_agents,)),
            episode_lengths=jnp.zeros((self._env.num_agents,)),
            returned_episode_returns=jnp.zeros((self._env.num_agents,)),
            returned_episode_lengths=jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


class MPELogWrapper(LogWrapper):
    """Times reward signal by number of agents to match on-policy codebase."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        rewardlog = jax.tree.map(lambda x: x * self._env.num_agents, reward)
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(rewardlog)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


def get_space_dim(space):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, (Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        raise NotImplementedError(
            'Wrapper works only with Discrete/MultiDiscrete/Box spaces'
        )


class CTRolloutManager(JaxMARLWrapper):
    """
    Rollout Manager for Centralized Training with Parameter Sharing.
    - Batchifies multiple environments.
    - Adds global state (obs["__all__"]) and global reward (rewards["__all__"]).
    - Pads observations to equal length.
    - Adds one-hot agent id to observations.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        batch_size: int,
        training_agents: List = None,
        preprocess_obs: bool = True,
    ):
        super().__init__(env)
        self.batch_size = batch_size
        self.training_agents = self.agents if training_agents is None else training_agents
        self.preprocess_obs = preprocess_obs

        self.batch_samplers = {
            agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0))
            for agent in self.agents
        }

        self.max_obs_length = max(
            get_space_dim(s) for s in self.observation_spaces.values()
        )
        self.max_action_space = max(
            get_space_dim(s) for s in self.action_spaces.values()
        )
        self.obs_size = self.max_obs_length
        if self.preprocess_obs:
            self.obs_size += len(self.agents)

        self.agents_one_hot = {
            a: oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))
        }
        self.valid_actions = {a: jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh = {
            a: jnp.concatenate(
                (jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n))
            )
            for a, u in self.action_spaces.items()
        }

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs_, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.tree.map(
                self._preprocess_obs,
                {agent: obs_[agent] for agent in self.agents},
                self.agents_one_hot,
            )
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs_, state, reward, done, infos = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.tree.map(
                self._preprocess_obs,
                {agent: obs_[agent] for agent in self.agents},
                self.agents_one_hot,
            )
            obs = jax.tree.map(
                lambda d, o: jnp.where(d, 0.0, o),
                {agent: done[agent] for agent in self.agents},
                obs,
            )
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs, state):
        return jnp.concatenate([obs[agent] for agent in self.agents], axis=-1)

    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack([reward[agent] for agent in self.training_agents]).sum(axis=0)

    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size)).astype(int)

    @partial(jax.jit, static_argnums=0)
    def get_valid_actions(self, state):
        return {
            agent: jnp.tile(actions, self.batch_size).reshape(self.batch_size, -1)
            for agent, actions in self.valid_actions_oh.items()
        }

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        arr = arr.flatten()
        pad_width = [(0, 0)] * (arr.ndim - 1) + [
            (0, max(0, self.max_obs_length - arr.shape[-1]))
        ]
        arr = jnp.pad(arr, pad_width, mode='constant', constant_values=0)
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
