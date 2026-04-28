from functools import partial

import jax
import jax.numpy as jnp
from jaxmarl.environments.smax import HeuristicEnemySMAX
from jaxmarl.wrappers.baselines import JaxMARLWrapper


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
        return jnp.all(~ally_alive) | jnp.all(~enemy_alive)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        # Use step_env to access pre-auto-reset state for timeout detection.
        step_key, reset_key = jax.random.split(key)
        obs_st, state_st, reward, done, info = self._env.step_env(step_key, state, action)

        # Compute bad_transition before losing stepped state.
        raw_step_state = state_st.state
        battle_done = self._battle_terminal(raw_step_state)
        timeout_done = raw_step_state.time >= self._env.max_steps
        bad_transition = done["__all__"] & timeout_done & ~battle_done
        info["bad_transition"] = jnp.full((self._env.num_allies,), bad_transition)

        # Manual auto-reset matching MultiAgentEnv.step.
        obs_reset, state_reset = self._env.reset(reset_key)
        env_state = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), state_reset, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), obs_reset, obs_st
        )

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
