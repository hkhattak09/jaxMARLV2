"""JAX Assembly Adapter — drop-in replacement for AssemblySwarmWrapper.

Wraps AssemblyEnv (JAX/GPU) and exposes the same API as AssemblySwarmWrapper
so the PyTorch MADDPG training loop requires only minimal changes.

Key design:
  • N parallel environments run as a single vmapped GPU kernel.
  • JAX DeviceArrays are converted to NumPy (D2H) once per step, so the
    existing CPU-side PyTorch rollout path stays unchanged.
  • Prior actions come from the JAX robot_policy (Reynolds flocking) which
    runs on-device before the D2H copy, adding zero extra PCIe traffic.

I/O conventions (match original AssemblySwarmWrapper / AssemblySwarmEnv):
  reset()  → obs           (obs_dim, n_envs*n_a)  np.float32
  step(a)  → obs, rew, done, info, a_prior
      a:       (n_envs*n_a, 2)  — format produced by maddpg.step
      obs:     (obs_dim,  n_envs*n_a)  np.float32
      rew:     (1,        n_envs*n_a)  np.float32
      done:    (1,        n_envs*n_a)  np.bool_
      a_prior: (2,        n_envs*n_a)  np.float32  ← (act_dim, n_agents)
"""

import sys
import os

import jax
import jax.numpy as jnp
import numpy as np

# Allow importing jaxmarl from the sibling JaxMARL directory when this module
# is used inside MARL-LLM without a package install.
_JAXMARL_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../..",  # repo root
    "JaxMARL",
)
if os.path.isdir(_JAXMARL_DIR) and _JAXMARL_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_JAXMARL_DIR))

from jaxmarl.environments.mpe.assembly import AssemblyEnv, AssemblyState  # noqa: E402


class _DummySpace:
    """Minimal gym.Space duck-type to satisfy MADDPG.init_from_env shape queries."""

    def __init__(self, shape):
        self.shape = shape


class JaxAssemblyAdapter:
    """Drop-in replacement for AssemblySwarmWrapper backed by the JAX environment.

    Args:
        jax_env:  Constructed AssemblyEnv instance (with results_file already loaded).
        n_envs:   Number of parallel environments to run via jax.vmap.
        seed:     Base random seed for PRNG key initialisation.
        alpha:    Initial regularisation coefficient (read by MADDPG update).
    """

    def __init__(
        self,
        jax_env: AssemblyEnv,
        n_envs: int = 1,
        seed: int = 0,
        alpha: float = 1.0,
    ):
        self.env = jax_env
        self.n_envs = n_envs
        self._n_a_per_env = jax_env.n_a

        # Exposed as total agents so training loop slices work correctly:
        #   start_stop_num = [slice(0, env.n_a)]
        self.n_a = n_envs * jax_env.n_a
        self.num_agents = self.n_a

        # MADDPG uses agent_types to decide how many policy networks to create.
        # One homogeneous team — keeps MADDPG.nagents == 1.
        self.agents = [_DummyAgent() for _ in range(self.num_agents)]
        self.agent_types = ["agent"]

        # Regularisation coefficient read by maddpg.update
        self.alpha = alpha

        # Spaces: shape[0] is used by MADDPG for network input/output dims
        self.observation_space = _DummySpace((jax_env.obs_dim, self.num_agents))
        self.action_space = _DummySpace((2, self.num_agents))

        # PRNG key (owned by adapter, advanced each call)
        self._key = jax.random.PRNGKey(seed)
        self._states: AssemblyState = None  # set by reset()
        self._agent_list = jax_env.agents  # Cache agent list

        # JIT-compile vmapped functions once at construction time
        if n_envs == 1:
            self._jit_reset = jax.jit(jax_env.reset)
            self._jit_step = jax.jit(jax_env.step_env)
            self._jit_prior = jax.jit(jax_env.robot_policy)
            
            # JIT-compiled output conversion (single env)
            @jax.jit
            def _convert_outputs(obs_dict, rew_dict, done_dict, prior):
                """Convert dicts to stacked arrays in one JIT call."""
                obs = jnp.stack([obs_dict[a] for a in self._agent_list], axis=0).T  # [obs_dim, n_a]
                rew = jnp.stack([rew_dict[a] for a in self._agent_list])[None, :]   # [1, n_a]
                done = jnp.stack([done_dict[a] for a in self._agent_list])[None, :] # [1, n_a]
                prior_out = prior.T  # [2, n_a]
                return obs, rew, done, prior_out
            self._jit_convert = _convert_outputs
        else:
            self._jit_reset = jax.jit(jax.vmap(jax_env.reset))
            self._jit_step = jax.jit(jax.vmap(jax_env.step_env))
            self._jit_prior = jax.jit(jax.vmap(jax_env.robot_policy))
            
            # JIT-compiled output conversion (batched envs)
            @jax.jit
            def _convert_outputs_batched(obs_dict, rew_dict, done_dict, prior):
                """Convert dicts to stacked arrays in one JIT call."""
                obs = jnp.stack([obs_dict[a] for a in self._agent_list], axis=1)  # [N, n_a, obs_dim]
                obs_flat = obs.reshape(n_envs * jax_env.n_a, jax_env.obs_dim).T   # [obs_dim, N*n_a]
                rew = jnp.stack([rew_dict[a] for a in self._agent_list], axis=1)  # [N, n_a]
                rew_flat = rew.reshape(1, -1)                                      # [1, N*n_a]
                done = jnp.stack([done_dict[a] for a in self._agent_list], axis=1) # [N, n_a]
                done_flat = done.reshape(1, -1)                                    # [1, N*n_a]
                prior_flat = prior.reshape(n_envs * jax_env.n_a, 2).T              # [2, N*n_a]
                return obs_flat, rew_flat, done_flat, prior_flat
            self._jit_convert = _convert_outputs_batched

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset all parallel environments.

        Returns:
            obs: (obs_dim, n_envs*n_a)  NumPy float32
        """
        keys = jax.random.split(self._key, self.n_envs + 1)
        self._key = keys[0]
        env_keys = keys[1:]  # [n_envs, 2]

        if self.n_envs == 1:
            obs_dict, self._states = self._jit_reset(env_keys[0])
        else:
            obs_dict, self._states = self._jit_reset(env_keys)

        return self._obs_dict_to_np(obs_dict)

    def step(self, actions: np.ndarray):
        """Step all parallel environments.

        Args:
            actions: (n_envs*n_a, 2)  NumPy — as produced by maddpg.step.

        Returns:
            obs:     (obs_dim,  n_envs*n_a)  np.float32
            rew:     (1,        n_envs*n_a)  np.float32
            done:    (1,        n_envs*n_a)  np.bool_
            info:    {}  (empty dict, kept for API compatibility)
            a_prior: (2,        n_envs*n_a)  np.float32
        """
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)  # (n_envs*n_a, 2)

        if self.n_envs == 1:
            actions_dict = {
                a: actions_jax[i] for i, a in enumerate(self.env.agents)
            }
            key, step_key = jax.random.split(self._key)
            self._key = key
            obs_dict, new_state, rew_dict, done_dict, _ = self._jit_step(
                step_key, self._states, actions_dict
            )
            self._states = new_state
            a_prior_jax = self._jit_prior(new_state)  # [n_a, 2]
        else:
            # Reshape flat (N*n_a, 2) → (N, n_a, 2) and build per-env action dicts
            actions_reshaped = actions_jax.reshape(
                self.n_envs, self._n_a_per_env, 2
            )  # [N, n_a, 2]
            actions_dict = {
                a: actions_reshaped[:, i, :]
                for i, a in enumerate(self.env.agents)
            }  # {agent_i: [N, 2]}

            keys = jax.random.split(self._key, self.n_envs + 1)
            self._key = keys[0]
            step_keys = keys[1:]  # [N, 2]

            obs_dict, new_states, rew_dict, done_dict, _ = self._jit_step(
                step_keys, self._states, actions_dict
            )
            self._states = new_states
            a_prior_jax = self._jit_prior(new_states)  # [N, n_a, 2]

        # JIT-compiled conversion: dicts → stacked arrays (single JAX dispatch)
        obs_jax, rew_jax, done_jax, prior_jax = self._jit_convert(
            obs_dict, rew_dict, done_dict, a_prior_jax
        )
        
        # JAX → NumPy
        obs_np = np.asarray(obs_jax)
        rew_np = np.asarray(rew_jax)
        done_np = np.asarray(done_jax)
        prior_np = np.asarray(prior_jax)

        return obs_np, rew_np, done_np, {}, prior_np

    def render(self, *args, **kwargs):
        """No-op — JAX env has no renderer yet."""
        pass

    # ── Evaluation metrics (proxy onto JAX env methods) ───────────────────

    def coverage_rate(self) -> float:
        if self.n_envs == 1:
            return float(self.env.coverage_rate(self._states))
        return float(jnp.mean(jax.vmap(self.env.coverage_rate)(self._states)))

    def distribution_uniformity(self) -> float:
        if self.n_envs == 1:
            return float(self.env.distribution_uniformity(self._states))
        return float(jnp.mean(jax.vmap(self.env.distribution_uniformity)(self._states)))

    def voronoi_based_uniformity(self) -> float:
        if self.n_envs == 1:
            return float(self.env.voronoi_based_uniformity(self._states))
        return float(jnp.mean(jax.vmap(self.env.voronoi_based_uniformity)(self._states)))


class _DummyAgent:
    """Minimal agent object for API compatibility (adversary flag only)."""

    def __init__(self):
        self.adversary = False
