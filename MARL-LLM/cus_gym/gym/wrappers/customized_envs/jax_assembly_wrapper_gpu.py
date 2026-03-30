"""JAX Assembly Adapter with GPU-only DLPack transfers.

GPU-optimized version that keeps all tensors on GPU and uses DLPack for
zero-copy sharing between JAX and PyTorch. No CPU transfers during rollout.

Key differences from jax_assembly_wrapper.py:
  • All data stays on GPU (JAX and PyTorch share same device)
  • DLPack zero-copy transfers (no PCIe overhead)
  • Returns PyTorch CUDA tensors instead of NumPy arrays
  • Replay buffer must handle GPU tensors

Requirements:
  • Single GPU setup (cuda:0 for both JAX and PyTorch)
  • JAX 0.7.2+ with GPU support
  • PyTorch with CUDA enabled

I/O conventions (GPU tensors throughout):
  reset()  → obs_torch         (obs_dim, n_envs*n_a)  torch.cuda.FloatTensor
  step(a)  → obs, rew, done, info, a_prior (all torch.cuda tensors)
      a:       (n_envs*n_a, 2)  torch.cuda.FloatTensor
      obs:     (obs_dim,  n_envs*n_a)  torch.cuda.FloatTensor
      rew:     (1,        n_envs*n_a)  torch.cuda.FloatTensor
      done:    (1,        n_envs*n_a)  torch.cuda.BoolTensor
      a_prior: (2,        n_envs*n_a)  torch.cuda.FloatTensor
"""

import sys
import os

import jax
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack

_JAXMARL_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../../..",
    "JaxMARL",
)
if os.path.isdir(_JAXMARL_DIR) and _JAXMARL_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_JAXMARL_DIR))

from jaxmarl.environments.mpe.assembly import AssemblyEnv, AssemblyState


class _DummySpace:
    """Minimal gym.Space duck-type to satisfy MADDPG.init_from_env shape queries."""
    def __init__(self, shape):
        self.shape = shape


class JaxAssemblyAdapterGPU:
    """GPU-optimized adapter using DLPack for zero-copy JAX ↔ PyTorch transfers.
    
    All data stays on GPU throughout rollout. No CPU transfers.
    
    Args:
        jax_env:  Constructed AssemblyEnv instance.
        n_envs:   Number of parallel environments to run via jax.vmap.
        seed:     Base random seed for PRNG key initialisation.
        alpha:    Initial regularisation coefficient.
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

        self.n_a = n_envs * jax_env.n_a
        self.num_agents = self.n_a

        self.agents = [_DummyAgent() for _ in range(self.num_agents)]
        self.agent_types = ["agent"]

        self.alpha = alpha

        self.observation_space = _DummySpace((jax_env.obs_dim, self.num_agents))
        self.action_space = _DummySpace((2, self.num_agents))

        self._key = jax.random.PRNGKey(seed)
        self._states: AssemblyState = None
        self._agent_list = jax_env.agents  # Cache agent list

        # JIT-compile vmapped functions
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
            
            # JIT-compiled obs conversion for reset (single env)
            @jax.jit
            def _convert_obs(obs_dict):
                """Convert obs dict to stacked array."""
                return jnp.stack([obs_dict[a] for a in self._agent_list], axis=0).T  # [obs_dim, n_a]
            self._jit_convert_obs = _convert_obs
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
            
            # JIT-compiled obs conversion for reset (batched envs)
            @jax.jit
            def _convert_obs_batched(obs_dict):
                """Convert obs dict to stacked array."""
                obs = jnp.stack([obs_dict[a] for a in self._agent_list], axis=1)  # [N, n_a, obs_dim]
                return obs.reshape(n_envs * jax_env.n_a, jax_env.obs_dim).T        # [obs_dim, N*n_a]
            self._jit_convert_obs = _convert_obs_batched

    def _obs_dict_to_torch(self, obs_dict) -> torch.Tensor:
        """Convert observation dict to PyTorch GPU tensor.
        
        Args:
            obs_dict: Dict of observations from JAX environment
            
        Returns:
            obs_torch: (obs_dim, n_envs*n_a)  torch.cuda.FloatTensor
        """
        # JIT-compiled conversion: dict → stacked JAX array
        obs_jax = self._jit_convert_obs(obs_dict)
        
        # DLPack zero-copy: JAX GPU → PyTorch GPU
        obs_torch = torch_from_dlpack(obs_jax)
        
        return obs_torch

    def reset(self) -> torch.Tensor:
        """Reset all parallel environments.
        
        Returns:
            obs_torch: (obs_dim, n_envs*n_a)  torch.cuda.FloatTensor
        """
        keys = jax.random.split(self._key, self.n_envs + 1)
        self._key = keys[0]
        env_keys = keys[1:]

        if self.n_envs == 1:
            obs_dict, self._states = self._jit_reset(env_keys[0])
        else:
            obs_dict, self._states = self._jit_reset(env_keys)

        return self._obs_dict_to_torch(obs_dict)

    def step(self, actions_torch: torch.Tensor):
        """Step all parallel environments.
        
        Args:
            actions_torch: (n_envs*n_a, 2)  torch.cuda.FloatTensor
            
        Returns:
            obs_torch:   (obs_dim,  n_envs*n_a)  torch.cuda.FloatTensor
            rew_torch:   (1,        n_envs*n_a)  torch.cuda.FloatTensor
            done_torch:  (1,        n_envs*n_a)  torch.cuda.BoolTensor
            info:        {}  (empty dict)
            prior_torch: (2,        n_envs*n_a)  torch.cuda.FloatTensor
        """
        # DLPack: PyTorch GPU → JAX GPU (zero-copy, JAX 0.7+ API)
        # .contiguous() ensures C-contiguous layout; from_dlpack requires row-major
        actions_jax = jax_dlpack.from_dlpack(actions_torch.contiguous())

        if self.n_envs == 1:
            actions_dict = {
                a: actions_jax[i] for i, a in enumerate(self._agent_list)
            }
            key, step_key = jax.random.split(self._key)
            self._key = key
            obs_dict, new_state, rew_dict, done_dict, _ = self._jit_step(
                step_key, self._states, actions_dict
            )
            self._states = new_state
            a_prior_jax = self._jit_prior(new_state)
        else:
            # Reshape flat (N*n_a, 2) → (N, n_a, 2)
            actions_reshaped = actions_jax.reshape(
                self.n_envs, self._n_a_per_env, 2
            )
            actions_dict = {
                a: actions_reshaped[:, i, :]
                for i, a in enumerate(self._agent_list)
            }

            keys = jax.random.split(self._key, self.n_envs + 1)
            self._key = keys[0]
            step_keys = keys[1:]

            obs_dict, new_states, rew_dict, done_dict, _ = self._jit_step(
                step_keys, self._states, actions_dict
            )
            self._states = new_states
            a_prior_jax = self._jit_prior(new_states)

        # JIT-compiled conversion: dicts → stacked arrays (single JAX dispatch)
        obs_jax, rew_jax, done_jax, prior_jax = self._jit_convert(
            obs_dict, rew_dict, done_dict, a_prior_jax
        )
        
        # DLPack zero-copy: JAX GPU → PyTorch GPU
        obs_torch = torch_from_dlpack(obs_jax)
        rew_torch = torch_from_dlpack(rew_jax)
        done_torch = torch_from_dlpack(done_jax)
        prior_torch = torch_from_dlpack(prior_jax)

        return obs_torch, rew_torch, done_torch, {}, prior_torch

        return obs_torch, rew_torch, done_torch, {}, prior_torch

    def render(self, *args, **kwargs):
        """No-op — JAX env has no renderer yet."""
        pass

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
    """Minimal agent object for API compatibility."""
    def __init__(self):
        self.adversary = False
