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

        # JIT-compile vmapped functions
        if n_envs == 1:
            self._jit_reset = jax.jit(jax_env.reset)
            self._jit_step = jax.jit(jax_env.step_env)
            self._jit_prior = jax.jit(jax_env.robot_policy)
        else:
            self._jit_reset = jax.jit(jax.vmap(jax_env.reset))
            self._jit_step = jax.jit(jax.vmap(jax_env.step_env))
            self._jit_prior = jax.jit(jax.vmap(jax_env.robot_policy))

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
                a: actions_jax[i] for i, a in enumerate(self.env.agents)
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
                for i, a in enumerate(self.env.agents)
            }

            keys = jax.random.split(self._key, self.n_envs + 1)
            self._key = keys[0]
            step_keys = keys[1:]

            obs_dict, new_states, rew_dict, done_dict, _ = self._jit_step(
                step_keys, self._states, actions_dict
            )
            self._states = new_states
            a_prior_jax = self._jit_prior(new_states)

        # Convert all outputs via DLPack (zero-copy, GPU → GPU)
        obs_torch = self._obs_dict_to_torch(obs_dict)
        rew_torch = self._rew_dict_to_torch(rew_dict)
        done_torch = self._done_dict_to_torch(done_dict)
        prior_torch = self._prior_to_torch(a_prior_jax)

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

    # ──────────────────────────────────────────────────────────────────────
    # JAX → PyTorch conversions via DLPack (zero-copy on GPU)
    # ──────────────────────────────────────────────────────────────────────

    def _obs_dict_to_torch(self, obs_dict) -> torch.Tensor:
        """Convert obs dict to (obs_dim, N*n_a) torch.cuda.FloatTensor via DLPack."""
        if self.n_envs == 1:
            obs_arr = jnp.stack(
                [obs_dict[a] for a in self.env.agents], axis=0
            )  # [n_a, obs_dim]
            obs_transposed = obs_arr.T  # [obs_dim, n_a]
        else:
            obs_arr = jnp.stack(
                [obs_dict[a] for a in self.env.agents], axis=1
            )  # [N, n_a, obs_dim]
            obs_flat = obs_arr.reshape(self.n_envs * self._n_a_per_env, self.env.obs_dim)
            obs_transposed = obs_flat.T  # [obs_dim, N*n_a]

        # DLPack zero-copy: JAX GPU → PyTorch GPU (JAX 0.7+ API)
        return torch_from_dlpack(obs_transposed)

    def _rew_dict_to_torch(self, rew_dict) -> torch.Tensor:
        """Convert reward dict to (1, N*n_a) torch.cuda.FloatTensor via DLPack."""
        if self.n_envs == 1:
            rew = jnp.stack([rew_dict[a] for a in self.env.agents])  # [n_a]
            rew_reshaped = rew[None, :]  # [1, n_a]
        else:
            rew = jnp.stack(
                [rew_dict[a] for a in self.env.agents], axis=1
            )  # [N, n_a]
            rew_reshaped = rew.reshape(1, -1)  # [1, N*n_a]

        return torch_from_dlpack(rew_reshaped)

    def _done_dict_to_torch(self, done_dict) -> torch.Tensor:
        """Convert done dict to (1, N*n_a) torch.cuda.BoolTensor via DLPack."""
        if self.n_envs == 1:
            done = jnp.stack([done_dict[a] for a in self.env.agents])  # [n_a]
            done_reshaped = done[None, :]  # [1, n_a]
        else:
            done = jnp.stack(
                [done_dict[a] for a in self.env.agents], axis=1
            )  # [N, n_a]
            done_reshaped = done.reshape(1, -1)  # [1, N*n_a]

        return torch_from_dlpack(done_reshaped)

    def _prior_to_torch(self, a_prior_jax) -> torch.Tensor:
        """Convert robot_policy output to (2, N*n_a) torch.cuda.FloatTensor via DLPack."""
        if self.n_envs == 1:
            prior_transposed = a_prior_jax.T  # [n_a, 2] → [2, n_a]
        else:
            prior_flat = a_prior_jax.reshape(self.n_envs * self._n_a_per_env, 2)
            prior_transposed = prior_flat.T  # [N*n_a, 2] → [2, N*n_a]

        return torch_from_dlpack(prior_transposed)


class _DummyAgent:
    """Minimal agent object for API compatibility."""
    def __init__(self):
        self.adversary = False
