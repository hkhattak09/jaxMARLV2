"""JAX Assembly Swarm Environment.

Direct port of AssemblySwarmEnv (MARL-LLM/cus_gym) to JAX for parallel simulation.
Physics and reward exactly match the active C++ implementation in AssemblyEnv.cpp.

Active configuration (matches assembly_cfg.py defaults):
  - is_boundary = True      (wall forces, not periodic)
  - is_con_self_state = True (own pos/vel included in obs)
  - is_feature_norm = False  (no normalisation)
  - dynamics_mode = Cartesian
  - penalize_entering = True, penalize_interaction = True, penalize_exploration = True
  - n_frames = 1             (single physics substep per env step)
  - sensitivity = 1          (raw action goes directly to force)
  - No aerodynamic drag      (c_aero is defined in original but NOT applied in step())
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import chex
from typing import Tuple, Dict
from functools import partial

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box


@chex.dataclass
class AssemblyState:
    p_pos: chex.Array        # [n_a, 2]      agent positions
    p_vel: chex.Array        # [n_a, 2]      agent velocities
    grid_center: chex.Array  # [2, n_g_max]  padded grid cell positions (current episode)
    valid_mask: chex.Array   # [n_g_max]     True = real cell, False = padding
    l_cell: float            # grid cell side-length for this episode
    shape_index: int         # which shape is active (for logging)
    done: chex.Array         # [n_a]
    step: int


@chex.dataclass
class CachedDistances:
    """Pre-computed distances to avoid redundant argsort calls.
    
    Computed ONCE per step, then reused for obs, rewards, and prior.
    This eliminates 4× redundant argsort calls per agent per step.
    """
    # Agent-to-agent
    agent_dists: chex.Array       # [n_a, n_a] pairwise distances
    nei_idx: chex.Array           # [n_a, K] indices of K nearest neighbors per agent
    nei_dists: chex.Array         # [n_a, K] distances to K nearest neighbors
    
    # Agent-to-grid  
    a2g_dist: chex.Array          # [n_a, n_g_max] distance from each agent to each grid cell
    nearest_grid_idx: chex.Array  # [n_a] index of nearest grid cell per agent
    nearest_grid_dist: chex.Array # [n_a] distance to nearest grid cell
    in_flag: chex.Array           # [n_a] whether agent is inside a grid cell


class AssemblyEnv(MultiAgentEnv):
    """JAX Assembly Swarm Environment, functionally identical to AssemblySwarmEnv."""

    def __init__(
        self,
        results_file: str,
        n_a: int = 30,
        topo_nei_max: int = 6,
        num_obs_grid_max: int = 80,
        grid_obs_fraction: float = None,
        dt: float = 0.1,
        vel_max: float = 0.8,
        k_ball: float = 2000.0,
        c_ball: float = 30.0,
        k_wall: float = 100.0,
        c_wall: float = 5.0,
        c_drag: float = 1.5,
        size_a: float = 0.035,
        d_sen: float = 0.4,
        r_avoid: float = 0.10,
        boundary_half: float = 2.4,
        max_steps: int = 200,
    ):
        super().__init__(num_agents=n_a)

        self.n_a = n_a
        self.topo_nei_max = topo_nei_max
        self.dt = dt
        self.vel_max = vel_max
        self.k_ball = k_ball
        self.c_ball = c_ball
        self.k_wall = k_wall
        self.c_wall = c_wall
        self.c_drag = c_drag
        self.size_a = size_a
        self.d_sen = d_sen
        # r_avoid: personal space radius. Spacing violation: dist < 2*r_avoid.
        # Cell covered when agent centre within r_avoid of cell centre.
        # Preferred value: floor(size_a + 2*size_a) = 3*size_a ≈ 0.105 → 0.10.
        self.r_avoid = r_avoid
        self.boundary_half = boundary_half
        self.max_steps = max_steps

        # boundary_pos layout matches C++: [x_min, y_max, x_max, y_min]
        self.boundary_pos = jnp.array([
            -boundary_half,   # 0: x_min
             boundary_half,   # 1: y_max
             boundary_half,   # 2: x_max
            -boundary_half,   # 3: y_min
        ])

        # ── Load target shapes from pickle ──────────────────────────────────
        with open(results_file, 'rb') as f:
            loaded = pickle.load(f)

        # loaded['grid_coords'] is a list of [n_g_i, 2] numpy arrays (row = cell, col = x/y)
        # loaded['l_cell']      is a list of floats
        l_cells_np = np.array(loaded['l_cell'], dtype=np.float32)
        grid_coords_list = loaded['grid_coords']
        self.num_shapes = len(l_cells_np)

        n_gs = [int(g.shape[0]) for g in grid_coords_list]
        self.n_g_max = int(max(n_gs))

        # Pad shapes to [num_shapes, 2, n_g_max]
        all_gc = np.zeros((self.num_shapes, 2, self.n_g_max), dtype=np.float32)
        all_vm = np.zeros((self.num_shapes, self.n_g_max), dtype=bool)
        for s, (gc, ng) in enumerate(zip(grid_coords_list, n_gs)):
            # gc is [n_g, 2]; store as [2, n_g] (matches C++ column-major layout)
            all_gc[s, :, :ng] = gc[:ng].T
            all_vm[s, :ng] = True

        self.all_grid_centers = jnp.array(all_gc)   # [num_shapes, 2, n_g_max]
        self.all_valid_masks  = jnp.array(all_vm)   # [num_shapes, n_g_max]
        self.all_l_cells      = jnp.array(l_cells_np)  # [num_shapes]

        # ── Resolve num_obs_grid_max from fraction if provided ───────────────
        # grid_obs_fraction in (0, 1]: fraction of baseline M=80 visible per agent.
        # e.g. 0.1 → M=8, 0.5 → M=40, 1.0 → M=80.
        # Each value produces a different obs_dim and is trained from scratch.
        self.grid_obs_fraction = grid_obs_fraction
        if grid_obs_fraction is not None:
            num_obs_grid_max = max(1, int(grid_obs_fraction * 80))
        self.num_obs_grid_max = num_obs_grid_max  # set after fraction override

        # ── Spaces ──────────────────────────────────────────────────────────
        self.agents     = [f"agent_{i}" for i in range(n_a)]
        self.agent_range = jnp.arange(n_a)

        # obs: 4*(topo_nei_max+1) + 2 (tgt_pos_rel) + 2 (tgt_vel_rel) + 2*num_obs_grid_max
        self.obs_dim = 4 * (topo_nei_max + 1) + 4 + 2 * num_obs_grid_max

        self.observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for a in self.agents
        }
        self.action_spaces = {
            a: Box(-1.0, 1.0, (2,)) for a in self.agents
        }

    # ────────────────────────────────────────────────────────────────────────
    # Reset
    # ────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, AssemblyState]:
        key_shape, key_angle, key_offset, key_pos, key_cluster, key_choice, key_vel = (
            jax.random.split(key, 7)
        )

        # Sample shape index
        shape_index = jax.random.randint(key_shape, (), 0, self.num_shapes)

        # Retrieve padded shape data (scale = 1, matches original)
        grid_center_origin = self.all_grid_centers[shape_index]  # [2, n_g_max]
        valid_mask         = self.all_valid_masks[shape_index]   # [n_g_max]
        l_cell             = self.all_l_cells[shape_index]       # scalar

        # Random rotation (domain generalisation 3)
        rand_angle = jnp.pi * jax.random.uniform(key_angle, (), minval=-1.0, maxval=1.0)
        cos_a = jnp.cos(rand_angle)
        sin_a = jnp.sin(rand_angle)
        # Rotation matrix matches original: [[cos, sin], [-sin, cos]]
        rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])  # [2, 2]
        grid_center_rot = rot @ grid_center_origin           # [2, n_g_max]

        # Random offset (domain generalisation 4)
        rand_offset = jax.random.uniform(
            key_offset, (2,),
            minval=-self.boundary_half + 1.0,
            maxval= self.boundary_half - 1.0,
        )
        grid_center = grid_center_rot + rand_offset[:, None]  # [2, n_g_max]

        # Agent initial positions: 50/50 uniform-in-boundary vs clustered
        p_pos_uniform = jax.random.uniform(
            key_pos, (self.n_a, 2),
            minval=-self.boundary_half, maxval=self.boundary_half,
        )
        cluster_center = jax.random.uniform(
            key_cluster, (2,),
            minval=-self.boundary_half + 1.0, maxval=self.boundary_half - 1.0,
        )
        p_pos_clustered = (
            jax.random.uniform(key_pos, (self.n_a, 2), minval=-1.0, maxval=1.0)
            + cluster_center
        )
        use_uniform = jax.random.uniform(key_choice, (), minval=-1.0, maxval=1.0) > 0.0  # prob 0.5 each
        p_pos = jax.lax.cond(use_uniform, lambda: p_pos_uniform, lambda: p_pos_clustered)

        # Initial velocity uniform in [-0.5, 0.5]
        p_vel = jax.random.uniform(key_vel, (self.n_a, 2), minval=-0.5, maxval=0.5)

        state = AssemblyState(
            p_pos=p_pos,
            p_vel=p_vel,
            grid_center=grid_center,
            valid_mask=valid_mask,
            l_cell=l_cell,
            shape_index=shape_index,
            done=jnp.zeros(self.n_a, dtype=bool),
            step=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0, 2])
    def reset_eval(self, key: chex.PRNGKey, shape_index: int) -> Tuple[Dict, AssemblyState]:
        """Reset for evaluation: specific shape, no rotation, no offset.
        
        Args:
            key: PRNG key for agent position initialization only.
            shape_index: Which shape to use (0 to num_shapes-1).
            
        Returns:
            obs: Dict of observations per agent.
            state: Initial AssemblyState with shape centered at origin.
        """
        key_pos, key_vel, key_cluster, key_choice = jax.random.split(key, 4)

        # Retrieve shape data (no random selection)
        grid_center = self.all_grid_centers[shape_index]  # [2, n_g_max]
        valid_mask  = self.all_valid_masks[shape_index]   # [n_g_max]
        l_cell      = self.all_l_cells[shape_index]       # scalar

        # No rotation, no offset - shape stays centered at origin

        # Agent initial positions: 50/50 uniform-in-boundary vs clustered
        p_pos_uniform = jax.random.uniform(
            key_pos, (self.n_a, 2),
            minval=-self.boundary_half, maxval=self.boundary_half,
        )
        cluster_center = jax.random.uniform(
            key_cluster, (2,),
            minval=-self.boundary_half + 1.0, maxval=self.boundary_half - 1.0,
        )
        p_pos_clustered = (
            jax.random.uniform(key_pos, (self.n_a, 2), minval=-1.0, maxval=1.0)
            + cluster_center
        )
        use_uniform = jax.random.uniform(key_choice, (), minval=-1.0, maxval=1.0) > 0.0  # prob 0.5 each
        p_pos = jax.lax.cond(use_uniform, lambda: p_pos_uniform, lambda: p_pos_clustered)

        # Initial velocity uniform in [-0.5, 0.5]
        p_vel = jax.random.uniform(key_vel, (self.n_a, 2), minval=-0.5, maxval=0.5)

        state = AssemblyState(
            p_pos=p_pos,
            p_vel=p_vel,
            grid_center=grid_center,
            valid_mask=valid_mask,
            l_cell=l_cell,
            shape_index=jnp.int32(shape_index),
            done=jnp.zeros(self.n_a, dtype=bool),
            step=0,
        )

        return self.get_obs(state), state

    # ────────────────────────────────────────────────────────────────────────
    # Step
    # ────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: AssemblyState,
        actions: Dict,
    ) -> Tuple[Dict, AssemblyState, Dict, Dict, chex.Array]:
        """Standard step returning dicts (for compatibility)."""
        # Actions: dict → [n_a, 2], clipped to [-1, 1]
        u = jnp.stack([actions[a] for a in self.agents])  # [n_a, 2]
        u = jnp.clip(u, -1.0, 1.0)

        # Physics (single substep, n_frames=1)
        p_pos, p_vel = self._world_step(state, u)

        done = jnp.full((self.n_a,), state.step + 1 >= self.max_steps)

        new_state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            done=done,
            step=state.step + 1,
        )

        # Pre-compute all distances ONCE (eliminates redundant argsort calls)
        cached = self._compute_cached_distances(new_state)

        obs     = self._get_obs_fast(new_state, cached)
        rewards = self._rewards_fast(new_state, cached)

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = jnp.all(done)

        # Compute prior actions using cached distances
        prior = self._robot_policy_fast(new_state, cached)

        return obs, new_state, rewards, dones, prior
    
    @partial(jax.jit, static_argnums=[0])
    def step_env_array(
        self,
        key: chex.PRNGKey,
        state: AssemblyState,
        actions: chex.Array,  # [n_a, 2] instead of dict
    ) -> Tuple[chex.Array, AssemblyState, chex.Array, chex.Array, chex.Array]:
        """Pure-array step for maximum JIT performance.
        
        Returns arrays instead of dicts to avoid Python dict overhead in JIT:
          obs:    [n_a, obs_dim]
          rew:    [n_a]
          done:   [n_a]
          prior:  [n_a, 2]
        """
        # Actions already [n_a, 2], just clip
        u = jnp.clip(actions, -1.0, 1.0)

        # Physics (single substep, n_frames=1)
        p_pos, p_vel = self._world_step(state, u)

        done = jnp.full((self.n_a,), state.step + 1 >= self.max_steps)

        new_state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            done=done,
            step=state.step + 1,
        )

        # Pre-compute all distances ONCE (eliminates redundant argsort calls)
        cached = self._compute_cached_distances(new_state)

        # Return arrays directly, no dict conversion
        obs = self._get_obs_vectorized(new_state, cached)   # [n_a, obs_dim]
        rew = self._rewards_vectorized(new_state, cached)   # [n_a]
        prior = self._robot_policy_vectorized(new_state, cached)  # [n_a, 2]

        return obs, new_state, rew, done, prior

    def compute_prior(self, state: AssemblyState) -> chex.Array:
        """Compute Reynolds flocking prior from current state without stepping.

        Identical to the prior returned by step_env_array, but computed from
        the state that is passed in rather than the post-step state. Use this
        to get the synchronous prior for the current observation before acting.

        Returns:
            prior: [n_a, 2] clipped to [-1, 1]
        """
        cached = self._compute_cached_distances(state)
        return self._robot_policy_vectorized(state, cached)

    def _compute_cached_distances(self, state: AssemblyState) -> CachedDistances:
        """Compute all distance metrics ONCE per step.
        
        Replaces 4× redundant argsort per agent with a single batched computation.
        Uses jax.lax.top_k which is O(n × k) instead of O(n log n) for argsort.
        """
        # ── Agent-to-agent distances [n_a, n_a] ──
        delta = state.p_pos[:, None, :] - state.p_pos[None, :, :]  # [n_a, n_a, 2]
        agent_dists = jnp.linalg.norm(delta, axis=-1)  # [n_a, n_a]
        
        # Set self-distance to inf to exclude from neighbors
        eye_mask = jnp.eye(self.n_a, dtype=bool)
        agent_dists_excl = jnp.where(eye_mask, jnp.inf, agent_dists)
        
        # top_k on NEGATIVE distances gives K nearest (top_k returns largest values)
        neg_dists = -agent_dists_excl
        _, nei_idx = jax.lax.top_k(neg_dists, self.topo_nei_max)  # [n_a, K]
        nei_dists = jnp.take_along_axis(agent_dists_excl, nei_idx, axis=1)  # [n_a, K]
        
        # ── Agent-to-grid distances [n_a, n_g_max] ──
        grid_pos = state.grid_center.T  # [n_g_max, 2]
        a2g = grid_pos[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)  # [n_a, n_g_max]
        
        # Mask invalid grid cells
        a2g_dist_masked = jnp.where(state.valid_mask[None, :], a2g_dist, jnp.inf)
        
        # Nearest grid cell per agent
        nearest_grid_idx = jnp.argmin(a2g_dist_masked, axis=1)  # [n_a]
        nearest_grid_dist = jnp.min(a2g_dist_masked, axis=1)    # [n_a]
        
        # in_flag: whether agent is inside a grid cell
        threshold = jnp.sqrt(2.0) * state.l_cell / 2.0
        in_flag = nearest_grid_dist < threshold  # [n_a]
        
        return CachedDistances(
            agent_dists=agent_dists,
            nei_idx=nei_idx,
            nei_dists=nei_dists,
            a2g_dist=a2g_dist,
            nearest_grid_idx=nearest_grid_idx,
            nearest_grid_dist=nearest_grid_dist,
            in_flag=in_flag,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Physics
    # ────────────────────────────────────────────────────────────────────────

    def _world_step(self, state: AssemblyState, u: chex.Array):
        """Integrate one timestep using 4 substeps at dt/4.

        k_ball=2000 with dt_sub=dt/4=0.025 gives stability margin k*dt²=1.25<2.
        Worst-case head-on collision at vel_max produces ~0.02 units max penetration
        (29% of agent diameter) before the spring reverses the agents — no tunneling.

        Physics forces:
          - Ball-to-ball: spring repulsion (k_ball) + dashpot damping (c_ball)
          - Ball-to-wall: spring repulsion (k_wall) + damping (c_wall)
          - Aerodynamic drag: F_drag = -c_drag * v (viscous medium, dissipates KE)
        """
        dt_sub = self.dt / 4.0
        p_pos, p_vel = state.p_pos, state.p_vel
        for _ in range(4):
            sf_b2b, df_b2b = self._ball_to_ball_force(p_pos, p_vel)
            sf_b2w, df_b2w = self._ball_to_wall_force(p_pos, p_vel)
            F_drag = -self.c_drag * p_vel
            F     = u + sf_b2b + df_b2b + sf_b2w + df_b2w + F_drag
            p_vel = jnp.clip(p_vel + F * dt_sub, -self.vel_max, self.vel_max)
            p_pos = p_pos + p_vel * dt_sub
        return p_pos, p_vel

    def _ball_to_ball_force(self, p_pos: chex.Array, p_vel: chex.Array
                            ) -> Tuple[chex.Array, chex.Array]:
        """Spring repulsion + dashpot damping between overlapping agents.

        Spring (k_ball): elastic repulsion proportional to overlap depth.
        Dashpot (c_ball): velocity-dependent damping along collision normal.
        Together they form a spring-dashpot contact model that dissipates
        kinetic energy during collisions, preventing perpetual bouncing.

        delta[i,j] = p[j] - p[i]; force on i from j is repulsion (-dir_ij).
        """
        # delta[i,j] = p[j] - p[i]
        delta = p_pos[None, :, :] - p_pos[:, None, :]   # [n_a, n_a, 2]
        dist  = jnp.linalg.norm(delta, axis=-1)          # [n_a, n_a]

        overlap    = 2.0 * self.size_a - dist             # [n_a, n_a]
        is_collide = (dist < 2.0 * self.size_a) & (dist > 1e-8)  # exclude self
        collide_f  = is_collide[:, :, None].astype(jnp.float32)

        safe_dist = jnp.maximum(dist, 1e-8)
        dir_ij    = delta / safe_dist[:, :, None]          # [n_a, n_a, 2]

        # Spring: repulsion away from j  → -dir_ij * k * overlap
        sf_b2b = collide_f * overlap[:, :, None] * self.k_ball * (-dir_ij)

        # Dashpot: damp relative velocity along collision normal.
        # v_rel[i,j] = v[j] - v[i]; project onto dir_ij (i→j).
        # When closing (v_rel_n < 0): extra repulsion. When separating: slows separation.
        v_rel = p_vel[None, :, :] - p_vel[:, None, :]     # [n_a, n_a, 2]
        v_rel_n = jnp.sum(v_rel * dir_ij, axis=-1)        # [n_a, n_a]
        df_b2b = collide_f * (self.c_ball * v_rel_n[:, :, None]) * dir_ij

        return jnp.sum(sf_b2b, axis=1), jnp.sum(df_b2b, axis=1)  # [n_a, 2] each

    def _ball_to_wall_force(
        self, p_pos: chex.Array, p_vel: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Spring + damping wall forces (k_wall, c_wall).

        Matches sf_b2w / df_b2w computation in assembly.py step().
        Wall order: [x_min, y_max, x_max, y_min].
        """
        r   = self.size_a
        bnd = self.boundary_pos  # [x_min, y_max, x_max, y_min]

        # Signed distances: positive = clear, negative = penetrating
        d_left   =  p_pos[:, 0] - r - bnd[0]       # [n_a]
        d_top    =  bnd[1] - (p_pos[:, 1] + r)      # [n_a]
        d_right  =  bnd[2] - (p_pos[:, 0] + r)      # [n_a]
        d_bottom =  p_pos[:, 1] - r - bnd[3]        # [n_a]

        col_l = (d_left   < 0.0).astype(jnp.float32)
        col_t = (d_top    < 0.0).astype(jnp.float32)
        col_r = (d_right  < 0.0).astype(jnp.float32)
        col_b = (d_bottom < 0.0).astype(jnp.float32)

        abs_l = jnp.abs(d_left)
        abs_t = jnp.abs(d_top)
        abs_r = jnp.abs(d_right)
        abs_b = jnp.abs(d_bottom)

        # Spring: [[1,0,-1,0],[0,-1,0,1]] @ (col * |d|) * k_wall
        sf_x = (col_l * abs_l - col_r * abs_r) * self.k_wall
        sf_y = (-col_t * abs_t + col_b * abs_b) * self.k_wall
        sf_b2w = jnp.stack([sf_x, sf_y], axis=-1)  # [n_a, 2]

        # Damping: [[-1,0,-1,0],[0,-1,0,-1]] @ (col * dp_concat) * c_wall
        df_x = -(col_l + col_r) * p_vel[:, 0] * self.c_wall
        df_y = -(col_t + col_b) * p_vel[:, 1] * self.c_wall
        df_b2w = jnp.stack([df_x, df_y], axis=-1)  # [n_a, 2]

        return sf_b2w, df_b2w

    # ────────────────────────────────────────────────────────────────────────
    # Observations
    # ────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: AssemblyState) -> Dict[str, chex.Array]:
        """Build observations for all agents.

        Matches _get_observation() with:
          condition[0]=False (is_periodic), condition[1]=True (is_Cartesian),
          condition[2]=True (is_con_self_state), condition[3]=False (is_feature_norm).
        """
        obs_arr = jax.vmap(self._obs_single, in_axes=(0, None))(
            self.agent_range, state
        )  # [n_a, obs_dim]
        return {a: obs_arr[i] for i, a in enumerate(self.agents)}
    
    def _get_obs_fast(self, state: AssemblyState, cached: CachedDistances) -> Dict[str, chex.Array]:
        """Fully vectorized observation - NO vmap, pure batched ops."""
        obs_arr = self._get_obs_vectorized(state, cached)  # [n_a, obs_dim]
        return {a: obs_arr[i] for i, a in enumerate(self.agents)}
    
    def _get_obs_vectorized(self, state: AssemblyState, cached: CachedDistances) -> chex.Array:
        """Fully vectorized observations for ALL agents at once.
        
        No vmap - uses advanced indexing and broadcasting.
        Returns [n_a, obs_dim] array.
        """
        n_a = self.n_a
        K = self.topo_nei_max
        M = self.num_obs_grid_max
        
        # ══════════════════════════════════════════════════════════════════════
        # Part 1: Neighbor observations [n_a, 4*(K+1)]
        # ══════════════════════════════════════════════════════════════════════
        
        # Gather neighbor positions/velocities for all agents at once
        # nei_idx: [n_a, K] - indices of K nearest neighbors per agent
        nei_pos = state.p_pos[cached.nei_idx]  # [n_a, K, 2]
        nei_vel = state.p_vel[cached.nei_idx]  # [n_a, K, 2]
        
        # Relative to each agent
        rel_pos = nei_pos - state.p_pos[:, None, :]  # [n_a, K, 2]
        rel_vel = nei_vel - state.p_vel[:, None, :]  # [n_a, K, 2]
        
        # Zero out neighbors outside sensor range
        in_range = (cached.nei_dists < self.d_sen)[:, :, None]  # [n_a, K, 1]
        rel_pos = rel_pos * in_range
        rel_vel = rel_vel * in_range
        
        # Own state: [n_a, 4]
        own_state = jnp.concatenate([state.p_pos, state.p_vel], axis=-1)
        
        # Neighbor cols: [n_a, K, 4]
        nei_cols = jnp.concatenate([rel_pos, rel_vel], axis=-1)
        
        # Stack own + neighbors, flatten: [n_a, 4*(K+1)]
        obs_agent = jnp.concatenate([own_state[:, None, :], nei_cols], axis=1)  # [n_a, K+1, 4]
        obs_agent_flat = obs_agent.reshape(n_a, -1)  # [n_a, 4*(K+1)]
        
        # ══════════════════════════════════════════════════════════════════════
        # Part 2: Target grid cell [n_a, 4]
        # ══════════════════════════════════════════════════════════════════════
        
        grid_pos = state.grid_center.T  # [n_g_max, 2]
        
        # Gather nearest grid position for each agent
        nearest_grid_pos = grid_pos[cached.nearest_grid_idx]  # [n_a, 2]
        
        # Target position: self if in_flag, else nearest grid
        in_flag = cached.in_flag[:, None]  # [n_a, 1]
        target_pos = jnp.where(in_flag, state.p_pos, nearest_grid_pos)  # [n_a, 2]
        target_vel = jnp.where(in_flag, state.p_vel, jnp.zeros_like(state.p_vel))  # [n_a, 2]
        
        target_rel_pos = target_pos - state.p_pos  # [n_a, 2]
        target_rel_vel = target_vel - state.p_vel  # [n_a, 2]
        
        # ══════════════════════════════════════════════════════════════════════
        # Part 3: Sensed unoccupied grid cells [n_a, M*2]
        # ══════════════════════════════════════════════════════════════════════
        
        # a2g_dist: [n_a, n_g_max]
        # in_sensor: cells within d_sen for each agent
        in_sensor = (cached.a2g_dist < self.d_sen) & state.valid_mask[None, :]  # [n_a, n_g_max]
        
        # is_nearby: agents within d_sen + r_avoid of each agent (could own a visible cell)
        is_nearby = cached.agent_dists < (self.d_sen + self.r_avoid)  # [n_a, n_a]

        # Cell occupied if any nearby agent is within r_avoid of it
        # For each agent i, check if any agent j (where is_nearby[i,j]) is close to each grid cell
        # is_nearby: [n_a, n_a], a2g_dist: [n_a, n_g_max]
        # We need: for each agent i, for each grid cell g: any(is_nearby[i,j] & (a2g_dist[j,g] < r_avoid))
        cell_agent_close = cached.a2g_dist < self.r_avoid  # [n_a, n_g_max]
        # is_occupied_by_nearby[i,g] = any_j(is_nearby[i,j] & cell_agent_close[j,g])
        # This is: is_nearby @ cell_agent_close (treating bools as 0/1)
        is_occupied_by_nearby = jnp.matmul(
            is_nearby.astype(jnp.float32), 
            cell_agent_close.astype(jnp.float32)
        ) > 0  # [n_a, n_g_max]
        
        is_occupied = cached.in_flag[:, None] & is_occupied_by_nearby  # [n_a, n_g_max]
        is_sensed_unoccupied = in_sensor & ~is_occupied  # [n_a, n_g_max]
        
        # For each agent, get M nearest sensed unoccupied cells
        # Set non-sensed to inf, then take top_k of negative distances
        sort_dist = jnp.where(is_sensed_unoccupied, cached.a2g_dist, jnp.inf)  # [n_a, n_g_max]
        _, sensed_idx = jax.lax.top_k(-sort_dist, M)  # [n_a, M]
        
        # Gather grid positions for sensed cells
        # sensed_idx: [n_a, M] - need to gather grid_pos[sensed_idx[i, m]] for each (i, m)
        sensed_pos_abs = grid_pos[sensed_idx]  # [n_a, M, 2]
        
        # Check validity and compute relative positions
        sensed_valid = jnp.take_along_axis(is_sensed_unoccupied, sensed_idx, axis=1)  # [n_a, M]
        sensed_rel = jnp.where(
            sensed_valid[:, :, None],
            sensed_pos_abs - state.p_pos[:, None, :],
            jnp.zeros((n_a, M, 2))
        )  # [n_a, M, 2]

        # Per-shape cell visibility cap: zero out cells ranked >= effective_M.
        # effective_M = floor(grid_obs_fraction * n_cells_this_episode).
        # This branch is resolved at trace time (self is static_argnums).
        if self.grid_obs_fraction is not None:
            n_cells = jnp.sum(state.valid_mask).astype(jnp.float32)
            effective_M = jnp.maximum(
                1, jnp.floor(self.grid_obs_fraction * n_cells).astype(jnp.int32)
            )
            rank_mask = (jnp.arange(M) < effective_M)[None, :, None]  # [1, M, 1]
            sensed_rel = sensed_rel * rank_mask

        sensed_flat = sensed_rel.reshape(n_a, -1)  # [n_a, M*2]
        
        # ══════════════════════════════════════════════════════════════════════
        # Concatenate all parts
        # ══════════════════════════════════════════════════════════════════════
        return jnp.concatenate([
            obs_agent_flat,    # [n_a, 4*(K+1)]
            target_rel_pos,    # [n_a, 2]
            target_rel_vel,    # [n_a, 2]
            sensed_flat,       # [n_a, M*2]
        ], axis=-1)  # [n_a, obs_dim]

    def _obs_single(self, i: int, state: AssemblyState) -> chex.Array:
        """Observation for agent i."""
        # ── Relative positions/velocities of all agents w.r.t. agent i ──
        rel_pos = state.p_pos - state.p_pos[i]   # [n_a, 2]
        rel_vel = state.p_vel - state.p_vel[i]   # [n_a, 2]
        dists   = jnp.linalg.norm(rel_pos, axis=-1)  # [n_a]

        # Exclude self (set to inf so it sorts last)
        dists_excl = jnp.where(jnp.arange(self.n_a) == i, jnp.inf, dists)

        # K nearest within d_sen (argsort, take top topo_nei_max)
        sorted_idx  = jnp.argsort(dists_excl)[:self.topo_nei_max]  # [K]
        nei_rel_pos = rel_pos[sorted_idx]                           # [K, 2]
        nei_rel_vel = rel_vel[sorted_idx]                           # [K, 2]
        nei_dists   = dists_excl[sorted_idx]                        # [K]

        # Zero out neighbours outside sensor range (pad with zeros)
        in_range    = (nei_dists < self.d_sen)[:, None]
        nei_rel_pos = nei_rel_pos * in_range
        nei_rel_vel = nei_rel_vel * in_range

        # obs_agent: column-major flatten of [4, topo_nei_max+1]
        # Columns: [own, nei_1, ..., nei_K]; rows: [px, py, vx, vy]
        # → each column is [px, py, vx, vy] → stacked then flattened
        own_col  = jnp.concatenate([state.p_pos[i], state.p_vel[i]])  # [4]
        nei_cols = jnp.concatenate([nei_rel_pos, nei_rel_vel], axis=-1)  # [K, 4]
        obs_agent_flat = jnp.concatenate(
            [own_col[None], nei_cols], axis=0
        ).flatten()  # [4*(K+1)]

        # ── Target grid cell ─────────────────────────────────────────────
        # grid_center: [2, n_g_max]; grid positions relative to agent i
        grid_rel  = state.grid_center.T - state.p_pos[i]  # [n_g_max, 2]
        grid_dist = jnp.linalg.norm(grid_rel, axis=-1)    # [n_g_max]

        # Mask padding cells
        grid_dist_masked = jnp.where(state.valid_mask, grid_dist, jnp.inf)
        nearest_idx = jnp.argmin(grid_dist_masked)
        min_dist    = grid_dist_masked[nearest_idx]

        in_flag = min_dist < (jnp.sqrt(2.0) * state.l_cell / 2.0)

        # When in_flag: target = self position; else: nearest grid cell
        target_pos = jnp.where(in_flag, state.p_pos[i], state.grid_center.T[nearest_idx])
        target_vel = jnp.where(in_flag, state.p_vel[i], jnp.zeros(2))

        target_grid_pos_rel = target_pos - state.p_pos[i]  # [2]
        target_grid_vel_rel = target_vel - state.p_vel[i]  # [2]

        # ── Sensed unoccupied grid cells ─────────────────────────────────
        # Cells within d_sen, not "occupied" by a nearby agent when in_flag
        in_sensor = (grid_dist < self.d_sen) & state.valid_mask  # [n_g_max]

        # Nearby agents: within d_sen + r_avoid of agent i (could own a visible cell)
        agent_dists = jnp.linalg.norm(state.p_pos - state.p_pos[i], axis=-1)  # [n_a]
        is_nearby   = agent_dists < (self.d_sen + self.r_avoid)                # [n_a]

        # Agent-to-grid distances: [n_a, n_g_max]
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)                          # [n_a, n_g_max]

        # Cell is occupied if any nearby agent is within r_avoid of it
        is_occupied_by_nearby = jnp.any(
            is_nearby[:, None] & (a2g_dist < self.r_avoid), axis=0
        )  # [n_g_max]

        # Only apply occupancy filter when agent i itself is in a grid cell
        is_occupied = in_flag & is_occupied_by_nearby  # [n_g_max]

        is_sensed_unoccupied = in_sensor & ~is_occupied  # [n_g_max]

        # Sort cells by distance, take top num_obs_grid_max
        sort_dist    = jnp.where(is_sensed_unoccupied, grid_dist, jnp.inf)
        sensed_idx   = jnp.argsort(sort_dist)[:self.num_obs_grid_max]       # [M]
        sensed_valid = is_sensed_unoccupied[sensed_idx]                      # [M]

        sensed_pos_abs = state.grid_center.T[sensed_idx]  # [M, 2]
        sensed_rel = jnp.where(
            sensed_valid[:, None],
            sensed_pos_abs - state.p_pos[i],
            jnp.zeros(2),
        )  # [M, 2]

        # Column-major flatten: [grid1_x, grid1_y, grid2_x, ...]
        sensed_flat = sensed_rel.flatten()  # [M*2]

        return jnp.concatenate([
            obs_agent_flat,       # 4*(topo_nei_max+1) = 28
            target_grid_pos_rel,  # 2
            target_grid_vel_rel,  # 2
            sensed_flat,          # num_obs_grid_max * 2 = 160
        ])  # total = obs_dim

    # ────────────────────────────────────────────────────────────────────────
    # Rewards
    # ────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=[0])
    def rewards(self, state: AssemblyState) -> Dict[str, float]:
        """Compute per-agent reward.

        Matches active _get_reward() in AssemblyEnv.cpp:
          +1.0 if agent is in a grid cell AND has no b2b collision with a neighbour
               AND its sensed unoccupied cells are uniformly distributed (|v_exp| < 0.05).
        """
        rew_arr = jax.vmap(self._reward_single, in_axes=(0, None))(
            self.agent_range, state
        )  # [n_a]
        return {a: rew_arr[i] for i, a in enumerate(self.agents)}
    
    def _rewards_fast(self, state: AssemblyState, cached: CachedDistances) -> Dict[str, float]:
        """Fully vectorized rewards - NO vmap."""
        rew_arr = self._rewards_vectorized(state, cached)  # [n_a]
        return {a: rew_arr[i] for i, a in enumerate(self.agents)}
    
    def _rewards_vectorized(self, state: AssemblyState, cached: CachedDistances) -> chex.Array:
        """Fully vectorized rewards for ALL agents at once.

        Three continuous components:
        1. proximity         — continuous attraction toward shape (pull from outside)
        2. coverage_score    — physical Voronoi territory balance (inside AND sensed)
        3. crowding          — all-pairs continuous density penalty

        No vmap - uses broadcasting and batched operations.
        Returns [n_a] array.
        """
        n_a = self.n_a

        # ══════════════════════════════════════════════════════════════════════
        # Component 1: Shape Proximity (continuous attraction)
        # ══════════════════════════════════════════════════════════════════════
        dist = cached.nearest_grid_dist  # [n_a]
        cell_thresh = jnp.sqrt(2.0) * state.l_cell / 2.0  # half-diagonal of grid cell

        # Broad pull: attracts agents toward shape from up to d_sen away
        broad = jnp.clip(1.0 - dist / self.d_sen, 0.0, 1.0)
        # Sharp on-cell bonus: high reward for sitting on a cell
        sharp = jnp.clip(1.0 - dist / cell_thresh, 0.0, 1.0)

        proximity = 0.3 * broad + 0.7 * sharp  # [n_a], range [0, 1]

        # ══════════════════════════════════════════════════════════════════════
        # Component 2: Crowding Penalty (all-pairs, continuous, density-aware)
        # ══════════════════════════════════════════════════════════════════════
        dists_excl = jnp.where(jnp.eye(n_a, dtype=bool), jnp.inf, cached.agent_dists)

        # Quadratic overlap: smooth at boundary, scales with proximity AND density
        raw_overlap = jnp.maximum(0.0, 2.0 * self.r_avoid - dists_excl) / (2.0 * self.r_avoid)
        overlap = raw_overlap ** 2  # [n_a, n_a], smooth gradient at boundary
        crowding = jnp.sum(overlap, axis=1)  # [n_a]

        # ══════════════════════════════════════════════════════════════════════
        # Component 3: Physical Voronoi Coverage Score
        #
        # Replaces geometric Voronoi territory. An agent earns credit for a cell
        # only if it is BOTH the nearest agent to that cell AND within d_sen of it.
        # This enforces physical presence — boundary agents cannot claim interior
        # cells they are not actually sensing. Naturally incentivises medial-axis
        # placement: agents that move inward sense more exclusive cells.
        # ══════════════════════════════════════════════════════════════════════
        n_valid = jnp.sum(state.valid_mask).astype(jnp.float32)
        ideal   = n_valid / n_a

        a2g_masked    = jnp.where(state.valid_mask[None, :], cached.a2g_dist, jnp.inf)
        nearest_agent = jnp.argmin(a2g_masked, axis=0)                         # [n_g_max]
        agent_ids     = jnp.arange(n_a)[:, None]                               # [n_a, 1]
        is_mine       = (agent_ids == nearest_agent[None, :])                   # [n_a, n_g_max]
        covers_cell   = cached.a2g_dist < self.d_sen                            # [n_a, n_g_max]
        exclusive_coverage = jnp.sum(
            is_mine & covers_cell & state.valid_mask[None, :], axis=1
        ).astype(jnp.float32)                                                   # [n_a]

        # Score: 1.0 at ideal share, linear falloff, 0.0 at >=50% deviation
        coverage_score = jnp.clip(
            1.0 - jnp.abs(exclusive_coverage - ideal) / (0.5 * ideal + 1e-8),
            0.0, 1.0,
        )                                                                        # [n_a]

        # ══════════════════════════════════════════════════════════════════════
        # Physical contact count (safety penalty)
        # ══════════════════════════════════════════════════════════════════════
        is_touching = (cached.agent_dists < 2.0 * self.size_a) & ~jnp.eye(n_a, dtype=bool)
        n_touching = jnp.sum(is_touching.astype(jnp.float32), axis=1)  # [n_a]

        # ══════════════════════════════════════════════════════════════════════
        # Settling penalty (velocity near shape)
        # ══════════════════════════════════════════════════════════════════════
        # Penalise speed only when on/near the shape. Free to move fast in open space.
        speed_sq = jnp.sum(state.p_vel ** 2, axis=1)  # [n_a]
        settling = proximity * speed_sq                # [n_a], gated by proximity

        # ══════════════════════════════════════════════════════════════════════
        # Combined reward
        # ══════════════════════════════════════════════════════════════════════
        reward = (0.20 * proximity        # pull agents toward shape from outside
                + 0.60 * coverage_score   # physical territory balance (replaces geometric Voronoi)
                - 0.25 * crowding         # increased from 0.15 — violations were 2.5× too high
                - 0.05 * n_touching
                - 0.10 * settling)
        return reward

    def _reward_single(self, i: int, state: AssemblyState) -> chex.Array:
        """Reference single-agent reward. Matches _rewards_vectorized logic."""
        n_a = self.n_a

        # ── Component 1: Shape Proximity ─────────────────────────────────
        grid_rel  = state.grid_center.T - state.p_pos[i]   # [n_g_max, 2]
        grid_dist = jnp.linalg.norm(grid_rel, axis=-1)      # [n_g_max]
        grid_dist_masked = jnp.where(state.valid_mask, grid_dist, jnp.inf)
        dist = jnp.min(grid_dist_masked)
        cell_thresh = jnp.sqrt(2.0) * state.l_cell / 2.0

        broad = jnp.clip(1.0 - dist / self.d_sen, 0.0, 1.0)
        sharp = jnp.clip(1.0 - dist / cell_thresh, 0.0, 1.0)
        proximity = 0.3 * broad + 0.7 * sharp

        # ── Component 2: Crowding Penalty (all-pairs) ────────────────────
        agent_dists = jnp.linalg.norm(state.p_pos - state.p_pos[i], axis=-1)  # [n_a]
        agent_dists_excl = jnp.where(jnp.arange(n_a) == i, jnp.inf, agent_dists)

        raw_overlap = jnp.maximum(0.0, 2.0 * self.r_avoid - agent_dists_excl) / (2.0 * self.r_avoid)
        overlap = raw_overlap ** 2
        crowding = jnp.sum(overlap)

        # ── Component 3: Physical Voronoi Coverage Score ─────────────────
        # Mirrors _rewards_vectorized: credit only for cells nearest to i
        # AND within d_sen (physical presence enforced).
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)  # [n_a, n_g_max]
        a2g_masked = jnp.where(state.valid_mask[None, :], a2g_dist, jnp.inf)
        nearest_agent = jnp.argmin(a2g_masked, axis=0)  # [n_g_max]

        n_valid = jnp.sum(state.valid_mask).astype(jnp.float32)
        ideal   = n_valid / n_a
        is_mine     = (nearest_agent == i) & state.valid_mask          # [n_g_max]
        covers_cell = a2g_dist[i] < self.d_sen                         # [n_g_max]
        exclusive_coverage = jnp.sum(is_mine & covers_cell).astype(jnp.float32)
        coverage_score = jnp.clip(
            1.0 - jnp.abs(exclusive_coverage - ideal) / (0.5 * ideal + 1e-8),
            0.0, 1.0,
        )

        # ── Physical contact ─────────────────────────────────────────────
        is_touching = (agent_dists < 2.0 * self.size_a) & (jnp.arange(n_a) != i)
        n_touching = jnp.sum(is_touching.astype(jnp.float32))

        # ── Settling penalty ─────────────────────────────────────────────
        speed_sq = jnp.sum(state.p_vel[i] ** 2)
        settling = proximity * speed_sq

        # ── Combined reward ──────────────────────────────────────────────
        reward = (0.20 * proximity
                + 0.60 * coverage_score
                - 0.25 * crowding
                - 0.05 * n_touching
                - 0.10 * settling)
        return reward

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rho_cos_dec(z: chex.Array, r: float) -> chex.Array:
        """Cosine decay weight: 1 at z=0, 0 at z>=r.  Matches _rho_cos_dec(z, delta=0, r)."""
        return jnp.where(z < r, 0.5 * (1.0 + jnp.cos(jnp.pi * z / r)), 0.0)

    # ────────────────────────────────────────────────────────────────────────
    # Evaluation Metrics  (mirrors AssemblySwarmWrapper in MARL-LLM)
    # ────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=[0])
    def sensing_coverage(self, state: AssemblyState) -> chex.Array:
        """Fraction of valid target grid cells sensed by at least one agent.

        A cell is sensed when any agent centre lies within d_sen of the cell centre.
        Reaches 1.0 when all agents collectively observe the entire shape.
        No r_avoid dependency — purely geometric coverage by sensing radius.
        """
        # a2g[i, j] = distance from agent i to grid cell j
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)                          # [n_a, n_g_max]

        # Cell sensed if ANY agent is within d_sen
        cell_sensed = jnp.any(a2g_dist < self.d_sen, axis=0)   # [n_g_max]

        # Only count valid (non-padding) cells
        n_g = jnp.sum(state.valid_mask.astype(jnp.float32))
        n_sensed = jnp.sum((cell_sensed & state.valid_mask).astype(jnp.float32))
        return n_sensed / n_g

    @partial(jax.jit, static_argnums=[0])
    def distribution_uniformity(self, state: AssemblyState) -> chex.Array:
        """Uniformity of minimum inter-agent distances, in [0, 1].

        For each agent, finds its nearest neighbour distance; computes
        uniformity as 1 / (1 + coefficient_of_variation). Returns 1.0 when
        all nearest-neighbour distances are identical (perfect uniformity),
        approaching 0.0 as variance increases.
        """
        # Pairwise distances [n_a, n_a]
        delta = state.p_pos[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_a, 2]
        dists = jnp.linalg.norm(delta, axis=-1)                     # [n_a, n_a]

        # Exclude self-distance
        dists_excl = jnp.where(jnp.eye(self.n_a, dtype=bool), jnp.inf, dists)

        # Nearest neighbour distance per agent
        min_dists = jnp.min(dists_excl, axis=1)  # [n_a]

        mean_d = jnp.mean(min_dists)
        std_d = jnp.std(min_dists)
        # Coefficient of variation: std/mean; uniformity = 1/(1+cv)
        # Higher uniformity (closer to 1) means more uniform spacing
        return 1.0 / (1.0 + std_d / jnp.maximum(mean_d, 1e-8))

    @partial(jax.jit, static_argnums=[0])
    def voronoi_based_uniformity(self, state: AssemblyState) -> chex.Array:
        """Uniformity of per-agent Voronoi cell counts, in [0, 1].

        Assigns each valid grid cell to its nearest agent (Voronoi partition),
        counts cells per agent, then computes uniformity as 1 / (1 + cv).
        Returns 1.0 when all agents have equal cell counts (perfect uniformity),
        approaching 0.0 as variance increases.
        """
        # a2g_dist[i, j] = distance from agent i to grid cell j
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)                          # [n_a, n_g_max]

        # Mask padding cells with inf so argmin ignores them
        a2g_dist_masked = jnp.where(
            state.valid_mask[None, :], a2g_dist, jnp.inf
        )  # [n_a, n_g_max]

        # Nearest agent index for each cell  [n_g_max]
        nearest_agent = jnp.argmin(a2g_dist_masked, axis=0)

        # Count valid cells assigned to each agent using one-hot accumulation
        one_hot = jax.nn.one_hot(nearest_agent, self.n_a)         # [n_g_max, n_a]
        voronoi_counts = jnp.sum(
            one_hot * state.valid_mask[:, None].astype(jnp.float32), axis=0
        )  # [n_a]

        mean_c = jnp.mean(voronoi_counts)
        std_c = jnp.std(voronoi_counts)
        # Coefficient of variation: std/mean; uniformity = 1/(1+cv)
        # Higher uniformity (closer to 1) means more balanced cell distribution
        return 1.0 / (1.0 + std_c / jnp.maximum(mean_c, 1e-8))

    @partial(jax.jit, static_argnums=[0])
    def r_avoid_violation_count(self, state: AssemblyState) -> chex.Array:
        """Count unique agent pairs violating the minimum spacing (dist < 2*r_avoid).

        Counts pairs (i < j) where dist < 2*r_avoid, so each pair is counted once.
        A single two-agent violation = 1, three mutually-violating agents = 3 pairs.
        """
        dists = jnp.linalg.norm(
            state.p_pos[:, None, :] - state.p_pos[None, :, :], axis=-1
        )  # [n_a, n_a]
        # Upper-triangle mask: counts each pair (i,j) with i < j exactly once
        upper = jnp.triu(jnp.ones((self.n_a, self.n_a), dtype=bool), k=1)
        is_violation = (dists < 2.0 * self.r_avoid) & upper
        return jnp.sum(is_violation.astype(jnp.float32))

    @partial(jax.jit, static_argnums=[0])
    def mean_neighbor_distance(self, state: AssemblyState) -> chex.Array:
        """Mean nearest-neighbour distance across all agents.

        Measures the absolute magnitude of agent spacing — unlike
        distribution_uniformity which only measures the CoV of those distances.
        Higher values mean agents are more spread out (less clustering).
        """
        delta = state.p_pos[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_a, 2]
        dists = jnp.linalg.norm(delta, axis=-1)                     # [n_a, n_a]
        dists_excl = jnp.where(jnp.eye(self.n_a, dtype=bool), jnp.inf, dists)
        min_dists = jnp.min(dists_excl, axis=1)                     # [n_a]
        return jnp.mean(min_dists)


    @partial(jax.jit, static_argnums=[0])
    def agents_in_shape(self, state: AssemblyState) -> chex.Array:
        """Fraction of agents physically inside the shape.

        An agent is counted as "in shape" when its nearest valid grid cell is
        within l_cell/2 — i.e., the agent is overlapping a cell square.
        This is stricter than d_sen and excludes agents hovering just outside
        the shape boundary. Returns a value in [0, 1].
        """
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)                          # [n_a, n_g_max]

        # Mask padding cells with inf so they don't attract agents
        a2g_dist_masked = jnp.where(state.valid_mask[None, :], a2g_dist, jnp.inf)  # [n_a, n_g_max]

        # Agent is in shape if nearest valid cell is within half a cell width
        min_dist_to_shape = jnp.min(a2g_dist_masked, axis=1)  # [n_a]
        in_shape = (min_dist_to_shape < state.l_cell * 0.5).astype(jnp.float32)
        return jnp.mean(in_shape)

    @partial(jax.jit, static_argnums=[0])
    def springboard_collision_count(self, state: AssemblyState) -> chex.Array:
        """Count unique agent pairs currently in physical body contact (spring force active).

        A springboard collision occurs when two agents overlap: dist < 2 * size_a (= 0.07).
        This is the threshold at which k_ball spring repulsion kicks in (_ball_to_ball_force).
        Counts unique pairs (i < j), so a two-agent collision = 1, three mutually-overlapping
        agents = 3 pairs. Accumulate per step over an episode for a running total.
        """
        dists = jnp.linalg.norm(
            state.p_pos[:, None, :] - state.p_pos[None, :, :], axis=-1
        )  # [n_a, n_a]
        # Upper-triangle mask: counts each pair (i,j) with i < j exactly once
        upper = jnp.triu(jnp.ones((self.n_a, self.n_a), dtype=bool), k=1)
        is_spring = (dists < 2.0 * self.size_a) & upper
        return jnp.sum(is_spring.astype(jnp.float32))

    @partial(jax.jit, static_argnums=[0])
    def eval_metrics(self, state: AssemblyState) -> Dict[str, chex.Array]:
        """Return all evaluation metrics as a dict."""
        return {
            "sensing_coverage":        self.sensing_coverage(state),
            "distribution_uniformity": self.distribution_uniformity(state),
            "voronoi_uniformity":      self.voronoi_based_uniformity(state),
            "mean_neighbor_distance":  self.mean_neighbor_distance(state),
            "r_avoid_violation_count": self.r_avoid_violation_count(state),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Prior Policy  (JAX port of C++ robotPolicy / calculateActionPrior)
    # ────────────────────────────────────────────────────────────────────────

    def _robot_policy_single(self, i: int, state: AssemblyState) -> chex.Array:
        """Reynolds flocking prior action for agent i.

        Exactly mirrors C++ robotPolicy():
          - Target attraction:    2.0 × normalised direction to nearest grid cell
          - Neighbour repulsion:  3.0 × (r_avoid/dist − 1) for neighbours < r_avoid
          - Velocity sync:        2.0 × (mean_neighbour_vel − own_vel)
        Neighbours = top topo_nei_max nearest agents (same set as neighbour_index in C++).
        Total force is clamped to [−1, 1].
        """
        pos_i = state.p_pos[i]  # [2]
        vel_i = state.p_vel[i]  # [2]

        # ── Target position (matches _get_target_grid_state in C++) ──────
        grid_rel  = state.grid_center.T - pos_i       # [n_g_max, 2]
        grid_dist = jnp.linalg.norm(grid_rel, axis=-1)  # [n_g_max]
        grid_dist_masked = jnp.where(state.valid_mask, grid_dist, jnp.inf)
        nearest_idx = jnp.argmin(grid_dist_masked)
        min_dist    = grid_dist_masked[nearest_idx]
        in_flag     = min_dist < (jnp.sqrt(2.0) * state.l_cell / 2.0)

        # When already in a cell: target = self → zero attraction
        target_pos = jnp.where(in_flag, pos_i, state.grid_center.T[nearest_idx])

        # ── Attraction: attraction_strength=2.0 ──────────────────────────
        dir_to_tgt = target_pos - pos_i               # [2]
        d_tgt      = jnp.linalg.norm(dir_to_tgt)
        attraction = jnp.where(
            d_tgt > 0,
            2.0 * dir_to_tgt / jnp.maximum(d_tgt, 1e-8),
            jnp.zeros(2),
        )  # [2]

        # ── Topological neighbours (top topo_nei_max nearest) ────────────
        agent_dists      = jnp.linalg.norm(state.p_pos - pos_i, axis=-1)   # [n_a]
        agent_dists_excl = jnp.where(jnp.arange(self.n_a) == i, jnp.inf, agent_dists)
        sorted_nei_idx   = jnp.argsort(agent_dists_excl)[:self.topo_nei_max]  # [K]
        nei_dists        = agent_dists_excl[sorted_nei_idx]                    # [K]
        nei_pos          = state.p_pos[sorted_nei_idx]                         # [K, 2]
        nei_vel          = state.p_vel[sorted_nei_idx]                         # [K, 2]

        # ── Repulsion: repulsion_strength=3.0 ────────────────────────────
        # Fires when dist < 2*r_avoid (matches reward's spacing violation threshold).
        # Direction from neighbour toward self (pointing away from each neighbour)
        dir_away      = pos_i - nei_pos                          # [K, 2]
        safe_nei_dist = jnp.maximum(nei_dists, 1e-8)
        unit_away     = dir_away / safe_nei_dist[:, None]        # [K, 2]
        rep_factor    = jnp.where(
            (nei_dists > 0) & (nei_dists < 2.0 * self.r_avoid),
            3.0 * (2.0 * self.r_avoid / safe_nei_dist - 1.0),
            0.0,
        )  # [K]
        repulsion = jnp.sum(rep_factor[:, None] * unit_away, axis=0)  # [2]

        # ── Velocity sync: sync_strength=2.0 ─────────────────────────────
        # C++ averages over ALL topological neighbours regardless of d_sen range
        avg_nei_vel = jnp.mean(nei_vel, axis=0)          # [2]
        sync        = 2.0 * (avg_nei_vel - vel_i)         # [2]

        total = attraction + repulsion + sync
        return jnp.clip(total, -1.0, 1.0)

    @partial(jax.jit, static_argnums=[0])
    def robot_policy(self, state: AssemblyState) -> chex.Array:
        """Prior actions for all agents via Reynolds flocking.

        Returns:
            [n_a, 2] array of clamped actions, one per agent.
        """
        return jax.vmap(self._robot_policy_single, in_axes=(0, None))(
            self.agent_range, state
        )
    
    def _robot_policy_fast(self, state: AssemblyState, cached: CachedDistances) -> chex.Array:
        """Fully vectorized prior actions - NO vmap."""
        return self._robot_policy_vectorized(state, cached)
    
    def _robot_policy_vectorized(self, state: AssemblyState, cached: CachedDistances) -> chex.Array:
        """Fully vectorized Reynolds flocking prior for ALL agents.
        
        No vmap - uses broadcasting and batched operations.
        Returns [n_a, 2] array.
        """
        n_a = self.n_a
        K = self.topo_nei_max
        
        # ══════════════════════════════════════════════════════════════════════
        # Target attraction
        # ══════════════════════════════════════════════════════════════════════
        
        grid_pos = state.grid_center.T  # [n_g_max, 2]
        nearest_grid_pos = grid_pos[cached.nearest_grid_idx]  # [n_a, 2]
        
        # Target: self if in_flag, else nearest grid
        in_flag = cached.in_flag[:, None]  # [n_a, 1]
        target_pos = jnp.where(in_flag, state.p_pos, nearest_grid_pos)  # [n_a, 2]
        
        # Direction to target
        dir_to_tgt = target_pos - state.p_pos  # [n_a, 2]
        d_tgt = jnp.linalg.norm(dir_to_tgt, axis=1, keepdims=True)  # [n_a, 1]
        attraction = jnp.where(
            d_tgt > 0,
            2.0 * dir_to_tgt / jnp.maximum(d_tgt, 1e-8),
            jnp.zeros_like(dir_to_tgt)
        )  # [n_a, 2]
        
        # ══════════════════════════════════════════════════════════════════════
        # Neighbor repulsion
        # ══════════════════════════════════════════════════════════════════════
        
        # Gather neighbor positions/velocities
        nei_pos = state.p_pos[cached.nei_idx]  # [n_a, K, 2]
        nei_vel = state.p_vel[cached.nei_idx]  # [n_a, K, 2]
        nei_dists = cached.nei_dists  # [n_a, K]
        
        # Direction away from each neighbor
        dir_away = state.p_pos[:, None, :] - nei_pos  # [n_a, K, 2]
        safe_nei_dist = jnp.maximum(nei_dists, 1e-8)  # [n_a, K]
        unit_away = dir_away / safe_nei_dist[:, :, None]  # [n_a, K, 2]
        
        # Repulsion factor: fires when dist < 2*r_avoid (matches reward's spacing violation threshold).
        rep_factor = jnp.where(
            (nei_dists > 0) & (nei_dists < 2.0 * self.r_avoid),
            3.0 * (2.0 * self.r_avoid / safe_nei_dist - 1.0),
            0.0
        )  # [n_a, K]
        
        repulsion = jnp.sum(rep_factor[:, :, None] * unit_away, axis=1)  # [n_a, 2]
        
        # ══════════════════════════════════════════════════════════════════════
        # Velocity sync
        # ══════════════════════════════════════════════════════════════════════
        
        avg_nei_vel = jnp.mean(nei_vel, axis=1)  # [n_a, 2]
        sync = 2.0 * (avg_nei_vel - state.p_vel)  # [n_a, 2]
        
        # ══════════════════════════════════════════════════════════════════════
        # Total
        # ══════════════════════════════════════════════════════════════════════
        
        total = attraction + repulsion + sync
        return jnp.clip(total, -1.0, 1.0)  # [n_a, 2]
