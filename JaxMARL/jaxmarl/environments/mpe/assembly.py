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


class AssemblyEnv(MultiAgentEnv):
    """JAX Assembly Swarm Environment, functionally identical to AssemblySwarmEnv."""

    def __init__(
        self,
        results_file: str,
        n_a: int = 30,
        topo_nei_max: int = 6,
        num_obs_grid_max: int = 80,
        dt: float = 0.1,
        vel_max: float = 0.8,
        k_ball: float = 30.0,
        k_wall: float = 100.0,
        c_wall: float = 5.0,
        size_a: float = 0.035,
        d_sen: float = 0.4,
        boundary_half: float = 2.4,
        max_steps: int = 200,
    ):
        super().__init__(num_agents=n_a)

        self.n_a = n_a
        self.topo_nei_max = topo_nei_max
        self.num_obs_grid_max = num_obs_grid_max
        self.dt = dt
        self.vel_max = vel_max
        self.k_ball = k_ball
        self.k_wall = k_wall
        self.c_wall = c_wall
        self.size_a = size_a
        self.d_sen = d_sen
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

        # r_avoid: same formula as original __reinit__
        min_n_g   = float(min(n_gs))
        min_l_cell = float(np.min(l_cells_np))
        self.r_avoid = round(np.sqrt(4.0 * min_n_g / (n_a * np.pi)) * min_l_cell, 2)

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
        use_uniform = jax.random.uniform(key_choice, ()) > 0.0  # prob 0.5 each
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

    # ────────────────────────────────────────────────────────────────────────
    # Step
    # ────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: AssemblyState,
        actions: Dict,
    ) -> Tuple[Dict, AssemblyState, Dict, Dict, Dict]:

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

        obs     = self.get_obs(new_state)
        rewards = self.rewards(new_state)

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = jnp.all(done)

        return obs, new_state, rewards, dones, self.eval_metrics(new_state)

    # ────────────────────────────────────────────────────────────────────────
    # Physics
    # ────────────────────────────────────────────────────────────────────────

    def _world_step(self, state: AssemblyState, u: chex.Array):
        """Integrate one timestep. Matches step() in assembly.py (C++ backed)."""
        sf_b2b          = self._ball_to_ball_force(state.p_pos)
        sf_b2w, df_b2w  = self._ball_to_wall_force(state.p_pos, state.p_vel)

        # sensitivity=1 in original, m_a=1
        F = u + sf_b2b + sf_b2w + df_b2w    # [n_a, 2]
        p_vel = state.p_vel + F * self.dt    # ddp = F/1, integrate
        p_vel = jnp.clip(p_vel, -self.vel_max, self.vel_max)
        p_pos = state.p_pos + p_vel * self.dt
        return p_pos, p_vel

    def _ball_to_ball_force(self, p_pos: chex.Array) -> chex.Array:
        """Spring repulsion between overlapping agents (k_ball).

        Matches _sf_b2b_all() in AssemblyEnv.cpp.
        delta[i,j] = p[j] - p[i]; force on i from j is repulsion (-dir_ij).
        """
        # delta[i,j] = p[j] - p[i]
        delta = p_pos[None, :, :] - p_pos[:, None, :]   # [n_a, n_a, 2]
        dist  = jnp.linalg.norm(delta, axis=-1)          # [n_a, n_a]

        overlap    = 2.0 * self.size_a - dist             # [n_a, n_a]
        is_collide = (dist < 2.0 * self.size_a) & (dist > 1e-8)  # exclude self

        safe_dist = jnp.maximum(dist, 1e-8)
        dir_ij    = delta / safe_dist[:, :, None]          # [n_a, n_a, 2]

        # Force on i from j: repulsion away from j  → -dir_ij
        force = (
            is_collide[:, :, None].astype(jnp.float32)
            * overlap[:, :, None]
            * self.k_ball
            * (-dir_ij)
        )  # [n_a, n_a, 2]

        return jnp.sum(force, axis=1)  # [n_a, 2]

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

        # Nearby agents: within d_sen + r_avoid/2 of agent i
        agent_dists = jnp.linalg.norm(state.p_pos - state.p_pos[i], axis=-1)  # [n_a]
        is_nearby   = agent_dists < (self.d_sen + self.r_avoid / 2.0)          # [n_a]

        # Agent-to-grid distances: [n_a, n_g_max]
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)                          # [n_a, n_g_max]

        # Cell is occupied if any nearby agent is within r_avoid/2 of it
        is_occupied_by_nearby = jnp.any(
            is_nearby[:, None] & (a2g_dist < self.r_avoid / 2.0), axis=0
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

    def _reward_single(self, i: int, state: AssemblyState) -> chex.Array:
        # ── in_flag ──────────────────────────────────────────────────────
        grid_rel  = state.grid_center.T - state.p_pos[i]   # [n_g_max, 2]
        grid_dist = jnp.linalg.norm(grid_rel, axis=-1)      # [n_g_max]
        grid_dist_masked = jnp.where(state.valid_mask, grid_dist, jnp.inf)
        min_dist  = jnp.min(grid_dist_masked)
        in_flag   = min_dist < (jnp.sqrt(2.0) * state.l_cell / 2.0)

        # ── is_collision: penalize_interaction (condition[3]) ─────────────
        # C++ uses neighbor_index[agent]: top topo_nei_max nearest within d_sen.
        # We replicate that by sorting and taking the K nearest, then checking
        # if any of those K neighbours is closer than r_avoid.
        agent_dists      = jnp.linalg.norm(state.p_pos - state.p_pos[i], axis=-1)  # [n_a]
        agent_dists_excl = jnp.where(jnp.arange(self.n_a) == i, jnp.inf, agent_dists)

        # Top topo_nei_max nearest (same set as neighbor_index in C++)
        sorted_nei_idx = jnp.argsort(agent_dists_excl)[:self.topo_nei_max]   # [K]
        nei_dists_topo = agent_dists_excl[sorted_nei_idx]                     # [K]
        in_nei_range   = nei_dists_topo < self.d_sen                          # [K]
        is_collision   = jnp.any(in_nei_range & (nei_dists_topo < self.r_avoid))

        # ── is_uniform: penalize_exploration (condition[4]) ───────────────
        # Only evaluated when in_flag=True.
        # Sensed unoccupied cells (same logic as obs):
        in_sensor  = (grid_dist < self.d_sen) & state.valid_mask

        is_nearby  = agent_dists < (self.d_sen + self.r_avoid / 2.0)
        a2g        = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist   = jnp.linalg.norm(a2g, axis=-1)
        is_occupied_by_nearby = jnp.any(
            is_nearby[:, None] & (a2g_dist < self.r_avoid / 2.0), axis=0
        )
        is_occupied           = in_flag & is_occupied_by_nearby
        is_sensed_unoccupied  = in_sensor & ~is_occupied  # [n_g_max]

        # psi = rho_cos_dec(dist, delta=0, r=d_sen)
        psi       = self._rho_cos_dec(grid_dist, self.d_sen)  # [n_g_max]
        psi_valid = jnp.where(is_sensed_unoccupied, psi, 0.0)

        numerator   = jnp.sum(psi_valid[:, None] * grid_rel, axis=0)  # [2]
        denominator = jnp.sum(psi_valid) + 1e-8
        v_exp       = numerator / denominator                           # [2]
        v_exp_norm  = jnp.linalg.norm(v_exp)

        any_sensed = jnp.any(is_sensed_unoccupied)
        is_uniform = in_flag & any_sensed & (v_exp_norm < 0.05)

        # Final reward (all three conditions must hold)
        return jnp.where(in_flag & ~is_collision & is_uniform, 1.0, 0.0)

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
    def coverage_rate(self, state: AssemblyState) -> chex.Array:
        """Fraction of valid target grid cells occupied by at least one agent.

        A cell is considered occupied when any agent centre lies within
        r_avoid / 2 of the cell centre — matching AssemblySwarmWrapper.coverage_rate().
        """
        # a2g[i, j] = distance from agent i to grid cell j
        a2g = state.grid_center.T[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_g_max, 2]
        a2g_dist = jnp.linalg.norm(a2g, axis=-1)                          # [n_a, n_g_max]

        # Cell occupied if ANY agent is within r_avoid/2
        cell_occupied = jnp.any(a2g_dist < self.r_avoid / 2.0, axis=0)   # [n_g_max]

        # Only count valid (non-padding) cells
        n_g = jnp.sum(state.valid_mask.astype(jnp.float32))
        n_occupied = jnp.sum((cell_occupied & state.valid_mask).astype(jnp.float32))
        return n_occupied / n_g

    @partial(jax.jit, static_argnums=[0])
    def distribution_uniformity(self, state: AssemblyState) -> chex.Array:
        """Normalised variance of minimum inter-agent distances.

        For each agent, finds its nearest neighbour distance; computes
        variance across all agents and normalises to [0, 1] — matching
        AssemblySwarmWrapper.distribution_uniformity().
        """
        # Pairwise distances [n_a, n_a]
        delta = state.p_pos[None, :, :] - state.p_pos[:, None, :]  # [n_a, n_a, 2]
        dists = jnp.linalg.norm(delta, axis=-1)                     # [n_a, n_a]

        # Exclude self-distance
        dists_excl = jnp.where(jnp.eye(self.n_a, dtype=bool), jnp.inf, dists)

        # Nearest neighbour distance per agent
        min_dists = jnp.min(dists_excl, axis=1)  # [n_a]

        variance = jnp.var(min_dists)
        min_d    = jnp.min(min_dists)
        max_d    = jnp.max(min_dists)
        # Guard against degenerate case where all nearest distances are equal
        return (variance - min_d) / jnp.maximum(max_d - min_d, 1e-8)

    @partial(jax.jit, static_argnums=[0])
    def voronoi_based_uniformity(self, state: AssemblyState) -> chex.Array:
        """Normalised variance of per-agent Voronoi cell counts.

        Assigns each valid grid cell to its nearest agent (Voronoi partition),
        counts cells per agent, then normalises the variance — matching
        AssemblySwarmWrapper.voronoi_based_uniformity().
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

        variance = jnp.var(voronoi_counts)
        min_c    = jnp.min(voronoi_counts)
        max_c    = jnp.max(voronoi_counts)
        return (variance - min_c) / jnp.maximum(max_c - min_c, 1e-8)

    @partial(jax.jit, static_argnums=[0])
    def eval_metrics(self, state: AssemblyState) -> Dict[str, chex.Array]:
        """Return all three evaluation metrics as a dict."""
        return {
            "coverage_rate":           self.coverage_rate(state),
            "distribution_uniformity": self.distribution_uniformity(state),
            "voronoi_uniformity":      self.voronoi_based_uniformity(state),
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
        # Direction from neighbour toward self (pointing away from each neighbour)
        dir_away      = pos_i - nei_pos                          # [K, 2]
        safe_nei_dist = jnp.maximum(nei_dists, 1e-8)
        unit_away     = dir_away / safe_nei_dist[:, None]        # [K, 2]
        rep_factor    = jnp.where(
            (nei_dists > 0) & (nei_dists < self.r_avoid),
            3.0 * (self.r_avoid / safe_nei_dist - 1.0),
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
