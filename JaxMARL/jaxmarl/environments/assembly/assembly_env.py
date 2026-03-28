"""
Assembly Swarm Environment - JAX implementation.

Agents must assemble into target formations while avoiding collisions.
Adapted from MARL-LLM C++ environment to pure JAX.
"""

import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box
import chex
from flax import struct
from typing import Tuple, Dict, Optional
from functools import partial


# Default parameters (matches original assembly environment)
DT = 0.1
MAX_STEPS = 200
AGENT_RADIUS = 0.035
AGENT_MASS = 1.0
MAX_SPEED = 0.8
K_BALL = 30.0
K_WALL = 100.0
C_WALL = 5.0
C_AERO = 1.2
DAMPING = 0.25
CONTACT_FORCE = 100.0
CONTACT_MARGIN = 0.1
D_SEN = 3.0  # Sensing range
R_AVOID = 0.15  # Collision avoidance radius
BOUNDARY_WIDTH = 2.4
BOUNDARY_HEIGHT = 2.4


@struct.dataclass
class AssemblyState:
    """State for assembly environment."""
    p_pos: chex.Array  # Agent positions (n_agents, 2)
    p_vel: chex.Array  # Agent velocities (n_agents, 2)
    grid_centers: chex.Array  # Target grid cells (n_grid, 2)
    l_cell: float  # Grid cell size
    done: chex.Array  # Done flags (n_agents,)
    step: int  # Current step


class AssemblyEnv(MultiAgentEnv):
    """Assembly swarm environment - agents form target shapes."""
    
    def __init__(
        self,
        num_agents=30,
        target_shape="circle",  # 'circle', 'line', 'square', or array of grid positions
        num_grid_cells=30,
        cell_size=0.06,
        max_steps=MAX_STEPS,
        dt=DT,
        d_sen=D_SEN,
        r_avoid=R_AVOID,
        boundary_width=BOUNDARY_WIDTH,
        boundary_height=BOUNDARY_HEIGHT,
        **kwargs,
    ):
        super().__init__(num_agents=num_agents)
        
        # Environment parameters
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.max_steps = max_steps
        self.dt = dt
        self.d_sen = d_sen
        self.r_avoid = r_avoid
        self.boundary_width = boundary_width
        self.boundary_height = boundary_height
        
        # Physics parameters
        self.agent_radius = AGENT_RADIUS
        self.agent_mass = AGENT_MASS
        self.max_speed = MAX_SPEED
        self.k_ball = K_BALL
        self.k_wall = K_WALL
        self.c_wall = C_WALL
        self.c_aero = C_AERO
        self.damping = DAMPING
        self.contact_force = CONTACT_FORCE
        self.contact_margin = CONTACT_MARGIN
        
        # Target shape
        self.num_grid_cells = num_grid_cells
        self.cell_size = cell_size
        self.target_shape = target_shape
        
        # Action and observation spaces (continuous control)
        self.action_spaces = {agent: Box(-1.0, 1.0, (2,)) for agent in self.agents}
        
        # Observation: own_vel (2) + own_pos (2) + neighbors (6*2) + grids (10*2)
        obs_dim = 2 + 2 + 6*2 + 10*2
        self.observation_spaces = {agent: Box(-jnp.inf, jnp.inf, (obs_dim,)) for agent in self.agents}
        
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, AssemblyState]:
        """Reset environment."""
        key_agents, key_shape = jax.random.split(key)
        
        # Random agent positions
        p_pos = jax.random.uniform(
            key_agents, (self.num_agents, 2),
            minval=-self.boundary_width * 0.8,
            maxval=self.boundary_width * 0.8
        )
        p_vel = jnp.zeros((self.num_agents, 2))
        
        # Generate target shape
        grid_centers, l_cell = self._generate_target_shape(key_shape)
        
        state = AssemblyState(
            p_pos=p_pos,
            p_vel=p_vel,
            grid_centers=grid_centers,
            l_cell=l_cell,
            done=jnp.full((self.num_agents,), False),
            step=0,
        )
        
        return self.get_obs(state), state
    
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: AssemblyState, actions: Dict
    ) -> Tuple[Dict, AssemblyState, Dict, Dict, Dict]:
        """Step the environment."""
        # Convert actions dict to array
        actions_array = jnp.array([actions[agent] for agent in self.agents])
        
        # Physics step
        p_pos, p_vel = self._physics_step(state, actions_array)
        
        # Compute rewards
        rewards = self._compute_rewards(p_pos, state.grid_centers, state.l_cell)
        
        # Update state
        done = state.step >= self.max_steps
        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            done=jnp.full((self.num_agents,), done),
            step=state.step + 1,
        )
        
        # Get observations
        obs = self.get_obs(state)
        
        # Format outputs
        rewards_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        info = {}
        
        return obs, state, rewards_dict, dones, info
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: AssemblyState) -> Dict:
        """Compute observations for all agents."""
        
        @partial(jax.vmap, in_axes=(0,))
        def _single_agent_obs(agent_idx: int):
            # Own state
            own_vel = state.p_vel[agent_idx]
            own_pos = state.p_pos[agent_idx]
            
            # Nearest 6 neighbors
            rel_pos = state.p_pos - own_pos
            distances = jnp.linalg.norm(rel_pos, axis=-1)
            distances = jnp.where(jnp.arange(self.num_agents) == agent_idx, jnp.inf, distances)
            neighbor_indices = jnp.argsort(distances)[:6]
            neighbor_rel_pos = rel_pos[neighbor_indices]
            
            # Nearest 10 grid cells
            grid_rel_pos = state.grid_centers - own_pos
            grid_dist = jnp.linalg.norm(grid_rel_pos, axis=-1)
            grid_indices = jnp.argsort(grid_dist)[:10]
            grid_rel_pos_nearest = grid_rel_pos[grid_indices]
            
            return jnp.concatenate([
                own_vel.flatten(),
                own_pos.flatten(),
                neighbor_rel_pos.flatten(),
                grid_rel_pos_nearest.flatten(),
            ])
        
        obs_array = _single_agent_obs(jnp.arange(self.num_agents))
        return {agent: obs_array[i] for i, agent in enumerate(self.agents)}
    
    def _physics_step(self, state: AssemblyState, actions: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Update physics (positions and velocities)."""
        # Action forces (simple velocity control)
        action_forces = actions * 5.0  # Scale actions
        
        # Boundary forces
        boundary_forces = self._compute_boundary_forces(state.p_pos, state.p_vel)
        
        # Collision forces (simplified - just between agents)
        collision_forces = self._compute_collision_forces(state.p_pos)
        
        # Total forces
        total_forces = action_forces + boundary_forces + collision_forces
        
        # Integrate
        p_vel_new = state.p_vel * (1 - self.damping) + (total_forces / self.agent_mass) * self.dt
        
        # Clip velocity
        speeds = jnp.linalg.norm(p_vel_new, axis=-1, keepdims=True)
        p_vel_new = jnp.where(speeds > self.max_speed, p_vel_new * self.max_speed / speeds, p_vel_new)
        
        # Update position
        p_pos_new = state.p_pos + p_vel_new * self.dt
        
        return p_pos_new, p_vel_new
    
    def _compute_boundary_forces(self, p_pos: chex.Array, p_vel: chex.Array) -> chex.Array:
        """Compute forces from boundary collisions."""
        # Penetration into walls
        x_min_pen = jnp.maximum(0, self.agent_radius - (p_pos[:, 0] + self.boundary_width))
        x_max_pen = jnp.maximum(0, self.agent_radius - (self.boundary_width - p_pos[:, 0]))
        y_min_pen = jnp.maximum(0, self.agent_radius - (p_pos[:, 1] + self.boundary_height))
        y_max_pen = jnp.maximum(0, self.agent_radius - (self.boundary_height - p_pos[:, 1]))
        
        # Spring-damper forces
        fx = self.k_wall * (x_min_pen - x_max_pen) - self.c_wall * p_vel[:, 0]
        fy = self.k_wall * (y_min_pen - y_max_pen) - self.c_wall * p_vel[:, 1]
        
        return jnp.stack([fx, fy], axis=-1)
    
    def _compute_collision_forces(self, p_pos: chex.Array) -> chex.Array:
        """Compute agent-agent collision forces (simplified)."""
        # Pairwise distances
        rel_pos = p_pos[:, None, :] - p_pos[None, :, :]  # (n, n, 2)
        distances = jnp.linalg.norm(rel_pos, axis=-1)  # (n, n)
        
        # Collision threshold
        collision_dist = 2 * self.agent_radius
        penetration = jnp.maximum(0, collision_dist - distances)
        
        # Force direction (normalized)
        safe_dist = jnp.maximum(distances, 1e-8)
        directions = rel_pos / safe_dist[:, :, None]
        
        # Repulsion forces
        force_mag = self.k_ball * penetration
        forces = directions * force_mag[:, :, None]
        
        # Sum forces from all neighbors (excluding self)
        mask = jnp.eye(self.num_agents)
        forces = forces * (1 - mask[:, :, None])
        total_forces = jnp.sum(forces, axis=1)
        
        return total_forces
    
    def _compute_rewards(self, p_pos: chex.Array, grid_centers: chex.Array, l_cell: float) -> chex.Array:
        """Compute rewards for all agents."""
        # Check if in target region
        rel_pos = grid_centers[None, :, :] - p_pos[:, None, :]
        distances = jnp.linalg.norm(rel_pos, axis=-1)
        min_dist = jnp.min(distances, axis=1)
        in_target = min_dist < l_cell
        
        # Check collisions
        pairwise_dist = jnp.linalg.norm(p_pos[:, None, :] - p_pos[None, :, :], axis=-1)
        collision_dist = 2 * self.r_avoid
        colliding = jnp.any(
            (pairwise_dist < collision_dist) & (jnp.arange(self.num_agents)[:, None] != jnp.arange(self.num_agents)[None, :]),
            axis=1
        )
        
        # Reward: in target AND not colliding
        rewards = (in_target & ~colliding).astype(jnp.float32)
        
        return rewards
    
    def _generate_target_shape(self, key: chex.PRNGKey) -> Tuple[chex.Array, float]:
        """Generate target shape grid."""
        if self.target_shape == "circle":
            angles = jnp.linspace(0, 2 * jnp.pi, self.num_grid_cells, endpoint=False)
            radius = (self.num_grid_cells * self.cell_size) / (2 * jnp.pi)
            grid_centers = jnp.stack([
                radius * jnp.cos(angles),
                radius * jnp.sin(angles)
            ], axis=-1)
        elif self.target_shape == "line":
            x_coords = jnp.linspace(-self.num_grid_cells * self.cell_size / 2,
                                   self.num_grid_cells * self.cell_size / 2,
                                   self.num_grid_cells)
            grid_centers = jnp.stack([x_coords, jnp.zeros(self.num_grid_cells)], axis=-1)
        elif self.target_shape == "square":
            side_len = int(jnp.sqrt(self.num_grid_cells))
            x = jnp.linspace(-side_len * self.cell_size / 2, side_len * self.cell_size / 2, side_len)
            y = jnp.linspace(-side_len * self.cell_size / 2, side_len * self.cell_size / 2, side_len)
            xx, yy = jnp.meshgrid(x, y)
            grid_centers = jnp.stack([xx.flatten()[:self.num_grid_cells],
                                     yy.flatten()[:self.num_grid_cells]], axis=-1)
        else:
            # Random positions
            grid_centers = jax.random.uniform(
                key, (self.num_grid_cells, 2),
                minval=-self.boundary_width * 0.5,
                maxval=self.boundary_width * 0.5
            )
        
        return grid_centers, self.cell_size
