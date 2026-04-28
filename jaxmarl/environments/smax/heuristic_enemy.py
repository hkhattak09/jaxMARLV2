import jax.numpy as jnp
import jax
from flax import struct
import chex
from functools import partial


@struct.dataclass
class HeuristicPolicyState:
    default_target: chex.Array  # the place we are headed for
    last_attacked_enemy: int  # needed to remember where we attacked last
    self_target_slot: int  # own-team action slot for support units

    def __eq__(self, other):
        return jnp.all(other.default_target == self.default_target) & (
            other.last_attacked_enemy == self.last_attacked_enemy
        )


def create_heuristic_policy(
    env, team: int, shoot: bool = True, attack_mode: str = "closest"
):
    """
    Args:
        env (_type_): the SMAX environment to operate in
        team (int): 0 for allies, 1 for enemies
        shoot (bool, optional): Whether or not the agents should shoot. Defaults to True.
        attack_mode (bool, optional):  How the agents should choose targets.
         Options are 'closest' or 'random'. Defaults to 'closest'.

    Returns: a heuristic policy to micromanage SC2 units.
    """
    num_unit_features = len(env.unit_features)
    num_move_actions = env.num_movement_actions

    def get_heuristic_action(
        key: jax.random.PRNGKey, state: HeuristicPolicyState, obs: chex.Array
    ):
        """Generate a heuristic action based on an observation.
        Follows the following scheme:
            -- If you can attack:
                -- Find all the enemies that are in range
                -- Attack one either at random or the closest, depending
                   on the attack mode
            -- If you can't attack:
                -- Go to just past the middle of the enemy's half, or
                   follow a random enemy you can see.
        """
        unit_type = jnp.nonzero(obs[-env.unit_type_bits :], size=1, fill_value=None)[0][
            0
        ]
        initial_state = get_heuristic_policy_initial_state()
        is_initial_state = initial_state == state
        teams = {0: env.num_allies, 1: env.num_enemies}
        team_size = teams[team]
        other_team_size = teams[1 - team]
        total_units = env.num_allies + env.num_enemies
        attack_range = env.unit_type_attack_ranges[unit_type]
        first_enemy_idx = (team_size - 1) * num_unit_features
        own_feats_idx = (total_units - 1) * num_unit_features

        def scaled_position_to_map(position, x_scale, y_scale):
            return position * jnp.array([x_scale, y_scale])

        own_position = scaled_position_to_map(
            obs[own_feats_idx + 1 : own_feats_idx + 3],
            env.map_width,
            env.map_height,
        )
        enemy_positions = jnp.zeros((other_team_size, 2))
        enemy_positions = enemy_positions.at[:, 0].set(
            obs[first_enemy_idx + 1 : own_feats_idx : num_unit_features],
        )
        enemy_positions = enemy_positions.at[:, 1].set(
            obs[first_enemy_idx + 2 : own_feats_idx : num_unit_features]
        )
        enemy_positions = scaled_position_to_map(
            enemy_positions,
            env.unit_type_sight_ranges[unit_type],
            env.unit_type_sight_ranges[unit_type],
        )

        # visible if health is > 0. Otherwise out of range or dead
        visible_enemy_mask = obs[first_enemy_idx:own_feats_idx:num_unit_features] > 0
        shootable_enemy_mask = (
            jnp.linalg.norm(enemy_positions, axis=-1) < attack_range
        ) & visible_enemy_mask
        can_shoot = jnp.any(shootable_enemy_mask)
        is_medivac = env._is_medivac(unit_type)

        ally_health = obs[:first_enemy_idx:num_unit_features]
        ally_positions = jnp.zeros((team_size - 1, 2))
        ally_positions = ally_positions.at[:, 0].set(
            obs[1:first_enemy_idx:num_unit_features],
        )
        ally_positions = ally_positions.at[:, 1].set(
            obs[2:first_enemy_idx:num_unit_features]
        )
        ally_positions = scaled_position_to_map(
            ally_positions,
            env.unit_type_sight_ranges[unit_type],
            env.unit_type_sight_ranges[unit_type],
        )
        ally_features = obs[:first_enemy_idx].reshape((team_size - 1, num_unit_features))
        ally_unit_types = jnp.argmax(
            ally_features[:, 7 : 7 + env.unit_type_bits], axis=-1
        )
        if env.medivac_type_idx is None:
            ally_is_medivac = jnp.zeros((team_size - 1,), dtype=jnp.bool_)
        else:
            ally_is_medivac = ally_unit_types == env.medivac_type_idx
        healable_ally_mask = (
            (ally_health > 0)
            & (ally_health < 1.0)
            & jnp.logical_not(ally_is_medivac)
            & (
                jnp.linalg.norm(ally_positions, axis=-1)
                < env.unit_type_attack_ranges[unit_type]
            )
        )
        can_heal = jnp.any(healable_ally_mask)
        heal_dist = jnp.linalg.norm(ally_positions, axis=-1)
        heal_dist = jnp.where(
            healable_ally_mask,
            heal_dist,
            jnp.linalg.norm(jnp.array([env.map_width, env.map_height])),
        )
        heal_target_obs_idx = jnp.argmin(heal_dist)
        heal_action = heal_target_obs_idx + num_move_actions
        heal_action = jnp.where(
            (state.self_target_slot >= 0)
            & (heal_target_obs_idx >= state.self_target_slot),
            heal_action + 1,
            heal_action,
        )

        key, key_attack = jax.random.split(key)
        random_attack_action = jax.random.choice(
            key_attack,
            jnp.arange(num_move_actions, other_team_size + num_move_actions),
            p=(shootable_enemy_mask / jnp.sum(shootable_enemy_mask)),
        )
        enemy_dist = jnp.linalg.norm(enemy_positions, axis=-1)
        enemy_dist = jnp.where(
            shootable_enemy_mask,
            enemy_dist,
            jnp.linalg.norm(jnp.array([env.map_width, env.map_height])),
        )
        closest_attack_action = jnp.argmin(enemy_dist)
        closest_attack_action += num_move_actions
        new_attack_action = jax.lax.select(
            attack_mode == "random", random_attack_action, closest_attack_action
        )
        # Want to keep attacking the same enemy until it is dead.
        attack_action = jax.lax.select(
            (state.last_attacked_enemy != -1)
            & shootable_enemy_mask[state.last_attacked_enemy],
            state.last_attacked_enemy + num_move_actions,
            new_attack_action,
        )
        attacked_idx = attack_action - num_move_actions
        state = state.replace(
            last_attacked_enemy=jax.lax.select(
                shootable_enemy_mask[attacked_idx], attacked_idx, -1
            )
        )
        # compute the correct movement action.
        random_enemy_target = jax.random.choice(
            key,
            enemy_positions + own_position,
            p=(visible_enemy_mask / jnp.sum(visible_enemy_mask)),
        )
        can_see = jnp.any(visible_enemy_mask)

        # Rotate the current position 180 degrees about the centre of the map
        # to get the default target.
        # This means that in surrounded and reflect scenarios we will always
        # pass through the centre, and therefore are likely to get involved
        # in the action. From there the behaviour of chasing enemies should
        # take over to produce sensible behaviour.
        centre = jnp.array([env.map_width / 2, env.map_height / 2])
        default_target = jax.lax.select(
            is_initial_state,
            jnp.array([[-1, 0], [0, -1]]) @ (own_position - centre) + centre,
            state.default_target,
        )
        state = state.replace(default_target=default_target)
        target = jax.lax.cond(
            can_see, lambda: random_enemy_target, lambda: state.default_target
        )
        vector_to_target = target - own_position
        action_vectors = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        similarity = jnp.dot(action_vectors, vector_to_target)
        move_action = jnp.argmax(similarity)
        medivac_action = jax.lax.cond(
            can_heal & shoot,
            lambda: heal_action,
            lambda: move_action,
        )
        combat_action = jax.lax.cond(
            can_shoot & shoot, lambda: attack_action, lambda: move_action
        )
        return (
            jax.lax.cond(is_medivac, lambda: medivac_action, lambda: combat_action),
            state,
        )

    return get_heuristic_action


def get_heuristic_policy_initial_state(self_target_slot=-1):
    return HeuristicPolicyState(
        default_target=jnp.array([0.0, 0.0]),
        last_attacked_enemy=-1,
        self_target_slot=self_target_slot,
    )
