"""Custom curriculum utilities for the locomotion task."""

from __future__ import annotations

import torch


def contact_curriculum_active(env) -> bool:
    """Whether the contact curriculum is still in its first phase, see cfg.contact_curriculum_iterations."""
    if not env.cfg.use_contact_curriculum:
        return False
    num_episode_from_start = env.common_step_counter / 24. #self.max_episode_length #HACK this should be taken from rsl rl
    return num_episode_from_start < env.cfg.contact_curriculum_iterations


def update_terrain_levels_with_replay(
    env,
    env_ids: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
    replay_probability: float = 0.2,
    start_level: int = 0,
) -> torch.Tensor:
    """Update curriculum progress and occasionally replay an earlier terrain level.

    The progress level is kept separately from the terrain level assigned to the
    current episode. Therefore, an episode replayed below the progress frontier
    neither promotes nor demotes that frontier.

    Returns:
        A boolean mask identifying environments for which random level sampling
        was selected. The sampled level can still equal the progress level because
        the interval is inclusive.
    """
    if not 0.0 <= replay_probability <= 1.0:
        raise ValueError(
            f"curriculum replay probability must be in [0, 1], got {replay_probability}."
        )

    terrain = env._terrain
    if not hasattr(env, "_curriculum_frontier_levels"):
        env._curriculum_frontier_levels = terrain.terrain_levels.clone()
        env._curriculum_replay_mask = torch.zeros_like(terrain.terrain_levels, dtype=torch.bool)
        env._curriculum_random_level_mask = torch.zeros_like(
            terrain.terrain_levels, dtype=torch.bool
        )

    # Outcomes from an easier replay episode must not alter the true curriculum
    # frontier. Restore that frontier before using Isaac Lab's standard update.
    previous_replay = env._curriculum_replay_mask[env_ids]
    terrain.terrain_levels[env_ids] = env._curriculum_frontier_levels[env_ids]
    terrain.update_env_origins(
        env_ids,
        move_up & ~previous_replay,
        move_down & ~previous_replay,
    )
    frontier_levels = terrain.terrain_levels[env_ids].clone()
    env._curriculum_frontier_levels[env_ids] = frontier_levels

    # Uniform integer sampling over the inclusive interval [start, frontier].
    bounded_start_level = max(0, int(start_level))
    start_levels = torch.minimum(
        torch.full_like(frontier_levels, bounded_start_level), frontier_levels
    )
    interval_sizes = frontier_levels - start_levels + 1
    sampled_levels = start_levels + torch.floor(
        torch.rand(frontier_levels.shape, device=frontier_levels.device) * interval_sizes
    ).to(frontier_levels.dtype)
    random_level_mask = (
        torch.rand(frontier_levels.shape, device=frontier_levels.device) < replay_probability
    )
    assigned_levels = torch.where(random_level_mask, sampled_levels, frontier_levels)

    terrain.terrain_levels[env_ids] = assigned_levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        assigned_levels, terrain.terrain_types[env_ids]
    ]

    env._curriculum_random_level_mask[env_ids] = random_level_mask
    env._curriculum_replay_mask[env_ids] = assigned_levels < frontier_levels
    return random_level_mask
