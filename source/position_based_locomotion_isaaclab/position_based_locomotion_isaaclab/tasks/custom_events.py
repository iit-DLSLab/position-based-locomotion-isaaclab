# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv



def randomize_joint_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if env_ids != slice(None) and joint_ids != slice(None):
        env_ids_for_slice = env_ids[:, None]
    else:
        env_ids_for_slice = env_ids

    # sample joint properties from the given ranges and set into the physics simulation
    # joint friction coefficient
    if friction_distribution_params is not None:
        friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )

        # ensure the friction coefficient is non-negative
        friction_coeff = torch.clamp(friction_coeff, min=0.0)

        # Always set static friction (indexed once)
        static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

        # Randomize raw tensors
        #dynamic_friction_coeff = _randomize_prop_by_op(
        #    asset.data.default_joint_dynamic_friction_coeff.clone(),
        #    friction_distribution_params,
        #    env_ids,
        #    joint_ids,
        #    operation=operation,
        #    distribution=distribution,
        #)
        viscous_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_viscous_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )

        # Clamp to non-negative
        #dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
        viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

        # Ensure dynamic ≤ static (same shape before indexing)
        #dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

        # Index once at the end
        #dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
        viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]


        # Newton exposes a single dry-friction value, while PhysX backends also
        # provide a separate dynamic-friction coefficient.
        if hasattr(asset, "write_joint_dynamic_friction_coefficient_to_sim_index"):
            asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=static_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        else:
            asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

    # joint armature
    if armature_distribution_params is not None:
        armature = _randomize_prop_by_op(
            asset.data.default_joint_armature.clone(),
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim_index(
            armature=armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
        )


def randomize_joint_friction_model(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    first_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    second_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model. 
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                friction = actuator.friction_static.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_static[env_ids[:, None], actuator_joint_ids] = friction

                friction = actuator.friction_dynamic.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_dynamic[env_ids[:, None], actuator_joint_ids] = friction

    if armature_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                armature = actuator.armature.to(asset.device).clone()
                armature = _randomize_prop_by_op(
                    armature, armature_distribution_params, env_ids, torch.arange(armature.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.armature[env_ids[:, None], actuator_joint_ids] = armature



def _init_box_goal_cells(self) -> None:
    """Precompute where goals snap to a box center on the ``boxes`` terrain.

    With the curriculum layout, Isaac Lab fills each terrain column with one sub-terrain
    type in order of the proportions, and every box grid is centered on its sub-terrain
    origin. All box terrains therefore share the same cell centers relative to the env origin.
    """
    self._box_cell_centers = None
    generator_cfg = self.cfg.terrain.terrain_generator
    if generator_cfg is None or not generator_cfg.curriculum or "boxes" not in generator_cfg.sub_terrains:
        return

    # Same column -> sub-terrain rule as TerrainGenerator._generate_curriculum_terrains.
    names = list(generator_cfg.sub_terrains)
    proportions = np.array([sub_cfg.proportion for sub_cfg in generator_cfg.sub_terrains.values()])
    cumulative = np.cumsum(proportions / np.sum(proportions))
    column_types = [np.min(np.where(col / generator_cfg.num_cols + 0.001 < cumulative)[0]) for col in range(generator_cfg.num_cols)]
    self._box_terrain_columns = torch.tensor([names[i] == "boxes" for i in column_types], device=self.device)

    # Same grid as mesh_terrains.random_grid_terrain (square, centered on the origin).
    grid_width = generator_cfg.sub_terrains["boxes"].grid_width
    num_boxes = int(generator_cfg.size[0] / grid_width)
    offsets = (torch.arange(num_boxes, device=self.device) + 0.5 - num_boxes / 2) * grid_width
    self._box_cell_centers = torch.cartesian_prod(offsets, offsets)


def _randomize_box_spawn(self, env_ids: torch.Tensor, root_state: torch.Tensor) -> None:
    """Spawn on a random lower flat grid cell, preserving the default base clearance.

    TerrainGenerator stores patches in world coordinates. Index the current
    level after the curriculum update, including episodes replayed at an easier
    level. Only rows corresponding to ``boxes`` environments are modified.
    """
    if self._box_cell_centers is None:
        return
    patches = self._terrain.flat_patches.get("box_spawn")
    if patches is None:
        return
    is_box = self._box_terrain_columns[self._terrain.terrain_types[env_ids]]
    box_ids = env_ids[is_box]
    if box_ids.numel() == 0:
        return
    patch_ids = torch.randint(patches.shape[2], (box_ids.numel(),), device=self.device)
    spawn_positions = patches[
        self._terrain.terrain_levels[box_ids], self._terrain.terrain_types[box_ids], patch_ids
    ]
    root_state[is_box, :3] = spawn_positions + self._robot.data.default_root_state[box_ids, :3]


def _sample_random_goals(self, env_ids: torch.Tensor) -> None:
    """Sample one world-frame goal per episode around the spawn position."""
    num_goals = len(env_ids)
    distance = torch.empty(num_goals, device=self.device).uniform_(*self.cfg.goal_distance_range)
    bearing = torch.empty(num_goals, device=self.device).uniform_(*self.cfg.goal_bearing_range)

    # A random fraction of the episodes starts at the goal with a random heading to track,
    # so the policy learns to rotate in place (the joystick command at deploy time).
    # Drawn per reset, not per env index, which would tie them to the first terrain columns.
    near_goal = torch.rand(num_goals, device=self.device) < self.cfg.near_goal_episode_probability
    distance[near_goal] = torch.empty_like(distance[near_goal]).uniform_(0.0, self.cfg.goal_radius)
    self._near_goal_episodes[env_ids] = near_goal
    self._goal_heading_locked[env_ids] = near_goal
    # Only kept by near-goal episodes: the others follow the bearing to the goal.
    self._goal_headings_w[env_ids] = torch.empty(num_goals, device=self.device).uniform_(-torch.pi, torch.pi)

    offset_w = torch.stack((distance * torch.cos(bearing), distance * torch.sin(bearing)), dim=1)
    self._goal_positions_w[env_ids] = self._robot.data.root_pos_w[env_ids, :2] + offset_w
    _snap_goals_to_box_centers(self, env_ids[~near_goal])


def _snap_goals_to_box_centers(self, env_ids: torch.Tensor) -> None:
    """Move goals on ``boxes`` terrains to a random box center within the goal distance range."""
    if self._box_cell_centers is None:
        return
    env_ids = env_ids[self._box_terrain_columns[self._terrain.terrain_types[env_ids]]]
    centers_w = self._terrain.env_origins[env_ids, None, :2] + self._box_cell_centers
    distances = torch.linalg.vector_norm(centers_w - self._robot.data.root_pos_w[env_ids, None, :2], dim=2)
    in_range = (distances >= self.cfg.goal_distance_range[0]) & (distances <= self.cfg.goal_distance_range[1])

    # Random center among those in range; keep the sampled goal if none is.
    choice = torch.argmax(torch.rand_like(distances) * in_range, dim=1)
    has_center = in_range.any(dim=1)
    self._goal_positions_w[env_ids[has_center]] = centers_w[has_center, choice[has_center]]


def _update_goal_commands(self) -> None:
    """Update the goal command, distance and heading error of every environment.

    The policy command follows AME-2's deployable representation: relative goal position
    in the base frame (clipped in norm), followed by sine/cosine of the heading error.
    Unlike AME-2, the desired heading is the bearing to the goal, so the quadruped
    approaches the target with its front side.
    """
    goal_delta_w = self._goal_positions_w - self._robot.data.root_pos_w[:, :2]
    goal_distance = torch.linalg.vector_norm(goal_delta_w, dim=1)

    # Lock the heading once inside the goal radius (near-goal episodes start locked on
    # their random heading) and unlock only beyond goal_heading_radius. Small drifts or
    # overshoots then neither flip the target nor replace it with atan2 noise.
    self._goal_heading_locked[:] = (self._goal_heading_locked | (goal_distance <= self.cfg.goal_radius)) & (
        goal_distance <= self.cfg.goal_heading_radius
    )
    bearing_w = torch.atan2(goal_delta_w[:, 1], goal_delta_w[:, 0])
    self._goal_headings_w[:] = torch.where(self._goal_heading_locked, self._goal_headings_w, bearing_w)

    _, _, base_yaw_w = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w.torch)
    heading_error = math_utils.wrap_to_pi(self._goal_headings_w - base_yaw_w)
    cos_yaw = torch.cos(base_yaw_w)
    sin_yaw = torch.sin(base_yaw_w)
    self._goal_delta_b[:] = torch.stack(
        (
            cos_yaw * goal_delta_w[:, 0] + sin_yaw * goal_delta_w[:, 1],
            -sin_yaw * goal_delta_w[:, 0] + cos_yaw * goal_delta_w[:, 1],
        ),
        dim=1,
    )
    command_scale = torch.clamp(self.cfg.goal_command_distance_clip / goal_distance.clamp_min(1.0e-6), max=1.0)

    self._commands[:] = torch.cat(
        (
            self._goal_delta_b * command_scale.unsqueeze(1),
            torch.sin(heading_error).unsqueeze(1),
            torch.cos(heading_error).unsqueeze(1),
        ),
        dim=1,
    )
    self._goal_distances[:] = goal_distance
    self._goal_heading_errors[:] = heading_error


def randomize_pace_actuator_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    min_delay: int = 0,
    max_delay: int | None = None,
):
    """Randomize integer delay (in simulation steps) for Pace actuators.

    The sampled delay is applied per-environment at reset through each actuator's
    ``update_time_lags`` method.
    """
    if min_delay < 0:
        raise ValueError(f"min_delay must be >= 0, got {min_delay}.")

    asset: Articulation = env.scene[asset_cfg.name]

    # If max_delay is not specified in the event params, use the maximum default
    # max_delay declared in the Pace actuator configs attached to this asset.
    if max_delay is None:
        inferred_max_delay = 0
        for actuator in asset.actuators.values():
            actuator_cfg = getattr(actuator, "cfg", None)
            actuator_max_delay = getattr(actuator_cfg, "max_delay", None)
            if actuator_max_delay is not None:
                inferred_max_delay = max(inferred_max_delay, int(actuator_max_delay))
        max_delay = inferred_max_delay

    if max_delay < min_delay:
        raise ValueError(f"max_delay must be >= min_delay, got [{min_delay}, {max_delay}].")

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if len(env_ids) == 0:
        return

    delays = torch.randint(
        low=min_delay,
        high=max_delay + 1,
        size=(len(env_ids),),
        dtype=torch.int,
        device=asset.device,
    )

    for actuator in asset.actuators.values():
        if hasattr(actuator, "update_time_lags"):
            actuator.update_time_lags(delays, env_ids)
