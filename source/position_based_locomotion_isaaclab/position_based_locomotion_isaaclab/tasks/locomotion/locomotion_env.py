# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import gymnasium as gym
import math
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    Imu,
    Pva,
    PvaCfg,
    MultiMeshRayCaster,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
    RayCaster,
    RayCasterCamera,
    RayCasterCameraCfg,
    RayCasterCfg,
    TiledCamera,
    TiledCameraCfg,
    patterns,
)
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModel

from isaaclab_newton.physics import NewtonCfg


from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

from position_based_locomotion_isaaclab.tasks import (
    custom_curriculums,
    custom_events,
    custom_observations,
    custom_rewards,
)
from position_based_locomotion_isaaclab.tasks.supervised_learning_networks import FrozenRandomMlpEncoder, create_supervised_network

class LocomotionEnv(DirectRLEnv):

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._edge_map_visualizer = None
        self._feet_edge_swing_visualizer = None
        self._goal_visualizer = None
        # Isaac Lab 3.0: the InteractiveScene owns the construction and the cloning of the assets and sensors,
        # so they must be declared on cfg.scene before the scene is created
        self._populate_scene_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        # Output of the exponential moving average action filter
        self._filtered_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Position-based command: clipped relative goal x/y in the base frame,
        # followed by sine/cosine of the heading error.
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_positions_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_headings_w = torch.zeros(self.num_envs, device=self.device)
        self._goal_distances = torch.full(
            (self.num_envs,), self.cfg.goal_distance_range[1], device=self.device
        )
        self._goal_heading_errors = torch.zeros(self.num_envs, device=self.device)
        self._goal_delta_b = torch.zeros(self.num_envs, 2, device=self.device)  # unclipped, for the critic
        self._goal_heading_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._near_goal_episodes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        custom_events._init_box_goal_cells(self)

        # Swing peak
        self._swing_peak = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        self._swing_peak_periodic = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        
        # Desired Hip Offset
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        
        # Periodic gait
        self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0


        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)

        # RMA
        if(cfg.use_rma == True):
            self._rma_network = create_supervised_network(
                cfg.rma_observation_space,
                cfg.rma_output_space,
                network_type=getattr(cfg, "rma_network_type", "mlp"),
                sequence_length=cfg.rma_history_length,
            )
            self._rma_network.to(self.device)
            
            if self.cfg.rma_use_latent_space:
                self._rma_latent_encoder = FrozenRandomMlpEncoder(
                    cfg.rma_privileged_observation_space,
                    cfg.rma_output_space,
                    hidden_features=getattr(cfg, "rma_latent_encoder_hidden_features", 128),
                    seed=getattr(cfg, "rma_latent_encoder_seed", 0),
                )
                self._rma_latent_encoder.to(self.device)
            self._observation_history_rma = torch.zeros(self.num_envs, cfg.rma_history_length, cfg.single_rma_observation_space, device=self.device)

        # Learned State Estimator
        if(cfg.use_concurrent_state_est == True):
            self._concurrent_state_est_network = create_supervised_network(
                cfg.concurrent_state_est_observation_space,
                cfg.concurrent_state_est_output_space,
                network_type=getattr(cfg, "concurrent_state_est_network_type", "mlp"),
                sequence_length=cfg.concurrent_state_est_history_length,
            )
            self._concurrent_state_est_network.to(self.device)
            self._observation_history_concurrent_state_est = torch.zeros(self.num_envs, cfg.concurrent_state_est_history_length, cfg.single_concurrent_state_est_observation_space, device=self.device)

        # Observation noise with a separate std for each term, see cfg.observation_noise_std.
        # The terms must follow the same order used to build the observations
        if self.cfg.observation_noise_model:
            num_joints = cfg.action_space
            proprio_terms = [
                ("base_linear", 3),
                ("base_ang_vel", 3),
                ("projected_gravity", 3),
                ("commands", 4),
                ("joint_pos", num_joints),
                ("joint_vel", num_joints),
                ("actions", num_joints),
            ]
            policy_terms = (proprio_terms + [("clock", 8 if cfg.use_clock_signal else 0)]) * cfg.history_length
            # the height map takes whatever is left of the observation space
            rma_size = cfg.rma_output_space if cfg.use_rma else 0
            height_map_size = cfg.observation_space - sum(size for _, size in policy_terms) - rma_size
            policy_terms += [("height_map", height_map_size), ("rma", rma_size)]

            self._observation_noise_model = self._make_observation_noise_model(policy_terms)
            if cfg.use_rma:
                self._observation_noise_model_rma = self._make_observation_noise_model(
                    proprio_terms * cfg.rma_history_length
                )
            if cfg.use_concurrent_state_est:
                self._observation_noise_model_concurrent_state_est = self._make_observation_noise_model(
                    proprio_terms * cfg.concurrent_state_est_history_length
                )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_height_exp",
                "track_lin_vel_z_l2",
                "track_orientation_l2",
                "track_ang_vel_xy_l2",
                "track_ang_vel_z_exp",
                "goal_position_tracking",
                "goal_heading_tracking",
                "moving_to_goal",

                "undesired_contacts",
                "action_rate_l2",
                "action_smoothness_l2",
                
                "joints_hip_pos_l2",
                "joints_thigh_pos_l2",
                "joints_calf_pos_l2",
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",
                
                "feet_air_time",
                "feet_air_time_variance",

                "feet_height_clearance_periodic",
                "feet_height_clearance_aperiodic",
                "feet_height_clearance_mujoco_periodic",
                "feet_height_clearance_mujoco_aperiodic",
                "feet_swing_trajectory_periodic",
                "feet_swing_trajectory_aperiodic",
                "feet_slide",
                "feet_landing_vel_l2",
                "feet_to_hip_distance_l2",
                "feet_edge_stance",
                "feet_edge_swing",
                "feet_vertical_surface_contacts",

                "periodic_contact_suggestion",
                "stance_contact_suggestion",
            ]
        }
        # Get specific body indices
        self._base_contact_sensor_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)
        self._thigh_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"], preserve_order=True)
        # Bodies penalized by the undesired contacts reward during the contact curriculum (the head is part of the base)
        self._undesired_contact_body_ids = self._base_contact_sensor_id + self._hip_contact_sensor_ids

        
        self._feet_ids_robot, _ = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_ids_robot, _ = self._robot.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)

        # Ensure the order is consistent with the one expected in the cfg
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]

        # Nominal (pre-randomization) explicit-actuator PD gains. Captured here, before any
        # "reset" mode event has run, since asset.data.default_joint_stiffness/damping is a
        # deprecated live snapshot that reads 0 for explicit actuators like PaceDCMotor
        # (the solver's own PD gains are zeroed; the actuator computes effort in Python).
        self._nominal_actuator_stiffness = {}
        self._nominal_actuator_damping = {}
        for joint_type in ("hip", "thigh", "calf"):
            actuator = self._robot.actuators[joint_type]
            self._nominal_actuator_stiffness[joint_type] = actuator.stiffness.clone()
            self._nominal_actuator_damping[joint_type] = actuator.damping.clone()

        if (
            getattr(self.cfg, "visualize_edge_map", False)
            or getattr(self.cfg, "visualize_feet_edge_swing_zone", False)
            or getattr(self.cfg, "visualize_goal", False)
        ):
            self.set_debug_vis(True)


    @staticmethod
    def _populate_scene_cfg(cfg):
        """Declare the robot, the sensors, the terrain and the light on cfg.scene.

        In Isaac Lab 3.0 the InteractiveScene builds the clone plan from the declared entities, constructs them and
        replicates env_0 (including the PhysX collision filtering with the terrain). The entities keep the names used
        so far, and the env cfg fields (cfg.robot, cfg.terrain, ...) are left in place, so the rest of the code and the
        hydra overrides are unchanged.
        """
        scene = cfg.scene
        scene.robot = cfg.robot
        scene.contact_sensor = cfg.contact_sensor

        # The head collision shapes belong to the base body, so they need shape-level contact sensing (Newton only).
        # PhysX would report the whole base instead, so there the head is not penalized.
        if getattr(cfg, "head_contact_sensor", None) is not None:
            if isinstance(cfg.sim.physics, NewtonCfg):
                scene.head_contact_sensor = cfg.head_contact_sensor
            else:
                print("[WARN] Head contact sensing needs the Newton backend; head contacts are not penalized.")

        # Keep the base-centered scanner for base-height and terrain-orientation terms.
        scene.pose_height_scanner = cfg.pose_height_scanner

        # Use one small height map centered on each foot for the clearance rewards.
        for foot_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            # Preserve the configured hierarchy, replacing the leg prefix in each link.
            leg_prefix = foot_name.split("_", 1)[0]
            scanner_cfg = cfg.foot_height_scanner.replace(
                prim_path=cfg.foot_height_scanner.prim_path.replace("FL_", f"{leg_prefix}_"),
                visualizer_cfg=cfg.foot_height_scanner.visualizer_cfg.replace(
                    prim_path=f"/Visuals/{foot_name}HeightScanner"
                ),
            )
            setattr(scene, f"{foot_name.lower()}_height_scanner", scanner_cfg)

        # Add the perceptive scanner only for vision-based locomotion.
        if getattr(cfg, "perceptive_height_scanner", None) is not None:
            scene.perceptive_height_scanner = cfg.perceptive_height_scanner
        # The edge scanner is only used by the feet edge reward (never observed), so it is available for blind locomotion as well
        if getattr(cfg, "edge_height_scanner", None) is not None:
            scene.edge_height_scanner = cfg.edge_height_scanner
        # we add a depth camera if needed for vision-based locomotion
        if getattr(cfg, "use_depth_camera", False):
            scene.depth_camera = cfg.depth_camera

        # we add an imu
        scene.imu = cfg.imu
        # Report ideal projected gravity in the same frame as the IMU measurements.
        scene.pva = PvaCfg(
            prim_path=cfg.imu.prim_path,
            update_period=cfg.imu.update_period,
            offset=PvaCfg.OffsetCfg(pos=cfg.imu.offset.pos, rot=cfg.imu.offset.rot),
        )

        # the scene sets num_envs and env_spacing of the terrain
        scene.terrain = cfg.terrain
        # add lights
        scene.light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )

    def _setup_scene(self):
        # The entities are constructed and cloned by the InteractiveScene, see _populate_scene_cfg
        self._robot = self.scene["robot"]
        self._contact_sensor = self.scene["contact_sensor"]
        if "head_contact_sensor" in self.scene.sensors:
            self._head_contact_sensor = self.scene["head_contact_sensor"]
        self._pose_height_scanner = self.scene["pose_height_scanner"]
        self._foot_height_scanners = [
            self.scene[f"{foot_name.lower()}_height_scanner"] for foot_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        ]
        if getattr(self.cfg, "perceptive_height_scanner", None) is not None:
            self._perceptive_height_scanner = self.scene["perceptive_height_scanner"]
        if getattr(self.cfg, "edge_height_scanner", None) is not None:
            self._edge_height_scanner = self.scene["edge_height_scanner"]
        if getattr(self.cfg, "use_depth_camera", False):
            self._depth_camera = self.scene["depth_camera"]
        self._imu = self.scene["imu"]
        self._pva = self.scene["pva"]
        self._terrain = self.scene.terrain


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        default_joint_pos_ordered = self._robot.data.default_joint_pos[:, self._ids_joints_order]
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action with an exponential moving average (low-pass on the filter output, not on the raw actions)
        if(self.cfg.use_filter_actions):
            alpha = self.cfg.action_filter_alpha
            self._filtered_actions = alpha * self._actions + (1 - alpha) * self._filtered_actions
            self._processed_actions = self.cfg.action_scale * self._filtered_actions + default_joint_pos_ordered
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + default_joint_pos_ordered


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._ids_joints_order)


    def _make_observation_noise_model(self, terms: list[tuple[str, int]]) -> NoiseModel:
        # Expand the (noise std, bias std) of each term to one value per observation dimension
        noise_std = [self.cfg.observation_noise_std[name][0] for name, size in terms for _ in range(size)]
        bias_std = [self.cfg.observation_noise_std[name][1] for name, size in terms for _ in range(size)]

        noise_model_cfg = copy.deepcopy(self.cfg.observation_noise_model)
        noise_model_cfg.noise_cfg.std = torch.tensor(noise_std, device=self.device)
        noise_model_cfg.bias_noise_cfg.std = torch.tensor(bias_std, device=self.device)
        noise_model = noise_model_cfg.class_type(noise_model_cfg, num_envs=self.num_envs, device=self.device)

        # A first call sizes the bias to the observation, otherwise the first reset fails
        noise_model(torch.zeros(self.num_envs, len(noise_std), device=self.device))
        return noise_model

    def _get_observations(self) -> dict:
        custom_events._update_goal_commands(self)


        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            self._phase_signal += self.step_dt * self._step_freq
            self._phase_signal = self._phase_signal % 1.0
            # Encode the phase as sin/cos so the signal stays continuous at the 1 -> 0 wrap: [sin FL..RR, cos FL..RR]
            clock_data = torch.cat([torch.sin(2.0 * torch.pi * self._phase_signal), torch.cos(2.0 * torch.pi * self._phase_signal)], dim=1)
            # for all the envs that are not moving, we put -1 (outside the unit circle, so distinguishable from any phase)
            should_move = custom_rewards._should_move(self)
            clock_data[:, :] = clock_data[:, :]*should_move.unsqueeze(1).expand(-1, 8) + -1.0* ~should_move.unsqueeze(1).expand(-1, 8)
            

        # Choosing the main source of observation
        if(self.cfg.use_concurrent_state_est):
            # If concurrent SE/Learned State Estimator, we predict linear and angular vel from IMU
            base_linear = custom_observations._get_concurrent_state_estimation(self)
            base_ang_vel = self._imu.data.ang_vel_b
            projected_gravity_b = self._pva.data.projected_gravity_b
        elif(self.cfg.use_imu):
            # Using directly the IMU
            base_linear = self._imu.data.lin_acc_b
            base_ang_vel = self._imu.data.ang_vel_b
            projected_gravity_b = self._pva.data.projected_gravity_b
        else:
            #Using a model-based state estimation
            base_linear = self._robot.data.root_lin_vel_b
            base_ang_vel = self._robot.data.root_ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b
        
        
        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    base_linear,
                    base_ang_vel,
                    projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order],
                    self._actions,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)


        observations = {"common": obs}


        # Add heightmap data to obs if needed
        if(getattr(self.cfg, "perceptive_height_scanner", None) is not None):
            height_data = (
                self._perceptive_height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self._perceptive_height_scanner.data.ray_hits_w[..., 2]
                - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)   


        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            remaining_time_s = torch.clamp(
                (self.max_episode_length - self.episode_length_buf).to(torch.float) * self.step_dt,
                min=0.0,
            )
            critic_command = torch.cat(
                (
                    self._goal_delta_b,
                    torch.sin(self._goal_heading_errors).unsqueeze(1),
                    torch.cos(self._goal_heading_errors).unsqueeze(1),
                    remaining_time_s.unsqueeze(1),
                ),
                dim=1,
            )
            obs_critic = custom_observations._get_privileged_observation_asymmetric(self)
            # The critic gets a 5-D unclipped command with remaining seconds.
            observations["critic"] = torch.cat((obs, critic_command, obs_critic), dim=-1)
        else:
            observations["critic"] = obs


        # If RMA, we add some other predicted obs AFTER the critic asymmetric obs to avoid duplication
        if(self.cfg.use_rma):
            # Predict the RMA observation
            obs_rma = custom_observations._get_rma(self)
            obs = torch.cat((obs, obs_rma), dim=-1)


        # Actor OBS - here after the critic to avoid duplication with rma obs
        # if asymmetric ppo is used
        observations["policy"] = obs    
        # --------------------------------------------------------------------------------------------
        return observations


    def _get_rewards(self) -> torch.Tensor:

        track_height_exp = custom_rewards.track_height_exp(self)
        track_lin_vel_z_l2 = custom_rewards.track_lin_vel_z_l2(self)
        track_orientation_l2 = custom_rewards.track_orientation_l2(self)
        track_ang_vel_xy_l2 = custom_rewards.track_ang_vel_xy_l2(self)
        track_ang_vel_z_exp = custom_rewards.track_ang_vel_z_exp(self)
        goal_position_tracking = custom_rewards.goal_position_tracking(self)
        goal_heading_tracking = custom_rewards.goal_heading_tracking(self)
        moving_to_goal = custom_rewards.moving_to_goal(self)

        undesired_contacts = custom_rewards.undesired_contacts(self)
        action_rate_l2 = custom_rewards.action_rate_l2(self)
        action_smoothness_l2 = custom_rewards.action_smoothness_l2(self)

        joints_hip_pos_l2 = custom_rewards.joints_hip_pos_l2(self)
        joints_thigh_pos_l2 = custom_rewards.joints_thigh_pos_l2(self)
        joints_calf_pos_l2 = custom_rewards.joints_calf_pos_l2(self)
        joints_acc_l2 = custom_rewards.joints_acc_l2(self)
        joints_torques_l2 = custom_rewards.joints_torques_l2(self)
        joints_energy_l1 = custom_rewards.joints_energy_l1(self)

        feet_air_time = custom_rewards.feet_air_time(self)
        feet_air_time_variance = custom_rewards.feet_air_time_variance(self)

        feet_slide = custom_rewards.feet_slide(self)
        feet_landing_vel_l2 = custom_rewards.feet_landing_vel_l2(self)
        feet_edge_stance = custom_rewards.feet_edge_stance(self)
        feet_edge_swing = custom_rewards.feet_edge_swing(self)
        periodic_contact_suggestion = custom_rewards.periodic_contact_suggestion(self)
        stance_contact_suggestion = custom_rewards.stance_contact_suggestion(self)
        feet_height_clearance_mujoco_aperiodic = custom_rewards.feet_height_clearance_mujoco_aperiodic(self)
        feet_height_clearance_mujoco_periodic = custom_rewards.feet_height_clearance_mujoco_periodic(self)
        feet_swing_trajectory_periodic = custom_rewards.feet_swing_trajectory_periodic(self)
        feet_swing_trajectory_aperiodic = custom_rewards.feet_swing_trajectory_aperiodic(self)
        feet_height_clearance_periodic = custom_rewards.feet_height_clearance_periodic(self)
        feet_height_clearance_aperiodic = custom_rewards.feet_height_clearance_aperiodic(self)
        feet_to_hip_distance_l2 = custom_rewards.feet_to_hip_distance_l2(self)
        feet_vertical_surface_contacts = custom_rewards.feet_vertical_surface_contacts(self)

        rewards = {
            "track_height_exp": track_height_exp * self.cfg.height_reward_scale * self.step_dt,
            "track_lin_vel_z_l2": track_lin_vel_z_l2 * self.cfg.z_vel_reward_scale * self.step_dt,
            "track_orientation_l2": track_orientation_l2 * self.cfg.orientation_reward_scale * self.step_dt,
            "track_ang_vel_xy_l2": track_ang_vel_xy_l2 * self.cfg.ang_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": track_ang_vel_z_exp * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "goal_position_tracking": goal_position_tracking * self.cfg.goal_position_tracking_reward_scale * self.step_dt,
            "goal_heading_tracking": goal_heading_tracking * self.cfg.goal_heading_tracking_reward_scale * self.step_dt,
            "moving_to_goal": moving_to_goal * self.cfg.moving_to_goal_reward_scale * self.step_dt,

            "undesired_contacts": undesired_contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "action_rate_l2": action_rate_l2 * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness_l2 * self.cfg.action_smoothness_reward_scale * self.step_dt,

            "joints_hip_pos_l2": joints_hip_pos_l2 * self.cfg.joints_hip_position_reward_scale * self.step_dt,
            "joints_thigh_pos_l2": joints_thigh_pos_l2 * self.cfg.joints_thigh_position_reward_scale * self.step_dt,
            "joints_calf_pos_l2": joints_calf_pos_l2 * self.cfg.joints_calf_position_reward_scale * self.step_dt,
            "joints_acc_l2": joints_acc_l2 * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques_l2 * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy_l1 * self.cfg.joints_energy_reward_scale * self.step_dt,

            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "feet_air_time_variance": feet_air_time_variance * self.cfg.feet_air_time_variance_reward_scale * self.step_dt,
            
            "feet_height_clearance_aperiodic": feet_height_clearance_aperiodic * self.cfg.feet_height_clearance_aperiodic_reward_scale * self.step_dt,
            "feet_height_clearance_periodic": feet_height_clearance_periodic * self.cfg.feet_height_clearance_periodic_reward_scale * self.step_dt,
            "feet_height_clearance_mujoco_aperiodic": feet_height_clearance_mujoco_aperiodic * self.cfg.feet_height_clearance_mujoco_aperiodic_reward_scale * self.step_dt,
            "feet_height_clearance_mujoco_periodic": feet_height_clearance_mujoco_periodic * self.cfg.feet_height_clearance_mujoco_periodic_reward_scale * self.step_dt,
            "feet_swing_trajectory_periodic": feet_swing_trajectory_periodic * self.cfg.feet_swing_trajectory_periodic_reward_scale * self.step_dt,
            "feet_swing_trajectory_aperiodic": feet_swing_trajectory_aperiodic * self.cfg.feet_swing_trajectory_aperiodic_reward_scale * self.step_dt,
            
            "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale * self.step_dt,
            "feet_landing_vel_l2": feet_landing_vel_l2 * self.cfg.feet_landing_vel_reward_scale * self.step_dt,
            "feet_to_hip_distance_l2": feet_to_hip_distance_l2 * self.cfg.feet_to_hip_distance_reward_scale * self.step_dt,
            "feet_edge_stance": feet_edge_stance * self.cfg.feet_edge_stance_reward_scale * self.step_dt,
            "feet_edge_swing": feet_edge_swing * getattr(self.cfg, "feet_edge_swing_reward_scale", 0.0) * self.step_dt,
            "feet_vertical_surface_contacts": feet_vertical_surface_contacts * self.cfg.feet_vertical_surface_contacts_reward_scale * self.step_dt,

            "periodic_contact_suggestion": periodic_contact_suggestion * self.cfg.periodic_contact_suggestion_reward_scale * self.step_dt,
            "stance_contact_suggestion": stance_contact_suggestion * self.cfg.stance_contact_suggestion_reward_scale * self.step_dt,
            
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Check for NaNs and Infs
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("NaN or Inf detected in reward computation. Setting reward to zero for affected environments.")
            breakpoint()  # For debugging purposes
            reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_contact_sensor_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died = torch.logical_or(died_check_base, died_check_hips)
        # After the contact curriculum, contacts do not terminate the episode anymore
        if not custom_curriculums.contact_curriculum_active(self):
            died = torch.zeros_like(died)
        # A fallen robot always terminates the episode
        base_tilt = torch.acos(torch.clamp(-self._robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        self._tilt_terminated = base_tilt > self.cfg.tilt_termination_angle
        died = torch.logical_or(died, self._tilt_terminated)
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Isaac Lab 3.0 compat: ids may arrive (or _ALL_INDICES may be) warp arrays
        def _to_torch_ids(ids):
            if ids is not None and not torch.is_tensor(ids):
                import warp as wp
                ids = wp.to_torch(ids)
            return ids.to(dtype=torch.long) if ids is not None else ids

        env_ids = _to_torch_ids(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = _to_torch_ids(self._robot._ALL_INDICES)

        final_goal_distance = self._goal_distances[env_ids].clone()
        goal_reached = final_goal_distance <= self.cfg.goal_radius*1.2
        # Episodes that started at the goal reach it for free and say nothing about the terrain.
        far_goal_episode = ~self._near_goal_episodes[env_ids]

        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            valid_episode = (self.episode_length_buf[env_ids] > 0) & far_goal_episode
            # Use the same goal-reaching condition reported in Episode_Metrics/goal_reached.
            move_up = valid_episode & goal_reached
            move_down = valid_episode & (final_goal_distance > self.cfg.goal_radius*3.0)
            move_down *= ~move_up
            custom_curriculums.update_terrain_levels_with_replay(
                self,
                env_ids,
                move_up,
                move_down,
                replay_probability=getattr(self.cfg, "curriculum_replay_probability", 0.2),
                start_level=getattr(self.cfg, "curriculum_start_level", 0),
            )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        # Reset actions and action filtering
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0
        self._filtered_actions[env_ids] = 0.0
        
        # Reset swing peak
        self._swing_peak[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self._swing_peak_periodic[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0

        # Reset observation history
        self._observation_history[env_ids] *= 0.0

        # Reset obs and noise concurrent
        if(self.cfg.use_concurrent_state_est):
            self._observation_history_concurrent_state_est[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est.reset(env_ids)
        
        # Reset obs and noise rma
        if(self.cfg.use_rma):
            self._observation_history_rma[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.2, 0.2)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        custom_events._randomize_box_spawn(self, env_ids, default_root_state)
        default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample a new fixed position goal only after the new base pose is written.
        custom_events._sample_random_goals(self, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        if hasattr(self, "_tilt_terminated"):
            # base_contact counts every termination, tilt the ones of a fallen robot
            extras["Episode_Termination/tilt"] = torch.count_nonzero(self._tilt_terminated[env_ids]).item()
        if torch.any(far_goal_episode):
            # Skipped when only near-goal episodes reset; rsl_rl averages over the steps that log it.
            extras["Episode_Metrics/final_goal_distance"] = torch.mean(final_goal_distance[far_goal_episode])
            extras["Episode_Metrics/goal_reached"] = torch.mean(goal_reached[far_goal_episode].to(torch.float))
        extras["Episode_Curriculum/contact_curriculum_active"] = float(custom_curriculums.contact_curriculum_active(self))
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
            extras["Episode_Curriculum/frontier_levels"] = torch.mean(
                self._curriculum_frontier_levels.float()
            )
            extras["Episode_Curriculum/random_level_fraction"] = torch.mean(
                self._curriculum_random_level_mask[env_ids].float()
            )
        
        self.extras["log"].update(extras)


    def _set_debug_vis_impl(self, debug_vis: bool):
        custom_rewards._set_debug_vis_impl(self, debug_vis)

        if not getattr(self.cfg, "visualize_goal", False):
            if self._goal_visualizer is not None:
                self._goal_visualizer.set_visibility(False)
            return

        if debug_vis:
            if self._goal_visualizer is None:
                marker_radius = getattr(self.cfg, "goal_visualization_radius", 0.12)
                goal_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Goal",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                        "goal_reached": sim_utils.SphereCfg(
                            radius=marker_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        ),
                    },
                )
                self._goal_visualizer = VisualizationMarkers(goal_marker_cfg)
            self._goal_visualizer.set_visibility(True)
        elif self._goal_visualizer is not None:
            self._goal_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        custom_rewards._debug_vis_callback(self, event)

        if self._goal_visualizer is None or not self._goal_visualizer.is_visible():
            return

        goal_positions_w = torch.zeros(self.num_envs, 3, device=self.device)
        goal_positions_w[:, :2] = self._goal_positions_w
        estimated_ground_height = self._robot.data.root_pos_w[:, 2] - self.cfg.desired_base_height
        goal_positions_w[:, 2] = estimated_ground_height + getattr(self.cfg, "goal_visualization_height_offset", 0.15)
        marker_indices = (self._goal_distances <= self.cfg.goal_radius).long()
        self._goal_visualizer.visualize(translations=goal_positions_w, marker_indices=marker_indices)
