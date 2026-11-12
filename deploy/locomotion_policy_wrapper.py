# Description: Wrapper of the locomotion policy

# Authors:
# Giulio Turrisi

import time
import copy
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from tqdm import tqdm
import mujoco
import onnxruntime as ort

from deploy import config

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")


def load_supervised_network(checkpoint_path):
    """Load a supervised network (concurrent state estimator, RMA) as a function obs -> prediction.

    Uses the ONNX export saved next to the checkpoint (same name, .onnx) with onnxruntime.
    """
    onnx_path = os.path.splitext(checkpoint_path)[0] + ".onnx"
    options = ort.SessionOptions()
    # Small network called once per step: one thread, no worker threads spinning between the calls
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Network loaded from {onnx_path}")
    return lambda obs: session.run(None, {input_name: np.asarray(obs, dtype=np.float32).reshape(1, -1)})[0].squeeze()


class LocomotionPolicyWrapper:
    def __init__(self, mjModel):

        self.policy = ort.InferenceSession(config.locomotion_policy_folder_path + "/exported/policy.onnx")
        self.Kp_walking = config.Kp_walking
        self.Kd_walking = config.Kd_walking
        self.Kp_stand_up_and_down = config.Kp_stand_up_and_down
        self.Kd_stand_up_and_down = config.Kd_stand_up_and_down

        self.RL_FREQ = 1./(config.training_env["sim"]["dt"]*config.training_env["decimation"])  # Hz, frequency of the RL controller


        # RL controller initialization -------------------------------------------------------------
        self.action_scale = config.training_env["action_scale"]
        self.past_rl_actions = np.zeros(12) # only legs
        

        keyframe_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        standUp_qpos = mjModel.key_qpos[keyframe_id]
        self.default_joint_pos_leg = standUp_qpos[7:19]

        self.default_joint_pos_arm = standUp_qpos[19:25]


        # Observation space initialization -------------------------------------------------------
        self.use_clock_signal = config.training_env["use_clock_signal"]


        self.step_freq = config.training_env.get("desired_step_freq", 1.4)
        self.duty_factor = config.training_env.get("desired_duty_factor", 0.65)
        self.phase_offset = np.array(config.training_env.get("desired_phase_offset", [0.0, 0.5, 0.5, 0.0]))
        self.phase_signal = self.phase_offset.copy()


        self.desired_clip_actions = config.training_env["desired_clip_actions"]


        self.use_filter_actions = config.training_env["use_filter_actions"]
        self.action_filter_alpha = config.training_env["action_filter_alpha"]
        self.filtered_rl_actions = np.zeros(12)


        self.use_observation_history = config.training_env["use_observation_history"]
        if(self.use_observation_history):
            single_observation_space = config.training_env["single_observation_space"]
            self.history_length = config.training_env["history_length"]
            self._observation_history = np.zeros((self.history_length, single_observation_space), dtype=np.float32)


        # vision policies are the ones trained with the perceptive height scanner
        self.use_vision = config.training_env.get("perceptive_height_scanner") is not None

        # Learned State Estimator
        if(config.training_env["base_state_source"] == "concurrent_se"):
            self._concurrent_state_est_network = load_supervised_network(config.concurrent_state_est_network)
            single_concurrent_state_est_observation_space = int(config.training_env["single_concurrent_state_est_observation_space"])
            concurrent_state_est_history_length = int(config.training_env.get("concurrent_state_est_history_length", self.history_length))
            self._observation_history_concurrent_state_est = np.zeros((concurrent_state_est_history_length, single_concurrent_state_est_observation_space), dtype=np.float32)


        # Desired joint vector
        self.desired_joint_pos = np.zeros(12) # only legs
        self._goal_heading_w = 0.0
        self._goal_manual_heading_initialized = False
        self._goal_approaching_initialized = False
        self._goal_heading_locked = False


    def _get_projected_gravity(self, quat_wxyz):        
        # Get the projected gravity in the base frame: the unit gravity vector rotated by the inverse of quat_wxyz
        # (same formula as quat_rotate_inverse of Isaac Lab)
        v = np.array([0.0, 0.0, -1.0])
        q_w = float(quat_wxyz[0])
        q_vec = np.asarray(quat_wxyz[1:], dtype=np.float64)
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        return a - b + c


    def _update_goal_commands(self, base_pos, base_yaw_w, ref_goal_pose_world):
        """Build the deploy goal command with the same semantics used in Isaac Lab."""
        goal_delta_w = ref_goal_pose_world[0:2] - base_pos[0:2]
        goal_distance = np.linalg.vector_norm(goal_delta_w)

        base_yaw_w = np.arctan2(np.sin(base_yaw_w), np.cos(base_yaw_w))

        if self._goal_manual_heading_initialized:
            # A manual yaw command is a pure rotation in place.
            self._goal_heading_w = ref_goal_pose_world[2]
            goal_delta_w = np.zeros(2, dtype=np.float64)
            goal_distance = 0.0
        else:
            bearing_w = np.arctan2(goal_delta_w[1], goal_delta_w[0])
            goal_radius = config.training_env["goal_radius"]
            # Policies trained before the hysteresis unlock as soon as they leave the goal radius.
            unlock_radius = config.training_env.get("goal_heading_radius", goal_radius)

            # Match training: follow the bearing while approaching, lock it inside the
            # goal radius and unlock only beyond the unlock radius (e.g. after an overshoot).
            if not self._goal_approaching_initialized:
                # A new goal. One already within the goal radius (the startup goal, or a goal set
                # at the current position) starts locked, like the near-goal episodes of training,
                # on the current heading: the bearing of a near-zero offset is just atan2 noise
                # (a 2 cm offset behind the base would turn the robot by 180 deg).
                self._goal_heading_locked = goal_distance <= goal_radius
                self._goal_heading_w = base_yaw_w if self._goal_heading_locked else bearing_w
                self._goal_approaching_initialized = True
            else:
                if goal_distance > unlock_radius:
                    self._goal_heading_locked = False
                if goal_distance <= goal_radius:
                    self._goal_heading_locked = True
                if not self._goal_heading_locked:
                    self._goal_heading_w = bearing_w

        heading_error = self._goal_heading_w - base_yaw_w
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        cos_yaw = np.cos(base_yaw_w)
        sin_yaw = np.sin(base_yaw_w)
        goal_delta_b = np.stack(
            (
                cos_yaw * goal_delta_w[0] + sin_yaw * goal_delta_w[1],
                -sin_yaw * goal_delta_w[0] + cos_yaw * goal_delta_w[1],
            )
        )
        command_scale = min(
            config.training_env["goal_command_distance_clip"] / max(goal_distance, 1.0e-6),
            1.0,
        )
        goal_delta_b = goal_delta_b * command_scale

        commands = np.array([
            goal_delta_b[0],
            goal_delta_b[1],
            np.sin(heading_error),
            np.cos(heading_error),
        ])
        return commands, goal_distance, heading_error


    def compute_control(self, 
            base_pos, 
            base_ori_euler_xyz, 
            base_quat_wxyz,
            base_lin_vel, 
            base_ang_vel, 
            heading_orientation_SO3,
            joints_pos_leg, 
            joints_vel_leg,
            ref_goal_pose_world,
            imu_linear_acceleration=None,
            imu_angular_velocity=None,
            imu_orientation=None,
            heightmap_data=None):


        # Update Observation ----------------------        
        if(config.training_env["base_state_source"] != "ground_truth"):
            base_projected_gravity = self._get_projected_gravity(imu_orientation)
            if(config.training_env["base_state_source"] == "imu"):
                # In training the IMU linear acceleration is zeroed (see locomotion_env.py), so do the same here
                base_linear = np.zeros(3)
            else:
                base_linear = imu_linear_acceleration
            base_ang_vel = imu_angular_velocity
        else:
            base_projected_gravity = self._get_projected_gravity(base_quat_wxyz)
            base_linear = base_lin_vel
            base_ang_vel = base_ang_vel

            
        # Fill the observation vector
        joints_pos_delta = joints_pos_leg - self.default_joint_pos_leg
        joints_pos_delta_FL = joints_pos_delta[0:3]
        joints_pos_delta_FR = joints_pos_delta[3:6]
        joints_pos_delta_RL = joints_pos_delta[6:9]
        joints_pos_delta_RR = joints_pos_delta[9:12]

        joints_vel_FL = joints_vel_leg[0:3]
        joints_vel_FR = joints_vel_leg[3:6]
        joints_vel_RL = joints_vel_leg[6:9]
        joints_vel_RR = joints_vel_leg[9:12]



        base_yaw_w = base_ori_euler_xyz[2]
        commands, goal_distance, heading_error = self._update_goal_commands(
            base_pos=base_pos,
            base_yaw_w=base_yaw_w,
            ref_goal_pose_world=ref_goal_pose_world,
        )


        # Phase Signal
        if(self.use_clock_signal):
            self.phase_signal += self.step_freq * (1 / (self.RL_FREQ))
            self.phase_signal = self.phase_signal % 1.0
            # sin and cos of the phase of each leg: [sin FL..RR, cos FL..RR]
            clock_data = np.concatenate((np.sin(2.0 * np.pi * self.phase_signal), np.cos(2.0 * np.pi * self.phase_signal)))
            
            position_reached = goal_distance <= config.training_env["goal_radius"]
            heading_reached = np.abs(heading_error) <= config.training_env["goal_forward_heading_tolerance"]
            if(position_reached and heading_reached):
                clock_data = -np.ones(8, dtype=np.float32)


        if(config.training_env["base_state_source"] == "concurrent_se"):
            obs_concurrent_state_est = np.concatenate([
                #base_linear, # the concurrent SE does not use the IMU linear acceleration
                base_ang_vel, 
                base_projected_gravity,
                commands,
                [joints_pos_delta_FL[0]], [joints_pos_delta_FR[0]], [joints_pos_delta_RL[0]], [joints_pos_delta_RR[0]],
                [joints_pos_delta_FL[1]], [joints_pos_delta_FR[1]], [joints_pos_delta_RL[1]], [joints_pos_delta_RR[1]],
                [joints_pos_delta_FL[2]], [joints_pos_delta_FR[2]], [joints_pos_delta_RL[2]], [joints_pos_delta_RR[2]],
                [joints_vel_FL[0]], 
                [joints_vel_FR[0]], 
                [joints_vel_RL[0]], 
                [joints_vel_RR[0]],

                [joints_vel_FL[1]],
                [joints_vel_FR[1]], 
                [joints_vel_RL[1]], 
                [joints_vel_RR[1]],
                
                [joints_vel_FL[2]], 
                [joints_vel_FR[2]], 
                [joints_vel_RL[2]], 
                [joints_vel_RR[2]],
                self.past_rl_actions.copy(),
            ])
            #the bottom element is the newest observation!!
            past_concurrent_state_est = self._observation_history_concurrent_state_est[1:,:]
            self._observation_history_concurrent_state_est = np.vstack((past_concurrent_state_est, copy.deepcopy(obs_concurrent_state_est)))
            obs_concurrent_state_est = self._observation_history_concurrent_state_est.flatten()
            
            # QUERY THE NETOWRK
            base_linear = self._concurrent_state_est_network(obs_concurrent_state_est)
            #base_linear = base_lin_vel
        
        
        obs = np.concatenate([
            base_linear, # this could be imu linear acc if base_state_source is "imu" or linear vel from state est
            base_ang_vel, # this could be imu angular vel if base_state_source is "imu" or angular vel from state est
            base_projected_gravity,
            commands,
            [joints_pos_delta_FL[0]], [joints_pos_delta_FR[0]], [joints_pos_delta_RL[0]], [joints_pos_delta_RR[0]],
            [joints_pos_delta_FL[1]], [joints_pos_delta_FR[1]], [joints_pos_delta_RL[1]], [joints_pos_delta_RR[1]],
            [joints_pos_delta_FL[2]], [joints_pos_delta_FR[2]], [joints_pos_delta_RL[2]], [joints_pos_delta_RR[2]],
            [joints_vel_FL[0]], 
            [joints_vel_FR[0]], 
            [joints_vel_RL[0]], 
            [joints_vel_RR[0]],

            [joints_vel_FL[1]], 
            [joints_vel_FR[1]], 
            [joints_vel_RL[1]], 
            [joints_vel_RR[1]],
            
            [joints_vel_FL[2]], 
            [joints_vel_FR[2]], 
            [joints_vel_RL[2]], 
            [joints_vel_RR[2]],
            self.past_rl_actions.copy(),
            clock_data if self.use_clock_signal else np.array([]),
        ])


        if(self.use_observation_history):
            #the bottom element is the newest observation!!
            past = self._observation_history[1:,:]
            self._observation_history = np.vstack((past, copy.deepcopy(obs)))
            obs = self._observation_history.flatten()

        
        if(self.use_vision):
            # Flatten heightmap with bottom-right at [0], then points going upward
            heightmap_2d = heightmap_data[..., 2][:, :, 0]  # Remove the last dimension
            
            # Flip vertically (so bottom row becomes first) and horizontally (so rightmost becomes first)
            heightmap_flipped = np.flip(heightmap_2d, axis=(0, 1))
            
            # Flatten column-wise so bottom-right is [0], then element above it is [1], etc.
            heightmap_data_isaac_convention = heightmap_flipped.flatten(order='F')

            height_data = (base_pos[2] - heightmap_data_isaac_convention - 0.5)
            height_data = height_data.clip(-1.0, 1.0)
            obs = np.concatenate((obs, height_data), axis=0)
            
        
        # RL Prediction
        obs = obs.reshape(1, -1)
        obs = obs.astype(np.float32)
        rl_action_temp = self.policy.run(None, {'obs': obs})[0][0]
        rl_action_temp = np.clip(rl_action_temp, -self.desired_clip_actions, self.desired_clip_actions)
        

        # Action Filtering
        if(self.use_filter_actions):
            # Exponential moving average, as in training
            alpha = self.action_filter_alpha
            self.past_rl_actions = rl_action_temp.copy()
            self.filtered_rl_actions = alpha * rl_action_temp + (1-alpha) * self.filtered_rl_actions
            rl_action_temp = self.filtered_rl_actions.copy()
        else:
            self.past_rl_actions = rl_action_temp.copy()

        # Use this print to understand the order of joints in the locomotion policy
        # first HIP, then THIGH, then CALF for each leg
        # but mujoco expect per leg order FL, FR, RL, RR for each joint
        #print("order of policy joint:", config.training_env["desired_joints_order"])
        rl_action = np.array([
            rl_action_temp[0], rl_action_temp[4], rl_action_temp[8],
            rl_action_temp[1], rl_action_temp[5], rl_action_temp[9],
            rl_action_temp[2], rl_action_temp[6], rl_action_temp[10],
            rl_action_temp[3], rl_action_temp[7], rl_action_temp[11],
        ])

        # Impedence Loop
        #TODO fix order of joints in locomotion env
        self.desired_joint_pos = self.default_joint_pos_leg + rl_action*self.action_scale

        
        return self.desired_joint_pos
