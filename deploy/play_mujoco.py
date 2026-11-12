# Description: This script is used to simulate the full model of the robot in mujoco

import time
import numpy as np


import sys
import os
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path+"/mujoco_utils/")
sys.path.append(file_path+"/../")

# Simulation related imports
import mujoco
import mujoco.viewer
import mujoco_utils
from heightmap import HeightMap

# Controller related imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper
from deploy import config
import threading

class PlayMujoco:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)

        self.simulation_dt = 0.002


        # Load the model and data
        spec = mujoco.MjSpec.from_file(file_path + "/../robot_model/" + config.robot + "/" + config.scene + ".xml")
        self.mjModel = spec.compile()
        self.mjData = mujoco.MjData(self.mjModel)
        keyframe_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.mjData.qpos = self.mjModel.key_qpos[keyframe_id]
        mujoco.mj_forward(self.mjModel, self.mjData)
        self.viewer = mujoco.viewer.launch_passive(
                        self.mjModel,
                        self.mjData,
                        show_left_ui=False,
                        show_right_ui=False,
                        #key_callback=lambda x: self._key_callback(x),
                    )



        mujoco.mjv_defaultFreeCamera(self.mjModel, self.viewer.cam)
        self.last_render_time = time.time()
        self.RENDER_FREQ = 30.0  # Hz


        # Initialization of variables used in the main control loop --------------------------------
        self.locomotion_policy = LocomotionPolicyWrapper(mjModel=self.mjModel)

        if(self.locomotion_policy.use_vision):
            resolution_heightmap = config.training_env["perceptive_height_scanner"]["pattern_cfg"]["resolution"]
            num_rows_heightmap = round(config.training_env["perceptive_height_scanner"]["pattern_cfg"]["size"][0]/resolution_heightmap) + 1
            num_cols_heightmap = round(config.training_env["perceptive_height_scanner"]["pattern_cfg"]["size"][1]/resolution_heightmap) + 1
            heightmap_offset = config.training_env["perceptive_height_scanner"]["offset"]
            self.heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=self.mjModel, mj_data=self.mjData)

        self.legs_joints_position = np.zeros(12)  # 12 leg joints
        self.legs_joints_velocity = np.zeros(12)  # 12 leg joints

        self.desired_joint_pos_leg = self.mjData.qpos[7:19]
        self.Kp_legs = 0
        self.Kd_legs = 0

        # --------------------------------------------------------------
        self.ref_goal_pose_world = np.array([0.0, 0.0, 0.0])  # Desired goal pose (x, y, yaw)


        # Interactive Command Line
        from console_controller import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

        self.console.isDown = False  # Only in this play_mujoco script
        self.console.isRLActivated = True  # Only in this play_mujoco script


    def run(self):
        step_num = 1
        # Run the simulation
        while self.viewer.is_running():
            step_start = time.time()

            # Get the current state of the robot -----------------------------------------------------
            qpos, qvel = self.mjData.qpos, self.mjData.qvel
            base_pos = mujoco_utils.base_pos(self.mjData)

            joints_pos_leg = qpos[7:19]
            joints_vel_leg = qvel[6:18]

            # RL controller --------------------------------------------------------------
            if self.console.isRLActivated and step_num % round(1 / (self.locomotion_policy.RL_FREQ * self.simulation_dt)) == 0:
                # The policy reads the base state from the simulator ground truth.
                policy_data = self.mjData
                base_quat_wxyz = policy_data.qpos[3:7]
                policy_base_pos = base_pos

                if(config.training_env["base_state_source"] != "ground_truth"):
                    imu_linear_acceleration = self.mjData.sensordata[0:3]
                    imu_angular_velocity = self.mjData.sensordata[3:6]
                    imu_orientation = self.mjData.sensordata[9:13]
                else:
                    imu_linear_acceleration = np.zeros(3)
                    imu_angular_velocity = np.zeros(3)
                    imu_orientation = np.zeros(4)

                heightmap_data = None
                if(self.locomotion_policy.use_vision):
                    # The terrain is scanned around the robot, as run_simulator_ros2 does.
                    heading_orientation_SO3 = mujoco_utils.heading_orientation_SO3(self.mjData)
                    offset_world_frame = config.training_env["perceptive_height_scanner"]["offset"]["pos"] @ heading_orientation_SO3.T
                    self.heightmap.update_height_map(self.mjData.qpos[0:3] + offset_world_frame, yaw=mujoco_utils.base_ori_euler_xyz(self.mjData)[2])
                    heightmap_data = self.heightmap.data

                self.desired_joint_pos_leg = self.locomotion_policy.compute_control(
                            base_pos=policy_base_pos,
                            base_ori_euler_xyz=mujoco_utils.base_ori_euler_xyz(policy_data),
                            base_quat_wxyz=base_quat_wxyz,
                            base_lin_vel=mujoco_utils.base_lin_vel(policy_data, frame='base'),
                            base_ang_vel=mujoco_utils.base_ang_vel(policy_data, frame='base'),
                            heading_orientation_SO3=mujoco_utils.heading_orientation_SO3(policy_data),
                            joints_pos_leg=joints_pos_leg,
                            joints_vel_leg=joints_vel_leg,
                            ref_goal_pose_world=self.ref_goal_pose_world,
                            imu_linear_acceleration=imu_linear_acceleration,
                            imu_angular_velocity=imu_angular_velocity,
                            imu_orientation=imu_orientation,
                            heightmap_data=heightmap_data)

                self.Kp_legs = self.locomotion_policy.Kp_walking
                self.Kd_legs = self.locomotion_policy.Kd_walking

            elif not self.console.isRLActivated:
                # Go up-and-down
                self.Kp_legs = self.locomotion_policy.Kp_stand_up_and_down
                self.Kd_legs = self.locomotion_policy.Kd_stand_up_and_down

            # Torque saturation for the legs
            max_torque = self.mjModel.actuator_ctrlrange[0:12, 1]
            max_torque = max_torque*0.95  # A margin for safety
            lower = (-max_torque + self.Kd_legs * joints_vel_leg) / self.Kp_legs
            upper = ( max_torque + self.Kd_legs * joints_vel_leg) / self.Kp_legs

            self.desired_joint_pos_leg = np.clip(
                self.desired_joint_pos_leg,
                joints_pos_leg + lower,
                joints_pos_leg + upper
            )

            # PD controller --------------------------------------------------------------
            error_joints_pos_leg = self.desired_joint_pos_leg - joints_pos_leg
            tau_leg = self.Kp_legs*error_joints_pos_leg - self.Kd_legs*joints_vel_leg


            # Set control and mujoco step ----------------------------------------------------------------------
            self.mjData.ctrl[0:12] = tau_leg
            mujoco.mj_step(self.mjModel, self.mjData)
            step_num = step_num + 1


            # Sleep to match real-time ---------------------------------------------------------
            loop_elapsed_time = time.time() - step_start
            if(loop_elapsed_time < self.simulation_dt):
                time.sleep(self.simulation_dt - (loop_elapsed_time))


            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / self.RENDER_FREQ:
                # Update the camera position
                self.viewer.cam.lookat[:] = base_pos
                self.viewer.sync()
                self.last_render_time = time.time()

                if(self.locomotion_policy.use_vision):
                    if self.heightmap.data is not None:
                        for i in range(self.heightmap.data.shape[0]):
                            for j in range(self.heightmap.data.shape[1]):
                                self.heightmap.geom_ids[i, j] = mujoco_utils.render_sphere(
                                    viewer=self.viewer,
                                    position=([self.heightmap.data[i][j][0][0], self.heightmap.data[i][j][0][1], self.heightmap.data[i][j][0][2]]),
                                    diameter=0.02,
                                    color=[0, 1, 0, 0.5],
                                    geom_id=self.heightmap.geom_ids[i, j],
                                )




if __name__ == "__main__":
    play_mujoco = PlayMujoco()
    play_mujoco.run()
