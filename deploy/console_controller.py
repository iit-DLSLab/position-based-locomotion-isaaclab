import readline
import readchar
import time

import numpy as np
import copy
import mujoco


import sys
import os 
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path+"/mujoco_utils/")
sys.path.append(file_path+"/../")
import mujoco_utils


class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
        self.isRLActivated = False

        # Autocomplete setup
        self.commands = [
            "help", "ictp", "goUp", "goDown", "activateRL", "ictp", "setKp", "setKd", "setGoalPose",
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None

    def goUp(self, ): 
        if(not self.isDown):
            print("The robot is already up")
            return

                            
        start_time = time.time()
        time_motion = 5.


        initial_joint_positions = copy.deepcopy(self.controller_node.legs_joints_position)

        keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "home")
        standUp_qpos = self.controller_node.mjModel.key_qpos[keyframe_id]
        reference_joint_positions = standUp_qpos[7:19]

        while(time.time() - start_time < time_motion):
            time_diff = time.time() - start_time
            alpha = time_diff / time_motion
            interpolated_positions = [
                (1 - alpha) * initial + alpha * reference
                for initial, reference in zip(initial_joint_positions, reference_joint_positions)
            ]

            self.controller_node.desired_joint_pos_leg = np.array(interpolated_positions)

            time.sleep(0.01)

        self.isDown = False

    def goDown(self, ):
        if(self.isDown):
            print("The robot is already down")
            return

        self.isDown = True
        self.isRLActivated = False

        start_time = time.time()
        time_motion = 5.

        temp = copy.deepcopy(self.controller_node.legs_joints_position)
        initial_joint_positions = temp
        
        keyframe_id = mujoco.mj_name2id(self.controller_node.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        goDown_qpos = self.controller_node.mjModel.key_qpos[keyframe_id]
        reference_joint_positions = goDown_qpos[7:19]

        while(time.time() - start_time < time_motion):
            time_diff = time.time() - start_time
            alpha = time_diff / time_motion
            interpolated_positions = [
                (1 - alpha) * initial + alpha * reference
                for initial, reference in zip(initial_joint_positions, reference_joint_positions)
            ]

            self.controller_node.desired_joint_pos_leg = np.array(interpolated_positions)

            time.sleep(0.01)


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "goUp"):
                    print("Going Up")
                    self.goUp()

                elif(input_string == "goDown"):
                    print("Going Down")
                    self.goDown()

                elif(input_string == "activateRL"):
                    self.isRLActivated = not self.isRLActivated

                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller_node.locomotion_policy.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kp_stand_up_and_down= float(temp)

                    print("Kp walking: ", self.controller_node.locomotion_policy.Kp_walking)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kp_walking = float(temp)


                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.locomotion_policy.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_stand_up_and_down = float(temp)

                    print("Kd walking: ", self.controller_node.locomotion_policy.Kd_walking)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.locomotion_policy.Kd_walking = float(temp)

                
                elif(input_string == "ictp"):
                    print("Interactive Keyboard Control")
                    print("w: Move Forward")
                    print("s: Move Backward")
                    print("a: Move Left")
                    print("d: Move Right")
                    print("q: Rotate Left")
                    print("e: Rotate Right")
                    print("0: Stop")
                    print("1: Pitch Up")
                    print("2: Reset Pitch")
                    print("3: Pitch Down")
                    print("Press any other key to exit")
                    while True:
                        command = readchar.readkey()
                        if(command == "w"):
                            self.controller_node.ref_base_lin_vel_H[0] += 0.1
                            print("w")
                        elif(command == "s"):
                            self.controller_node.ref_base_lin_vel_H[0] -= 0.1
                            print("s")
                        elif(command == "a"):
                            self.controller_node.ref_base_lin_vel_H[1] += 0.1
                            print("a")
                        elif(command == "d"):
                            self.controller_node.ref_base_lin_vel_H[1] -= 0.1
                            print("d")
                        elif(command == "q"):
                            self.controller_node.ref_base_ang_yaw_dot += 0.1
                            print("q")
                        elif(command == "e"):
                            self.controller_node.ref_base_ang_yaw_dot -= 0.1
                            print("e")
                        elif(command == "0"):
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0 
                            print("0")
                        else:
                            self.controller_node.ref_base_lin_vel_H[0] = 0
                            self.controller_node.ref_base_lin_vel_H[1] = 0
                            self.controller_node.ref_base_ang_yaw_dot = 0 
                            break
                
                elif input_string == "whereAmI":
                    base_pos = self.controller_node.mjData.qpos[0:3]
                    base_ori_euler = mujoco_utils.base_ori_euler_xyz(self.controller_node.mjData)
                    print("Current Position world: ", base_pos)
                    print("Current Orientation world (Euler XYZ): ", base_ori_euler)

                elif input_string == "setGoalXY":
                    print("Current Goal Pose world: ", self.controller_node.ref_goal_pose_world)
                    temp = input("Enter x: ")
                    if(temp != ""):
                        goal_x_horizontal = float(temp)
                    else:
                        goal_x_horizontal = 0.0
                    temp = input("Enter y: ")
                    if(temp != ""):
                        goal_y_horizontal = float(temp)
                    else:
                        goal_y_horizontal = 0.0

                    self.controller_node.locomotion_policy._goal_manual_heading_initialized = False
                    self.controller_node.locomotion_policy._goal_approaching_initialized = False
                    
                    # Trasform from horizontal frame to world frame
                    base_pos = self.controller_node.mjData.qpos[0:3]
                    R_heading = mujoco_utils.heading_orientation_SO3(self.controller_node.mjData)

                    offset_world = R_heading @ np.array([goal_x_horizontal,
                                                        goal_y_horizontal,
                                                        0.0])

                    goal_pos_world = base_pos + offset_world
                    

                    self.controller_node.ref_goal_pose_world = np.array([goal_pos_world[0],
                                                                    goal_pos_world[1],
                                                                    0.0])
                    
                    print("New Goal Pose world: ", self.controller_node.ref_goal_pose_world)

                elif input_string == "setGoalYAW":
                    print("Current Goal Pose world: ", self.controller_node.ref_goal_pose_world)
                    temp = input("Enter yaw: ")
                    if(temp != ""):
                        goal_yaw_horizontal = float(temp)  
                        self.controller_node.locomotion_policy._goal_approaching_initialized = False
                        self.controller_node.locomotion_policy._goal_manual_heading_initialized = True
                    else:
                        goal_yaw_horizontal = 0.0

                    # Trasform from horizontal frame to world frame
                    base_pos = self.controller_node.mjData.qpos[0:3]
                    R_heading = mujoco_utils.heading_orientation_SO3(self.controller_node.mjData)

                    offset_world = R_heading @ np.array([0.0,
                                                        0.0,
                                                        0.0])

                    goal_pos_world = base_pos + offset_world
                    
                    base_yaw = mujoco_utils.base_ori_euler_xyz(self.controller_node.mjData)[2]  # oppure equivalente
                    goal_yaw_world = base_yaw + goal_yaw_horizontal

                    self.controller_node.ref_goal_pose_world = np.array([goal_pos_world[0],
                                                                    goal_pos_world[1],
                                                                    goal_yaw_world])
                    
                    print("New Goal Pose world: ", self.controller_node.ref_goal_pose_world)
                
         
                elif input_string == "closeConsole":
                    print("Closing Console")
                    break


            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable keyboard commands:")
        print("help: Display all available messages")
        print("ictp: Interactive Keyboard Control")
        print("goUp: Move the robot to standing position")
        print("goDown: Move the robot to crouch position")
        print("activate: Activate/Deactivate RL policy for locomotion")
        print("setKp: Set the Kp values for the legs")
        print("setKd: Set the Kd values for the legs")
        print("whereAmI: Print the current position and orientation of the robot in world frame")
        print("setGoalXY: Set desired goal x and y in base frame")
        print("setGoalYAW: Set desired goal yaw in base frame\n")
        print("\nAvailable joystick commands:")
