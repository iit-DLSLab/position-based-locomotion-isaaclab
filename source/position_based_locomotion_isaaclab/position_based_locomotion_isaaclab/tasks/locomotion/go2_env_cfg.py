import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    RayCasterCfg,
    MultiMeshRayCasterCfg,
    MultiMeshRayCasterCameraCfg,
    TiledCameraCfg,
    patterns,
)
from isaaclab.sim import SimulationCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from position_based_locomotion_isaaclab.assets.go2_asset import GO2_CFG 
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import position_based_locomotion_isaaclab.tasks.custom_events as custom_events
import position_based_locomotion_isaaclab.tasks.custom_curriculums as custom_curriculums

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    
    
    joint_parameters = EventTerm(
        func=custom_events.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    pace_actuator_delay = EventTerm(
        func=custom_events.randomize_pace_actuator_delay,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "min_delay": 0,
        },
    )
    
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                                   "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
    )



@configclass
class PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=2**23)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=2**23)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default = newton_mjwarp



@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 14.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 3 # base linear velocity
    observation_space += 3 # base angular velocity  
    observation_space += 3 # projected gravity in base frame
    observation_space += 4 # goal in base frame (x/y) and relative heading (sin/cos)
    observation_space += 12 # joint positions
    observation_space += 12 # joint velocities
    observation_space += 12 # last actions

    use_clock_signal = True
    if(use_clock_signal):
        observation_space += 8 # clock signal for periodic gait (sin and cos of the phase of each leg)

    single_observation_space = observation_space # Usefull for concatenating history

    # observation history
    use_observation_history = True
    if(use_observation_history):
        history_length = 5
        observation_space *= history_length
    else:
        history_length = 1


    base_state_source = "ground_truth"  # "ground_truth" | "imu" | "concurrent_se"
    if(base_state_source == "concurrent_se"):
        concurrent_state_est_network_type = "tcn" # "mlp" or "tcn"
        
        concurrent_state_est_output_space = 3 #lin_vel_b
        
        #single_concurrent_state_est_observation_space = 3 # base linear acceleration
        single_concurrent_state_est_observation_space = 3 # base angular velocity  
        single_concurrent_state_est_observation_space += 3 # projected gravity in base frame
        single_concurrent_state_est_observation_space += 4 # position/heading goal command
        single_concurrent_state_est_observation_space += 12 # joint positions
        single_concurrent_state_est_observation_space += 12 # joint velocities
        single_concurrent_state_est_observation_space += 12 # last actions
        concurrent_state_est_history_length = 5 
        concurrent_state_est_observation_space = single_concurrent_state_est_observation_space*concurrent_state_est_history_length
        
        concurrent_state_est_batch_size = 512
        concurrent_state_est_train_epochs = 1000
        concurrent_state_est_lr = 1e-3
        concurrent_state_est_ep_saving_interval = 1000
        concurrent_state_est_ep_saving_start = 6000


    use_rma = False
    if(use_rma):
        rma_network_type = "mlp" # "mlp" or "tcn"
        rma_use_latent_space = False
        if(rma_use_latent_space):
            rma_latent_space = 8
            rma_latent_encoder_hidden_features = 128
            rma_latent_encoder_seed = 0
        
        rma_privileged_observation_space = 12 # P gain
        rma_privileged_observation_space += 12 # D gain
        rma_privileged_observation_space += 3 # clean linear velocity
        rma_privileged_observation_space += 1 # base height error
        rma_privileged_observation_space += 1 # terrain pitch
        rma_privileged_observation_space += 4 # foot contacts

        rma_output_space = rma_latent_space if rma_use_latent_space else rma_privileged_observation_space
        observation_space += rma_output_space

        #single_rma_observation_space = 3 # base linear acceleration
        single_rma_observation_space = 3 # base angular velocity  
        single_rma_observation_space += 3 # projected gravity in base frame
        single_rma_observation_space += 4 # position/heading goal command
        single_rma_observation_space += 12 # joint positions
        single_rma_observation_space += 12 # joint velocities
        single_rma_observation_space += 12 # last actions
        rma_history_length = 5
        rma_observation_space = single_rma_observation_space*rma_history_length
    
        rma_batch_size = 512
        rma_train_epochs = 1000
        rma_lr = 1e-3
        rma_ep_saving_interval = 1000
        rma_ep_saving_start = 6000


    # Base-centered height scanner for pose-related rewards and privileged observations.
    pose_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 2.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )

    # Template copied onto each foot link to measure the terrain immediately around that foot.
    foot_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base/FL_hip/FL_thigh/FL_calf/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )


    # asymmetric ppo
    use_asymmetric_ppo = True
    if(use_asymmetric_ppo):
        state_space = observation_space
        state_space += 5 # full position/heading goal command and remaining seconds
        state_space += 12 #P gain
        state_space += 12 #D gain
        state_space += 2 #base pitch and height
        state_space += 3 #clean lin vel b
        state_space += 4 #contacts foot
        state_space += 8 #feet air and contact time
        state_space += 4 #foot error

        pattern_cfg = pose_height_scanner.pattern_cfg
        height_map_x_points = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        state_space += height_map_x_points * height_map_y_points
    else:
        state_space = 0


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # "max" (not "multiply"): Newton/MJWarp ignores "multiply" and always uses max (IsaacLab PR #8877).
            # With max + low ground mu, effective mu = max(0.2, robot mu) -> spans the full randomized range on all backends.
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.0,
        ),
        physics=PhysicsCfg(),
        use_newton_actuators=False,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # "max" (not "multiply"): Newton/MJWarp ignores "multiply" and always uses max (IsaacLab PR #8877).
            # With max + low ground mu, effective mu = max(0.2, robot mu) -> spans the full randomized range on all backends.
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # an imu sensor in case we don't want any state estimator (for now we can't use sites from the xml)
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base", 
        offset=ImuCfg.OffsetCfg(
            pos=(-0.02557, 0, 0.04232)
        ), 
        debug_vis=False)


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()


    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset.
    # The std is set separately for each observation term with observation_noise_std
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, operation="abs"),
    )
    # (noise std, bias std) for each observation term
    observation_noise_std = {
        "base_linear": (0.05, 0.01),
        "base_ang_vel": (0.05, 0.01),
        "projected_gravity": (0.05, 0.01),
        "commands": (0.01, 0.01),
        "joint_pos": (0.01, 0.01),
        "joint_vel": (0.1, 0.01),
        "actions": (0.01, 0.01),
        "clock": (0.01, 0.01),
        "height_map": (0.02, 0.01),
        "rma": (0.02, 0.01),
    }

    # robot
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    # Contacts of the head (the head_upper and head_lower collision geoms of the base in go2.xml) against the
    # terrain, used by the undesired contacts reward. Shape-level contact sensing is Newton only.
    head_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base",
        sensor_shape_prim_expr=[
            "/World/envs/env_.*/Robot/Geometry/base/head_upper",
            "/World/envs/env_.*/Robot/Geometry/base/head_lower",
        ],
        filter_shape_prim_expr=["/World/ground/.*"],
        history_length=3,
        update_period=0.005,
    )

    desired_joints_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                           'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',  
                           'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    # Desired tracking variables
    desired_base_height = 0.30
    desired_feet_height = 0.05
    foot_radius = 0.022 # radius of the foot collision sphere

    # Position-goal command, one goal per episode. The desired heading follows the line
    # connecting the current base position to the goal.
    goal_distance_range = (1.0, 3.0)
    goal_bearing_range = (-3.141592653589793, 3.141592653589793)
    goal_command_distance_clip = 2.0
    goal_radius = 0.2
    goal_forward_heading_tolerance = 0.35
    # The desired heading locks once the base is within goal_radius and unlocks only
    # beyond this distance. It also gates the heading reward (AME-2 uses 0.5 m).
    goal_heading_radius = 0.5
    # Fraction of episodes that start at the goal with a random heading to track,
    # i.e. rotation in place as commanded by the joystick at deploy time.
    near_goal_episode_probability = 0.12

    # AME-2 goal-reaching reward parameters.
    goal_position_tracking_window_s = 4.0
    goal_heading_tracking_window_s = 2.0
    goal_moving_velocity_range = (0.2, 0.4)
    goal_velocity_alignment_cosine = 0.5

    visualize_goal = True
    goal_visualization_radius = 0.12
    goal_visualization_height_offset = 0.15


    # Desired clip actions
    desired_clip_actions = 6.0
    use_filter_actions = True
    action_filter_alpha = 0.8 # exponential moving average, lower is smoother but adds delay


    # Position-based task rewards from AME-2 (Table I).
    goal_position_tracking_reward_scale = 20.0
    goal_heading_tracking_reward_scale = 10.0
    moving_to_goal_reward_scale = 2.0
        

    # Tracking reward scale
    yaw_rate_reward_scale = 0.1
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.25
    orientation_reward_scale = -10.0
    height_reward_scale = 1.0


    # Joint reward scale
    joints_torque_reward_scale = -2.5e-6 
    joints_accel_reward_scale = -2.5e-7
    joints_energy_reward_scale = -1e-4
    joints_hip_position_reward_scale = 0.0
    joints_thigh_position_reward_scale = 0.0
    joints_calf_position_reward_scale = 0.0
   
    
    # Undesired contacts reward scale
    undersired_contact_reward_scale = -1.0
    # Contact curriculum, in learning iterations. Before it ends, trunk and hip contacts terminate the episode and the
    # undesired contacts reward penalizes trunk, hips and head. Afterwards contacts do not terminate the episode and
    # only the head is penalized. Without the curriculum (play, dataset collection) the env starts from the final phase.
    use_contact_curriculum = True
    contact_curriculum_iterations = 500
    # The episode terminates when the base tilts more than this angle from the vertical [rad], in every phase of the
    # contact curriculum, so that a fallen robot does not keep collecting penalties until the time out.
    # pi/2: the trunk is upside down (its up axis points below the horizontal)
    tilt_termination_angle = 1.5708
    action_rate_reward_scale = -0.01
    action_smoothness_reward_scale = -0.001


    # Feet reward scale
    feet_air_time_reward_scale = 0.25
    feet_air_time_variance_reward_scale = -1.0  * 0.0

    feet_height_clearance_aperiodic_reward_scale = 0.25 * 0.0  
    feet_height_clearance_periodic_reward_scale = 0.25 * 0.0
    
    feet_height_clearance_mujoco_aperiodic_reward_scale = 0.25 * 0.0
    feet_swing_trajectory_aperiodic_reward_scale = 0.5 * 0.0

    feet_height_clearance_mujoco_periodic_reward_scale = 0.25 * 0.0
    feet_swing_trajectory_periodic_reward_scale = 0.5
    
    feet_slide_reward_scale = -0.25 * 0.0

    feet_landing_vel_reward_scale = -0.1
    feet_landing_height = 0.02 # foot height above terrain below which the downward velocity is penalized
    
    feet_to_hip_distance_reward_scale = 1.5
    # This is used in loocmotion_env.py for the above reward
    desired_hip_offset = 0.095

    feet_edge_stance_reward_scale = 0.0
    feet_edge_swing_reward_scale = -1.0
    feet_edge_swing_margin = 0.04 # desired horizontal distance of the swing foot from a riser face [m]
    feet_edge_swing_height_margin = 0.01 # the foot bottom has to be this much below the riser top to be penalized, and this much above its base to count as touching it [m]
    feet_edge_height_threshold = 0.05 # minimum height difference between two neighboring cells of the edge height map to mark an edge (riser) between them [m]
    feet_edge_radius_px = 0
    visualize_edge_map = False
    visualize_feet_edge_swing_zone = False # transparent red cylinders where a swing foot is penalized by feet_edge_swing


    feet_vertical_surface_contacts_reward_scale = -2.5


    # Contact suggestion reward scale
    periodic_contact_suggestion_reward_scale =  1.0
    # Desired step freq and duty factor (if periodic gait contact suggestion is used)
    desired_step_freq = 1.4
    desired_duty_factor = 0.65
    desired_phase_offset = [0.0, 0.5, 0.5, 0.0] #FL, FR, RL, RR

    stance_contact_suggestion_reward_scale = 2.0



import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
@configclass
class Go2RoughBlindEnvCfg(Go2FlatEnvCfg):

    curriculum_replay_probability = 0.2
    curriculum_start_level = 0

    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.1
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=1.5, grid_height_range=(0.05, 0.40), platform_width=2.0,
                flat_patch_sampling={
                    "box_spawn": FlatPatchSamplingCfg(
                        num_patches=128,
                        patch_radius=[0.0, 0.2, 0.45],
                        x_range=(-3.0, 3.0),
                        y_range=(-3.0, 3.0),
                        # Heights are relative to the elevated central platform.
                        # Exclude it so episodes can start on lower grid cells.
                        z_range=(-1.0, -0.01),
                        max_height_diff=0.001,
                    ),
                },
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.05, 0.22), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.05, 0.22), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
        },
    )

    """Rough terrains configuration."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # "max" (not "multiply"): Newton/MJWarp ignores "multiply" and always uses max (IsaacLab PR #8877).
            # With max + low ground mu, effective mu = max(0.2, robot mu) -> spans the full randomized range on all backends.
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            static_friction=0.2,
            dynamic_friction=0.2,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for the feet edge reward (only used in the rewards, so also for blind locomotion)
    edge_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.8, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )



@configclass
class Go2RoughVisionEnvCfg(Go2RoughBlindEnvCfg):

    def __post_init__(self) -> None:
        pattern_cfg = self.perceptive_height_scanner.pattern_cfg
        height_map_x_points = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        self.observation_space = self.observation_space + height_map_x_points * height_map_y_points
        # the critic receives the actor observation as well, perceptive heightmap included
        if self.use_asymmetric_ppo:
            self.state_space = self.state_space + height_map_x_points * height_map_y_points

        self.feet_edge_stance_reward_scale = -1.0

    # we add a height scanner for perceptive locomotion
    perceptive_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 2.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )

    #camera_usd = CAMERA_USD_CFG
    use_depth_camera = False
    depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base",
        # Raycast whenever it is read (sensors update lazily): collect_dataset.py of iekf-quadruped-isaaclab reads it at
        # the visual odometry rate.
        update_period=0.0,
        # RealSense D435(i) mount: looking forward, 20 degrees down. The quaternion is xyzw (Isaac Lab 3) in the ROS
        # convention; keep it equal to camera_quat_base_wxyz in deploy/config.py of iekf-quadruped-isaaclab, which the
        # IEKF uses.
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.33, 0.0, 0.08), rot=(-0.579228, 0.579228, -0.405579, 0.405579)),
        mesh_prim_paths=[
            "/World/ground",
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/base/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FL_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FR_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RL_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RR_.*/visuals"),
        ],
        # D435 depth stream at 848x480 with a nominal 87 degrees horizontal field of view (fx = 424 / tan(43.5 deg)),
        # downsampled by 8 to 106x60 (about 87 x 57 degrees).
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=[55.85, 0.0, 53.0, 0.0, 55.85, 30.0, 0.0, 0.0, 1.0],
            width=106,
            height=60,
        ),
        # As the D435, no depth beyond its range: 0 marks an invalid pixel.
        max_distance=10.0,
        depth_clipping_behavior="zero",
        debug_vis=True,
    )
