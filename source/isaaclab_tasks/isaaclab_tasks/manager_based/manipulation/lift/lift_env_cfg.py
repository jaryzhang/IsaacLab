# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCameraCfg,CameraCfg

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # object_id :int=0

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_1",
        # offset=TiledCameraCfg.OffsetCfg(pos=(1.5, 0, 0.2), rot=(0,0,0,-1), convention="world"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(1.66, 0.0, 1.12), rot=((0.63004, 0.32102, 0.32102, 0.63004)), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0, 0.6), rot=(0,-0.4332,0,0.9013), convention="world"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(1.3, 0.0, 0.9), rot=((0.63281, 0.31551, 0.31551, 0.63281)), convention="opengl"),
        # data_types=["rgb"],
        # spawn=sim_utils.PinholeCameraCfg(
        #     focal_length=48.9, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        # ),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.5), rot=((0.63281, 0.31551, 0.31551, 0.63281)), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0, 0.0, 0.06), rot=((-0.52133, -0.47771, 0.47771, 0.52133)), convention="opengl"),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.193681, 0.0575158), rot=((0.4912, 0.50865, -0.50865, -0.4912)), convention="opengl"),
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.6, 0.0, 0.3), rot=((0.6509, 0.27629, 0.27629, 0.6509)), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.6, focus_distance=400.0, horizontal_aperture=36, clipping_range=(0.1, 20.0),vertical_aperture=25.45
        ),
        # spawn=sim_utils.PinholeCameraCfg(
        #     focal_length=1.8, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        # ),
        width=200,
        height=150,
    )

    # tiled_camera2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera_2",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(1.3, 0.0, 0.9), rot=((0.63281, 0.31551, 0.31551, 0.63281)), convention="opengl"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=38.3, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=1000,
    #     height=800,
    # )



##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            # pos_x=(0.25, 0.35),
            # pos_y=(-0.05, 0.05),
            # pos_z=(0.25, 0.5),
            # roll=(0.0, 0.0),
            # pitch=(0.0, 0.0),
            # yaw=(0.0, 0.0),
            
            pos_x=(0.3, 0.3),
            pos_y=(-0.01, 0.01),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    '''
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    '''



    # observation groups
    policy: PolicyCfg = PolicyCfg()
    #rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class DepthObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        """Observations for policy group with depth images."""

        image = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "distance_to_camera"}
        )

    policy: ObsGroup = DepthCameraPolicyCfg()


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"},
        )

        # image = ObsTerm(
        #     func=mdp.image_features,
        #     params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb","model_name": "resnet18"}
        # )

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # def __post_init__(self):
        #     self.enable_corruption = False
        #     self.concatenate_terms = False

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",
                "model_device": "cuda:0",
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            # "pose_range": {"x": (-0.05, 0.05), "y": (-0.25, 0.25), "z": (0.0, 0.0)},

            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=20,  # 2.0
        # weight=20.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.008},
        weight=50.0,   # 1500  150
    )

    lifting_object1 = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.015},
        weight=10000.0,   # 1500  150
    )

    lifting_object2 = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.024},
        weight=20000.0,   # 1500  150
    )

    lifting_object3 = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=50000.0,   # 1500  150
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.015, "command_name": "object_pose"},
        weight=200, # 16.0
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        #params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        params={"std": 0.05, "minimal_height": 0.015, "command_name": "object_pose"},
        weight=0.5,  # 5.0
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    grip_object = RewTerm(
        func=mdp.grip_object,
        weight=10,  # 10.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    #action_rate = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    #)

    #joint_vel = CurrTerm(
    #    func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    #)


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=128, env_spacing=2.5)
    # Basic settings
    # observations: ObservationsCfg = ObservationsCfg()
    #observations: TheiaTinyObservationCfg = TheiaTinyObservationCfg()
    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 20  # 2 20 48
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
