# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg,CameraCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from .lift_env_cfg import LiftEnvCfg,ObjectTableSceneCfg

@configclass
class ObjectTableRGBCameraSceneCfg(ObjectTableSceneCfg):

    # add camera to the scene
    # 前方
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_1",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.5, 0, 0.2), rot=(0,0,0,-1), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )

    # tiled_camera2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera_2",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(1.5, 0, 0.2), rot=(0,0,0,-1), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=100,
    #     height=100,
    # )

    # #上方
    # tiled_camera2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera_2",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.25, 0, 3), rot=(0.71,0,0.71,0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=100,
    #     height=100,
    # )

    # Set table view camera
    # tiled_camera :CameraCfg = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/table_cam",
    #     update_period=0.0333,
    #     height=84,
    #     width=84,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.33), rot=(-0.3799, 0.5963, 0.5963, -0.3799), convention="ros"),
    # )

    #左后
    tiled_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_2",
        offset=TiledCameraCfg.OffsetCfg(pos=(-1.0, 1.0, 0.2), rot=(0.92388,0,0,-0.38268), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )
    #右后
    # tiled_camera2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera_2",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(-1.0, -1.0, 0.2), rot=(0.92388,0,0,0.38268), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=100,
    #     height=100,
    # )

@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        # 除图像外的observationimage_features
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)

        # image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})
        
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()

@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""
        # 除图像外的observation
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)
        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb","model_name": "resnet18"},
        )
        image2 = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera2"), "data_type": "rgb", "model_name": "resnet18"},
        )
        # image = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("tiled_camera2"), "data_type": "rgb"},
        # )
        
        # def __post_init__(self) -> None:
        #     self.enable_corruption = True
        #     self.concatenate_terms = False
        # image2 = ObsTerm(
        #     func=mdp.image_features,
        #     params={"sensor_cfg": SceneEntityCfg("tiled_camera2"), "data_type": "rgb", "model_name": "resnet18"},
        # )
    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()

@configclass
class LiftRGBCameraEnvCfg(LiftEnvCfg):
    """Configuration for the lift environment with RGB camera."""

    scene: ObjectTableRGBCameraSceneCfg = ObjectTableRGBCameraSceneCfg(num_envs=256, env_spacing=20)
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)

@configclass
class LiftResNet18CameraEnvCfg(LiftRGBCameraEnvCfg):
    """Configuration for the lift environment with ResNet18 features as observations."""

    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()


