# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import math

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # print("height: ",object.data.root_pos_w[:, 2].mean().item())
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),    
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # print("ee_w: ", ee_w[:,2].mean().item())
    # print("cube_pos_w: ", cube_pos_w)
    if torch.isnan(ee_w).any():
        print("ee_w存在 NaN 值:", ee_w)
    if torch.isnan(cube_pos_w).any():
        print("cube_pos_w 存在 NaN 值:", cube_pos_w)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    with open('output_formres1.txt', 'a') as f:
       f.write(f"step {env.common_step_counter} dis: {torch.mean(object_ee_distance).item()},"
               f"reward: {torch.mean(1 - torch.tanh(object_ee_distance / std)).item()}, std: {std}\n")
    return 1 - torch.tanh(object_ee_distance/std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def print_debug_info(env:ManagerBasedRLEnv):
    """自定义打印函数"""
    if env.common_step_counter % 1 == 0:
        print(f"\nStep {env.common_step_counter}")
        print(f"Object position: {env.scene.object.data.root_pos_w[0].cpu().numpy()}")
        print(f"EE position: {env.scene.ee_frame.data.target_pos_w[0].cpu().numpy()}")
        print(f"Rewards: {env.reward_buf[0].item():.2f}")
    return 0  # 必须返回一个值，但不影响奖励


def grip_object(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg= SceneEntityCfg("object") , ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
)-> torch.Tensor:
    """Reward the agent for closing the gripper when object is within finger"""
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object = env.scene[object_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    cur_angle = env.scene['robot'].data.joint_pos_target[:, 5]
    #条件一：当前dis小于0.01
    cond1 = object_ee_distance < 0.025
    #条件二：在接近物体
    cond2 = object_ee_distance < env.last_dis
    #条件三：夹爪在逐渐闭合
    # cond3 = cur_angle < env.last_grip
    #new 条件三：夹爪开放
    cond3 = cur_angle < env.last_grip
    #new条件四：夹爪逐渐闭合
    cond4 = cur_angle > env.last_grip
    #new条件五：夹爪全部或部分闭合True
    cond5 = cur_angle >= 1.5
    #new条件六：夹爪开启
    cond6 = cur_angle <=0

    cond7 = object_ee_distance <0.05
   
    reward_mask1 = cond1 & cond2 & cond3 
    reward_mask3 = cond1 & (~cond2) & (~cond3)
    reward_mask4 = cond1 & cond2 & (~cond2) &(cond3)
    # print("reward_mask1: ", reward_mask1)
    reward1 = torch.where(reward_mask1, torch.tensor(1.5), torch.tensor(0.8))
    reward3 = torch.where(reward_mask3, torch.tensor(1.2), torch.tensor(0.0))
    reward_mask2 = (~cond1) & cond2 & (cond4 | cond5) & cond7
    reward2 = torch.where(reward_mask2, torch.tensor(0.8), torch.tensor(0.0))
    with open('output_formres2.txt', 'a') as f:
       f.write(f"step {env.common_step_counter} dis: {cur_angle.mean().item()}\n")
    reward4 = torch.where(reward_mask4, torch.tensor(1.8), torch.tensor(0.8))
    #更新last_dis和last_angle
    env.last_dis = object_ee_distance
    env.last_grip = cur_angle
    # print("reward1: ", reward1.mean().item())
    # print("reward2: ", reward2.mean().item())
    # print("reward3: ", reward3.mean().item())   
    # print("reward4: ", reward4.mean().item())

    return reward1+reward2+reward3+reward4

def contain_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_frame_1_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_1"),
    finger_frame_2_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_2"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    finger_frame_1: FrameTransformer = env.scene[finger_frame_1_cfg.name]
    finger_frame_2: FrameTransformer = env.scene[finger_frame_2_cfg.name]

    cube_pos_w = object.data.root_pos_w
    finger_w_1 = finger_frame_1.data.target_pos_w[..., 0, :]
    finger_w_2 = finger_frame_2.data.target_pos_w[..., 0, :]

    vector_1 = finger_w_1 - cube_pos_w
    vector_2 = finger_w_2 - cube_pos_w
    dot_products = torch.sum(vector_1 * vector_2, dim=1)
    norm_1 = torch.norm(vector_1, dim=1)
    norm_2 = torch.norm(vector_2, dim=1)
    cos_theta = dot_products / (norm_1 * norm_2 + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta = torch.acos(cos_theta)
    reward = theta / np.pi

    return reward


def clamp_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_frame_1_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_1"),
    finger_frame_2_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame_2"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    finger_frame_1: FrameTransformer = env.scene[finger_frame_1_cfg.name]
    finger_frame_2: FrameTransformer = env.scene[finger_frame_2_cfg.name]

    cube_pos_w = object.data.root_pos_w
    finger_w_1 = finger_frame_1.data.target_pos_w[..., 0, :]
    finger_w_2 = finger_frame_2.data.target_pos_w[..., 0, :]

    vector_1 = finger_w_1 - cube_pos_w
    vector_2 = finger_w_2 - cube_pos_w
    dot_products = torch.sum(vector_1 * vector_2, dim=1)
    norm_1 = torch.norm(vector_1, dim=1)
    norm_2 = torch.norm(vector_2, dim=1)
    cos_theta = dot_products / (norm_1 * norm_2 + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta = torch.acos(cos_theta)

    cur_angle = env.scene['robot'].data.joint_pos_target[:, 5]
    cond1 = cur_angle < 0.001
    cond2 = theta > np.pi * 4.5 / 6
    mask = cond1 & cond2

    # print('cur_state: ', cur_angle, theta)

    reward = torch.where(mask, torch.tensor(1.0), torch.tensor(0.0))

    return reward