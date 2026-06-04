# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ATEC 官方 B2Piper 机器人资产配置。

该文件把 ATEC 仓库中的 B2 + Piper USD、关节顺序和执行器参数搬到
LeggedManip_Lab 命名空间下，便于直接注册并训练 WBC/Flat 底层策略。
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


LEG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
    "arm_joint7",
    "arm_joint8",
]

ARM_TRACKING_JOINT_NAMES = ARM_JOINT_NAMES[:6]
JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
TRACKING_JOINT_NAMES = LEG_JOINT_NAMES + ARM_TRACKING_JOINT_NAMES

BASE_BODY_NAME = "base_link"
EE_BODY_NAME = "gripper_base"
ARM_ROOT_BODY_NAME = "base_link"
EE_LOCAL_OFFSET = (0.13, 0.0, 0.0)
# 将 ee-cmd 坐标系绕 y 轴旋到 gripper_base 坐标系。
EE_LOCAL_ROT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)


def _find_atec_b2_piper_usd() -> str:
    """从当前文件向上查找 ATEC 官方 B2Piper USD 路径。"""
    rel_path = Path("b2_piper.usda")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / rel_path
        if candidate.is_file():
            return str(candidate)
    # 保留一个基于当前仓库布局的 fallback，方便错误信息指向期望位置。
    return str(Path(__file__).resolve().parents[6] / rel_path)


B2_PIPER_USD = _find_atec_b2_piper_usd()


B2_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=B2_PIPER_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "arm_joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_hip_thigh": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit_sim=200,
            velocity_limit_sim=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
        "base_calf": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit_sim=320,
            velocity_limit_sim=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr="arm_joint.*",
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
            friction=0.01,
            armature=0.01,
        ),
    },
)
