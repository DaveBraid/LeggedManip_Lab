"""ATEC B2Piper 的 WBC 底层执行器训练环境。"""

from isaaclab.utils import configclass

from LeggedManip_Lab.assets.b2_piper.b2_piper_articulation_cfg import (
    ARM_ROOT_BODY_NAME,
    B2_PIPER_CFG,
    BASE_BODY_NAME,
    EE_BODY_NAME,
    EE_LOCAL_OFFSET,
    EE_LOCAL_ROT,
    TRACKING_JOINT_NAMES,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import *

from ...leggedmanip_lab_env_cfg import LeggedManipLabEnvCfg


@configclass
class WBCCommandsCfg:
    """B2Piper WBC command 配置。"""

    ee_pose = mdp.command_cfg.UniformPoseWBCCommandCfg(
        asset_name="robot",
        body_name=EE_BODY_NAME,
        body_offset=EE_LOCAL_OFFSET,
        body_rot=EE_LOCAL_ROT,
        link_name=ARM_ROOT_BODY_NAME,
        resampling_time_range=(8.0, 10.0),
        debug_vis=True,
        ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            pos_x=(0.7, 0.75),
            pos_y=(-0.05, 0.05),
            pos_z=(0.45, 0.55),
            roll=(-0.0, 0.0),
            pitch=(-0.0, -0.0),
            yaw=(-0.0, -0.0),
        ),
        limit_ranges=mdp.command_cfg.UniformPoseWBCCommandCfg.Ranges(
            pos_x=(0.7, 1.2),
            pos_y=(-0.25, 0.25),
            pos_z=(0.05, 0.8),
            roll=(-3.14 / 3, 3.14 / 3),
            pitch=(-3.14 / 2, 3.14 / 2),
            yaw=(-3.14 / 6, 3.14 / 6),
        ),
    )

    base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        heading_command=False,
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-0.0, 0.0),
        ),
        limit_ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-1.2, 1.2),
            ang_vel_z=(-2.0, 2.0),
            heading=(-0.0, 0.0),
        ),
    )


@configclass
class B2PiperWBCEnvCfg(LeggedManipLabEnvCfg):
    """使用 ATEC B2Piper 资产的 WBC command-following 环境。"""

    commands: WBCCommandsCfg = WBCCommandsCfg()

    def __post_init__(self):
        """
        初始化 B2Piper WBC 环境。

        该函数保留 LeggedManip 的 WBC 训练目标，同时把机器人资产、
        body 名称、action 关节和基础高度对齐到 ATEC B2Piper。
        """
        super().__post_init__()

        self.scene.robot: ArticulationCfg = B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.events.push_robot = None
        self.events.base_com.params["asset_cfg"].body_names = BASE_BODY_NAME
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY_NAME
        self.commands.ee_pose.curriculum_enabled = True
        self.commands.base_velocity.curriculum_enabled = True

        self.observations.policy.joint_pos.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.policy.joint_vel.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.critic.joint_pos.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.critic.joint_vel.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.critic.ee_link0_rel_pose.params = {
            "ee_body_name": EE_BODY_NAME,
            "ee_local_offset": EE_LOCAL_OFFSET,
            "ee_local_rot": EE_LOCAL_ROT,
            "root_body_name": ARM_ROOT_BODY_NAME,
        }

        self.actions.joint_pos.joint_names = TRACKING_JOINT_NAMES
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-10.0, 10.0)}

        self.rewards.end_effector_position_tracking_exp.func = mdp.position_command_error_exp
        self.rewards.end_effector_position_tracking_exp.weight = 4.5
        self.rewards.end_effector_position_tracking_exp.params["asset_cfg"].body_names = EE_BODY_NAME
        self.rewards.end_effector_position_tracking_exp.params["ee_local_offset"] = EE_LOCAL_OFFSET
        self.rewards.end_effector_position_tracking_exp.params["ee_local_rot"] = EE_LOCAL_ROT
        self.rewards.end_effector_position_tracking_exp.params["link0_name"] = ARM_ROOT_BODY_NAME
        self.rewards.end_effector_orientation_tracking.weight = -4.0
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = EE_BODY_NAME
        self.rewards.end_effector_orientation_tracking.params["ee_local_rot"] = EE_LOCAL_ROT
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.track_base_height_exp.weight = 0.35
        self.rewards.track_base_height_exp.params["target_height"] = 0.4
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.feet_long_air.weight = -1.0
        self.rewards.air_time_variance.weight = -1.0
        self.rewards.joint_power.weight = -4e-5
        self.rewards.dof_torques_l2.weight = -2e-5
        self.rewards.calf_torques_max.weight = -5e-5
        self.rewards.joint_mirror.weight = -0.2
        self.rewards.arm_deviation.params["asset_cfg"].joint_names = ["arm_joint.*"]

        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY_NAME

        self.disable_zero_weight_rewards()


class B2PiperWBCEnvCfg_PLAY(B2PiperWBCEnvCfg):
    """B2Piper WBC 的小规模可视化/验证环境。"""

    def __post_init__(self) -> None:
        """
        初始化 Play 环境。

        Play 配置减少环境数量并使用最大 command 范围，便于快速检查
        WBC 目标点、当前末端位置和基础稳定性。
        """
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges

        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
