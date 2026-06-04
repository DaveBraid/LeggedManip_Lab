"""ATEC B2Piper 的 Flat 底层执行器训练环境。"""

from isaaclab.utils import configclass

from LeggedManip_Lab.assets.b2_piper.b2_piper_articulation_cfg import (
    ARM_ROOT_BODY_NAME,
    B2_PIPER_CFG,
    BASE_BODY_NAME,
    EE_BODY_NAME,
    EE_LOCAL_OFFSET,
    TRACKING_JOINT_NAMES,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import *

from ...leggedmanip_lab_env_cfg import LeggedManipLabEnvCfg


@configclass
class B2PiperFlatEnvCfg(LeggedManipLabEnvCfg):
    """使用 ATEC B2Piper 资产的 Flat command-following 环境。"""

    def __post_init__(self):
        """
        初始化 B2Piper Flat 环境。

        该函数把 LeggedManip 默认的 Z1/Piper 命名假设替换为 ATEC B2Piper
        的 body、joint 和 action 顺序。
        """
        super().__post_init__()

        self.scene.robot: ArticulationCfg = B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.commands.ee_pose.body_name = EE_BODY_NAME
        self.commands.ee_pose.body_offset = EE_LOCAL_OFFSET
        self.commands.ee_pose.root_name = ARM_ROOT_BODY_NAME
        self.commands.ee_pose.limit_ranges.pos_x = (0.45, 0.85)
        self.commands.ee_pose.limit_ranges.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.limit_ranges.pos_z = (-0.20, 0.55)

        self.events.push_robot = None
        self.events.base_com.params["asset_cfg"].body_names = BASE_BODY_NAME
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY_NAME

        self.observations.policy.joint_pos.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.policy.joint_vel.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.critic.joint_pos.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.critic.joint_vel.params = {"joint_names": TRACKING_JOINT_NAMES}
        self.observations.critic.ee_link0_rel_pose.params = {
            "ee_body_name": EE_BODY_NAME,
            "ee_local_offset": EE_LOCAL_OFFSET,
            "root_body_name": ARM_ROOT_BODY_NAME,
        }

        self.actions.joint_pos.joint_names = TRACKING_JOINT_NAMES
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-10.0, 10.0)}

        self.rewards.end_effector_position_tracking_exp.params["asset_cfg"].body_names = EE_BODY_NAME
        self.rewards.end_effector_position_tracking_exp.params["ee_local_offset"] = EE_LOCAL_OFFSET
        self.rewards.end_effector_position_tracking_exp.params["root_body_name"] = ARM_ROOT_BODY_NAME
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = EE_BODY_NAME
        self.rewards.track_base_height_exp.params["target_height"] = 0.58
        self.rewards.arm_deviation.params["asset_cfg"].joint_names = ["arm_joint.*"]

        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY_NAME

        self.disable_zero_weight_rewards()


class B2PiperFlatEnvCfg_PLAY(B2PiperFlatEnvCfg):
    """B2Piper Flat 的小规模可视化/验证环境。"""

    def __post_init__(self) -> None:
        """
        初始化 Play 环境。

        Play 配置减少环境数量并放开 command 范围，便于先做可视化和
        zero/random agent 验证。
        """
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges

        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
