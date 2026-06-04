"""RSL-RL 配置兼容层。

LeggedManip_Lab 原配置使用新版 IsaacLab-RL 的 `RslRlMLPModelCfg`。
当前 ATEC 环境的 IsaacLab v2.3.x 没有导出该类，但本机 rsl_rl 5.x
runner 已经需要 actor/critic 的 MLPModel 字典。本文件提供一个同名
configclass，让原始 agent 配置无需改动网络超参数即可导入和训练。
"""

from isaaclab.utils import configclass


@configclass
class RslRlMLPModelCfg:
    """兼容 rsl_rl 5.x `MLPModel` 的模型配置。"""

    class_name: str = "MLPModel"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: object | None = None

    @configclass
    class GaussianDistributionCfg:
        """兼容 rsl_rl 5.x `GaussianDistribution` 的分布配置。"""

        class_name: str = "GaussianDistribution"
        init_std: float = 1.0
        std_type: str = "scalar"
