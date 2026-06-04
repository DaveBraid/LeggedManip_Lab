"""B2Piper 的 RSL-RL PPO 配置。

该配置面向 IsaacLab v2.3.x + rsl_rl 5.x 的旧式 IsaacLab-RL API，
使用 `policy = RslRlPpoActorCriticCfg(...)`，避免依赖新版
`RslRlMLPModelCfg`。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class B2PiperBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """B2Piper Flat/WBC 共用 PPO 超参数。"""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "b2_piper"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class B2PiperFlatPPORunnerCfg(B2PiperBasePPORunnerCfg):
    """B2Piper Flat PPO 配置。"""

    def __post_init__(self):
        """设置 Flat 实验名。"""
        super().__post_init__()
        self.experiment_name = "b2_piper_flat"


@configclass
class B2PiperWBCPPORunnerCfg(B2PiperBasePPORunnerCfg):
    """B2Piper WBC PPO 配置。"""

    def __post_init__(self):
        """设置 WBC 实验名。"""
        super().__post_init__()
        self.experiment_name = "b2_piper_wbc"
