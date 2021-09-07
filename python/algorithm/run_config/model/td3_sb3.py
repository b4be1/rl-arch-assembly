import numpy as np
import stable_baselines3

from algorithm.baselines.baseline import Baseline
from algorithm.run_config.model import ModelConfig


def get_model_config(num_actions: int) -> ModelConfig:
    action_noise = stable_baselines3.common.noise.NormalActionNoise(mean=np.zeros(num_actions),
                                                                    sigma=0.01 * np.ones(num_actions))
    model_parameters = {"policy": "MlpPolicy", "learning_rate": 1e-3,
                        "action_noise": action_noise, "gamma": 0.99, "buffer_size": 500000, "learning_starts": 4096,
                        "batch_size": 256, "train_freq": 1024, "gradient_steps": 1024,
                        "policy_kwargs": dict(net_arch=[400, 300]), "n_episodes_rollout": 0}

    algorithm_args = {"algorithm_class": stable_baselines3.td3.TD3, "model_parameters": model_parameters}

    return ModelConfig(Baseline, algorithm_args, {})
