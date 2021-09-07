import numpy as np
import stable_baselines3
import stable_baselines3.common.noise
import stable_baselines3.her

from algorithm.baselines.baseline import Baseline
from algorithm.run_config.model import ModelConfig


def get_model_config(num_actions: int) -> ModelConfig:
    action_noise = stable_baselines3.common.noise.NormalActionNoise(mean=np.zeros(num_actions),
                                                                    sigma=0.01 * np.ones(num_actions))
    model_parameters = {"policy": "MlpPolicy", "model_class": stable_baselines3.td3.TD3, "n_sampled_goal": 4,
                        "goal_selection_strategy": "future", "max_episode_length": 60, "learning_rate": 1e-3,  # TODO: max_episode_length should be extracted from the gym environment
                        "action_noise": action_noise, "gamma": 0.99, "buffer_size": 500000, "learning_starts": 4096,
                        "batch_size": 256, "train_freq": 1024, "gradient_steps": 1024,
                        "policy_kwargs": dict(net_arch=[400, 300]), "n_episodes_rollout": 0}

    algorithm_args = {"algorithm_class": stable_baselines3.her.HER, "model_parameters": model_parameters}

    return ModelConfig(Baseline, algorithm_args, {})
