import numpy as np
import stable_baselines3

from algorithm.baselines.baseline import Baseline
from algorithm.run_config.model import ModelConfig


def get_model_config(num_actions: int) -> ModelConfig:
    model_parameters = {"policy": stable_baselines3.sac.MlpPolicy, "learning_rate": 3e-4, "buffer_size": int(1e6),
                        "batch_size": 256, "ent_coef": "auto", "gamma": 0.98, "train_freq": 1, "tau": 0.01,
                        "gradient_steps": 1, "learning_starts": 5000}

    algorithm_args = {"algorithm_class": stable_baselines3.sac.SAC, "model_parameters": model_parameters}

    return ModelConfig(Baseline, algorithm_args, {})
