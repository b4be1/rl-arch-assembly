import numpy as np
import stable_baselines3

from algorithm.baselines.baseline import Baseline
from algorithm.run_config.model import ModelConfig


def get_model_config(num_actions: int) -> ModelConfig:
    model_parameters = {"policy": "MlpPolicy", "policy_kwargs": dict(net_arch=[400, 300])}

    algorithm_args = {"algorithm_class": stable_baselines3.a2c.A2C, "model_parameters": model_parameters}

    return ModelConfig(Baseline, algorithm_args, {})
