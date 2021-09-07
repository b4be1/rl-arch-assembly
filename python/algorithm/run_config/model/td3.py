from algorithm.run_config.model import ModelConfig
from algorithm.td3 import TD3, Actor, Critic


def get_model_config() -> ModelConfig:
    algorithm_args = {
        "actor": Actor(1, (400, 300)),
        "critic": Critic((400, 300)),
        "max_action": 1,
        "discount": 0.99,
        "tau": 0.005,
        "exploration_noise": 0.01,
        "noise_clip": 0.03,
        "policy_delay": 2,
        "training_starts_at": 1024 * 128,
        "train_interval": 1024,
        "gradient_steps": 1024,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "replay_buffer_size": int(1e6)
    }

    return ModelConfig(TD3, algorithm_args, {})
