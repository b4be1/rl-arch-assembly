from algorithm.run_config.model import ModelConfig
from algorithm.planet_wrapper import PlanetWrapper


def get_model_config() -> ModelConfig:
    model_parameters = {
        "activation_function": "relu",
        "learning_rate": 1e-3,
        "learning_rate_schedule": 0,
        "batch_size": 50,
        "embedding_size": 1024,
        "hidden_size": 200,
        "belief_size": 200,
        "state_size": 30,
        "adam_epsilon": 1e-4,
        "planning_horizon": 12,     # TODO: Does a planning horizon that is shorter than the episode length make sense with sparse rewards?
        "optimization_iters": 10,
        "candidates": 1000,
        "top_candidates": 100,
        "overshooting_distance": 50,
        "action_repeat": 2,
        "bit_depth": 5,
        "free_nats": 3,
        "seed": 1,
        "disable_cuda": False
    }

    train_parameters = {
        "chunk_size": 50,
        "render": False,
        "experience_size": 1000000,
        "nr_seed_episodes": 5,
        "action_noise": 0.3,
        "collect_interval": 100,
        "disable_testing": True,
        "test_interval": 25,
        "test_episodes": 10,
        "global_kl_beta": 0.0,
        "overshooting_kl_beta": 0.0,
        "overshooting_reward_scale": 0.0,
        "grad_clip_norm": 1000
    }

    algorithm_args = {"model_parameters": model_parameters}

    return ModelConfig(PlanetWrapper, algorithm_args, train_parameters)
