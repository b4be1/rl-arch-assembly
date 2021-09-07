from pathlib import Path

from algorithm.run_config.run_config import RunConfig
from task import StabilizeTask

# !copy-include
from algorithm.run_config.model.planet import get_model_config
# !copy-include
from algorithm.run_config.env.stabilize import get_env_config
from task.rewards.robot.end_effector_stabilize_reward import EndEffectorStabilizeReward


def load_config(project_root: Path) -> RunConfig:
    robot_parameters_path = project_root / "config" / "robot_parameters.json"

    def env_factory(_robot_parameters_path=robot_parameters_path) -> StabilizeTask:
        env_config = get_env_config(_robot_parameters_path)
        env_config.env_args["rewards"] = [
            EndEffectorStabilizeReward(
                intermediate_timestep_reward_scale=0.95 / env_config.env_args["time_limit_steps"], max_pos_distance=0.4)
        ]
        return env_config.env_class(**env_config.env_args)

    model_config = get_model_config()
    algorithm = model_config.algorithm_class(**model_config.algorithm_args)

    total_time_steps = int(100e6)

    return RunConfig(env_factory, algorithm, model_config.train_config, total_time_steps)
