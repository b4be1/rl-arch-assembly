import math
from pathlib import Path

import gym

from aggregation import AggregationProperties
from algorithm.learning_rate import exponential_wrapper, step_function, TimePortionWrapper
from algorithm.run_config.run_config import RunConfig
from task.rewards import ConstantReward
from task.rewards.robot import *
from task.rewards.part import *
from task.sensors.robot import *
from task.sensors.part import *
from task.wrappers.goal_env_wrapper import GoalEnvWrapper
from task.rewards.goal.current_part_marker_goal_reward import CurrentPartMarkerGoalReward

# !copy-include
from algorithm.run_config.model.her import get_model_config
# !copy-include
from algorithm.run_config.env.stacking import get_env_config


def load_config(project_root: Path) -> RunConfig:
    properties_path = project_root / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    task_path = project_root / "data" / "tasks" / "box_single_short"

    robot_parameters_path = project_root / "config" / "robot_parameters.json"

    def env_factory(_robot_parameters_path=robot_parameters_path, _properties=properties,
                    _task_path=task_path) -> gym.GoalEnv:
        env_config = get_env_config(_robot_parameters_path, aggregation_properties=_properties, task_path=_task_path,
                                    part_release_distance=0.01, time_limit_s=3.0)
        time_limit_steps = env_config.env_args["time_limit_steps"]
        max_marker_distance = 0.2
        # TODO: This should be moved to env_config
        env_config.env_args["rewards"] = [
            ConstantReward(intermediate_timestep_reward_scale=0.005 / time_limit_steps),
            ReleasedPartVelocityReward(intermediate_timestep_reward_scale=0.39 / 10, max_linear_velocity=0.1,
                                       max_angular_velocity=math.pi / 4),
            EndEffectorAccelerationReward(intermediate_timestep_reward_scale=0.005 / time_limit_steps)
        ]
        env_config.env_args["sensors"] = [ArmJointSinCosSensor(), ArmJointVelocitySensor(), GripperJointAngleSensor(),
                                          GripperJointVelocitySensor(), EndEffectorPoseSensor(),
                                          EndEffectorVelocitySensor(), CurrentPartVelocitySensor()]
        achieved_goal_sensor = CurrentPartPoseSensor()
        desired_goal_sensor = CurrentPartTargetPoseSensor()
        goal_rewards = [CurrentPartMarkerGoalReward(final_timestep_reward_scale=0.6,
                                                    max_marker_distance=max_marker_distance,
                                                    sparse=True)]
        return GoalEnvWrapper(env_config.env_class(**env_config.env_args), achieved_goal_sensor,
                              desired_goal_sensor, goal_rewards)

    env_config = get_env_config(robot_parameters_path, aggregation_properties=properties, task_path=task_path)

    model_config = get_model_config(7)

    total_time = 72 * 60 * 60
    total_time_steps = int(100e6)

    t = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    v = [-4, -4.5, -5, -5.5, -6, -6.5, -7]

    step_func = step_function(t, v)
    exp = exponential_wrapper(step_func, 10)
    lr_sched = TimePortionWrapper(exp, total_time)

    model_config.algorithm_args["model_parameters"]["learning_rate"] = lr_sched
    model_config.algorithm_args["model_parameters"]["max_episode_length"] = env_config.env_args["time_limit_steps"]

    algorithm = model_config.algorithm_class(**model_config.algorithm_args)
    return RunConfig(env_factory, algorithm, model_config.train_config, total_time_steps)
