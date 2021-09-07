import math
from pathlib import Path

import numpy as np

from aggregation import AggregationProperties
from algorithm.learning_rate import exponential_wrapper, step_function, TimePortionWrapper
from algorithm.run_config.run_config import RunConfig
from task import StackingTask
from task.sensors.part import FixedNumberPlacedPartsPoseSensor, FixedNumberPlacedPartsTargetPoseSensor
from task.rewards import *
from task.rewards.part import *
from task.rewards.robot import *
from task.controllers import *

# !copy-include
from algorithm.run_config.model.td3 import get_model_config
# !copy-include
from algorithm.run_config.env.stacking import get_env_config


def load_config(project_root: Path) -> RunConfig:
    properties_path = project_root / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    task_path = project_root / "data" / "tasks" / "box_double_short"

    robot_parameters_path = project_root / "config" / "robot_parameters.json"

    def env_factory(_robot_parameters_path=robot_parameters_path, _properties=properties,
                    _task_path=task_path) -> StackingTask:
        env_config = get_env_config(_robot_parameters_path, aggregation_properties=_properties, task_path=_task_path,
                                    part_release_distance=0.1, time_limit_s=2.0)
        env_config.env_args["terminate_episode_if_target_reached"] = True
        target_reached_marker_tolerance = 1e-3
        env_config.env_args["target_reached_marker_tolerance"] = target_reached_marker_tolerance
        env_config.env_args["sensors"].append(FixedNumberPlacedPartsPoseSensor(max_nr_placed_parts=1))
        env_config.env_args["sensors"].append(FixedNumberPlacedPartsTargetPoseSensor(max_nr_placed_parts=1))
        time_limit_steps = env_config.env_args["time_limit_steps"]
        env_config.env_args["rewards"] = \
            [ConstantReward(intermediate_timestep_reward_scale=0.01 / time_limit_steps),
             AllPartMarkerReward(final_timestep_reward_scale=0.65, max_marker_distance=0.5,
                                 marker_pos_tolerance=target_reached_marker_tolerance),
             CurrentPartVelocityReward(intermediate_timestep_reward_scale=0.0, final_timestep_reward_scale=0.3,
                                       max_linear_velocity=0.1, max_angular_velocity=math.pi / 4),
             EndEffectorAccelerationReward(intermediate_timestep_reward_scale=0.04 / time_limit_steps)]
        # Replace the gripper controller with a fixed action controller
        assert len([controller for controller in env_config.env_args["controllers"]
                    if controller.robot_component_name == "gripper"]) == 1
        constant_controller = ConstantActionController(
            JointPositionController("ur10", "gripper", np.zeros(1), np.ones(1)), constant_action=np.ones(1))
        env_config.env_args["controllers"] = [constant_controller if controller.robot_component_name == "gripper"
                                              else controller for controller in env_config.env_args["controllers"]]
        return env_config.env_class(**env_config.env_args)

    model_config = get_model_config()

    total_time = 24 * 60 * 60
    total_time_steps = int(100e6)

    t = [0.0, 0.4, 0.6, 0.8, 0.9, 0.95]
    v = [-4.5, -5, -5.5, -6, -6.5, -7]

    step_func = step_function(t, v)
    exp = exponential_wrapper(step_func, 10)
    lr_sched = TimePortionWrapper(exp, total_time)

    model_config.algorithm_args["learning_rate"] = lr_sched
    model_config.algorithm_args["gradient_steps"] = 1024

    algorithm = model_config.algorithm_class(**model_config.algorithm_args)
    return RunConfig(env_factory, algorithm, model_config.train_config, total_time_steps)
