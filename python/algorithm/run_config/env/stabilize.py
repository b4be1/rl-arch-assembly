import json
from pathlib import Path

import numpy as np

from algorithm.run_config.env import EnvConfig
from task import StabilizeTask
from task.controllers import *
from task.sensors.robot import *
from task.sensors import *
from task.rewards.robot import *


def get_env_config(robot_parameters_path: Path, use_end_effector_controller: bool = False,
                   time_limit_s: float = 5.0) -> EnvConfig:
    controller_interval = 0.05
    time_limit_steps = int(round(time_limit_s / controller_interval))

    # ===== Controller definitions =====
    with robot_parameters_path.open() as robot_parameters_file:
        robot_parameters = json.load(robot_parameters_file)

    if use_end_effector_controller:
        arm_controller = EndEffectorVelocityController.from_parameters("ur10", robot_parameters["ur10"])
    else:
        arm_controller = JointVelocityDifferenceController.from_parameters(
            "ur10", "arm", robot_parameters["ur10"], controller_interval)
    gripper_controller = ConstantActionController(
        JointPositionController("ur10", "gripper", np.zeros(1), np.ones(1)), np.array([0.6]))

    sensors = [ArmJointVelocitySensor(), ArmJointSinCosSensor(), EndEffectorPoseSensor(), EndEffectorVelocitySensor(),
               TargetPositionSensor(), TimeSensor()]

    rewards = [
        EndEffectorStabilizeReward(intermediate_timestep_reward_scale=0.95 / time_limit_steps, max_pos_distance=0.4),
        EndEffectorAccelerationReward(intermediate_timestep_reward_scale=0.05 / time_limit_steps)
    ]

    env_args = {
        "controllers": [arm_controller, gripper_controller],
        "sensors": sensors,
        "rewards": rewards,
        "time_step": controller_interval,
        "initial_position_var": 0.2,
        "time_limit_steps": time_limit_steps
    }

    return EnvConfig(env_class=StabilizeTask, env_args=env_args)
