import json
import math
from pathlib import Path

import numpy as np

from aggregation import AggregationProperties
from algorithm.run_config.env import EnvConfig
from task import StackingTask
from task.controllers import *
from task.controllers.joint_velocity_difference_controller import JointVelocityDifferenceController
from task.sensors.robot import *
from task.sensors.part import *
from task.sensors import *
from task.rewards.part import *
from task.rewards.robot import *
from task.rewards import *


def get_env_config(robot_parameters_path: Path, aggregation_properties: AggregationProperties, task_path: Path,
                   use_end_effector_controller: bool = False, time_limit_s: float = 5.0,
                   part_release_distance: float = 0.05, min_gripper_closure: float = 0.4,
                   max_gripper_closure: float = 1.0) -> EnvConfig:
    # ===== stacking_env parameters =====
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
    gripper_controller = JointPositionController(
        "ur10", "gripper", np.ones(1) * min_gripper_closure, np.ones(1) * max_gripper_closure)

    # ===== Sensor definitions =====
    sensors = [ArmJointSinCosSensor(), ArmJointVelocitySensor(), GripperJointAngleSensor(joint_limits_tolerance=0.0),
               GripperJointVelocitySensor(), EndEffectorPoseSensor(), EndEffectorVelocitySensor(),
               CurrentPartPoseSensor(), CurrentPartVelocitySensor(), CurrentPartTargetPoseSensor(), TimeSensor()]

    # ===== Reward definitions =====
    # In the worst case (all cost at the maximum), the reward sums to -1
    rewards = [ConstantReward(intermediate_timestep_reward_scale=0.01 / time_limit_steps),
               AllPartMarkerReward(final_timestep_reward_scale=0.6, max_marker_distance=0.2),
               ReleasedPartVelocityReward(intermediate_timestep_reward_scale=0.35 / 10,
                                          max_linear_velocity=0.1, max_angular_velocity=math.pi / 4),
               EndEffectorAccelerationReward(intermediate_timestep_reward_scale=0.04 / time_limit_steps)]

    env_args = {
        "controllers": [arm_controller, gripper_controller],
        "sensors": sensors,
        "rewards": rewards,
        "spawn_position_limits_table_offset": np.array([[-0.16, -0.14], [-0.01, 0.01], [0.19, 0.21]]),
        "spawn_orientation_limits_euler": np.array([[-5, 5], [-5, 5], [-5, 5]]) * 2 * math.pi / 360,
        "target_position_table_offset_xy": [-0.15, 0.0],
        "target_structure_yaw": math.pi / 2,
        "start_with_first_part_grasped": True,
        "time_step": controller_interval,
        "time_limit_steps": time_limit_steps,
        "tasks_path": task_path,
        "aggregation_properties": aggregation_properties,
        "part_release_distance": part_release_distance,
        "grasp_offset_limits_y": (-0.01, 0.01),
        "grasp_offset_limits_z": (-0.005, 0.005),
        "fix_pre_placed_parts": False
    }

    return EnvConfig(env_class=StackingTask, env_args=env_args)
