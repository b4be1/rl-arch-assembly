import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aggregation import AggregationProperties
from task import StackingTask
from task.controllers import JointPositionController
from task.controllers.endeffector_velocity_controller import EndEffectorVelocityController
from task.rewards import ConstantReward
from task.rewards.part import AllPartMarkerReward, AllPartVelocityReward
from task.rewards.robot import ArmTargetJointAccelerationReward
from task.sensors.robot import EndEffectorPoseSensor
from task.sensors.part import CurrentPartPoseSensor
from task.sensors.part.current_part_target_pose_sensor import CurrentPartTargetPoseSensor
from assembly_gym.environment.pybullet import PybulletEnvironment


def close(val, target):
    return abs(val - target) < 0.01


def restore_rot(rot_obs: np.ndarray) -> Rotation:
    rot = np.zeros((3, 3))
    rot[:, :2] = rot_obs
    rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
    rot[:, 2] /= np.linalg.norm(rot[:, 2])
    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]
    return Rotation.from_matrix(rot)


if __name__ == "__main__":
    task_path = Path("../data/tasks/box_single_short")

    properties_path = Path.cwd().parent / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    simulation_timestep = 0.005
    controller_interval = 0.05
    max_steps = int(round(5.0 / controller_interval))

    robot_parameters_path = Path.cwd() / ".." / "config" / "robot_parameters.json"
    with robot_parameters_path.open() as robot_parameters_file:
        robot_parameters = json.load(robot_parameters_file)

    sensors = [CurrentPartTargetPoseSensor(), EndEffectorPoseSensor(), CurrentPartPoseSensor()]
    rewards = [ConstantReward(intermediate_timestep_reward_scale=0.012 / max_steps),
               AllPartMarkerReward(final_timestep_reward_scale=0.349, max_marker_distance=0.5),
               AllPartVelocityReward(final_timestep_reward_scale=0.629),
               ArmTargetJointAccelerationReward(
                   intermediate_timestep_reward_scale=0.01 / max_steps)]
    arm_controller = EndEffectorVelocityController.from_parameters(robot_parameters["ur10"])
    gripper_controller = JointPositionController.from_parameters(robot_parameters["rh_p12_rn"])

    target_position_xy = np.array([0.3, 0.6])
    sim_env = PybulletEnvironment()
    substeps = int(controller_interval / simulation_timestep)
    spawn_position = np.zeros((3, 2))
    spawn_position[2] += 0.3
    env = StackingTask(arm_controller, gripper_controller, sensors, rewards, task_path, properties,
                      spawn_position_limits_table_offset=spawn_position,
                      spawn_orientation_limits_euler=np.zeros((3, 2)),
                      target_position_table_offset_xy=np.zeros((2, )),
                      start_with_first_part_grasped=True,
                      time_step=controller_interval, add_digit_sensors=False,
                      substeps_per_step=substeps, time_limit_steps=max_steps)
    env.headless = False
    env.initialize(sim_env, auto_restart=False)

    try:
        total_reward = 0
        next_step = time.time()
        obs = env.reset()
        placed = False
        while True:
            gripper_pos = obs["end_effector_pos"]
            gripper_rot = restore_rot(obs["end_effector_rot"])

            part_pos = obs["current_part_pos"]
            part_rot = restore_rot(obs["current_part_rot"])

            gripper_pos_part_frame = part_rot.apply(gripper_pos - part_pos, inverse=True)
            gripper_rot_part_frame = part_rot.inv() * gripper_rot

            part_target_pos = obs["current_part_target_pos"]
            part_target_rot = restore_rot(obs["current_part_target_rot"])

            gripper_target_pos = part_target_rot.apply(gripper_pos_part_frame) + part_target_pos
            gripper_target_rot = part_target_rot * gripper_rot_part_frame

            dist = gripper_target_pos - gripper_pos
            if np.linalg.norm(dist) < 0.002:
                placed = True

            if not placed:
                gripper_direction = dist
            else:
                gripper_direction = np.array([0, 0, 1])
            action_val = np.concatenate([gripper_direction * 10, np.zeros(3)])
            gripper_action = np.array([0.6 if placed else 0.75])

            gripper_action = 2 * gripper_action - 1

            action_val += np.random.normal(0, 0.01, size=action_val.shape)
            action = {"arm": action_val, "gripper": gripper_action}
            time.sleep(max(next_step - time.time(), 0))
            next_step += env.time_step
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print("Total reward: {}, final: {}".format(total_reward, reward))
                obs = env.reset()
                placed = False
                total_reward = 0

    except KeyboardInterrupt:
        env.close()
