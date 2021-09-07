import json
from pathlib import Path

import numpy as np
import gym.wrappers

from task.controllers import ConstantActionController, JointPositionController
from task.controllers.endeffector_velocity_controller import EndEffectorVelocityController
from task.controllers.torque_controller import TorqueController
from task.rewards.robot import EndEffectorAccelerationReward
from task.rewards.robot.end_effector_stabilize_reward import EndEffectorStabilizeReward
from task.sensors import EndEffectorPoseSensor, TargetPositionSensor
from task.stabilize_env import StabilizeTask
from assembly_gym.environment.pybullet import PybulletEnvironment


def create_env(headless: bool, simulation_timestep: float):
    robot_parameters_path = Path.cwd() / ".." / "config" / "robot_parameters.json"
    with robot_parameters_path.open() as robot_parameters_file:
        robot_parameters = json.load(robot_parameters_file)
    arm_controller = EndEffectorVelocityController.from_parameters(robot_parameters["ur10"])
    gripper_controller = ConstantActionController(
        JointPositionController.from_parameters(robot_parameters["rh_p12_rn"]), np.array([1.0]))
    sensors = [EndEffectorPoseSensor(), TargetPositionSensor()]
    rewards = [
        EndEffectorStabilizeReward(intermediate_timestep_reward_scale=0.95 / 500,
                                   max_pos_distance=0.4),
        EndEffectorAccelerationReward(intermediate_timestep_reward_scale=0.05 / 500)
    ]
    sim_env = PybulletEnvironment()
    env_original = StabilizeTask(arm_controller, gripper_controller, sensors=sensors, rewards=rewards,
                                time_step=simulation_timestep, initial_position_var=0.2,
                                add_digit_sensors=False)
    env_original.headless = headless
    env_original.initialize(sim_env, auto_restart=False)
    env_time_limit = gym.wrappers.TimeLimit(env_original, max_episode_steps=500)
    return env_time_limit


def close(val, target):
    return abs(val - target) < 0.01


if __name__ == "__main__":
    simulation_timestep = 0.05

    env = create_env(False, simulation_timestep)
    env_original: StabilizeTask = env.unwrapped
    try:
        total_reward = 0
        obs = env.reset()
        while True:
            position = obs["end_effector_pos"] * 2
            target_robot_frame = obs["target_position"]
            direction_robot_frame = target_robot_frame - position
            action_val = np.concatenate([direction_robot_frame * 10, [0, 0, 0]])
            action_val += np.random.normal(0, 0.01, size=action_val.shape)
            action = {"arm": action_val, "gripper": np.array([])}
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Total reward: {}, final: {}".format(total_reward, reward))
                for name, reward in info.items():
                    print("{}: {}".format(name, reward))
                env.reset()
                part1_done = False
                total_reward = 0

    except KeyboardInterrupt:
        env.close()

    visualization_env = create_env(False, simulation_timestep)
