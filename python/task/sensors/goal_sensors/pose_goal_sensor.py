from typing import List

import gym
import numpy as np
from scipy.spatial.transform import Rotation

from .goal_sensor import GoalSensor
from assembly_gym.util import Transformation


# TODO: Normalization missing
# TODO: Avoid duplicate code from PoseSensor
class PoseGoalSensor(GoalSensor[Transformation]):
    def observe(self, goal_state: Transformation) -> np.ndarray:
        translation_obs = goal_state.translation
        rotation_obs = goal_state.matrix[:3, :2].reshape(-1)
        return np.concatenate((translation_obs, rotation_obs))

    def get_state(self, goal_observation: np.ndarray) -> Transformation:
        translation = goal_observation[:3]
        rotation_obs = goal_observation[3:].reshape(3, 2)
        rotation_last_column = np.cross(rotation_obs[:, 0], rotation_obs[:, 1]).reshape(3, 1)
        rotation_matrix = np.concatenate((rotation_obs, rotation_last_column), axis=1)
        rotation = Rotation.from_matrix(rotation_matrix)
        return Transformation(translation, rotation)

    def _get_goal_observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-np.ones(9), np.ones(9))
