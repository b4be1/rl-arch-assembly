from abc import ABC, abstractmethod

from typing import Dict, TypeVar, Union, Optional, Generic, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from assembly_gym.util import Transformation
from .invertible_continuous_sensor import InvertibleContinuousSensor

TaskType = TypeVar("TaskType")


class PoseSensor(InvertibleContinuousSensor[TaskType, Transformation], Generic[TaskType], ABC):
    def __init__(self, name_prefixes: Sequence[str], position_bounds_lower: Union[float, np.ndarray] = -2,
                 position_bounds_upper: Union[float, np.ndarray] = 2, use_reduced_rotation_matrix: bool = True,
                 angular_noise_variance: float = 0, linear_noise_variance: float = 0,
                 default_pose: Optional[Transformation] = None):
        """
        @param name_prefixes:               Prefix to put in front of the names of this sensor's readings.
        @param position_bounds_lower:       Lower bounds of the position this sensor reads.
        @param position_bounds_upper:       Upper bounds of the position this sensor reads.
        @param use_reduced_rotation_matrix: Whether to use the reduced (2 instead of 3 columns) rotation matrix as
                                            output.
        @param angular_noise_variance:      Variance of the angular noise. The angular noise is computed as an angle
                                            drawn from a Gaussian distribution, which is then used to rotate the object
                                            around a uniformly randomly chosen axis.
        @param linear_noise_variance:       Variance of the linear additive noise.
        @param default_pose:                The pose that is observed if one of the poses in _observe_poses() is None.
                                            default_pose is assumed to be in the observer's frame. If None is passed,
                                            the default pose is at the lower limits of the possible positions.
        """
        super(PoseSensor, self).__init__()
        if np.isscalar(position_bounds_lower):
            position_bounds_lower = np.ones(3) * position_bounds_lower
        if np.isscalar(position_bounds_upper):
            position_bounds_upper = np.ones(3) * position_bounds_upper
        limits_pos = {"{}_pos".format(prefix): (position_bounds_lower, position_bounds_upper)
                      for prefix in name_prefixes}
        rot_obs_shape = (3, 2) if use_reduced_rotation_matrix else (3, 3)
        limits_rot = {"{}_rot".format(prefix): (-np.ones(rot_obs_shape), np.ones(rot_obs_shape))
                      for prefix in name_prefixes}
        self.__limits = {**limits_pos, **limits_rot}
        self.__position_bounds_lower = position_bounds_lower
        self.__position_bounds_upper = position_bounds_upper
        self.__use_reduced_rotation_matrix = use_reduced_rotation_matrix
        self.__angular_noise_stddev = np.sqrt(angular_noise_variance)
        self.__linear_noise_stddev = np.sqrt(linear_noise_variance)
        self.__name_prefixes = tuple(name_prefixes)
        if default_pose is not None:
            self.__default_pose = default_pose
        else:
            self.__default_pose = Transformation.from_pos_euler(position_bounds_lower, (0, 0, 0))

    def _get_limits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self.__limits

    def _reset_unnormalized(self) -> Dict[str, np.ndarray]:
        return self._observe_unnormalized()

    def _observe_unnormalized(self) -> Dict[str, np.ndarray]:
        observer_pose = self._observer_frame_pose()
        # Replace every None pose with self.__default_pose
        poses = self._observe_poses()

        # Transform the poses according to the observer pose and replace None with self.__default_pose
        poses_transformed = []
        for pose in poses:
            if pose is None:
                poses_transformed.append(self.__default_pose)   # self.__default_pose is already in the observer's frame
            elif observer_pose is not None:
                poses_transformed.append(observer_pose.transform(pose, inverse=True))
            else:
                poses_transformed.append(pose)

        pos_obs = {
            "{}_pos".format(prefix):
                p.translation + np.random.normal(scale=self.__linear_noise_stddev, size=(3,))
            for p, prefix in zip(poses_transformed, self.__name_prefixes)
        }

        rotation_noise_axes = np.random.uniform(-1, 1, size=(len(poses_transformed), 3))
        rotation_noise_axes /= np.linalg.norm(rotation_noise_axes, axis=1).reshape((-1, 1))
        rotation_noise_angles = np.random.normal(scale=self.__linear_noise_stddev, size=(len(poses_transformed, )))
        noise_rotvecs = rotation_noise_angles.reshape((-1, 1)) * rotation_noise_axes

        rotations = (p.rotation * Rotation.from_rotvec(rv) for p, rv in zip(poses_transformed, noise_rotvecs))

        rot_obs = {
            "{}_rot".format(prefix):
                r.as_matrix()[:, :2] if self.__use_reduced_rotation_matrix else r.as_matrix()
            for r, prefix in zip(rotations, self.__name_prefixes)
        }
        return {**pos_obs, **rot_obs}

    def _observer_frame_pose(self) -> Optional[Transformation]:
        return None

    @abstractmethod
    def _observe_poses(self) -> Sequence[Optional[Transformation]]:
        """
        Returns a sequence of transformations of the observed objects.
        :return:
        """
        pass

    def _observation_to_state_unnormalized(self, unnormalized_observation: Dict[str, np.ndarray]) \
            -> Dict[str, Transformation]:
        states = {}
        for prefix in self.__name_prefixes:
            rotation_vec = unnormalized_observation["{}_rot".format(prefix)]
            position = unnormalized_observation["{}_pos".format(prefix)]
            if self.__use_reduced_rotation_matrix:
                rotation_matrix_first_columns = rotation_vec.reshape(3, 2)
                rotation_last_column = np.cross(rotation_matrix_first_columns[:, 0],
                                                rotation_matrix_first_columns[:, 1]).reshape(3, 1)
                rotation_matrix = np.concatenate((rotation_matrix_first_columns, rotation_last_column), axis=1)
            else:
                rotation_matrix = rotation_vec.reshape(3, 3)
            states[prefix] = Transformation(position, Rotation.from_matrix(rotation_matrix))
        return states

    def __repr__(self):
        return "{}({}, {}, {})".format(type(self).__name__, self.__position_bounds_lower, self.__position_bounds_upper,
                                       self.__use_reduced_rotation_matrix)
