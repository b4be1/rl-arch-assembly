from assembly_gym.environment.simulation import SimulationObject, ShapeTypes
import math
from typing import Tuple, Dict, Iterable, Optional
import numpy as np

from task.controllers.controller import Controller
from .rewards.reward import Reward
from .sensors import Sensor
from assembly_gym.util import Transformation
from .simulated_task import SimulatedTask


class StabilizeTask(SimulatedTask):
    """
    Toy-task for sanity-checking the environment and learning algorithm. The robot starts in a slightly perturbed
    upright position and has to move the end-effector to a given target position.
    """

    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["StabilizeTask"]], rewards: Iterable[Reward["StabilizeTask"]],
                 time_step: float = 0.005, initial_position_var: float = 0.2, time_limit_steps: Optional[int] = None):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param initial_position_var:    The variance of the joint positions for the initial robot configuration
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        super(StabilizeTask, self).__init__(controllers, sensors, rewards, time_step, time_limit_steps=time_limit_steps)
        self.__initial_position_var: float = initial_position_var
        self.__target_pos_radius: Optional[float] = None
        self.__target_pos_mean: Optional[float] = None
        self.__target_pose_world_frame: Optional[Transformation] = None
        self.__target_marker: Optional[SimulationObject] = None

    def _initialize_scene(self) -> None:
        robot = self.environment.add_ur10_robot("ur10")
        robot.set_pose(Transformation.from_pos_euler(position=np.array([0.0, 0.0, 0.71])))
        socket_extents = [0.15, 0.15, 0.71]
        robot_socket_collision_shape = self.environment.create_collision_shape(
            ShapeTypes.BOX, box_extents=[socket_extents])
        robot_socket_visual_shape = self.environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[socket_extents], rgba_colors=(0.7, 0.7, 0.7, 1.0)
        )
        robot_socket = self.environment.add_simple_object(robot_socket_visual_shape, robot_socket_collision_shape)
        robot_socket.set_pose(Transformation.from_pos_euler(position=np.array([0, 0, 0.355])))

        gripper_pos = robot.gripper.pose.translation
        self.__target_pos_radius = 0.3
        self.__target_pos_mean = gripper_pos - np.array([0, 0, 0.3])

        self.__target_pose_world_frame: Transformation = Transformation()

        target_marker_visual = self.environment.create_visual_shape(
            ShapeTypes.SPHERE, sphere_radii=[0.03], rgba_colors=(1.0, 1.0, 0.0, 0.7))
        self.__target_marker = self.environment.add_simple_object(target_marker_visual)

    def _step_task(self) -> Tuple[bool, Dict]:
        done = False
        info = {
            "metrics": {
                "distance_to_target_position": np.linalg.norm(
                    self.target_position_world_frame - self.environment.robots["ur10"].gripper.pose.translation)
            }
        }
        return done, info

    def _reset_task(self) -> None:
        # TODO: np.random.normal expects a standard deviation but self._initial_position_var is assumed to be a variance
        joint_positions = np.random.normal(
            0, self.__initial_position_var, self.environment.robots["ur10"].arm.nr_joints)
        self.environment.robots["ur10"].arm.move_to_joint_positions(joint_positions)
        target_direction = np.random.uniform(-1, 1, (3,))
        target_direction /= np.linalg.norm(target_direction)
        target_distance = np.random.uniform(0, self.__target_pos_radius)
        self.__target_pose_world_frame = Transformation(target_distance * target_direction + self.__target_pos_mean,
                                                        self.__target_pose_world_frame.rotation)
        if self.__target_marker is not None:
            self.__target_marker.set_pose(self.__target_pose_world_frame)

    @staticmethod
    def calc_angle_difference(angle1: float, angle2: float) -> float:
        """
        Calculates the (minimum) distance between two angles.

        :param angle1:      the first angle (in radians)
        :param angle2:      the second angle (in radians)
        :return:            the distance (in radians) from the second angle to the first
        """
        return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi

    @property
    def target_position_arm_frame(self) -> np.ndarray:
        """
        Returns the target position of the end-effector in the coordinate system of the arm.

        :return:            the target position of the end-effector in the coordinate system of the arm
        """
        target_pose_arm_frame = self.environment.robots["ur10"].arm.pose.transform(
            self.__target_pose_world_frame, inverse=True)
        return target_pose_arm_frame.translation

    @property
    def target_position_world_frame(self) -> np.ndarray:
        """
        Returns the target position of the end-effector in the world coordinates.

        :return:            the target position of the end-effector in world coordinates
        """
        return self.__target_pose_world_frame.translation
