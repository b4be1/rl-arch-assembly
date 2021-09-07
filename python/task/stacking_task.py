import itertools
import json
import logging
import math
import random
from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple, List, Sequence, Any

import numpy as np
from scipy.spatial.transform import Rotation

from aggregation import AggregationPart, AggregationProperties, TemplatePart
from assembly_gym.environment.generic import JointMode
from scene.simulated_assembly_scene import SimulatedAssemblyScene
from .base_task import ResetFailedException
from .controllers.controller import Controller
from scene.part import Part
from .rewards.reward import Reward
from assembly_gym.environment.simulation import SimulationEnvironment, SimulationObject
from .simulated_task import SimulatedTask
from .sensors import Sensor
from assembly_gym.environment.simulation import SimulationRobot
from assembly_gym.util import Transformation


class StackingTask(SimulatedTask):
    """
    A task where the robot has to stack a given number of parts on a given existing structure.
    """

    def __init__(self, controllers: Iterable[Controller],
                 sensors: Iterable[Sensor["StackingTask"]], rewards: Iterable[Reward["StackingTask"]], tasks_path: Path,
                 aggregation_properties: AggregationProperties, spawn_position_limits_table_offset: np.ndarray,
                 spawn_orientation_limits_euler: np.ndarray, target_position_table_offset_xy: Sequence[float],
                 start_with_first_part_grasped: bool = False, add_digit_sensors: bool = False,
                 time_step=0.005, time_limit_steps: Optional[int] = None,
                 part_release_distance: float = 0.05, fix_pre_placed_parts: bool = False,
                 grasp_offset_limits_y: Sequence[float] = (0, 0), grasp_offset_limits_z: Sequence[float] = (0, 0),
                 target_structure_yaw: float = math.pi / 2, use_mesh_bounding_boxes: bool = True,
                 terminate_episode_if_target_reached: bool = False, target_reached_marker_tolerance: float = 1e-3):
        """
        :param controllers:                         A sequence of controller objects that define the actions on the
                                                    environment
        :param sensors:                             A sequence of sensors objects that provide observations to the agent
        :param rewards:                             A sequence of rewards objects that provide rewards to the agent
        :param tasks_path:                          the path to the task definitions
        :param aggregation_properties:              the properties of the aggregation that the robot should build
                                                    (contains the geometry of the parts and the target structure)
        :param spawn_position_limits_table_offset:  a (3, 2) array of limits on the position in which the parts spawn;
                                                    the coordinates are interpreted as relative (offsets) to the center
                                                    of the table surface
                                                    --> shape: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        :param spawn_orientation_limits_euler:      a (3, 2) array of limits on the orientation as euler angles
                                                    (see assembly_gym.util.transformation for the definition of the
                                                    euler angles)
                                                    --> shape: [[alpha_min, alpha_max], [beta_min, beta_max],
                                                        [gamma_min, gamma_max]]
        :param target_position_table_offset_xy:     a (2, ) array of the x- and y-position of the desired position of
                                                    the target structure; the coordinates are interpreted as relative
                                                    (offsets) to the center of the table surface
        :param start_with_first_part_grasped:       whether the first part should already be in the gripper at the
                                                    beginning of each episode
        :param add_digit_sensors:                   whether the DIGIT sensors should be added to the gripper fingers
        :param time_step:                           the time between two controller updates (actions of the agent)
        :param time_limit_steps:                    The number of steps until the episode terminates (if the robot does
                                                    not finish the task)
        :param part_release_distance:               the minimum distance from part to fingers at which the part is
                                                    considered to be released from the gripper
        :param fix_pre_placed_parts:                whether the positions of the parts that are already placed at the
                                                    beginning of the episode should be fixed (i.e. whether the
                                                    parts should be static)
        :param grasp_offset_limits_y:               a [lower, upper] sequence of the limits of the grasp position
                                                    randomization parallel to the "spikes" of the part
        :param grasp_offset_limits_z:               a [lower, upper] sequence of the limits of the grasp position
                                                    randomization along the height of the part
        :param target_structure_yaw:                the angle of the rotation of the target structure (around the
                                                    z-axis)
        :param use_mesh_bounding_boxes:             Whether to use the bounding boxes of each convex mesh instead of the
                                                    mesh itself
        :param terminate_episode_if_target_reached: whether the episode should terminate as soon as all bounding box
                                                    markers of the last part are in the tolerance (see
                                                    target_reached_marker_tolerance) of their target positions
        :param target_reached_marker_tolerance:     the position tolerance of the part bounding box markers; ignored
                                                    if terminate_episode_if_target_reached is False

        """
        super(StackingTask, self).__init__(controllers, sensors, rewards, time_step=time_step,
                                           time_limit_steps=time_limit_steps)

        self._aggregation_properties: AggregationProperties = aggregation_properties
        scene_config_path = Path(__file__).parent.parent.parent / "config" / "placing_scene_config.json"
        self.__scene = SimulatedAssemblyScene(aggregation_properties.template_parts, scene_config_path,
                                              use_part_bounding_boxes=use_mesh_bounding_boxes)

        self._task_dicts: Optional[Dict[int, Dict[str, Any]]] = None
        self.set_tasks(tasks_path)

        assert spawn_position_limits_table_offset.shape == (3, 2), \
            "spawn_position_limits_table_offset must be of shape (3, 2)"
        self._spawn_position_limits_table_offset = spawn_position_limits_table_offset
        assert spawn_orientation_limits_euler.shape == (3, 2), "spawn_orientation_limits_euler must be of shape (3, 2)"
        self._spawn_orientation_limits_euler = spawn_orientation_limits_euler

        self._spawn_poses: Optional[List[Transformation]] = None

        self.__table_surface_center = Transformation(
            (self.__scene.pickup_table_surface_center.translation
             + self.__scene.place_table_surface_center.translation) / 2,
            self.__scene.pickup_table_surface_center.rotation)    # TODO: Ugly hack, spawn positions should be specified relative to pickup table, place poses relative to place table
        target_position_table_offset = np.concatenate((target_position_table_offset_xy, np.zeros(1)))
        target_position = self.__table_surface_center.transform(target_position_table_offset)
        self._transformation_structure_to_world: Transformation = Transformation.from_pos_euler(
            target_position, np.array([0, 0, target_structure_yaw]))

        self._grasp_offset_limits: np.ndarray = np.concatenate((np.zeros((1, 2)),
                                                                np.array(grasp_offset_limits_y).reshape(1, 2),
                                                                np.array(grasp_offset_limits_z).reshape(1, 2)), axis=0)

        self._placed_parts: List[Part] = []  # the list of parts that are already placed
        self._current_part: Optional[Part] = None  # the part that is placed currently
        self._future_parts: List[Part] = []  # the list of parts that are placed afterwards

        self._start_with_first_part_grasped: bool = start_with_first_part_grasped

        self.__fix_pre_placed_parts: bool = fix_pre_placed_parts
        self.__logger: logging.Logger = logging.getLogger("env")
        self.__part_release_distance: float = part_release_distance

        self.__add_digit_sensors = add_digit_sensors
        self.__use_mesh_bounding_boxes = use_mesh_bounding_boxes

        self.__part_collision_shapes: Optional[Dict[TemplatePart, Any]] = None

        self.__terminate_episode_if_target_reached = terminate_episode_if_target_reached
        self.__target_reached_marker_tolerance = target_reached_marker_tolerance

        self.__current_task_idx: Optional[int] = None

    def _initialize_scene(self) -> None:
        self.__scene.initialize_scene(self.environment)

    def _step_task(self) -> Tuple[bool, Dict]:
        # The episode ends if the last part in the placing order is released from the gripper (i.e. it is more than
        # self.__part_release_distance away from the fingers of the gripper).
        # TODO: Find out how this can be done in reality
        part_released = max(
            self.robot.finger_distances_to_object(self._current_part.scene_object)) > self.__part_release_distance
        target_reached = np.all(np.linalg.norm(self._current_part.bounding_box_marker_positions -
                                               self._current_part.target_bounding_box_marker_positions, axis=-1) \
                                <= self.__target_reached_marker_tolerance)
        # TODO: This does not make sense if there is only one part and start_with_first_part_grasped = False
        # TODO: Target reached condition does not make sense with more than one part placed
        done = len(self._future_parts) == 0 and (part_released
                                                 or self.__terminate_episode_if_target_reached and target_reached)

        # If there is a next part, it is spawned when the previous part is released from the gripper.
        # TODO: Need to check that the part is not at the spawn position
        #  --> otherwise parts will get stuck into each other
        if part_released and len(self._future_parts) > 0:
            self._next_part()

        # === Calculate metrics for the evaluation ================
        current_part_pose_error = self._current_part.scene_object.pose.transform(
            self._current_part.target_pose, inverse=True)
        current_part_position_error = np.linalg.norm(current_part_pose_error.translation)
        current_part_angle_error = current_part_pose_error.quat_angle

        metrics = {
            "current_part_position_error": current_part_position_error,
            "current_part_angle_error": current_part_angle_error,
            "parts_placed": len(self._placed_parts)
        }

        if len(self._placed_parts) > 0:
            placed_part_pose_errors = [p.scene_object.pose.transform(p.target_pose, inverse=True)
                                       for p in self._placed_parts]
            placed_part_position_errors = np.linalg.norm([p.translation for p in placed_part_pose_errors], axis=-1)
            placed_part_angle_errors = [p.quat_angle for p in placed_part_pose_errors]
            placed_part_position_error_mean = np.mean(placed_part_position_errors)
            placed_part_angle_error_mean = np.mean(placed_part_angle_errors)
            metrics.update({
                "mean_placed_part_position_error": placed_part_position_error_mean,
                "mean_placed_part_angle_error": placed_part_angle_error_mean
            })

        info = {"metrics": metrics}

        return done, info

    def _reset_task(self) -> None:
        # ==== Clean up last episode ========================
        self._placed_parts = []
        self._current_part = None
        self.__scene.reset_scene()

        # ==== Prepare next episode =========================
        self.__current_task_idx = random.randint(0, len(self._task_dicts) - 1)
        current_task_dict = self._task_dicts[self.__current_task_idx]

        # Read the information about the aggregation from aggregation properties
        root = AggregationPart.from_tree_dict(current_task_dict["aggregation"],
                                              self._aggregation_properties.template_parts,
                                              Transformation.from_matrix(current_task_dict["root_transformation"]))
        self._parts_by_id = {
            p.tags["id"]: p for p in root.traverse()
        }
        placed_parts_aggregation = [self._parts_by_id[i] for i in current_task_dict["pre_placed_parts"]]
        future_parts_aggregation = [self._parts_by_id[i] for i in current_task_dict["construction_order"]]

        # Add the parts that are already placed (spawn pose is target pose) to the scene
        self._placed_parts = [
            Part(aggregation_part.base_part,
                 self._transformation_structure_to_world.transform(aggregation_part.pose))
            for aggregation_part in placed_parts_aggregation
        ]
        for part in self._placed_parts:
            self.__scene.spawn_part(part)

        # Sample spawn poses for all parts that the robot should place
        spawn_poses = [self._sample_spawn_pose(part) for part in future_parts_aggregation]
        self._future_parts = [
            Part(aggregation_part.base_part, self._transformation_structure_to_world.transform(aggregation_part.pose),
                 spawn_pose)
            for aggregation_part, spawn_pose in zip(future_parts_aggregation, spawn_poses)
        ]

        # The building plan from current_task_dict includes gaps between the parts so that they do not get stuck into
        # each other. To get sensible target positions (without the gaps in z-direction), add all parts to the scene
        # and let the simulator run for a couple of steps. Use the resulting positions of the parts as new target
        # positions.
        for part in self._future_parts:
            self.__scene.spawn_part(part, part.target_pose)

        for _ in range(3):
            self.environment.step()

        for part in itertools.chain(self._placed_parts, self._future_parts):
            # Copy the target pose from the current pose
            part.target_pose = part.pose.copy()

        # Sanity check that all target positions are on the table
        for part in itertools.chain(self._placed_parts, self._future_parts):
            if not self.__scene.is_pose_above_tables(part.target_pose):
                raise ResetFailedException("Task {}: Target position {} for part is not on the table."
                                           .format(self.__current_task_idx, part.target_pose.translation.tolist()))

        for part in self._future_parts:
            self.environment.remove_object(part.scene_object)
            part.scene_object = None

        # Fix the position of the already placed parts if self.__fix_pre_placed_parts is set
        if self.__fix_pre_placed_parts and isinstance(self.environment, SimulationEnvironment):
            for p in self._placed_parts:
                p.scene_object.set_static(True)

        assert len(self._future_parts) > 0, "Cannot build an empty structure"
        self._next_part()
        if self._start_with_first_part_grasped:
            grasping_successful = self._grasp_part()
            if not grasping_successful:
                raise ResetFailedException("Grasping the part was not successful.")

        self._prev_action = None

    def _sample_spawn_pose(self, part: AggregationPart) -> Transformation:
        """
        Draw a spawn pose for a given part uniformly from within the spawn position limits. The orientation is sampled
        by drawing euler angles uniformly from within the orientation limits.

        :param part:            the aggregation part that should be spawned
        :return:                the pose in which the part should be spawned as a Transformation object
        """
        spawn_position_offset = np.random.uniform(self._spawn_position_limits_table_offset[:, 0],
                                                  self._spawn_position_limits_table_offset[:, 1])
        spawn_position = self.__table_surface_center.transform(spawn_position_offset)
        randomized_euler = np.random.uniform(self._spawn_orientation_limits_euler[:, 0],
                                             self._spawn_orientation_limits_euler[:, 1])
        randomized_rotation = Transformation.from_pos_euler(euler_angles=randomized_euler).rotation
        # Rotate the part so that is in the target orientation and then apply the orientation randomization
        structure_rotation = self._transformation_structure_to_world.rotation
        part_target_rotation = structure_rotation * part.pose.rotation
        return Transformation(spawn_position, randomized_rotation * part_target_rotation)

    def _next_part(self) -> None:
        """
        Select the next part to be placed and update the target marker accordingly.
        """
        assert len(self._future_parts) > 0, "_next_part called with empty _future_parts"
        if self._current_part is not None:
            self._placed_parts.append(self._current_part)
        # Remove the next FuturePart from self._future_parts and place it in the scene
        self._current_part = self._future_parts.pop(0)
        self.__scene.spawn_part(self._current_part)
        self.__scene.update_target_marker(self._current_part)

    def _grasp_part(self) -> bool:
        """
        Move the gripper to the part and grasp it.

        :return:    whether grasping the part was successful
        """
        gripper = self.robot.gripper

        # Make the part static to ensure that the part remains at the spawn position
        self._current_part.scene_object.set_static(True)
        self._current_part.scene_object.set_collidable(False)
        # Make the robot static to allow setting its joint angles
        self.robot.arm.set_static(True)
        self.robot.gripper.set_static(True)
        successful = self._position_gripper_at_part() <= 1e-2

        # Test if arm is still moving
        joint_positions = self.robot.arm.joint_positions
        arm_moved = True
        i = 0
        while arm_moved:
            self.environment.step()
            arm_moved = np.max(np.abs(joint_positions - self.robot.arm.joint_positions)) > 0.01
            if arm_moved:
                self.robot.arm.move_to_joint_positions(joint_positions)
                i += 1
                if i == 20:
                    self.__logger.warning("Robot did not stop moving for {} iterations during the reset phase "
                                          "(task index: {}). Crashing simulator to force a restart."
                                          .format(self.__current_task_idx, i))
                    raise ValueError("Robot did not stop moving for {} iterations during the reset phase "
                                     "(task index: {}).".format(i, self.__current_task_idx))

        # Set the gripper position to almost touch the part
        # --> we cannot just close the gripper since the fingers would just get stuck in the part
        gripper.move_to_joint_positions(np.array([0.65]))
        # Enable dynamics again to grasp the part
        self._current_part.scene_object.set_collidable(True)
        self.robot.gripper.set_static(False)

        gripper_joint_mode = gripper.get_joint_mode()
        gripper.set_joint_mode(JointMode.POSITION_CONTROL)
        gripper.set_joint_target_positions(np.array([0.8]))
        for i in range(5):
            self.environment.step()
        gripper.set_joint_mode(gripper_joint_mode)
        self.robot.arm.set_static(False)
        self._current_part.scene_object.set_static(False)
        return successful

    # TODO: Only works for part short
    def _position_gripper_at_part(self) -> float:
        """
        Place the gripper at the part so that the fingers are around on of the middle "spikes" of the part. The position
        of the gripper will be offsetted uniformly randomly (inside the limits specified by self.__grasp_offset_limits)
        parallel to the "spikes" and along the height of the parts.

        :return:        the error in the gripper position
        """
        robot = self.robot
        assert isinstance(robot, SimulationRobot)
        randomized_grasp_offset = np.random.uniform(self._grasp_offset_limits[:, 0], self._grasp_offset_limits[:, 1])
        transformation = self._current_part.scene_object.pose.matrix
        if transformation[2, 2] > 0:  # The part is not flipped
            grasp_offset = np.array([0.035, 0.0, 0.035]) + randomized_grasp_offset
            gripper_orientation = np.array([np.pi, 0.0, np.pi / 2])
        else:  # The part is flipped
            grasp_offset = np.array([0.0, 0.0, -0.035]) + randomized_grasp_offset
            gripper_orientation = np.array([0.0, 0.0, -np.pi / 2])
        gripper_pose_part_frame = Transformation(translation=grasp_offset,
                                                 rotation=Rotation.from_euler("XYZ", angles=gripper_orientation))
        gripper_pose_world_frame = self._current_part.pose.transform(gripper_pose_part_frame)

        gripper = self.robot.gripper
        gripper_pos = np.array([0.7])
        gripper.move_to_joint_positions(gripper_pos)
        gripper.set_joint_target_positions(gripper_pos)

        gripper_position_error = robot.place_gripper_at(gripper_pose_world_frame)
        return gripper_position_error

    def set_tasks(self, task_path: Path) -> None:
        """
        Sets the tasks, i.e. the structure definitions that the robot should build.

        :param task_path:       the path to the task definitions
        """
        self._task_dicts = []
        task_dict_file_paths = list(task_path.iterdir())
        for task_dict_file_path in task_dict_file_paths:
            with task_dict_file_path.open() as f:
                self._task_dicts.append(json.load(f))
        random.shuffle(self._task_dicts)

    @property
    def current_part(self) -> Optional[Part]:
        """
        Returns the part that is currently placed by the robot.

        :return:            the part that is currently placed by the robot
        """
        return self._current_part

    @property
    def placed_parts(self) -> List[Part]:
        """
        The parts that are already placed correctly.

        :return:            the parts that are already placed correctly
        """
        return self._placed_parts

    @property
    def future_parts(self) -> List[Part]:
        """
        Returns the parts that the robot has to place.

        :return:        a list of parts that the robot still has to place
        """
        return self._future_parts

    @property
    def aggregation_properties(self) -> AggregationProperties:
        """
        The properties of the aggregation that the robot is building (contains the geometry of the parts and the target
        structure).

        :return:        the properties of the aggregation that the robot is building
        """
        return self._aggregation_properties

    @property
    def robot(self) -> SimulationRobot:
        return self.__scene.robot
