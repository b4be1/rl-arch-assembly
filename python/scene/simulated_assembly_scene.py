import json
from pathlib import Path
from typing import Sequence, Optional, Generic, TypeVar, Dict, List

import numpy as np

from aggregation import TemplatePart
from assembly_gym.environment.simulation import SimulationEnvironment, ShapeTypes, SimulationRobot, SimulationObject
from scene.part import Part
from assembly_gym.util import Transformation
from scene.scene_config_parser import SceneConfig

EnvType = TypeVar("EnvType", bound=SimulationEnvironment)
RobotType = TypeVar("RobotType", bound=SimulationRobot)


class SimulatedAssemblyScene(Generic[EnvType, RobotType]):
    """
    Helper class that creates the specific assembly scene in the provided (simulation) environment.
    """

    def __init__(self, template_parts: Sequence[TemplatePart], scene_config_path: Path,
                 calibrated_offsets_path: Optional[Path] = None, use_part_bounding_boxes: bool = False):
        self.__template_parts: Sequence[TemplatePart] = template_parts
        self.__use_part_bounding_boxes = use_part_bounding_boxes

        self._scene_config = SceneConfig(scene_config_path, calibrated_offsets_path)

        self.__environment: Optional[SimulationEnvironment] = None
        self.__part_collision_shapes = None
        self.__part_marker_shapes = None
        self.__current_part_pose_marker = None
        self.pickup_table = None
        self.place_table = None
        self.robot_socket = None

    def initialize_scene(self, environment: SimulationEnvironment) -> None:
        """
        Places robot and table and loads visual and collision shapes for parts.

        :param environment:         the (simulation) environment
        """
        self.__environment = environment
        robot = self.__environment.add_ur10_robot("ur10", rh_p12_rn_urdf_path=Path(__file__).parent.parent.parent /
                                                                             "pybullet" / "models" /
                                                                             "rh_p12_rn_digit_mount_scaled.urdf")
        robot.set_pose(Transformation((0.0, 0.0, 0.705)))
        socket_extents = [0.15, 0.15, 0.705]
        robot_socket_collision_shape = self.__environment.create_collision_shape(
            ShapeTypes.BOX, box_extents=[socket_extents])
        robot_socket_visual_shape = self.__environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[socket_extents], rgba_colors=(0.7, 0.7, 0.7, 1.0)
        )
        self.robot_socket = self.__environment.add_simple_object(robot_socket_visual_shape,
                                                                 robot_socket_collision_shape)
        self.robot_socket.set_pose(Transformation.from_pos_euler(position=np.array([0, 0, 0.3525])))

        self.pickup_table = self._spawn_table(self._scene_config.pickup_table_pose)
        self.place_table = self._spawn_table(self._scene_config.place_table_pose)

        self.__part_collision_shapes = {
            p: self.__environment.create_collision_shape(
                ShapeTypes.MESH, mesh_data=p.sub_meshes, mesh_scale=1e-3,
                use_mesh_bounding_boxes=self.__use_part_bounding_boxes)
            for p in self.__template_parts
        }

        self.__part_marker_shapes = {
            p: self.__environment.create_visual_shape(
                ShapeTypes.MESH, mesh_data=p.sub_meshes, mesh_scale=1e-3, rgba_colors=(0.0, 0.0, 1.0, 0.3))
            for p in self.__template_parts
        }

    def reset_scene(self) -> None:
        """
        Must be called whenever the environment is resetted.
        """
        self.__current_part_pose_marker = None

    def spawn_part(self, part: Part, pose: Optional[Transformation] = None) -> None:
        """
        Spawns the given part in the scene.

        :param part:        the part to spawn
        :param pose:        the pose in which the part should be spawned (default: part.pose)
        """
        if pose is None:
            pose = part.pose
        assert part.scene_object is None, "Part is already in the scene"
        scene_object = self.__environment.add_simple_object(
            collision_shape=self.__part_collision_shapes[part.base_part], mass=part.base_part.mass_kg)
        scene_object.set_pose(pose)
        part.scene_object = scene_object

    def update_target_marker(self, current_part: Part) -> None:
        """
        Update the target marker to visualize the target pose of the next part.

        :param current_part:        the part for which the target marker should be displayed.
        """
        if self.__current_part_pose_marker is not None:
            self.__environment.remove_object(self.__current_part_pose_marker)
        self.__current_part_pose_marker = self.__environment.add_simple_object(
            self.__part_marker_shapes[current_part.base_part])
        self.__current_part_pose_marker.set_pose(current_part.target_pose)

    def is_pose_above_tables(self, pose: Transformation) -> bool:
        """
        Checks whether a given position is on or above one of the tables.

        :param pose:            the pose to check
        :return:                True iff the pose is on or above the table
        """
        pickup_surface_position = self.pickup_table_surface_center.inv.transform(pose).translation
        place_surface_position = self.place_table_surface_center.inv.transform(pose).translation
        is_above_tables = False
        for surface_position in [pickup_surface_position, place_surface_position]:
            is_above_tables = is_above_tables or \
                              (abs(surface_position[0]) <= self._scene_config.table_extents[0] / 2
                               and abs(surface_position[1]) <= self._scene_config.table_extents[1] / 2
                               and surface_position[2] >= 0)
        return is_above_tables

    def _spawn_table(self, pose: Transformation) -> SimulationObject:
        table_collision_shape = self.__environment.create_collision_shape(
            ShapeTypes.BOX, box_extents=[self._scene_config.table_extents])
        table_visual_shape = self.__environment.create_visual_shape(
            ShapeTypes.BOX, box_extents=[self._scene_config.table_extents], rgba_colors=(0.4, 0.3, 0.3, 1.0))
        table = self.__environment.add_simple_object(table_visual_shape, table_collision_shape)
        table.set_pose(pose)
        return table

    def _get_table_surface_center(self, table_pose: Transformation):
        surface_center = table_pose.translation + np.array([0, 0, self._scene_config.table_extents[2] / 2])
        return Transformation(surface_center, table_pose.rotation)

    @property
    def pickup_table_surface_center(self) -> Transformation:
        """
        Returns the pose in the center of the pickup table surface that has the same orientation as the table.

        :return:        the pose in the center of the table surface
        """
        return self._get_table_surface_center(self._scene_config.pickup_table_pose)

    @property
    def place_table_surface_center(self) -> Transformation:
        """
        Returns the pose in the center of the place table surface that has the same orientation as the table.

        :return:        the pose in the center of the table surface
        """
        return self._get_table_surface_center(self._scene_config.place_table_pose)

    @property
    def robot(self) -> SimulationRobot:
        """
        Returns the robot that is used for assembly.

        :return:        the robot to be used for assembly
        """
        return self.__environment.robots["ur10"]
