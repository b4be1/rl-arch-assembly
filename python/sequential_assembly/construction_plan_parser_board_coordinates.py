import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation
from trimesh import Trimesh

from assembly_gym.util import Transformation
from scene.scene_config_parser import SceneConfig


class ConstructionPlanParserBoardCoordinates:
    def __init__(self, part_mesh: Trimesh, scene_config_path: Path):
        self._part_mesh = part_mesh
        self._scene_config = SceneConfig(scene_config_path)

        self._offset_x = -0.024
        self._offset_y = 0.0185
        self._coordinate_to_m = 0.003

    def parse(self, task_path: Path) -> Tuple[List[Transformation], List[Transformation]]:
        with task_path.open() as task_file:
            task_coordinates_dict = json.load(task_file)
        goal_poses = self._coordinates_to_poses(task_coordinates_dict["goal"], self._scene_config.place_table_pose)
        spawn_poses = self._coordinates_to_poses(task_coordinates_dict["spawn"], self._scene_config.pickup_table_pose)

        return spawn_poses, goal_poses

    def _coordinates_to_poses(self, coordinates: Sequence[Dict[str, Any]], table_pose) -> List[Transformation]:
        poses = []
        part_extents = self._part_mesh.bounding_box.extents
        bounding_box_vertices = np.array(self._part_mesh.bounding_box.vertices)
        bounding_box_center_to_origin = (np.max(bounding_box_vertices, axis=0) + np.min(bounding_box_vertices,
                                                                                        axis=0)) / 2
        transformation_table_corner_to_world = Transformation(
            table_pose.transform(np.array([-self._scene_config.table_extents[0] / 2,
                                           self._scene_config.table_extents[1] / 2,
                                           self._scene_config.table_extents[2] / 2])), table_pose.rotation)
        for curr_coordinates in coordinates:
            coordinate = curr_coordinates["position"]
            rotation = self._orientation_nr_to_rotation(curr_coordinates["orientation_nr"])
            corner_position = np.array([self._offset_y + self._coordinate_to_m * coordinate[1],
                                        self._offset_x - self._coordinate_to_m * coordinate[0],
                                        -0.01])
            rotated_extents = rotation.apply(part_extents)
            bounding_box_center_to_origin_rotated = rotation.apply(bounding_box_center_to_origin)
            bounding_box_center_pos = corner_position + np.array(
                [rotated_extents[0], -rotated_extents[1], rotated_extents[2]])
            pose_table_corner_frame = Transformation(
                rotation.apply(bounding_box_center_pos + bounding_box_center_to_origin_rotated, inverse=True), rotation)
            poses.append(transformation_table_corner_to_world.transform(pose_table_corner_frame))
        return poses

    def _orientation_nr_to_rotation(self, pose_nr: int) -> Rotation:
        if pose_nr == 3:
            euler = [np.pi / 2, 0, 0]
        else:
            euler = [0, 0, 0]
        return Rotation.from_euler("xyz", euler)
