from typing import Optional, TypeVar

import numpy as np

from aggregation import TemplatePart
from assembly_gym.environment.generic import Object
from assembly_gym.util import Transformation

ObjectType = TypeVar("ObjectType", bound=Object)


class Part:
    def __init__(self, base_part: TemplatePart, target_pose: Transformation,
                 spawn_pose: Optional[Transformation] = None, scene_object: Optional[ObjectType] = None):
        self.base_part = base_part
        self.target_pose = target_pose
        if spawn_pose is None:
            self.spawn_pose = target_pose
        else:
            self.spawn_pose = spawn_pose
        self.scene_object = scene_object

    @property
    def pose(self) -> Transformation:
        """
        Returns the current pose of the part. If the part is not in the scene, the spawn pose is returned.

        :return:                the current pose of the part
        """
        if self.scene_object is not None:
            return self.scene_object.pose
        else:
            return self.spawn_pose

    @property
    def bounding_box_marker_positions(self) -> np.ndarray:
        """
        Returns the current 3D positions of the eight corners of bounding box of the part. If the part is not in the
        scene, the positions of the target markers are returned as if the part is in its spawn pose.

        :return:                a (8, 3) array of the current 3D positions of the eight corners of the bounding box
        """
        return self.pose.transform(self.base_part.bounding_box_marker_positions)

    @property
    def target_bounding_box_marker_positions(self) -> np.ndarray:
        """
        Returns the 3D positions of where the eight corners of bounding box of the part are if the part is in its target
        position.

        :return:                a (8, 3) array of the target 3D positions of the eight corners of the bounding box
        """
        return self.target_pose.transform(self.base_part.bounding_box_marker_positions)
