import numpy as np

from scene.part import Part
from .placing_controller import PlacingController
from sequential_assembly.sl_sequential_assembly import PlacingFailedException


class MagicPlacingController(PlacingController):
    def place(self, part: Part):
        distance_to_transport_goal = np.linalg.norm(part.pose.translation - part.target_pose.translation)
        distance_rotations = np.linalg.norm(part.pose.matrix[:3, :3] - part.target_pose.matrix[:3, :3])
        if distance_to_transport_goal > 0.15:
            raise PlacingFailedException("Could not place part, {}m is too far away".format(distance_to_transport_goal))
        elif distance_rotations > 1.0:
            raise PlacingFailedException("Could not place part, wrong orientation (quaternion distance: {})"
                                         .format(distance_rotations))
        else:  # TODO: Check also orientation, reduce threshold
            part.scene_object.set_pose(part.target_pose)
            part.scene_object.set_static(True)
