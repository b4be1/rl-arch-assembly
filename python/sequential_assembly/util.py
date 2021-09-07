from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from assembly_gym.util import Transformation

# TODO: The robot should figure out which of the 180° rotated gripper orientations makes the most sense
def gripper_pose_from_sl_pose(part_pose: Transformation, curr_gripper_pose: Optional[Transformation] = None) \
        -> Transformation:
    if np.allclose(part_pose.matrix[:3, 1], np.array([0, 0, 1]), atol=1e-4):
        grasp_offset = np.array([-0.01, 0.02, 0.015])
        if np.allclose(part_pose.matrix[:3, 0], np.array([1, 0, 0]), atol=1e-4):
            gripper_orientation = np.array([np.pi / 2, 0.0, 0.0])
        else:
            gripper_orientation = np.array([np.pi / 2, np.pi, 0.0])
    elif np.allclose(part_pose.matrix[:3, 1], np.array([0, 0, -1]), atol=1e-4):
        if np.allclose(part_pose.matrix[:3, 0], np.array([1, 0, 0]), atol=1e-4):
            gripper_orientation = np.array([np.pi / 2, np.pi, np.pi])
        else:
            gripper_orientation = np.array([np.pi / 2, 0.0, np.pi])
        grasp_offset = np.array([0.025, -0.02, -0.015])
    else:
        raise ValueError("No gripper pose defined for SL block with orientation {}".format(part_pose))
    gripper_pose_part_frame = Transformation.from_pos_euler(grasp_offset, gripper_orientation)
    gripper_pose_world_frame = part_pose.transform(gripper_pose_part_frame)
    # if curr_gripper_pose is not None:               # TODO: This probably only works if the parts are placed in the same orientation as they are picked up!
    #     curr_yaw = curr_gripper_pose.euler[2]
    #     target_yaw = gripper_pose_world_frame.euler[2]
    #     residual = (target_yaw - curr_yaw) % np.pi
    #     new_yaw = curr_yaw + residual
    #     new_euler = [gripper_pose_world_frame.euler[0], gripper_pose_world_frame.euler[1], new_yaw]
    #     gripper_pose_world_frame = Transformation.from_pos_euler(gripper_pose_world_frame.translation, new_euler)
    return gripper_pose_world_frame
