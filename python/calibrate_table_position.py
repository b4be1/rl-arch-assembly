from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import json

from assembly_gym.util import Transformation
from rtde_receive import RTDEReceiveInterface

from scene.scene_config_parser import SceneConfig


def calibrate_table_pose(table_pose, table_extents):
    table_orientation = Transformation(rotation=table_pose.rotation)
    gripper_pos_robot_frame = np.array(rtde_receive.getActualTCPPose()[:3])
    gripper_pos_world_frame = Transformation.from_pos_euler(euler_angles=[0, 0, np.pi]).transform(gripper_pos_robot_frame)
    gripper_pos_table_orientation = table_orientation.transform(np.array(gripper_pos_world_frame), inverse=True)
    table_center_pos_table_orientation = \
        np.array([gripper_pos_table_orientation[0] + table_extents[0] / 2,
                  gripper_pos_table_orientation[1] - table_extents[1] / 2, 0])
    table_offset = table_orientation.transform(table_center_pos_table_orientation) - table_pose.translation
    return list(table_offset[:2])


parser = ArgumentParser("Calibrates the position of the table with the robot arm.")
parser.add_argument("robot_ip", type=str)
parser.add_argument("scene_config_path", type=str)

args = parser.parse_args()

rtde_receive = RTDEReceiveInterface(args.robot_ip)
scene_config_path = Path(args.scene_config_path)
scene_config = SceneConfig(scene_config_path)

table_extents = np.array(scene_config.table_extents)

calibrated_offsets = {}

input("Close the gripper and move the arm to the top-left corner of the pickup table (viewed from the robot's base)")
pickup_table_center_offset_xy = calibrate_table_pose(scene_config.pickup_table_pose, table_extents)
print(pickup_table_center_offset_xy)
calibrated_offsets["pickup_table"] = pickup_table_center_offset_xy

input("Close the gripper and move the arm to the top-left corner of the place table (viewed from the robot's base)")
place_table_center_pos_xy = calibrate_table_pose(scene_config.place_table_pose, table_extents)
print(place_table_center_pos_xy)
calibrated_offsets["place_table"] = pickup_table_center_offset_xy

calibrated_offsets_path = scene_config_path.parent / "calibrated_offsets.json"
with calibrated_offsets_path.open("w") as calibrated_offsets_file:
    json.dump(calibrated_offsets, calibrated_offsets_file)


