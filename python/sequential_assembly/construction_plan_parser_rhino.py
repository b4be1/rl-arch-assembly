import math
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from trimesh import Trimesh

from assembly_gym.util import Transformation


class ConstructionPlanParserRhino:
    def __init__(self, part_mesh: Trimesh):
        self._part_mesh = part_mesh

    def parse(self, task_dir: Path, pickup_table_transformation: Optional[Transformation] = None,
              place_table_transformation: Optional[Transformation] = None) \
            -> Tuple[List[Transformation], List[Transformation]]:
        spawn_poses = self._process_poses_file(task_dir / "spawn.txt")
        if pickup_table_transformation is not None:
            spawn_poses = [pickup_table_transformation.transform(p) for p in spawn_poses]
        goal_poses = self._process_poses_file(task_dir / "goal.txt")
        if place_table_transformation is not None:
            goal_poses = [place_table_transformation.transform(p) for p in goal_poses]
        return spawn_poses, goal_poses

    def _process_poses_file(self, poses_file: Path) -> List[Transformation]:
        poses = []
        with poses_file.open() as f:
            position_re = re.compile("O\([0-9\-,. ]*\)")
            orientation_re = re.compile("Z\([0-9\-,. ]*\)")
            for line in f.readlines():
                position_str = re.search(position_re, line)[0]
                position = np.array([float(val) for val in (position_str[2:-1]).split(",")])
                orientation_str = re.search(orientation_re, line)[0]
                orientation = np.array([float(val) for val in (orientation_str[2:-1]).split(",")])
                poses.append(Transformation.from_pos_euler(position, orientation))
                # poses.append(Transformation.from_pos_euler(position, [math.pi / 2, 0, 0]))

        return [self._from_rhino_pose(pose) for pose in poses]

    @staticmethod
    def _from_rhino_pose(pose):
        position_scaled = pose.translation / 1000 * np.array([-1, -1, 1])
        position_pybullet = position_scaled + np.array([0, 0, 0.705 - 0.005])   # TODO: Fix rhino scene. The socket is 0.5cm too high, but the lower mat is 1cm to high as well
        return Transformation(position_pybullet, pose.rotation)

