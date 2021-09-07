import json
from pathlib import Path
from typing import List, Dict, Sequence, Optional

import numpy as np

from assembly_gym.util import Transformation


class SceneConfig:
    def __init__(self, scene_config_json_path: Path, calibrated_offsets_json_path: Optional[Path] = None):
        with scene_config_json_path.open() as scene_config_json:
            self._scene_config = json.load(scene_config_json)
        if calibrated_offsets_json_path is not None:
            with calibrated_offsets_json_path.open() as calibrated_offsets_json:
                calibrated_offsets = json.load(calibrated_offsets_json)
            self._calibrated_offset_pickup_table_xy = calibrated_offsets["pickup_table"]
            self._calibrated_offset_place_table_xy = calibrated_offsets["place_table"]
        else:
            self._calibrated_offset_pickup_table_xy = np.zeros(2)
            self._calibrated_offset_place_table_xy = np.zeros(2)

    def _get_table_pose(self, table_config_dict: Dict[str, List[float]], calibrated_offsets: Sequence[float]) \
            -> Transformation:
        table_pos = np.array(table_config_dict["center_pos_xy"] + [self.table_extents[2] / 2]) \
                    + np.concatenate((calibrated_offsets, np.zeros(1)))
        table_euler = [0, 0, table_config_dict["yaw"]]
        return Transformation.from_pos_euler(table_pos, table_euler)

    @property
    def pickup_table_pose(self) -> Transformation:
        return self._get_table_pose(self._scene_config["tables"]["pickup_table"],
                                    self._calibrated_offset_pickup_table_xy)

    @property
    def place_table_pose(self) -> Transformation:
        return self._get_table_pose(self._scene_config["tables"]["place_table"],
                                    self._calibrated_offset_place_table_xy)

    @property
    def table_extents(self) -> np.ndarray:
        return np.array(self._scene_config["table_extents"])
