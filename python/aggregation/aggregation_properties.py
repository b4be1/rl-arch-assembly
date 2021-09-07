from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np

import trimesh
import trimesh.visual

from aggregation import TemplatePart
from trimesh import Trimesh

_AggregationPropertiesFields = NamedTuple("_AggregationPropertiesFields", (
    ("template_parts", Tuple[TemplatePart]), ("global_target_shape", Trimesh)))


class AggregationProperties(_AggregationPropertiesFields):
    """
    Contains aggregation properties as exported from Rhino
    """

    @classmethod
    def load(cls, path: Path) -> "AggregationProperties":
        """
        Loads the properties at the given path
        :param path: Path to load properties from
        :return: AggregationProperties instance containing the loaded properties
        """
        global_target_shape_mesh_path = path / "global_target_shape.obj"
        with global_target_shape_mesh_path.open() as f:
            global_target_shape = trimesh.load(f, file_type="obj", resolver=None).apply_scale(1e-3)
        visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.array([[200, 200, 250, 150]] * len(global_target_shape.vertices)))
        global_target_shape.visual = visual

        parts = TemplatePart.load_parts(path)

        return AggregationProperties(tuple(parts), global_target_shape)
