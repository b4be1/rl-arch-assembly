import json
import logging

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import trimesh
import trimesh.boolean
import trimesh.grouping

from .template_connection_point import TemplateConnectionPoint

logger = logging.getLogger(__file__)

class TemplatePart:
    """
    Represents a template part of a Grasshopper-Wasp-like aggregation
    """

    def __init__(self, name: str, connection_points: List[TemplateConnectionPoint], sub_meshes: List[trimesh.Trimesh],
                 visualization_mesh: trimesh.Trimesh, mass_kg: float, friction: float):
        """

        :param name:                Name of this part type
        :param connection_points:   Connection points of this part type
        :param sub_meshes:          Convex meshes this part type consists of
        :param visualization_mesh:  Full (potentially non-convex) mesh of this part type without simulator gaps
        :param mass_kg:             Mass of this part in kilograms
        :param friction             (Bullet) friction of this part
        """
        self._name = name
        self._connection_points = connection_points
        self._visualization_mesh = visualization_mesh
        self._sub_meshes = tuple(sub_meshes)
        self._mesh = trimesh.util.concatenate(self._sub_meshes)
        self._mass_kg = mass_kg
        self._friction = friction

    @classmethod
    def from_dict(cls, part_properties: Dict, rule_dicts: List[Dict], mesh_dir: Path,
                  cp_gap_size: float) -> "TemplatePart":
        """
        Loads a part type from a Rhino exported dictionary
        :param part_properties: Property dictionary of this part
        :param rule_dicts:      Dictionary containing all connection rules
        :param mesh_dir:        Directory the meshes are stored in
        :param cp_gap_size:     Gap size between the connection points of two parts (note that this number will be added
                                to both connection points and hence be effectively doubled)
        :return: TemplatePart instance representing this part type
        """
        name = part_properties["name"]

        connection_points = TemplateConnectionPoint.from_dict(
            name, part_properties["connections"], rule_dicts, cp_gap_size)

        part_mesh_dir = mesh_dir / name
        sub_meshes = []
        for shape_file_name in part_mesh_dir.iterdir():
            with shape_file_name.open() as shape_file:
                sub_meshes.append(trimesh.load(shape_file, file_type="obj").apply_scale(1e-3))
        mesh = trimesh.load((mesh_dir / "{}.obj".format(name)).open(), file_type="obj").apply_scale(1e-3)
        if "mass_kg" in part_properties:
            mass = part_properties["mass_kg"]
        else:
            mass = 1
            logger.warning("Did not find mass_kg for part {} in part_properties. Defaulting to {}.".format(name, mass))
        if "friction" in part_properties:
            friction = part_properties["friction"]
        else:
            friction = 0.5
            logger.warning("Did not find friction for part {} in part_properties. Defaulting to {}."
                           .format(name, friction))
        return TemplatePart(name, connection_points, sub_meshes, mesh, mass, friction)

    @classmethod
    def load_parts(cls, directory: Path) -> List["TemplatePart"]:
        """
        Load all parts in the given aggregation properties directory
        :param directory: Path to the aggregation properties directory
        :return: List of part types
        """
        properties_path = directory / "properties.json"
        with properties_path.open() as properties_file:
            properties = json.load(properties_file)

        return [
            cls.from_dict(part_properties, properties["rules"], directory, properties["gap_size"])
            for part_properties in properties["parts"]
        ]

    def __repr__(self) -> str:
        return "Part {}".format(self._name)

    @property
    def name(self) -> str:
        """
        Name of this part type
        :return:
        """
        return self._name

    @property
    def visualization_mesh(self) -> trimesh.Trimesh:
        """
        Full mesh of this part type without simulator gaps.
        :return:
        """
        return self._visualization_mesh

    @property
    def connection_points(self) -> List["TemplateConnectionPoint"]:
        """
        Connection points of this part type
        :return:
        """
        return self._connection_points

    @property
    def sub_meshes(self) -> Tuple[trimesh.Trimesh]:
        """
        Convex sub meshes this part consists of
        :return:
        """
        return self._sub_meshes

    @property
    def mesh(self) -> trimesh.Trimesh:
        """
        Full mesh of this part.
        :return:
        """
        return self._mesh

    @property
    def mass_kg(self) -> float:
        """
        Mass of this part
        :return:
        """
        return self._mass_kg

    @property
    def friction(self) -> float:
        """
        (Bullet) friction of this part
        :return:
        """
        return self._friction

    @property
    def bounding_box_marker_positions(self) -> np.ndarray:
        return np.array(self.mesh.bounding_box_oriented.vertices)
