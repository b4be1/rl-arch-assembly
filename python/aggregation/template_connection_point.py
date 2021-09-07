from typing import NamedTuple, List, Dict, Union, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from assembly_gym.util import Transformation

if TYPE_CHECKING:
    from .template_part import TemplatePart

AllowedConnection = NamedTuple("AllowedConnection", (("part_name", str), ("connection_index", int)))


class TemplateConnectionPoint:
    """
    Represents a connection point of a template part
    """

    def __init__(self, transformation: Transformation, allowed_connections: List[AllowedConnection], index: int,
                 gap_size: float):
        """

        :param transformation:      Transformation of this connection point relative to the part
        :param allowed_connections: List of allowed connections (part type and connection point) on this connection
                                    point
        :param index:               Index of this connection point
        :param gap_size:            Size of the gap between this connection point and the other connection point (gap
                                    sizes of two connection points get added when connecting them)
        """
        self._transformation = transformation
        self._allowed_connections = allowed_connections
        self._index = index
        self._gap_size = gap_size

    @staticmethod
    def from_dict(part_name: str, connection_point_dicts: List[Dict[str, List[float]]],
                  rule_dicts: List[Dict[str, Union[str, int]]], gap_size: float) \
            -> List["TemplateConnectionPoint"]:
        """
        Loads the connection points from the part description inside a Rhino exported dictionary
        :param part_name:               Name of the part this connection point is on
        :param connection_point_dicts:  Dictionaries of the parts connection points
        :param rule_dicts:              Dictionaries with the connection rules
        :param gap_size:                Size of the gap between this connection point and the attached connection point
        :return:
        """
        output = []
        for i, c_d in enumerate(connection_point_dicts):
            rotation_mat = np.zeros((3, 3))
            translation = np.array(c_d["origin"]) * 1e-3
            for j, a in enumerate("xyz"):
                rotation_mat[:, j] = c_d["{}_axis".format(a)] / np.linalg.norm(c_d["{}_axis".format(a)])
            transformation = Transformation(translation=translation, rotation=Rotation.from_matrix(rotation_mat))

            allowed_connections1 = {
                AllowedConnection(r["part2"], r["conn2"])
                for r in rule_dicts if r["part1"] == part_name and r["conn1"] == i
            }

            allowed_connections2 = {
                AllowedConnection(r["part1"], r["conn1"])
                for r in rule_dicts if r["part2"] == part_name and r["conn2"] == i
            }

            # Ensure that there are no redundant rules
            allowed_connections = allowed_connections1.union(allowed_connections2)
            output.append(TemplateConnectionPoint(transformation, list(allowed_connections), i, gap_size))
        return output

    @property
    def transformation(self) -> Transformation:
        """
        Transformation of this connection point relative to its part
        :return:
        """
        return self._transformation

    @property
    def allowed_connections(self) -> List[AllowedConnection]:
        """
        List of allowed connections (part type and connection point) on this connection point
        :return:
        """
        return self._allowed_connections

    @property
    def origin(self) -> np.ndarray:
        """
        Position of this connection point relative to its part
        :return:
        """
        return self._transformation[:3, 3]

    @property
    def gap_size(self) -> float:
        """
        Size of the gap between this connection point and the attached connection point
        :return:
        """
        return self._gap_size

    @property
    def index(self) -> int:
        """
        Index of this connection point
        :return:
        """
        return self._index

    def __repr__(self) -> str:
        return "ConnectionPoint at {}".format(self.origin)
