from typing import Optional, Dict, Iterable, List, Any

import numpy as np

from assembly_gym.util import Transformation
from .connection_point import ConnectionPoint
from .template_connection_point import TemplateConnectionPoint
from .template_part import TemplatePart
from .util import calculate_aggregation_transform


class AggregationPart:
    """
    Represents a specific part in the aggregation
    """

    def __init__(self, base_part: TemplatePart, all_part_templates: Iterable[TemplatePart],
                 transformation: Optional[Transformation] = None,
                 parent_connection_point: Optional[ConnectionPoint] = None,
                 connection_point_to_parent: Optional[TemplateConnectionPoint] = None,
                 tags: Optional[Dict[str, Any]] = None):
        """

        :param base_part:                   Template part this part is an instance of
        :param all_part_templates:          List of all template parts in the aggregation
        :param transformation:              Transformation of this part
        :param parent_connection_point:     Connection point of the parent this part is connected to
        :param connection_point_to_parent:  Connection point of this part the parent is connected to
        :param tags:                        Tags attached to this part
        """
        self._base_part = base_part
        self._all_part_templates = all_part_templates

        assert (parent_connection_point is None) == (connection_point_to_parent is None), \
            "Either both connection points must be None or neither."
        if connection_point_to_parent is not None:
            assert self._base_part.connection_points[connection_point_to_parent.index] == connection_point_to_parent

        if transformation is None:
            if parent_connection_point is None:
                self._transformation = Transformation()
            else:
                # to make pylint shut up
                assert connection_point_to_parent is not None

                # Compute the transformation of this part based on the transformation of the parent
                gap_size = parent_connection_point.base_connection_point.gap_size + connection_point_to_parent.gap_size
                self._transformation = calculate_aggregation_transform(
                    parent_connection_point.transformation,
                    connection_point_to_parent.transformation, gap_size=gap_size)
        else:
            self._transformation = transformation

        self._connection_points = [ConnectionPoint(self, cp, all_part_templates) for cp in base_part.connection_points]

        if connection_point_to_parent is not None and parent_connection_point is not None:
            self._connection_point_to_parent = self._connection_points[connection_point_to_parent.index]
            self._connection_points[connection_point_to_parent.index].connect(parent_connection_point)
        else:
            self._connection_point_to_parent = None

        self._tags = {} if tags is None else tags

    def add_child(self, own_connection_point: ConnectionPoint, child_part: TemplatePart,
                  child_connection_point: TemplateConnectionPoint,
                  child_tags: Optional[Dict[str, Any]] = None) -> "AggregationPart":
        """
        Adds a child part to this part
        :param own_connection_point:    Connection point of this part to attach the child to
        :param child_part:              Template part of the new child
        :param child_connection_point:  Connection point of the child to attach this part to
        :param child_tags:              Tags to pass to the child
        :return:
        """
        assert not own_connection_point.is_connected, "Connection already occupied"
        placed_part = AggregationPart(child_part, all_part_templates=self._all_part_templates,
                                      parent_connection_point=own_connection_point,
                                      connection_point_to_parent=child_connection_point,
                                      tags=child_tags)
        return placed_part

    def remove_child(self, connection_point: ConnectionPoint):
        """
        Remove the child on the given connection point
        :param connection_point:    Connection point to disconnect. Must not be the connection point the parent is
                                    attached to
        :return:
        """
        assert self._connection_points[connection_point.base_connection_point.index] == connection_point
        assert connection_point != self._connection_point_to_parent, "Cannot remove parent"
        connection_point.disconnect()

    def traverse(self) -> List["AggregationPart"]:
        """
        Compute a list of all parts of the tree with this part as root
        :return: List of parts
        """
        output_list = []
        self._traverse(output_list)
        return output_list

    def _traverse(self, output_list: List["AggregationPart"]):
        """
        Helper function to build the traversal list. All parts get attached to the given list.
        :param output_list: List of parts to attach descendants to.
        :return:
        """
        output_list.append(self)
        for cp in self.connection_points_to_children:
            child = cp.connected_to.parent_part
            child._traverse(output_list)

    def tree_dict(self) -> Dict:
        """
        Compute a dictionary describing this aggregation tree
        :return:
        """
        return {
            "name": self._base_part.name,
            "connections": [
                {
                    "own_cp_index": c.base_connection_point.index,
                    "child_cp_index": c.connected_to.base_connection_point.index,
                    "child": c.connected_to.parent_part.tree_dict()
                }
                for c in self.connection_points_to_children
            ],
            "tags": self._tags
        }

    @classmethod
    def from_tree_dict(cls, tree_dict: Dict, template_parts: Iterable[TemplatePart],
                       root_transformation: Optional[Transformation] = None) -> "AggregationPart":
        """
        Recover this aggregation tree from a tree dictionary
        :param tree_dict:           Dictionary describing the tree structure
        :param template_parts:      Template parts used in the tree
        :param root_transformation: Transformation of the root part
        :return:
        """
        template_parts_by_name = {p.name: p for p in template_parts}
        root = AggregationPart(template_parts_by_name[tree_dict["name"]], template_parts, tags=tree_dict["tags"],
                               transformation=root_transformation)
        cls._from_tree_dict(tree_dict, template_parts_by_name, root)
        return root

    @classmethod
    def _from_tree_dict(cls, tree_dict: Dict, template_parts_by_name: Dict[str, TemplatePart], root_part: "AggregationPart"):
        """
        Helper function for from_tree_dict, which additionally takes a root node as argument and hence can be called
        recursively
        :param tree_dict:               Dictionary describing the tree structure
        :param template_parts_by_name:  Dictionary mapping a parts name on the part itself
        :param root_part:               Root part of the tree to build
        :return:
        """
        for c_dict in tree_dict["connections"]:
            cp_own = root_part.connection_points[c_dict["own_cp_index"]]
            child_template_part = template_parts_by_name[c_dict["child"]["name"]]
            cp_child = child_template_part.connection_points[c_dict["child_cp_index"]]
            child = root_part.add_child(cp_own, child_template_part, cp_child, c_dict["child"]["tags"])
            cls._from_tree_dict(c_dict["child"], template_parts_by_name, child)

    @property
    def pose(self) -> Transformation:
        """
        Pose of this part
        :return:
        """
        return self._transformation

    @property
    def position(self) -> np.ndarray:
        """
        Position of this part
        :return:
        """
        return self._transformation.translation

    @property
    def quaternion(self):
        """
        Quaternion of this part
        :return:
        """
        return self._transformation.quaternion

    @property
    def base_part(self) -> TemplatePart:
        """
        Template part this part is an instance of
        :return:
        """
        return self._base_part

    @property
    def connection_point_to_parent(self) -> Optional[ConnectionPoint]:
        """
        Connection point of this part the parent is attached to
        :return:
        """
        return self._connection_point_to_parent

    @property
    def connection_points_to_children(self) -> List[ConnectionPoint]:
        """
        All connection points that are currently attached to children of this part
        :return:
        """
        return [cp for cp in self._connection_points if cp != self._connection_point_to_parent and cp.is_connected]

    @property
    def connection_points(self) -> List[ConnectionPoint]:
        """
        All connection points of this parts
        :return:
        """
        return self._connection_points

    @property
    def available_connection_points(self):
        """
        All free connection points of this part
        :return:
        """
        return [cp for cp in self._connection_points if not cp.is_fully_blocked]

    @property
    def tags(self) -> Dict[str, Any]:
        return self._tags

    def __repr__(self) -> str:
        return "PlacedPart {} at {}".format(self._base_part.name, self.position)
