from typing import Optional, Iterable, Dict, Set, TYPE_CHECKING

from assembly_gym.util import Transformation
from .template_connection_point import TemplateConnectionPoint
from .template_part import TemplatePart

import numpy as np

if TYPE_CHECKING:
    from .aggregation_part import AggregationPart


class ConnectionPoint:
    """
    Represents a specific connection point of a specific part
    """

    def __init__(self, parent_part: "AggregationPart", base_connection_point: TemplateConnectionPoint,
                 part_templates: Iterable[TemplatePart]):
        """

        :param parent_part:             Part this connection point is on
        :param base_connection_point:   Template connection point this connection point is an instance of
        :param part_templates:          Collection of all part templates of the aggregation
        """
        self._parent_part = parent_part
        self._base_connection_point = base_connection_point
        self._connected_to: Optional[ConnectionPoint] = None
        self._possible_connections = {
            p: set(p.connection_points[r.connection_index]
                   for r in base_connection_point.allowed_connections
                   if r.part_name == p.name)
            for p in part_templates}
        self._fully_blocked = False
        self._transformation = parent_part.pose.transform(base_connection_point.transformation)
        self._update_fully_blocked()

    def connect(self, other: "ConnectionPoint"):
        """
        Connects this point to a connection point of another part
        :param other: Connection point to connect to
        :return:
        """
        assert not self.is_connected
        assert not other.is_connected
        assert other.base_connection_point in self._possible_connections[other.parent_part.base_part]
        assert self._base_connection_point in other._possible_connections[self._parent_part.base_part]
        self._connected_to = other
        other._connected_to = self

    def disconnect(self):
        """
        Detaches this connection point from the point it was connected to
        :return:
        """
        assert self.is_connected
        self._connected_to._connected_to = None
        self._connected_to = None

    def mark_blocked(self, template_part: TemplatePart, template_connection_point: TemplateConnectionPoint):
        """
        Mark a part-type - connection-point tuple as blocked. That is, let this connection point remember that the given
        part on the given connection point cannot be attached to it
        :param template_part:               Other part
        :param template_connection_point:   Connection point of the other part
        :return:
        """
        self._possible_connections[template_part].remove(template_connection_point)
        if len(self._possible_connections[template_part]) == 0:
            self._update_fully_blocked()

    def mark_fully_blocked(self):
        """
        Mark this connection point as fully blocked. That is, let this connection point remember that no part can
        be attached to it.
        :return:
        """
        for p in self._possible_connections:
            self._possible_connections[p].clear()

    def _update_fully_blocked(self):
        """
        Helper function to update the fully blocked attribute
        :return:
        """
        self._fully_blocked = all(len(s) == 0 for s in self._possible_connections.values())

    @property
    def parent_part(self) -> "AggregationPart":
        """
        Part this connection point is on
        :return:
        """
        return self._parent_part

    @property
    def base_connection_point(self) -> TemplateConnectionPoint:
        """
        Template connection point this connection point is an instance of
        :return:
        """
        return self._base_connection_point

    @property
    def connected_to(self) -> Optional["ConnectionPoint"]:
        """
        Connection point connected to this connection point (or None if it is free)
        :return:
        """
        return self._connected_to

    @property
    def is_connected(self) -> bool:
        """
        True if this connection point is connected to any other connection point
        :return:
        """
        return self._connected_to is not None

    @property
    def possible_connections(self) -> Dict[TemplatePart, Set[TemplateConnectionPoint]]:
        """
        Part-type - connection-point tuples that could potentially be attached to this connection point
        :return:
        """
        if not self.is_connected:
            return self._possible_connections
        else:
            return {p: set() for p in self._possible_connections}

    @property
    def is_fully_blocked(self):
        """
        True if no part can be attached to this connection point
        :return:
        """
        return self._fully_blocked or self.is_connected

    @property
    def transformation(self) -> Transformation:
        """
        Transformation of this connection point relative to its part
        :return:
        """
        return self._transformation
