from abc import ABC
from typing import Dict, Optional

import numpy as np

from aggregation import TemplatePart
from .part_sensor import PartSensor
from ..discrete_sensor import DiscreteSensor
from task import StackingTask


class PartTypeSensor(DiscreteSensor[StackingTask], PartSensor, ABC):
    def __init__(self, **kwargs):
        super(PartTypeSensor, self).__init__(**kwargs)
        self.__template_parts_indices: Optional[Dict[TemplatePart, int]]

    def _get_nr_values(self) -> Dict[str, np.ndarray]:
        self.__template_parts_indices = {
            p: i for i, p in enumerate(self.task.aggregation_properties.template_parts)
        }
        nr_template_parts = len(self.task.aggregation_properties.template_parts)
        return {
            "{}_type".format(self.name_prefix): np.array(self._nr_observed_parts * [nr_template_parts])
        }

    def observe(self) -> Dict[str, np.ndarray]:
        template_part_indices = np.array([self.__template_parts_indices[part.base_part]
                                          for part in self._get_observed_parts()])
        return {
            "{}_type".format(self.name_prefix): template_part_indices
        }

    def reset(self) -> Dict[str, np.ndarray]:
        return self.observe()
