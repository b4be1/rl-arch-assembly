from abc import ABC, abstractmethod
from typing import Optional

from ..sl_sequential_assembly import SLSequentialAssembly


class PlacingController(ABC):
    def __init__(self):
        self._sl_sequential_assembly: Optional[SLSequentialAssembly] = None

    def initialize(self, sl_sequential_assembly: SLSequentialAssembly):
        self._sl_sequential_assembly = sl_sequential_assembly

    @abstractmethod
    def place(self, part):
        pass
