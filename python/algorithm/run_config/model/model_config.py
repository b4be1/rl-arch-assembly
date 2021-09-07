from dataclasses import dataclass
from typing import Dict, Any, Type

from algorithm import Algorithm


@dataclass
class ModelConfig:
    algorithm_class: Type[Algorithm]
    algorithm_args: Dict[str, Any]
    train_config: Dict[str, Any]
