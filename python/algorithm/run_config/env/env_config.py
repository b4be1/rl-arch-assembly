from dataclasses import dataclass
from typing import Dict, Any, Type

from task import BaseTask


@dataclass
class EnvConfig:
    env_class: Type[BaseTask]
    env_args: Dict[str, Any]
