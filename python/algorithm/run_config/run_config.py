from dataclasses import dataclass
from typing import Dict, Any, Union, Callable

from algorithm import Algorithm
from task import BaseTask


@dataclass
class RunConfig:
    env: Union[BaseTask, Callable[[], BaseTask]]
    algorithm: Algorithm
    train_config: Dict[str, Any]
    total_time_steps: int
