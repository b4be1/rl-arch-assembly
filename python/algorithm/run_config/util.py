import importlib
from pathlib import Path

import sys

from .run_config import RunConfig


def load_config(log_path: Path) -> RunConfig:
    assert log_path.exists()
    config_path = str(log_path.resolve())
    if config_path not in sys.path:
        sys.path.append(config_path)
    module = importlib.import_module("run_config.run_config")

    return module.load_config(Path(__file__).parents[3])
