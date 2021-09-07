from pathlib import Path

from aggregation import AggregationProperties
from algorithm.learning_rate import exponential_wrapper, step_function, TimePortionWrapper
from algorithm.run_config.run_config import RunConfig
from task import StackingTask

# !copy-include
from algorithm.run_config.model.planet import get_model_config
# !copy-include
from algorithm.run_config.env.stacking import get_env_config


def load_config(project_root: Path) -> RunConfig:
    properties_path = project_root / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    task_path = project_root / "data" / "tasks" / "box_single_short"

    robot_parameters_path = project_root / "config" / "robot_parameters.json"

    def env_factory(_robot_parameters_path=robot_parameters_path, _properties=properties,
                    _task_path=task_path) -> StackingTask:
        env_config = get_env_config(_robot_parameters_path, aggregation_properties=_properties, task_path=_task_path)
        return env_config.env_class(**env_config.env_args)

    model_config = get_model_config()
    algorithm = model_config.algorithm_class(**model_config.algorithm_args)

    total_time_steps = int(100e6)

    return RunConfig(env_factory, algorithm, model_config.train_config, total_time_steps)
