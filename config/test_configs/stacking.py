from pathlib import Path

from aggregation import AggregationProperties
from algorithm.run_config.env.stacking import get_env_config
from task import RobotEnv, StackingTask


def load_config(project_root: Path) -> RobotEnv:
    properties_path = project_root / "example" / "rhino_export" / "box"
    properties = AggregationProperties.load(properties_path)

    task_path = project_root / "data" / "tasks_test" / "box_single_short"

    robot_parameters_path = project_root / "config" / "robot_parameters.json"

    def env_factory(_robot_parameters_path=robot_parameters_path, _properties=properties, _task_path=task_path) \
            -> StackingTask:
        env_config = get_env_config(_robot_parameters_path, aggregation_properties=_properties, task_path=_task_path,
                                    part_release_distance=0.2, time_limit_s=5.0)
        return env_config.env_class(**env_config.env_args)

    return env_factory
