from pathlib import Path

from aggregation import AggregationProperties
from algorithm.learning_rate import exponential_wrapper, step_function, convert_to_done_portion_wrapper
from algorithm.run_config.run_config import RunConfig
from task import StackingTask

# !copy-include
from algorithm.run_config.model.a2c import get_model_config
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

    env_config = get_env_config(robot_parameters_path, aggregation_properties=properties, task_path=task_path)
    arm_controller = env_config.env_args["arm_controller"]
    gripper_controller = env_config.env_args["gripper_controller"]

    action_dims = arm_controller.action_space.shape[0] + gripper_controller.action_space.shape[0]

    model_config = get_model_config(action_dims)

    total_time_steps = int(50e6)

    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    v = [-4, -4.5, -5, -5.5, -6, -6.5, -7]

    step_func = step_function(t, v)
    exp = exponential_wrapper(step_func, 10)
    lr_sched = convert_to_done_portion_wrapper(exp, total_time_steps)

    model_config.algorithm_args["model_parameters"]["learning_rate"] = lr_sched

    algorithm = model_config.algorithm_class(**model_config.algorithm_args)
    return RunConfig(env_factory, algorithm, model_config.train_config, total_time_steps)
