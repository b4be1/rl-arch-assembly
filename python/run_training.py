import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence, Optional, Literal, Callable

from algorithm.run_config.run_config import RunConfig
from algorithm.run_config.util import load_config
from assembly_gym.environment.generic import Environment
from task import BaseTask


def _find_next_non_empty_line(lines: Sequence[str], start_index: int) -> Optional[int]:
    for i, l in enumerate(lines[start_index:]):
        if l.strip() != "" and not l.strip().startswith("#"):
            return i + start_index
    return None


def _process_import_statement(import_statement: str, dest_dir: Path) -> str:
    import_statement = import_statement.strip()
    import_statement_split = import_statement.split(" ")
    if import_statement_split[0] in ["import", "from"]:
        _, fc_name, *_ = import_statement.split(" ")
        if import_statement_split[0] == "import":
            assert \
                len(import_statement_split) >= 4 and \
                import_statement_split[3] == "as", \
                "Statements of the form import ... are not supported. Use import ... as ... instead."
    else:
        raise ValueError("Invalid import statement: {}".format(import_statement))
    module_spec = importlib.util.find_spec(fc_name)
    module_path = Path(module_spec.origin)
    _copy_run_config_program(module_path, dest_dir)
    import_statement_split[1] = "run_config." + module_path.stem
    return " ".join(import_statement_split)


def _copy_run_config_program(origin: Path, dest_dir: Path, new_name: Optional[str] = None):
    with origin.open() as f:
        config_program = f.read()
    lines = config_program.split("\n")
    include_line_numbers = [
        _find_next_non_empty_line(lines, i + 1)
        for i, l in enumerate(lines)
        if l.replace(" ", "").startswith("#!copy-include")
    ]
    include_line_numbers = {i for i in include_line_numbers if i is not None}
    new_import_statements = {i: _process_import_statement(lines[i], dest_dir) for i in include_line_numbers}
    for i, stm in new_import_statements.items():
        lines[i] = stm
    if new_name is None:
        dest_name = dest_dir / origin.name
    else:
        dest_name = dest_dir / new_name
    assert not dest_name.exists(), "{} already exists in the file system".format(dest_name)
    with dest_name.open("w") as f:
        f.write("\n".join(lines))


def get_environment(name: Literal["pybullet", "real"], real_time_factor: Optional[float] = None,
                    headless: bool = True) -> Environment:
    if name == "pybullet":
        from assembly_gym.environment.pybullet import PybulletEnvironment
        return PybulletEnvironment(real_time_factor=real_time_factor, headless=headless)

    elif name == "real":
        from assembly_gym.environment.real import RealEnvironment
        return RealEnvironment()

    else:
        raise ValueError("Unsupported environment '{}'".format(name))


def create_env_factory(run_config: RunConfig, environment: Literal["pybullet", "real"], headless: bool = True,
                       auto_restart: bool = False, real_time_factor: Optional[float] = None) \
        -> Callable[[], BaseTask]:
    if isinstance(run_config.env, BaseTask):
        env = get_environment(environment, real_time_factor)
        run_config.env.initialize(env=env, auto_restart=auto_restart)

        def env_factory(_env=run_config.env) -> BaseTask:
            return _env
    else:
        def env_factory(_func=run_config.env, _headless=headless, _auto_restart=auto_restart,
                        _simulator=environment, _real_time_factor=real_time_factor) -> BaseTask:
            env = _func()
            env.initialize(env=get_environment(_simulator, _real_time_factor, _headless), auto_restart=_auto_restart)
            return env
    return env_factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executes a run-configuration")
    parser.add_argument("run_config_path", type=str, help="The run-configuration to execute")
    parser.add_argument("-n", "--name", type=str, default="",
                        help="The name of the run to be added to the log directory")
    parser.add_argument("--train-headless", action="store_true",
                        help="Set this flag to disable rendering the training runs")
    parser.add_argument("--no-auto-restart", action="store_true",
                        help="Disable automatic simulator restarting on crash")
    parser.add_argument("-e", "--environment", type=str, choices=("pybullet", "real"), default="pybullet")
    parser.add_argument("-w", "--num-workers", type=int, default=1,
                        help="Number of workers to use. Is ignored for SAC and TD3.")
    parser.add_argument("-m", "--initial-model", type=str, help="Initial model to use for training.")

    args = parser.parse_args()

    log_dir = Path(__file__).parent.parent / "experiment_logs"

    train_logger = logging.getLogger("training")
    train_logger.setLevel(logging.DEBUG)
    train_logger.addHandler(logging.StreamHandler(sys.stdout))
    env_logger = logging.getLogger("env")
    env_logger.setLevel(logging.DEBUG)
    env_logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().handlers.clear()

    run_config_path = Path(args.run_config_path)

    slurm_job_id = os.getenv("SLURM_JOB_ID")
    experiment_name = "_".join([
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        str(run_config_path.stem),
        args.environment])
    if slurm_job_id is not None:
        experiment_name += "_" + slurm_job_id
    if args.name != "":
        experiment_name += "_" + args.name
    log_path = log_dir / experiment_name
    log_path.mkdir(exist_ok=True, parents=True)

    with (log_path / "settings.json").open("w") as f:
        json.dump(args.__dict__, f)
    package_path = log_path / "run_config"
    package_path.mkdir()
    _copy_run_config_program(run_config_path, package_path, new_name="run_config.py")

    run_config = load_config(log_path)
    env_factory = create_env_factory(run_config, args.environment, args.train_headless, not args.no_auto_restart)

    # link = (log_dir / "latest")
    # link.unlink(missing_ok=True)
    # link.symlink_to(log_path.relative_to(link.parent))

    print("Logging in directory: {}".format(str(log_path)))
    log_path.mkdir(parents=True, exist_ok=True)

    # ===== Create and train the model =====
    run_config.algorithm.num_workers = args.num_workers
    if args.initial_model is not None:
        run_config.algorithm.load_checkpoint(Path(args.initial_model), env_factory)
    else:
        run_config.algorithm.initialize_new_model(env_factory)

    try:
        run_config.algorithm.train(log_path, run_config.total_time_steps, run_config.train_config)
    finally:
        run_config.algorithm.shutdown()
