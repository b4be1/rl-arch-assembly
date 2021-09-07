import argparse
import csv
import importlib.util
import logging
import multiprocessing
import sys
from multiprocessing import Process, Queue
from logging import StreamHandler
from pathlib import Path
from typing import Dict, Callable, Optional, Sequence, Iterable, Literal

import numpy as np

import re

from algorithm.run_config import load_config
from task import BaseTask
from task.wrappers import FlattenWrapper
from run_training import create_env_factory, get_environment

REDUCTIONS = {
    "sum": np.sum,
    "mean": np.mean,
    "final": lambda l: l[-1]
}


def dict_reduce(d: Dict, reduction_func: Callable) -> Dict:
    return {k: dict_reduce(v, reduction_func) if isinstance(v, Dict) else reduction_func(v) for k, v in d.items()}


def append_dict(d: Dict, other_dict: Dict):
    for k, v in other_dict.items():
        if isinstance(v, Dict):
            if k not in d:
                d[k] = {}
            append_dict(d[k], v)
        else:
            if k not in d:
                d[k] = []
            d[k].append(v)


def concatenate_dicts(dicts: Iterable[Dict]) -> Dict:
    first_dict = next(dicts.__iter__())
    output_dict = {}
    for k, v in first_dict.items():
        if isinstance(v, Dict):
            output_dict[k] = concatenate_dicts([d[k] for d in dicts])
        else:
            output_dict[k] = [d[k] for d in dicts]
    return output_dict


def zip_dict(d1: Dict, d2: Dict) -> Dict:
    return {k: zip_dict(v, d2[k]) if isinstance(v, Dict) else list(zip(v, d2[k])) for k, v in d1.items()}


def store_dict(d: Dict, steps: Sequence[int], root_path: Path, suffix: str):
    root_path.mkdir(exist_ok=True, parents=True)
    for k, v in d.items():
        if isinstance(v, Dict):
            store_dict(v, steps, root_path / k, suffix)
        else:
            columns = [steps] + list(zip(*v))
            rows = list(zip(*columns))
            with (root_path / "{}_{}.csv".format(k, suffix)).open("w") as f:
                csv_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(["steps", "mean", "variance"])
                rows_str = [map(str, r) for r in rows]
                csv_writer.writerows(rows_str)


def get_step(model_name: str) -> Optional[int]:
    groups = re.findall("model_([0-9]+)\.pkl", model_name)
    if len(groups) != 0:
        return int(groups[0])
    return None


def worker(task_queue: Queue, result_queue: Queue, log_path: Path, environment: Literal["pybullet", "real"],
           evaluations_per_checkpoint: int, test_config: Optional[Path] = None):
    run_config = load_config(log_path)

    if test_config is not None:
        env_factory = create_test_config_factory(Path(test_config), environment, headless=True, auto_restart=False)
    else:
        env_factory = create_env_factory(run_config, environment, headless=True, auto_restart=False)
    env = env_factory()
    env_wrapped = FlattenWrapper(env)

    logging.basicConfig(level=logging.ERROR)

    try:
        step = 0

        while step is not None:
            model_path, step = task_queue.get()
            if step is not None:
                if step > 0:
                    run_config.algorithm.load_checkpoint(model_path, lambda: env)
                else:
                    run_config.algorithm.initialize_new_model(env_factory)
                info_model = {n: {} for n in REDUCTIONS}
                for _ in range(evaluations_per_checkpoint):
                    done = False
                    total_reward = 0
                    obs = env_wrapped.reset()
                    info_collection = {}
                    while not done:
                        action = run_config.algorithm.predict(obs, evaluation_mode=True)
                        obs, reward, done, info = env_wrapped.step(action)
                        append_dict(info_collection, info)
                        total_reward += reward
                    for n, f in REDUCTIONS.items():
                        append_dict(info_model[n], dict_reduce(info_collection, f))
                result = (
                    step,
                    {
                        n: {
                            "mean": dict_reduce(v, np.mean),
                            "var": dict_reduce(v, np.var)
                        }
                        for n, v in info_model.items()
                    },
                )
                result_queue.put(result)
    finally:
        env.close()


def load_test_config(test_config_path: Path) -> Callable[[], BaseTask]:
    spec = importlib.util.spec_from_file_location("{}.{}".format(
        str(test_config_path.parent.name), str(test_config_path.stem)), str(test_config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(Path(__file__).resolve().parent.parent)


def create_test_config_factory(test_config_path: Path, environment: Literal["pybullet", "real"], headless: bool = True,
                               auto_restart: bool = False, real_time_factor: Optional[float] = None) \
        -> Callable[[], BaseTask]:
    task_factory_inner = load_test_config(test_config_path)

    def task_factory(_func=task_factory_inner, _auto_restart=auto_restart, _environment=environment,
                    _headless=headless, _real_time_factor=real_time_factor) \
            -> BaseTask:
        task: BaseTask = _func()
        env = get_environment(_environment, _real_time_factor, _headless)
        task.initialize(env=env, auto_restart=auto_restart)
        return task

    return task_factory


def __print_output(queue: Queue):
    next_item = queue.get()
    while next_item is not None:
        print(next_item)
        next_item = queue.get()


def main(args):
    multiprocessing.set_start_method("spawn")

    log_path = Path(args.log_path).resolve()

    env_logger = logging.getLogger("env")
    env_logger.addHandler(StreamHandler(sys.stdout))
    env_logger.setLevel(logging.ERROR)

    output_dir = Path(args.output_dir)

    result_queue = Queue()
    task_queue = Queue()

    model_paths = sorted((log_path / "models").iterdir())
    steps = [get_step(m.name) for m in model_paths]
    valid_models_and_steps = [(m, s) for m, s in zip(model_paths, steps) if
                              s is not None and (args.max_step is None or s < args.max_step)]
    valid_steps = [s for m, s in valid_models_and_steps]

    for m in valid_models_and_steps:
        task_queue.put(m)

    print_queue = Queue()
    output_proc = Process(target=__print_output, args=(print_queue,))
    output_proc.start()

    workers = [
        Process(target=worker, args=(
            task_queue, result_queue, log_path, args.environment, args.evaluations_per_checkpoint, args.test_config))
        for _ in range(args.num_workers)]

    print("Using {} workers.".format(len(workers)))

    for p in workers:
        p.start()

    for _ in range(args.num_workers):
        task_queue.put((None, None))

    info_per_step = {}
    while len(info_per_step) < len(valid_steps):
        s, info_dict = result_queue.get()
        info_per_step[s] = info_dict
        print_queue.put("{: {}d}/{} steps complete\r".format(
            len(info_per_step), int(np.ceil(np.log10(len(valid_steps) + 1))), len(valid_steps)))

    print_queue.put("")
    print_queue.put(None)

    for p in workers:
        p.join()

    print("Done. Saving results...")

    for n in REDUCTIONS:
        cat_mean = concatenate_dicts([info_per_step[s][n]["mean"] for s in valid_steps])
        cat_var = concatenate_dicts([info_per_step[s][n]["var"] for s in valid_steps])
        zipped = zip_dict(cat_mean, cat_var)
        store_dict(zipped, valid_steps, output_dir, n)

    print("Results stored in \"{}\"".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads and evaluates a pre-trained model on test data.")
    parser.add_argument("log_path", type=str, help="The path where the logs of the training are stored")
    parser.add_argument("output_dir", type=str, help="Output directory for the CSV files.")
    parser.add_argument("-t", "--test-config", type=str, default=None, help="Path of the test config.")
    parser.add_argument("-n", "--evaluations-per-checkpoint", type=int, default=10,
                        help="Number of evaluations per model checkpoint")
    parser.add_argument("-v", "--visualize", action="store_true", help="Whether to visualize the episodes.")
    parser.add_argument("-e", "--environment", type=str, choices=("pybullet", "real"), default="pybullet")
    parser.add_argument("-w", "--num-workers", type=int, help="Number of worker processes", default=1)
    parser.add_argument("--max-step", type=int, help="Only evaluate to this step.", default=None)

    args = parser.parse_args()

    main(args)
