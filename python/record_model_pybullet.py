import argparse
import json
import logging
import shutil
import sys
from logging import StreamHandler
from pathlib import Path

import math
from typing import Callable

import pybullet

from PIL import Image

from algorithm.run_config import load_config
from evaluate_test_metrics import load_test_config
from task import FlattenWrapper, SimulationRobotEnv
from assembly_gym.environment.pybullet import PybulletEnvironment


def create_record_callback(log_path: Path) -> Callable[[PybulletEnvironment], None]:
    def callback(sim_env: PybulletEnvironment, _log_path=log_path):
        if episode_started:
            global step_no
            width, height, rgbImg, depthImg, segImg = sim_env.physics_client.call(
                pybullet.getCameraImage, width=1000, height=1000, viewMatrix=viewMatrix, lightDirection=[20, 20, 20],
                projectionMatrix=projectionMatrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

            image_name = "{:0{}d}.png".format(step_no, int(
                math.ceil(math.log10(env.time_limit_steps * sim_env.substeps_per_step))))
            image_path = _log_path / image_name

            image = Image.fromarray(rgbImg)
            with image_path.open("wb") as f:
                image.save(f, format="PNG")

            step_no += 1

    return callback


def reset_callback(obs):
    global episode_started
    episode_started = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads a pre-trained model and creates image sequences of its executions.")
    parser.add_argument("log_path", type=str, help="The path where the logs of the training are stored")
    parser.add_argument("output_path", type=str, help="The output path for the resulting image sequences.")
    parser.add_argument("-t", "--test-config", type=str, default=None, help="Path of the test config.")
    parser.add_argument("-n", type=int, default=10, help="Number of runs to record.")

    args = parser.parse_args()
    log_path = Path(args.log_path).resolve()

    env_logger = logging.getLogger("env")
    env_logger.addHandler(StreamHandler(sys.stdout))
    env_logger.setLevel(logging.DEBUG)

    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    run_config = load_config(log_path)

    sim_env = PybulletEnvironment()

    if args.test_config is not None:
        env_factory_inner = load_test_config(Path(args.test_config))
        env = env_factory_inner()
    else:
        if isinstance(run_config.env, SimulationRobotEnv):
            env = run_config.env
        else:
            env = run_config.env()

    env.headless = False
    env.initialize(env=sim_env, auto_restart=False)
    env_wrapped = FlattenWrapper(env)

    run_config.algorithm.load_checkpoint(log_path / "model.pkl", lambda: env)

    viewMatrix = sim_env.physics_client.call(
        pybullet.computeViewMatrix, cameraEyePosition=[0, 2, 0.6], cameraTargetPosition=[0, 0, 0.3],
        cameraUpVector=[0, 0, 1])
    projectionMatrix = sim_env.physics_client.call(
        pybullet.computeProjectionMatrixFOV, fov=60.0, aspect=1.0, nearVal=0.1, farVal=100)

    step_no = 0

    sim_env.virtual_substep_mode = True

    with (output_path / "metadata.json").open("w") as f:
        json.dump({
            "frame_interval": sim_env.time_step,
            "model": str(log_path)
        }, f, indent=True)

    try:
        for run_no in range(args.n):
            episode_started = False
            log_path = output_path / "{:0{}d}".format(run_no, int(math.ceil(math.log10(args.n))))
            log_path.mkdir()
            sim_env.on_step_event.handlers = [create_record_callback(log_path)]
            total_reward = run_config.algorithm.visualize_episode(
                env_wrapped, time_per_step=env.time_step, reset_callback=reset_callback)
            step_no = 0
    except KeyboardInterrupt:
        run_config.algorithm.shutdown()
