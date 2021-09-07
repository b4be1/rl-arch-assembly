import argparse
import logging
import sys
from logging import StreamHandler
from pathlib import Path

import gym

from algorithm.run_config import load_config
from evaluate_test_metrics import create_test_config_factory
from task.wrappers import FlattenWrapper, FlattenWrapperGoalEnv, GoalEnvWrapper
from run_training import create_env_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loads and executes a pre-trained model.")
    parser.add_argument("log_path", type=str, help="The path where the logs of the training are stored")
    parser.add_argument("-e", "--environment", type=str, choices=("pybullet", "real"), default="pybullet")
    parser.add_argument("-t", "--test-config", type=str, default=None, help="Path of the test config.")
    parser.add_argument(
        "-r", "--real-time-factor", type=float, default=1,
        help="Speedup factor of simulation to real time. -1 will run the simulator as fast as possible.")
    parser.add_argument("-m", "--model-name", type=str, help="Name of the model to test.")

    args = parser.parse_args()
    log_path = Path(args.log_path).resolve()

    env_logger = logging.getLogger("env")
    env_logger.addHandler(StreamHandler(sys.stdout))
    env_logger.setLevel(logging.DEBUG)

    run_config = load_config(log_path)

    real_time_factor = args.real_time_factor
    if real_time_factor <= 0:
        real_time_factor = None

    if args.test_config is not None:
        env_factory = create_test_config_factory(
            Path(args.test_config), args.environment, headless=False, auto_restart=False,
            real_time_factor=real_time_factor)
    else:
        env_factory = create_env_factory(run_config, args.environment, headless=False, auto_restart=False,
                                         real_time_factor=real_time_factor)

    env = env_factory()
    if isinstance(env, gym.GoalEnv):
        env_wrapped = FlattenWrapperGoalEnv(env)
    else:
        env_wrapped = FlattenWrapper(env)

    model_name = "model.pkl"
    if args.model_name is not None:
        model_name = Path("models") / args.model_name

    run_config.algorithm.load_checkpoint(log_path / model_name, lambda: env)



    def callback(obs, reward, done, info):
        if done:
            print(info["metrics"])


    try:
        if isinstance(env, GoalEnvWrapper):
            time_step = env.unwrapped.time_step
        else:
            time_step = env.time_step
        while True:
            total_reward = run_config.algorithm.visualize_episode(
                env_wrapped, time_per_step=time_step, step_callback=callback)
    except KeyboardInterrupt:
        run_config.algorithm.shutdown()
