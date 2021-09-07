import itertools
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Iterable, TYPE_CHECKING, Optional, TypeVar, Generic, Any

import gym
import numpy as np

from assembly_gym.environment.generic import Environment

if TYPE_CHECKING:
    from task.controllers import Controller
    from task.sensors import Sensor
    from task.rewards.reward import Reward


class ResetFailedException(Exception):
    pass


EnvironmentType = TypeVar("EnvironmentType", bound=Environment)


class BaseTask(gym.Env, ABC, Generic[EnvironmentType]):
    """
    An abstract base class for gym environments that use the robot (real or simulated).
    """

    def __init__(self, controllers: Iterable["Controller[BaseTask]"],
                 sensors: Iterable["Sensor[BaseTask]"], rewards: Iterable["Reward[BaseTask]"],
                 time_step: float, time_limit_steps: Optional[int] = None):
        """
        :param controllers:             A sequence of controller objects that define the actions on the environment
        :param sensors:                 A sequence of sensors objects that provide observations to the agent
        :param rewards:                 A sequence of rewards objects that provide rewards to the agent
        :param time_step:               The time between two controller updates (actions of the agent)
        :param time_limit_steps:        The number of steps until the episode terminates (if no other termination
                                        criterion is reached)
        """
        self.__cumulative_rewards_dict: Dict["Reward", float] = {}

        self.__sensors: Iterable[Sensor] = sensors
        self.__rewards: Iterable["Reward"] = rewards

        self.__time_limit_steps: int = time_limit_steps
        self.__current_step: Optional[int] = None    # needed to enforce time_limit_steps

        self.__env: Optional[EnvironmentType] = None

        self.__logger: logging.Logger = logging.getLogger("task")
        self.__time_step: float = time_step

        self.__controllers = tuple(controllers)

        self.__auto_restart: Optional[bool] = None

    def initialize(self, env: EnvironmentType, auto_restart: bool = False) -> None:
        """
        Initialize the gym environment. Needs to be called before reset() is called for the first time.

        :param env:             the environment object that should be used to interact with the (simulated or real)
                                scene
        :param auto_restart:    whether the environment should be restarted if an error occurred during reset() or
                                step()
        """
        self.__env = env
        self.__auto_restart = auto_restart
        self.__env.initialize(self.__time_step)

        self._initialize()

        obs_space_dict = {}
        for sensor in self.__sensors:
            new_obs_spaces = sensor.initialize(self)
            assert len(set(new_obs_spaces.keys()).intersection(set(obs_space_dict.keys()))) == 0, \
                "Duplicate observation name"
            obs_space_dict.update(new_obs_spaces)

        self.observation_space = gym.spaces.Dict(obs_space_dict)

        action_space_dict = {}
        for controller in self.__controllers:
            new_action_space = controller.initialize(self)
            assert controller.name not in action_space_dict, "Duplicate action name"
            action_space_dict[controller.name] = new_action_space
        self.action_space = gym.spaces.Dict(action_space_dict)

        for reward in self.__rewards:
            reward.initialize(self)

    def _initialize(self) -> None:
        pass

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Executes one action of the agent and calculates the resulting observation, reward, and done signal.

        :param action:          the action that should be executed
        :return:                the feedback from the environment as a (observation, reward, done, info) tuple
        """
        try:
            self.__current_step += 1

            for controller in self.__controllers:
                controller.actuate(action[controller.name])

            self.__env.step()

            done, task_info = self._step_task()
            if self.__time_limit_steps is not None:
                done = done or self.__time_limit_steps <= self.__current_step

            current_rewards = {reward: reward.calculate_reward(done) for reward in self.__rewards}
            reward = sum(current_rewards.values())

            reward_info = {reward.name: value for reward, value in current_rewards.items()}
            info = {
                "reward": reward_info,
                **task_info
            }

            obs = {k: v for k, v in itertools.chain(*(sensor.observe().items() for sensor in self.__sensors))}

            for reward_obj, value in current_rewards.items():
                self.__cumulative_rewards_dict[reward_obj] += value

            if done:
                self.environment.terminate_episode()

                cumulative_reward = sum(self.__cumulative_rewards_dict.values())
                current_reward = sum(current_rewards.values())
                self.__logger.info("")
                self.__logger.info("Cumulative reward: {: .6f} ({})".format(
                    cumulative_reward,
                    "  ".join([
                        "{}: {: .6f} [{: 8.4f}%]".format(r.name_abbreviation, v, v / cumulative_reward * 100)
                        for r, v in self.__cumulative_rewards_dict.items()])))
                self.__logger.info("Final reward:      {: .6f} ({})".format(
                    current_reward,
                    "  ".join([
                        "{}: {: .6f} [{: 8.4f}%]".format(r.name_abbreviation, v, v / current_reward * 100)
                        for r, v in current_rewards.items()])))
                for h in self.__logger.handlers:
                    h.flush()
            return obs, reward, done, info
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            if self.__auto_restart:
                self._restart_env()
            else:
                raise
            # Now obtain some valid observation
            self.reset()
            obs, reward, _, task_info = self.step(action)
            return obs, reward, True, task_info

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the gym environment. Must be called at the beginning of each episode.

        :return:        the initial observation for the episode
        """
        try:
            for i in range(10):
                self.__current_step = 0
                self.__env.reset_scene()

                for controller in self.__controllers:
                    controller.initialize(self)

                try:
                    self._reset_task()
                    for reward in self.__rewards:
                        reward.reset()

                    self.__cumulative_rewards_dict = {reward: 0 for reward in self.__rewards}

                    return {k: v for k, v in itertools.chain(*(sensor.reset().items() for sensor in self.__sensors))}
                except ResetFailedException as e:
                    self.__logger.warning("Failed to reset the task. {} Retrying...".format(str(e)))
            raise ResetFailedException()
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            if self.__auto_restart:
                self._restart_env()
            else:
                raise
            return self.reset()

    def close(self) -> None:
        """
        Shuts the environment down. Should be called if the environment is not used anymore.
        """
        self.__env.shutdown()

    @abstractmethod
    def _restart_env(self) -> None:
        """
        Restarts the environment in case an error occurred.
        """
        pass

    @abstractmethod
    def _step_task(self) -> Tuple[bool, Dict]:
        """
        Execute the task-specific components of gym.step().

        :return:                        a (done, info) tuple, where info contains the task-specific
                                        components of the infos, respectively
        """
        pass

    @abstractmethod
    def _reset_task(self) -> None:
        """
        Execute the task-specific components of gym.reset().
        """
        pass

    @property
    def controllers(self) -> Tuple["Controller", ...]:
        return self.__controllers

    @property
    def sensors(self) -> Iterable["Sensor"]:
        return self.__sensors

    @property
    def rewards(self) -> Iterable["Reward"]:
        return self.__rewards

    @property
    def environment(self) -> EnvironmentType:
        return self.__env

    @property
    def time_step(self) -> float:
        return self.__time_step

    @property
    def time_limit_steps(self) -> int:
        return self.__time_limit_steps

    @property
    def current_step(self) -> int:
        return self.__current_step

    @property
    def logger(self) -> logging.Logger:
        return self.__logger
