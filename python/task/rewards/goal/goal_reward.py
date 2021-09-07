from abc import abstractmethod
from typing import Generic, TypeVar, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from task import BaseTask

EnvType = TypeVar("EnvType", bound="BaseTask")
GoalType = TypeVar("GoalType")


# TODO: Avoid duplicated code
# TODO: GoalType should probably extracted from the environment
class GoalReward(Generic[EnvType, GoalType]):
    def __init__(self, name: str, intermediate_timestep_reward_scale: float,
                 final_timestep_reward_scale: Optional[float] = None, clip: bool = False):
        self.__name: str = name
        self.__env: Optional[EnvType] = None
        self.__intermediate_timestep_reward_scale: float = intermediate_timestep_reward_scale
        if final_timestep_reward_scale is None:
            self.__final_timestep_reward_scale: float = intermediate_timestep_reward_scale
        else:
            self.__final_timestep_reward_scale: float = final_timestep_reward_scale
        self.__clip = clip
        self.__env: Optional[EnvType] = None
        self.__min_reward: Optional[float] = None

    def initialize(self, env: EnvType) -> None:
        self.__env = env

    def reset(self):
        self.__min_reward = self._get_min_reward_unnormalized()

    # TODO: Check whether passing info directly can be avoided somehow
    def calculate_reward(self, achieved_goal: Dict[str, GoalType], desired_goal: Dict[str, GoalType],
                         goal_rewards_info: Dict[str, Any]) -> float:
        done = goal_rewards_info["done"]
        reward_normalized = self._calculate_reward_unnormalized(achieved_goal, desired_goal, goal_rewards_info) / \
                            abs(self.__min_reward)
        reward_clipped = max(reward_normalized, -1.0) if self.__clip else reward_normalized
        reward_scale = self.__final_timestep_reward_scale if done else self.__intermediate_timestep_reward_scale
        return reward_scale * reward_clipped

    @abstractmethod
    def _calculate_reward_unnormalized(self, achieved_goal: Dict[str, GoalType], desired_goal: Dict[str, GoalType],
                                       goal_rewards_info: Dict[str, Any]) -> float:
        pass

    @abstractmethod
    def _get_min_reward_unnormalized(self):
        pass

    @abstractmethod
    def information_to_store(self) -> Dict[str, Any]:
        pass

    @property
    def name(self) -> str:
        return self.__name

    @property
    def env(self) -> EnvType:
        return self.__env
