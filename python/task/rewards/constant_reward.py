from .reward import Reward


class ConstantReward(Reward):
    """
    A constant (negative) reward to punish slow solutions.
    """

    def __init__(self, intermediate_timestep_reward_scale: float = 1e-4):
        """
        :param intermediate_timestep_reward_scale:  scaling factor (applied to the reward at every step in which the gym
                                                    environment does not terminate)
        """
        super().__init__("constant_reward", intermediate_timestep_reward_scale, None, abbreviated_name="const")

    def _reset(self) -> None:
        return

    def _calculate_reward_unnormalized(self) -> float:
        return -1

    def _get_min_reward_unnormalized(self) -> float:
        return -1
