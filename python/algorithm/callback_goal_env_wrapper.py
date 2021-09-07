import gym

from algorithm.callback_env_wrapper import CallbackEnvWrapper


class CallbackGoalEnvWrapper(CallbackEnvWrapper, gym.GoalEnv):
    def __init__(self, wrapped_env: gym.GoalEnv):
        super().__init__(wrapped_env)

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert isinstance(self._wrapped_env, gym.GoalEnv)       # To make PyLint shut up
        return self._wrapped_env.compute_reward(achieved_goal, desired_goal, info)
