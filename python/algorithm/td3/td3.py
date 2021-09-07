import copy
import logging
import time
from pathlib import Path
from typing import Union, Dict, Optional, Callable, Sequence

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch

from algorithm.algorithm import Algorithm
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer


class TD3(Algorithm):
    def __init__(self, actor: Actor, critic: Critic, max_action: float, discount: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.03, exploration_noise=0.01, policy_delay: int = 2,
                 training_starts_at: int = 4096, train_interval: int = 1024, gradient_steps: int = 1024,
                 batch_size: int = 256, learning_rate: float = 1e-4, replay_buffer_size: int = int(1e6),
                 device: Optional[Union[torch.device, str]] = None):
        super(TD3, self).__init__()

        self.__actor_orig = actor
        self.__critic_orig = critic

        self.__actor = None
        self.__actor_target = None
        self.__actor_optimizer = None

        self.__critic = None
        self.__critic_target = None
        self.__critic_optimizer: Optional[torch.optim.optimizer.Optimizer] = None

        self.__max_action = max_action
        self.__discount = discount
        self.__tau = tau
        self.__policy_noise = policy_noise
        self.__noise_clip = noise_clip
        self.__policy_delay = policy_delay
        self.__training_starts_at = training_starts_at
        self.__train_interval = train_interval
        self.__gradient_steps = gradient_steps
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate if isinstance(learning_rate, Callable) else lambda i: learning_rate
        self.__replay_buffer_size = replay_buffer_size
        self.__exploration_noise = exploration_noise

        self.__total_critic_steps = 0

        self.__env: Optional[VecEnv] = None

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.__logger = logging.getLogger("training")

    def _train(self, log_path: Path, total_timesteps: int, train_parameters: Dict, summary_writer: SummaryWriter):
        self.__logger.info("Training on device \"{}\".".format(self.__device))
        replay_buffer = ReplayBuffer(self.__env.observation_space.shape[0], self.__env.action_space.shape[0],
                                     max_size=self.__replay_buffer_size, training_device=self.__device)

        obs_np = self.__env.reset()
        obs = torch.from_numpy(obs_np).float().to(self.__device)

        last_training = 0

        for t in range(0, total_timesteps, self.__env.num_envs):

            # Select action randomly or according to policy
            if t < self.__training_starts_at:
                action = torch.from_numpy(
                    np.array([self.__env.action_space.sample() for _ in range(self.__env.num_envs)])).to(self.__device)
            else:
                action = self.__select_action(obs, add_noise=True).detach()

            # Perform action
            # Env must ensure that done flag is set in the final step!
            new_obs_np, reward_np, done_np, _ = self.__env.step(action.cpu().numpy())
            reward = torch.from_numpy(reward_np)
            done = torch.from_numpy(done_np)
            new_obs = torch.from_numpy(new_obs_np).float()

            # Store data in replay buffer
            for o, a, ns, r, d in zip(obs, action, new_obs, reward, done):
                replay_buffer.add(o, a, ns, r, d)

            obs = new_obs.to(self.__device)

            # Train agent after collecting sufficient data
            if t >= self.__training_starts_at and t - last_training >= self.__train_interval:
                current_lr = self.__learning_rate(1 - t / total_timesteps)
                for optimizer in [self.__actor_optimizer, self.__critic_optimizer]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                losses = [
                    self.__train_single_step(replay_buffer, self.__batch_size) for _ in range(self.__gradient_steps)]
                critic_losses, actor_losses = zip(*losses)
                actor_losses = [l for l in actor_losses if l is not None]

                now = time.time()
                summary_writer.add_scalar("train/actor_loss", self.__safe_mean(critic_losses), t, now)
                summary_writer.add_scalar("train/critic_loss", self.__safe_mean(actor_losses), t, now)
                summary_writer.add_scalar("train/learning_rate", current_lr, t, now)

                last_training = t

    @staticmethod
    def __safe_mean(values: Sequence[float]):
        if len(values) == 0:
            return 0
        return np.mean(values)

    @torch.no_grad()
    def __select_action(self, obs: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        action_mean: torch.Tensor = self.__actor(obs)
        if add_noise:
            noise = torch.normal(
                0, self.__max_action * self.__exploration_noise, size=action_mean.size(), device=action_mean.device)
            action = (action_mean + noise).clamp(-self.__max_action, self.__max_action)
        else:
            action = action_mean
        return action

    def predict(self, obs: np.ndarray, evaluation_mode: bool = False) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.FloatTensor(obs).to(device)
        action = self.__select_action(state, not evaluation_mode)
        return action.detach().cpu().numpy()

    def save_checkpoint(self, path: Path):
        torch.save({
            "critic": self.__critic.state_dict(),
            "critic_optimizer": self.__critic_optimizer.state_dict(),
            "actor": self.__actor.state_dict(),
            "actor_optimizer": self.__actor_optimizer.state_dict()
        }, path)

    def __update_device(self):
        self.__actor = self.__actor.to(self.__device)
        self.__actor_target = self.__actor_target.to(self.__device)
        self.__critic = self.__critic.to(self.__device)
        self.__critic_target = self.__critic_target.to(self.__device)

    def _load_checkpoint(self, path: Path, env: Union[gym.Env, VecEnv]):
        self._initialize_new_model(env)

        state = torch.load(path, map_location=self.__device)
        self.__critic.load_state_dict(state["critic"])
        self.__critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.__critic_target = copy.deepcopy(self.__critic)

        self.__actor.load_state_dict(state["actor"])
        self.__actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.__actor_target = copy.deepcopy(self.__actor)

    def _initialize_new_model(self, env: Union[gym.Env, VecEnv]):
        self.__actor = copy.deepcopy(self.__actor_orig)
        self.__actor_target = copy.deepcopy(self.__actor)
        self.__actor.initialize(env.observation_space.shape[0], env.action_space.shape[0])
        self.__actor_target.initialize(env.observation_space.shape[0], env.action_space.shape[0])
        self.__actor_optimizer = torch.optim.Adam(self.__actor.parameters(), lr=3e-4)

        self.__critic = copy.deepcopy(self.__critic_orig)
        self.__critic_target = copy.deepcopy(self.__critic)
        self.__critic.initialize(env.observation_space.shape[0], env.action_space.shape[0])
        self.__critic_target.initialize(env.observation_space.shape[0], env.action_space.shape[0])
        self.__critic_optimizer = torch.optim.Adam(self.__critic.parameters(), lr=3e-4)

        self.__total_critic_steps = 0
        if isinstance(env, gym.Env):
            self.__env = DummyVecEnv([lambda: env])
        else:
            self.__env = env

        self.__update_device()

    def __train_single_step(self, replay_buffer: ReplayBuffer, batch_size: int = 100):
        self.__total_critic_steps += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            action_mean = self.__actor_target(next_state)

            noise = (
                    torch.randn_like(action, device=action_mean.device) * self.__policy_noise
            ).clamp(-self.__noise_clip, self.__noise_clip)

            next_action = (action_mean + noise).clamp(-self.__max_action, self.__max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.__critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.__discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.__critic(state, action)

        # Compute critic loss
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.__critic_optimizer.zero_grad()
        critic_loss.backward()
        self.__critic_optimizer.step()

        # Delayed policy updates
        if self.__total_critic_steps % self.__policy_delay == 0:

            # Compute actor losse
            actor_loss = -self.__critic.q_nets[0](torch.cat([state, self.__actor(state)], dim=1)).mean()

            # Optimize the actor
            self.__actor_optimizer.zero_grad()
            actor_loss.backward()
            self.__actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.__critic.parameters(), self.__critic_target.parameters()):
                target_param.data.copy_(self.__tau * param.data + (1 - self.__tau) * target_param.data)

            for param, target_param in zip(self.__actor.parameters(), self.__actor_target.parameters()):
                target_param.data.copy_(self.__tau * param.data + (1 - self.__tau) * target_param.data)
            return critic_loss.item(), actor_loss.item()
        return critic_loss.item(), None
