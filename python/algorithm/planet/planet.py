import logging
import time
from math import inf
from pathlib import Path
from typing import Optional

import gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

try:
    from torchvision.utils import make_grid
except ImportError:
    logging.warning("Failed to import torchvision needed for PlaNet.")

from algorithm.chunked_summary_writer import ChunkedSummaryWriter
from . import ExperienceReplay
from .models import RewardModel
from .models import TransitionModel
from .models.encoder import SymbolicEncoder, VisualEncoder
from .models.observation_model import SymbolicObservationModel, VisualObservationModel
from . import MPCPlanner
from .utils import bottle


class Planet:
    """
    An implementation of the model-based reinforcement learning algorithm PlaNet
    (http://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).
    The implementation was adapted from https://github.com/Kaixhin/PlaNet.
    """

    def __init__(self, env: gym.Env, symbolic_env: bool, activation_function: str = "relu",
                 learning_rate: float = 1e-3, learning_rate_schedule: int = 0, batch_size: int = 50,
                 embedding_size: int = 1024, hidden_size: int = 200, belief_size: int = 200, state_size: int = 30,
                 adam_epsilon: float = 1e-4,
                 planning_horizon: int = 12, optimization_iters: int = 10, candidates: int = 1000,
                 top_candidates: int = 100, overshooting_distance: int = 50, action_repeat: int = 2, bit_depth: int = 5,
                 free_nats: float = 3,
                 seed: Optional[int] = None, disable_cuda: bool = False):
        """

        :param env:                     the gym environment to train on
        :param symbolic_env:            whether the gym environment is a symbolic environment (i.e. the observation is a
                                        vector of state variables instead of an image)
        :param activation_function:     the activation function to use in all networks
        :param learning_rate:           the learning rate used to train all networks
        :param learning_rate_schedule:
        :param batch_size:              the number of chunks (consecutive (observation, action, reward, not done)
                                        tuples) that are drawn from the experience replay to calculate the loss in every
                                        training iteration (name in the paper: B)
        :param embedding_size:
        :param hidden_size:
        :param belief_size:
        :param state_size:
        :param adam_epsilon:            the epsilon parameter of the adam optimizer
        :param planning_horizon:        the number of steps the planner plans into the future (name in the paper: H)
        :param optimization_iters:      the number of iterations that the planner does each step to find a good action
                                        sequence (name in the paper: I)
        :param candidates:              the number of candidate action sequences that the planner evaluates each
                                        iteration (name in the paper: J)
        :param top_candidates:          the number of best candidate action sequences that the planner uses to determine
                                        the distribution of action sequences for the next optimization iteration (name
                                        in the paper: K)
        :param overshooting_distance:
        :param action_repeat:           the number of times each action is repeated (to keep the sequence length short)
                                        --> i.e. an action repeat of 2 means that the agent takes two steps in the
                                            environment with the same action for each controller update
        :param bit_depth:
        :param free_nats:
        :param seed:
        :param disable_cuda:
        """
        self.experience_replay = None
        self.env = env

        self.symbolic_env = symbolic_env
        # TODO: Move learning_rate to train
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.belief_size = belief_size
        self.state_size = state_size
        self.overshooting_distance = overshooting_distance
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

        # TODO: Does not support different limits for different actions
        self.max_action = float(max(self.env.action_space.high))
        self.min_action = float(min(self.env.action_space.low))

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if torch.cuda.is_available() and not disable_cuda:
            self.device = torch.device("cuda")
            if seed is not None:
                torch.cuda.manual_seed(seed)
        else:
            if not disable_cuda:
                logging.warning("CUDA is enabled but not available. Defaulting to training on CPU instead.")
            self.device = torch.device("cpu")

        assert len(self.env.action_space.shape) == 1, "Only 1D action spaces are supported at the moment"
        assert len(self.env.observation_space.shape), "Only 1D observation spaces are supported at the moment"
        self.action_size = self.env.action_space.shape[0]
        self.observation_size = self.env.observation_space.shape[0]
        # Initialise model parameters randomly
        self.transition_model = TransitionModel(belief_size, state_size, self.action_size, hidden_size,
                                                embedding_size, activation_function).to(device=self.device)
        self.reward_model = RewardModel(belief_size, state_size, hidden_size, activation_function) \
            .to(device=self.device)
        if symbolic_env:
            self.observation_model = SymbolicObservationModel(self.observation_size, belief_size,
                                                              state_size, embedding_size, activation_function)
            self.encoder = SymbolicEncoder(self.observation_size, embedding_size, activation_function)
        else:
            self.encoder = VisualEncoder(embedding_size, activation_function)
            self.observation_model = VisualObservationModel(belief_size, state_size, embedding_size,
                                                            activation_function)
        self.observation_model = self.observation_model.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(
            self.reward_model.parameters()) + list(self.encoder.parameters())
        self.optimizer = optim.Adam(self.param_list, lr=0 if learning_rate_schedule != 0 else learning_rate,
                                    eps=adam_epsilon)
        self.planner = MPCPlanner(self.action_size, planning_horizon, optimization_iters, candidates,
                                  top_candidates, self.transition_model, self.reward_model, self.min_action,
                                  self.max_action)
        self.global_prior = Normal(torch.zeros(batch_size, state_size, device=self.device),
                                   torch.ones(batch_size, state_size,
                                              device=self.device))  # Global prior N(0, I)
        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1,), free_nats, dtype=torch.float32, device=self.device)

    # TODO: Limits can probably be removed (depend on the environment)
    def update_belief_and_act(self, belief: torch.Tensor, posterior_state: torch.Tensor, action: torch.Tensor,
                              observation: torch.Tensor, min_action: float = -inf, max_action: float = inf,
                              explore: bool = False, action_noise: float = 0.3):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        belief, _, _, _, posterior_state, _, _ = \
            self.transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                  self.encoder(observation).unsqueeze(
                                      dim=0))  # Action and observation need extra time dimension
        # Remove time dimension from belief/state
        belief = belief.squeeze(dim=0)
        posterior_state = posterior_state.squeeze(dim=0)
        action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
        # TODO: explore is redundant
        if explore:
            # Add exploration noise ε ~ p(ε) to the action
            action = action + action_noise * torch.randn_like(action)
        action.clamp_(min=min_action, max=max_action)  # Clip action range
        # Perform environment step (action repeats handled internally)
        next_observation, reward, done, _ = self.env.step(action[0].cpu().numpy())
        next_observation = torch.tensor(next_observation).float()
        # TODO: Fix this with the occurrence above
        if next_observation.ndim == 1:
            next_observation = next_observation.unsqueeze(dim=0)
        return belief, posterior_state, action, next_observation, reward, done

    def test(self, episodes: int = 10, render: bool = False):
        # Set models to eval mode
        self.transition_model.eval()
        self.reward_model.eval()
        self.encoder.eval()
        with torch.no_grad():
            total_reward = 0
            for _ in range(episodes):
                observation = self.env.reset()
                belief = torch.zeros(1, self.belief_size, device=self.device)
                posterior_state = torch.zeros(1, self.state_size, device=self.device)
                action = torch.zeros(1, self.action_size, device=self.device)
                done = False
                while not done:
                    belief, posterior_state, action, observation, reward, done = \
                        self.update_belief_and_act(belief, posterior_state, action,
                                                   observation.to(device=self.device), self.min_action, self.max_action)
                    total_reward += reward
                    if render:
                        self.env.render()
                    if done:
                        break
        print("Average Reward:", total_reward / episodes)
        self.env.close()
        quit()

    def train(self, total_time_steps: int = 100000, chunk_size: int = 50, render: bool = False,
              experience_size: int = 1000000, nr_seed_episodes: int = 5, action_noise: float = 0.3,
              collect_interval: int = 100, disable_testing: bool = False, test_interval: int = 25,
              test_episodes: int = 10, global_kl_beta: float = 0.0, overshooting_kl_beta: float = 0.0,
              overshooting_reward_scale: float = 0.0, grad_clip_norm: float = 1000,
              summary_writer: Optional[SummaryWriter] = None, record_chunk_size: int = 1000):
        assert chunk_size <= self.overshooting_distance, "Overshooting distance cannot be greater than chunk size"

        writer = ChunkedSummaryWriter(summary_writer, record_chunk_size)

        # Initialise training environment and experience replay memory
        if self.experience_replay is not None:
            current_step = self.experience_replay.steps
            current_episode = self.experience_replay.episodes
        else:
            current_step = 0
            current_episode = 0
            self.experience_replay = ExperienceReplay(experience_size, self.symbolic_env, self.observation_size,
                                                      self.action_size, self.bit_depth, self.device)
            # Initialise dataset experience_replay with S random seed episodes
            for s in range(1, nr_seed_episodes + 1):
                observation = self.env.reset()
                done = False
                while not done:
                    action = self.env.action_space.sample()
                    next_observation, reward, done, _ = self.env.step(action)
                    self.experience_replay.append(observation, action, reward, done)
                    observation = next_observation
                    current_step += 1
                current_episode += 1

        while current_step < total_time_steps:
            # Model fitting
            losses = []
            for s in range(collect_interval):
                # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset
                # (including terminal flags)
                observations, actions, rewards, nonterminals = self.experience_replay.sample(
                    self.batch_size, chunk_size)  # Transitions start at time t = 0
                # Create initial belief and state for time t = 0
                init_belief = torch.zeros(self.batch_size, self.belief_size, device=self.device)
                init_state = torch.zeros(self.batch_size, self.state_size, device=self.device)
                # Update belief/state using posterior from previous belief/state, previous action and current
                # observation (over entire sequence at once)
                beliefs, prior_states, prior_means, prior_stds, posterior_states, posterior_means, posterior_stds \
                    = self.transition_model(init_state, actions[:-1], init_belief,
                                            bottle(self.encoder, (observations[1:],)), nonterminals[:-1])
                # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent
                # overshooting); sum over final dims, average over batch and time (original implementation, though paper
                # seems to miss 1/T scaling?)
                observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)),
                                              observations[1:], reduction="none").sum(dim=2 if self.symbolic_env
                else (2, 3, 4)).mean(dim=(0, 1))
                reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1],
                                         reduction="none").mean(dim=(0, 1))
                # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
                kl_loss = torch.max(
                    kl_divergence(Normal(posterior_means, posterior_stds), Normal(prior_means, prior_stds))
                        .sum(dim=2), self.free_nats).mean(dim=(0, 1))
                if global_kl_beta != 0:
                    kl_loss += global_kl_beta * kl_divergence(Normal(posterior_means, posterior_stds),
                                                              self.global_prior).sum(dim=2).mean(dim=(0, 1))
                # Calculate latent overshooting objective for t > 0
                if overshooting_kl_beta != 0:
                    overshooting_vars = []  # Collect variables for overshooting to process in batch
                    for t in range(1, chunk_size - 1):
                        d = min(t + self.overshooting_distance, chunk_size - 1)  # Overshooting distance
                        t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                        # Calculate sequence padding so overshooting terms can be calculated in one batch
                        seq_pad = [0, 0, 0, 0, 0, t - d + self.overshooting_distance]
                        # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                        overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad),
                                                  F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_],
                                                  F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad),
                                                  F.pad(posterior_stds[t_ + 1:d_ + 1].detach(), seq_pad, value=1),
                                                  F.pad(torch.ones(d - t, self.batch_size, self.state_size,
                                                                   device=self.device),
                                                        seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
                    overshooting_vars = tuple(zip(*overshooting_vars))
                    # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
                    beliefs, prior_states, prior_means, prior_stds = self.transition_model(
                        torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1),
                        torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
                    seq_mask = torch.cat(overshooting_vars[7], dim=1)
                    # Calculate overshooting KL loss with sequence mask
                    kl_loss += (1 / self.overshooting_distance) * overshooting_kl_beta * torch.max((kl_divergence(
                        Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                        Normal(prior_means, prior_stds)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1)) * (
                                       chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
                    # Calculate overshooting reward prediction loss with sequence mask
                    if overshooting_reward_scale != 0:
                        reward_loss += (1 / self.overshooting_distance) * overshooting_reward_scale * F.mse_loss(
                            bottle(self.reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0],
                            torch.cat(overshooting_vars[2], dim=1), reduction="none").mean(dim=(0, 1)) * (
                                               chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)

                # TODO: Change learning rate schedule to work with general functions
                # Apply linearly ramping learning rate schedule
                if self.learning_rate_schedule != 0:
                    for group in self.optimizer.param_groups:
                        group["lr"] = min(group["lr"] + self.learning_rate / self.learning_rate_schedule,
                                          self.learning_rate)
                # Update model parameters
                self.optimizer.zero_grad()
                (observation_loss + reward_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.param_list, grad_clip_norm, norm_type=2)
                self.optimizer.step()
                # Store (0) observation loss (1) reward loss (2) KL loss
                losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])

            # Update metrics
            losses = tuple(zip(*losses))
            if writer is not None:
                now = time.time()
                writer.record("train/observation_loss", losses[0], current_step, now)
                writer.record("train/reward_loss", losses[1], current_step, now)
                writer.record("kl_loss", losses[2], current_step, now)

            # Data collection
            with torch.no_grad():
                observation = torch.tensor(self.env.reset()).float()
                if observation.ndim == 1:
                    observation = observation.unsqueeze(dim=0)
                total_reward = 0
                belief = torch.zeros(1, self.belief_size, device=self.device)
                posterior_state = torch.zeros(1, self.state_size, device=self.device)
                action = torch.zeros(1, self.action_size, device=self.device)
                done = False
                while not done:
                    belief, posterior_state, action, next_observation, reward, done = \
                        self.update_belief_and_act(belief, posterior_state, action,
                                                   observation.to(device=self.device), self.min_action, self.max_action,
                                                   explore=True, action_noise=action_noise)
                    self.experience_replay.append(observation, action.cpu(), reward, done)
                    total_reward += reward
                    observation = next_observation
                    if render:
                        self.env.render()

            # Test model
            if not disable_testing and current_episode % test_interval == 0:
                # Set models to eval mode
                self.transition_model.eval()
                self.observation_model.eval()
                self.reward_model.eval()
                self.encoder.eval()

                rewards_per_episode = []
                for _ in range(test_episodes):
                    total_reward = 0
                    with torch.no_grad():
                        observation = torch.tensor(self.env.reset()).float()
                        # TODO: Find out why everything is a (1, N) tensor in the first place
                        if observation.ndim == 1:
                            observation = observation.unsqueeze(dim=0)
                        video_frames = []
                        belief = torch.zeros(1, self.belief_size, device=self.device)
                        posterior_state = torch.zeros(1, self.state_size, device=self.device)
                        action = torch.zeros(1, self.action_size, device=self.device)
                        done = False
                        while not done:
                            belief, posterior_state, action, next_observation, reward, done = \
                                self.update_belief_and_act(belief, posterior_state, action,
                                                           observation.to(device=self.device), self.min_action,
                                                           self.max_action, explore=False)
                            total_reward += reward
                            if not self.symbolic_env:  # Collect real vs. predicted frames for video
                                video_frames.append(make_grid(
                                    torch.cat([observation, self.observation_model(belief, posterior_state).cpu()],
                                              dim=3)
                                    + 0.5, nrow=5))  # Decentre
                            observation = next_observation
                    rewards_per_episode.append(total_reward)

                if not self.symbolic_env:
                    # TODO: this is likely to expensive
                    summary_writer.add_video("train/video", torch.tensor(video_frames), current_step,
                                             walltime=time.time())

                # Set models to train mode
                self.transition_model.train()
                self.observation_model.train()
                self.reward_model.train()
                self.encoder.train()

    def save_checkpoint(self, path: Path, store_replay_buffer: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "transition_model": self.transition_model.state_dict(),
            "observation_model": self.observation_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "encoder": self.encoder.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        if store_replay_buffer:
            # Warning: will fail with MemoryError with large memory sizes
            save_dict["replay_buffer"] = self.experience_replay
        torch.save(save_dict, path)

    def load_checkpoint(self, path: Path, load_replay_buffer: bool = True):
        model_dicts = torch.load(path)
        self.transition_model.load_state_dict(model_dicts["transition_model"])
        self.observation_model.load_state_dict(model_dicts["observation_model"])
        self.reward_model.load_state_dict(model_dicts["reward_model"])
        self.encoder.load_state_dict(model_dicts["encoder"])
        self.optimizer.load_state_dict(model_dicts["optimizer"])
        if "replay_buffer" in model_dicts and load_replay_buffer:
            self.experience_replay = model_dicts["replay_buffer"]
