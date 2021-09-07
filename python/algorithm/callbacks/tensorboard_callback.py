import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Sequence, Tuple, List, Optional

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from .callback import Callback
from ..chunked_summary_writer import ChunkedSummaryWriter


class TensorboardCallback(Callback):
    def __init__(self, summary_writer: SummaryWriter, record_chunk_size: int = 1000):
        super(TensorboardCallback, self).__init__()
        self.__writer = ChunkedSummaryWriter(summary_writer, record_chunk_size=record_chunk_size)
        self.__metrics_store = None
        self.__metrics_store_final = None
        self.__rewards_store = None
        self.__previous_time_step = 0
        self.__previous_step_time = None
        self.__episode_steps = None
        self.__training_started = False
        self.__record_store: Optional[Dict[str, Tuple[int, List[float], List[float]]]] = None

    def on_training_start(self):
        self.__training_started = False
        self.__previous_time_step = 0
        self.__previous_step_time = time.time()

    def on_episode_start(self, obs: Tuple[np.ndarray]):
        if not self.__training_started:
            self.__metrics_store = tuple(defaultdict(lambda: []) for _ in range(len(obs)))
            self.__metrics_store_final = tuple({} for _ in range(len(obs)))
            self.__rewards_store = tuple(defaultdict(lambda: []) for _ in range(len(obs)))
            self.__episode_steps = [0 for _ in range(len(obs))]
            self.__record_store = {}
            self.__training_started = True

    def on_step(self, current_time_step: int, action: Tuple[np.ndarray], obs: Tuple[np.ndarray], reward: Tuple[float],
                done: Tuple[float], info: Tuple[Dict[str, Any]]):
        now = time.time()
        for i in range(len(info)):
            self.__episode_steps[i] += 1
            if "metrics" in info[i]:
                for n, v in info[i]["metrics"].items():
                    self.__metrics_store[i][n].append(v)
                    if done[i]:
                        self.__metrics_store_final[i][n] = v
            for n, v in info[i]["reward"].items():
                self.__rewards_store[i][n].append(v)
            if "goal_reward" in info[i]:
                self.__rewards_store[i]["goal_reward"].append(info[i]["goal_reward"])

            if done[i]:
                s = current_time_step
                for n, v in self.__metrics_store[i].items():
                    self.__writer.record("metrics/mean/{}".format(n), np.asscalar(np.mean(v)), s, now)
                for n, v in self.__metrics_store_final[i].items():
                    self.__writer.record("metrics/final/{}".format(n), v, s, now)
                reward_sums = {n: np.sum(v) for n, v in self.__rewards_store[i].items()}
                total_reward = sum(reward_sums.values())
                self.__writer.record("reward/total/sum", np.asscalar(total_reward), s, now)
                for n, v in reward_sums.items():
                    self.__writer.record("reward/total/{}".format(n), v, s, now)
                    relative_reward = 0
                    if total_reward != 0:
                        relative_reward = v / total_reward
                    self.__writer.record("reward/relative/{}".format(n), relative_reward, s, now)
                self.__writer.record("general/episode_length", self.__episode_steps[i], s, now)

                self.__metrics_store[i].clear()
                self.__metrics_store_final[i].clear()
                self.__rewards_store[i].clear()
                self.__episode_steps[i] = 0

        time_diff = now - self.__previous_step_time
        steps_per_second = (current_time_step - self.__previous_time_step) / time_diff
        self.__writer.record("general/steps_per_second", steps_per_second, current_time_step, now, weight=time_diff)

        self.__previous_time_step = current_time_step
        self.__previous_step_time = now
