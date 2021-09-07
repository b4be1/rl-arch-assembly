from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ChunkedSummaryWriter:
    def __init__(self, summary_writer: SummaryWriter, record_chunk_size: int = 1000):
        self.__summary_writer = summary_writer
        self.__record_chunk_size = record_chunk_size
        self.__record_store = {}

    def record(self, name: str, value: float, step: int, wall_time: float, weight: float = 1.0):
        chunk = step // self.__record_chunk_size
        if name in self.__record_store:
            old_chunk, values, times, weights = self.__record_store[name]
            if old_chunk == chunk:
                values.append(value)
                times.append(wall_time)
                weights.append(weight)
            else:
                chunk_time = np.asscalar(np.max(times))
                self.__summary_writer.add_scalar(
                    name, np.average(values, weights=weights), global_step=old_chunk * self.__record_chunk_size,
                    walltime=chunk_time)
                del self.__record_store[name]
        if name not in self.__record_store:
            self.__record_store[name] = (chunk, [value], [wall_time], [weight])
