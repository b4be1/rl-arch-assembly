import time
from typing import Optional, Callable, Sequence

import numpy as np


def linear_schedule(initial_value: float, value_after_n_steps: float, n: float, sleep_time: float = 0,
                    min_val: Optional[float] = None, max_val: Optional[float] = None) -> Callable[[float], float]:
    def func(steps: float, _initial_value=initial_value, _value_after_n_steps=value_after_n_steps, _n=n,
             _sleep_time=sleep_time, _min_val=min_val, _max_val=max_val) -> float:
        if steps < _sleep_time:
            val = _initial_value
        else:
            val = _initial_value + (steps - _sleep_time) / (_n - _sleep_time) * (_value_after_n_steps - _initial_value)

        if _min_val is not None:
            val = max(_min_val, val)
        elif _max_val is not None:
            val = min(_max_val, val)

        return val

    return func


def step_function(steps: Sequence[float], step_values: Sequence[float]) -> Callable[[float], float]:
    steps_np = np.array(steps)
    sort_indices = np.argsort(steps)
    values = np.array(step_values)

    def func(x: float, _steps_np=steps_np[sort_indices], _values=values[sort_indices]) -> float:
        if x <= _steps_np[0]:
            return _values[0]
        return _values[np.searchsorted(_steps_np, x, side="right") - 1]

    return func


def exponential_wrapper(inner_schedule: Callable[[float], float], base: float) -> Callable[[float], float]:
    def func(x: float, _inner_schedule=inner_schedule, _base=base) -> float:
        inner_val = _inner_schedule(x)
        return _base ** inner_val

    return func


def convert_to_steps_wrapper(inner_schedule: Callable[[int], float], total_step_count: int) -> Callable[[float], float]:
    def func(remaining_portion: float, _inner_schedule=inner_schedule, _total_step_count=total_step_count) -> float:
        steps_done = int(round(_total_step_count * (1 - remaining_portion)))
        return _inner_schedule(steps_done)

    return func


def convert_to_done_portion_wrapper(inner_schedule: Callable[[float], float], total_step_count: int) \
        -> Callable[[float], float]:
    def func(remaining_portion: float, _inner_schedule=inner_schedule, _total_step_count=total_step_count) -> float:
        return _inner_schedule(1.0 - remaining_portion)

    return func


class TimePortionWrapper:
    def __init__(self, inner_schedule: Callable[[float], float], total_training_time: float):
        self.__total_training_time = total_training_time
        self.__start_time = None
        self.__inner_schedule = inner_schedule

    def __call__(self, *args, **kwargs):
        if self.__start_time is None:
            portion = 0.0
            self.__start_time = time.time()
        else:
            portion = max(min((time.time() - self.__start_time) / self.__total_training_time, 1.0), 0.0)
        return self.__inner_schedule(portion)
