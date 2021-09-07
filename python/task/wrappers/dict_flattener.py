from enum import Enum, auto
from typing import Dict, Tuple, NamedTuple, Union

import gym
import numpy as np


class SupportedSpaces(Enum):
    Box = auto()
    Discrete = auto()
    MultiDiscrete = auto()


_PackingInfoEntry = NamedTuple("_PackingInfoEntry",
                               (("vec_slice", slice), ("space_type", SupportedSpaces), ("shape", Tuple[int, ...])))
_PackingInfo = NamedTuple("_PackingInfo", (("entries", Dict[str, _PackingInfoEntry]), ("total_size", int)))


class DictFlattener:
    """
    Flattens a gym.spaces.Dict observation to a gym.spaces.Box observation. Allows to convert values from the Dict space
    to the Box space (pack_dict) and back (unpack_dict).
    """
    def __init__(self, dictionary: gym.spaces.Dict):
        self.__flattened_space, self.__packing_info = self._dict_to_box(dictionary)

    @staticmethod
    def _dict_to_box(dictionary: gym.spaces.Dict) -> Tuple[gym.spaces.Box, _PackingInfo]:
        o_c = [
            (n, np.product(b.shape), b.shape, b.low, b.high)
            for n, b in dictionary.spaces.items()
            if isinstance(b, gym.spaces.Box)
        ]

        if len(o_c) > 0:
            names_c, dims_c, shape_c, low_c, high_c = zip(*o_c)
        else:
            names_c = dims_c = shape_c = low_c = high_c = ()

        # Do a one hot encoding here
        o_d = [
            (n, d.n, np.zeros((d.n,)), np.ones((d.n,)))
            for n, d in dictionary.spaces.items()
            if isinstance(d, gym.spaces.discrete.Discrete)
        ]

        if len(o_d) > 0:
            names_d, dims_d, low_d, high_d = zip(*o_d)
        else:
            names_d = dims_d = low_d = high_d = ()

        o_md = [
            (n, np.sum(md.nvec), tuple(md.nvec), np.zeros(np.sum(md.nvec)), np.ones(np.sum(md.nvec)))
            for n, md in dictionary.spaces.items()
            if isinstance(md, gym.spaces.MultiDiscrete)
        ]

        if len(o_md) > 0:
            names_md, dims_md, shape_md, low_md, high_md = zip(*o_md)
        else:
            names_md = dims_md = shape_md = low_md = high_md = ()

        names = names_c + names_d + names_md
        dims = dims_c + dims_d + dims_md
        end_indices = np.cumsum(dims)
        start_indices = np.concatenate(([0], end_indices))[:-1]

        packing_info_entries = {
            n: _PackingInfoEntry(slice(s, e), t, sh)
            for n, s, e, t, sh in
            zip(names, start_indices, end_indices, [SupportedSpaces.Box] * len(names_c)
                + [SupportedSpaces.Discrete] * len(names_d) + [SupportedSpaces.MultiDiscrete] * len(names_md),
                shape_c + (None, ) * len(names_d) + shape_md)
        }

        total_dims = int(np.sum(dims))
        low = np.concatenate(list(map(lambda a: a.reshape((-1,)), low_c + low_d + low_md)))
        high = np.concatenate(list(map(lambda a: a.reshape((-1,)), high_c + high_d + high_md)))

        box = gym.spaces.Box(low, high, shape=(total_dims,))

        return box, _PackingInfo(packing_info_entries, total_dims)

    def pack_dict(self, dictionary: Dict[str, np.ndarray]) -> np.ndarray:
        output = np.zeros((self.__packing_info.total_size,))
        for n, e in self.__packing_info.entries.items():
            if e.space_type is SupportedSpaces.Box:
                output[e.vec_slice] = dictionary[n].reshape((-1,))
            elif e.space_type is SupportedSpaces.Discrete:
                output[e.vec_slice][dictionary[n]] = 1
            elif e.space_type is SupportedSpaces.MultiDiscrete:
                indices_one = dictionary[n] + np.cumsum(np.concatenate((np.zeros(1), e.shape)), dtype=np.int64)[:-1]
                output[e.vec_slice][indices_one] = 1
            else:
                raise RuntimeError("Unknown space type {}".format(e.space_type.name))
        return output

    def unpack_dict(self, vector: np.ndarray) -> Dict[str, np.ndarray]:
        output = {}
        for n, e in self.__packing_info.entries.items():
            if e.space_type is SupportedSpaces.Box:
                value = vector[e.vec_slice].reshape(e.shape)
            elif e.space_type is SupportedSpaces.Discrete:
                value = np.where(vector[e.vec_slice] == 1)[0][0]
            elif e.space_type is SupportedSpaces.MultiDiscrete:
                # TODO: Implement
                raise NotImplementedError("Unpacking MultiDiscrete spaces is currently not implemented")
            else:
                raise RuntimeError("Unknown space type {}".format(e.space_type.name))
            output[n] = value
        return output

    @property
    def flattened_space(self) -> gym.spaces.Box:
        return self.__flattened_space
