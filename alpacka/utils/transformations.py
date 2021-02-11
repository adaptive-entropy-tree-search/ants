"""Data transformation utils."""

import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    return scipy.signal.lfilter(
        [1], [1, float(-discount)], x[::-1], axis=0
    )[::-1]


def one_hot_encode(values, value_space_size, dtype=np.float32):
    target_shape = (len(values), value_space_size)

    result = np.zeros(target_shape, dtype=dtype)
    result[np.arange(target_shape[0]), values] = 1

    return result


def map_dict_keys(input_dict, mapper):
    result = {
        mapper(key): value
        for key, value in input_dict.items()
    }
    if len(result.keys()) != len(input_dict.keys()):
        raise ValueError(
            'There are collisions of keys after applying the mapper.'
        )
    return result


def zip_dicts_strict(dict1, dict2):
    if dict1.keys() != dict2.keys():
        raise ValueError(
            'Keys of dicts do not match.\n'
            f'{dict1.keys()} != {dict2.keys()}'
        )

    return {
        key: (val1, dict2[key])
        for key, val1 in dict1.items()
    }
