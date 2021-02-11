"""Uniform replay buffer."""

import functools

import numpy as np

from alpacka import data


class UniformReplayBuffer:

    def __init__(self, datapoint_signature, capacity):
        self._datapoint_shape = data.nested_map(
            lambda x: x.shape, datapoint_signature
        )
        self._capacity = capacity
        self._size = 0
        self._insert_index = 0

        def init_array(signature):
            shape = (self._capacity,) + signature.shape
            return np.zeros(shape, dtype=signature.dtype)

        self._data_buffer = data.nested_map(init_array, datapoint_signature)

    def add(self, stacked_datapoints):
        datapoint_shape = data.nested_map(
            lambda x: x.shape[1:], stacked_datapoints
        )
        if datapoint_shape != self._datapoint_shape:
            raise ValueError(
                'Datapoint shape mismatch: got {}, expected {}.'.format(
                    datapoint_shape, self._datapoint_shape
                )
            )

        n_elems = data.choose_leaf(data.nested_map(
            lambda x: x.shape[0], stacked_datapoints
        ))

        def insert_to_array(buf, elems):
            buf_size = buf.shape[0]
            assert elems.shape[0] == n_elems
            index = self._insert_index

            buf[index:min(index + n_elems, buf_size)] = elems[:buf_size - index]

            buf[:max(index + n_elems - buf_size, 0)] = elems[buf_size - index:]

        data.nested_zip_with(
            insert_to_array, (self._data_buffer, stacked_datapoints)
        )
        if self._size < self._capacity:
            self._size = min(self._insert_index + n_elems, self._capacity)
        self._insert_index = (self._insert_index + n_elems) % self._capacity

    def sample(self, batch_size):
        if self._data_buffer is None:
            raise ValueError('Cannot sample from an empty buffer.')
        indices = np.random.randint(self._size, size=batch_size)
        return data.nested_map(lambda x: x[indices], self._data_buffer)


class HierarchicalReplayBuffer:

    def __init__(self, datapoint_signature, capacity, hierarchy_depth):
        self._raw_buffer_fn = functools.partial(
            UniformReplayBuffer, datapoint_signature, capacity
        )

        assert not hierarchy_depth
        self._buffer_hierarchy = self._raw_buffer_fn()
        self._hierarchy_depth = hierarchy_depth

    def add(self, stacked_datapoints, buckets):
        assert len(buckets) == self._hierarchy_depth

        if self._hierarchy_depth:
            buffer_hierarchy = self._buffer_hierarchy
            for bucket in buckets[:-1]:
                buffer_hierarchy = buffer_hierarchy[bucket]

            bucket = buckets[-1]
            if bucket not in buffer_hierarchy:
                buffer_hierarchy[bucket] = self._raw_buffer_fn()
            buf = buffer_hierarchy[bucket]
        else:
            buf = self._buffer_hierarchy

        buf.add(stacked_datapoints)

    def _sample_one(self):
        buffer_hierarchy = self._buffer_hierarchy
        for _ in range(self._hierarchy_depth):
            buffer_hierarchy = buffer_hierarchy.random_value()
        return buffer_hierarchy.sample(batch_size=1)

    def sample(self, batch_size):
        return data.nested_concatenate(
            [self._sample_one() for _ in range(batch_size)]
        )
