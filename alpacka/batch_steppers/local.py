"""Single-threaded environment stepper."""

import collections
import functools

import numpy as np

from alpacka import data
from alpacka.batch_steppers import core


class _NetworkRequestBatcher:

    def __init__(self, requests):
        self._requests = requests
        self._model_request = None

    @property
    def batched_request(self):
        if self._model_request is not None:
            return self._model_request
        self._model_request = next(x for x in self._requests if x is not None)
        return self._model_request

    def unbatch_responses(self, batched_responses):
        return [batched_responses] * len(self._requests)


class _PredictionRequestBatcher:

    def __init__(self, requests):
        self._requests = requests
        self._n_agents = len(requests)
        self._batched_request = None
        self._batch_sizes = None

    @property
    def batched_request(self):
        if self._batched_request is not None:
            return self._batched_request

        def assert_not_scalar(x):
            assert np.array(x).shape, (
                'All arrays in a PredictRequest must be at least rank 1.'
            )

        data.nested_map(assert_not_scalar, self._requests)

        self._batched_request = data.nested_concatenate(self._requests)

        self._batch_sizes = [
            data.choose_leaf(request).shape[0]
            for request in self._requests
        ]
        return self._batched_request

    def unbatch_responses(self, batched_responses):
        def slice_responses(x, start_index, batch_size):
            return x[start_index:(start_index + batch_size)]

        unbatched_responses = []
        start_index = 0
        for batch_size in self._batch_sizes:
            unbatched_responses.append(data.nested_map(
                functools.partial(
                    slice_responses,
                    start_index=start_index,
                    batch_size=batch_size,
                ),
                batched_responses,
            ))
            start_index += batch_size

        return unbatched_responses


class LocalBatchStepper(core.BatchStepper):

    def __init__(self, env_class, agent_class, network_fn, n_envs, output_dir):
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        def make_env_and_agent():
            env = env_class()
            agent = agent_class()
            return (env, agent)

        self._envs_and_agents = [make_env_and_agent() for _ in range(n_envs)]
        self._request_handler = core.RequestHandler(network_fn)

    def _group_requests(self, requests):
        type_to_requests = collections.defaultdict(list)
        for (index, request) in enumerate(requests):
            type_to_requests[type(request)].append((request, index))

        return [

            list(zip(*requests_and_indices))
            for (type_, requests_and_indices) in type_to_requests.items()

            if type_ is not type(None)
        ]

    @staticmethod
    def _batch_requests(requests):
        assert requests
        if isinstance(requests[0], data.NetworkRequest):
            batcher_type = _NetworkRequestBatcher
        else:
            batcher_type = _PredictionRequestBatcher
        return batcher_type(requests)

    def _handle_requests(self, requests):
        request_groups = self._group_requests(requests)
        request_batchers = [
            (self._batch_requests(request_group), indices)
            for (request_group, indices) in request_groups
        ]

        responses = [None] * len(requests)
        for (batcher, indices) in request_batchers:
            batched_response = yield batcher.batched_request
            response_group = batcher.unbatch_responses(batched_response)

            for (response, index) in zip(response_group, indices):
                responses[index] = response

        return responses

    def _batch_coroutines(self, cors):
        episodes = [None] * len(cors)

        def store_transitions(i, cor):
            episodes[i] = yield from cor

            while True:
                yield None

        cors = [store_transitions(i, cor) for (i, cor) in enumerate(cors)]

        def all_finished(xs):
            return all(x is None for x in xs)

        requests = [next(cor) for cor in cors]
        while not all_finished(requests):
            responses = yield from self._handle_requests(requests)
            requests = [
                cor.send(response)
                for (cor, response) in zip(cors, responses)
            ]
        return episodes

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        episode_cor = self._batch_coroutines([
            agent.solve(env, **solve_kwargs)
            for (env, agent), solve_kwargs in zip(
                self._envs_and_agents, solve_kwargs_per_agent
            )
        ])
        return self._request_handler.run_coroutine(episode_cor, params)

    def close(self):
        for env, agent in self._envs_and_agents:
            env.close()
            agent.close()
