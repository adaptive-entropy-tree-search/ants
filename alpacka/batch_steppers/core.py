"""Environment steppers."""

from alpacka import data


class BatchStepper:

    def __init__(
            self, env_class, agent_class, network_fn, n_envs, output_dir
    ):
        del env_class
        del agent_class
        del network_fn
        del output_dir
        self.n_envs = n_envs

    def _prepare_solve_kwargs(self, batched_solve_kwargs, common_solve_kwargs):
        batched_solve_kwargs = batched_solve_kwargs or dict()
        for name, values in batched_solve_kwargs.items():
            assert len(values) == self.n_envs

        for name, value in common_solve_kwargs.items():
            assert name not in batched_solve_kwargs, f'duplicated parameter {name}'
            batched_solve_kwargs[name] = [value] * self.n_envs

        if batched_solve_kwargs:

            kwargs_names = batched_solve_kwargs.keys()
            solve_kwargs_per_agent = [
                dict(zip(kwargs_names, agent_kwargs_values))
                for agent_kwargs_values in zip(*batched_solve_kwargs.values())
            ]
        else:

            solve_kwargs_per_agent = [dict() for _ in range(self.n_envs)]
        return solve_kwargs_per_agent

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        raise NotImplementedError

    def run_episode_batch(
            self, params, batched_solve_kwargs=None, **common_solve_kwargs
    ):
        solve_kwargs_per_agent = self._prepare_solve_kwargs(
            batched_solve_kwargs, common_solve_kwargs)
        return self._run_episode_batch(params, solve_kwargs_per_agent)

    def close(self):
        pass


class RequestHandler:

    def __init__(self, network_fn):
        self.network_fn = network_fn

        self._network = None
        self._should_update_params = None

    def run_coroutine(self, episode_cor, params):
        self._should_update_params = True

        try:
            request = next(episode_cor)
            while True:
                if isinstance(request, data.NetworkRequest):
                    request_handler = self._handle_network_request
                else:
                    request_handler = self._handle_prediction_request

                response = request_handler(request, params)
                request = episode_cor.send(response)
        except StopIteration as e:
            return e.value

    def _handle_network_request(self, request, params):
        del request
        return self.network_fn, params

    def _handle_prediction_request(self, request, params):
        return self.get_network(params).predict(request)

    def get_network(self, params=None):
        if self._network is None:
            self._network = self.network_fn()
        if params is not None and self._should_update_params:
            self.network.params = params
            self._should_update_params = False
        return self._network

    network = property(get_network)
