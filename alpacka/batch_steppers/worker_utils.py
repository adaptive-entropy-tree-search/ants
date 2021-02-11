"""Utilities for BatchSteppers running in separate workers."""

import lzma
import pickle

import gin

from alpacka.batch_steppers import core

init_hooks = []


def register_init_hook(hook):
    init_hooks.append(hook)


def get_config(env_class, agent_class, network_fn):
    env_class()
    agent_class()
    network_fn()
    return gin.operative_config_str()


class Worker:

    def __init__(
            self,
            env_class,
            agent_class,
            network_fn,
            config,
            scope,
            init_hooks,
            compress_episodes,
    ):

        gin.parse_config(config, skip_unknown=True)

        for hook in init_hooks:
            hook()

        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        with gin.config_scope(scope):
            self.env = env_class()
            self.agent = agent_class()
            self._request_handler = core.RequestHandler(network_fn)

        self._compress_episodes = compress_episodes

    def run(self, params, solve_kwargs):
        episode_cor = self.agent.solve(self.env, **solve_kwargs)
        episode = self._request_handler.run_coroutine(episode_cor, params)
        if self._compress_episodes:
            episode = compress_episode(episode)
        return episode

    def close(self):
        self.env.close()
        self.agent.close()

    @property
    def network(self):
        return self._request_handler.network


def compress_episode(episode):
    return lzma.compress(pickle.dumps(episode))


def decompress_episode(data):
    return pickle.loads(lzma.decompress(data))
