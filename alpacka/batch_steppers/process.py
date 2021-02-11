"""Process-distributed environment stepper."""

import collections
import functools
import multiprocessing as _mp
import sys

import gin
from tblib import pickling_support

from alpacka.batch_steppers import core
from alpacka.batch_steppers import worker_utils

mp = _mp.get_context(method='spawn')

pickling_support.install()

_ExcInfo = collections.namedtuple('_ExcInfo', ('type', 'value', 'traceback'))


def _cloudpickle():
    import cloudpickle
    return cloudpickle


class ProcessBatchStepper(core.BatchStepper):

    def __init__(
            self,
            env_class,
            agent_class,
            network_fn,
            n_envs,
            output_dir,
            process_class=mp.Process,
            serialize_worker_fn=True,
    ):

        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        config = worker_utils.get_config(env_class, agent_class, network_fn)

        def start_worker():
            worker_fn = functools.partial(
                worker_utils.Worker,
                env_class=env_class,
                agent_class=agent_class,
                network_fn=network_fn,
                config=config,
                scope=gin.current_scope(),

                init_hooks=[],
                compress_episodes=False,
            )

            if serialize_worker_fn:
                worker_fn = _cloudpickle().dumps(worker_fn)
            queue_in = mp.Queue()
            queue_out = mp.Queue()
            process = process_class(
                target=_target,
                args=(worker_fn, queue_in, queue_out, serialize_worker_fn),
            )
            process.start()
            return (queue_in, queue_out)

        self._queues = [start_worker() for _ in range(n_envs)]

        for (_, queue_out) in self._queues:
            _receive(queue_out)

    def _run_episode_batch(self, params, solve_kwargs_per_agent):
        for ((queue_in, _), solve_kwargs) in zip(
                self._queues, solve_kwargs_per_agent
        ):
            queue_in.put((params, solve_kwargs))

        return [_receive(queue_out) for (_, queue_out) in self._queues]

    def close(self):
        for (queue_in, _) in self._queues:
            queue_in.put(None)


def _target(worker_fn, queue_in, queue_out, serialized):
    try:
        if serialized:
            worker_fn = _cloudpickle().loads(worker_fn)
        worker = worker_fn()
        queue_out.put(None)
        while True:
            msg = queue_in.get()
            if msg is None:
                worker.close()
                break
            (params, solve_kwargs) = msg
            episode = worker.run(params, solve_kwargs)
            queue_out.put(episode)
    except Exception:
        queue_out.put(_ExcInfo(*sys.exc_info()))


def _receive(queue):
    msg = queue.get()
    if isinstance(msg, _ExcInfo):
        exc = msg.value

        raise exc.with_traceback(msg.traceback)
    return msg
