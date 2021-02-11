"""Agent base classes."""

from alpacka import data
from alpacka import envs
from alpacka import metric_logging
from alpacka import utils
from alpacka.utils import space


class Agent:

    def __init__(self, parameter_schedules=None):
        self._parameter_schedules = parameter_schedules or {}

    def solve(self, env, epoch=None, init_state=None, time_limit=None):
        del env
        del init_state
        del time_limit
        for attr_name, schedule in self._parameter_schedules.items():
            param_value = schedule(epoch)
            utils.recursive_setattr(self, attr_name, param_value)
            metric_logging.log_scalar(
                'agent_param/' + attr_name, epoch, param_value
            )

        return
        yield

    def network_signature(self, observation_space, action_space):
        del observation_space
        del action_space
        return None

    def close(self):
        pass


class OnlineAgent(Agent):

    def __init__(self, callback_classes=(), **kwargs):
        super().__init__(**kwargs)

        self._action_space = None
        self._epoch = None
        self._callbacks = [
            callback_class(self) for callback_class in callback_classes
        ]

    def reset(self, env, observation):
        del observation
        self._action_space = env.action_space

        return
        yield

    def act(self, observation):
        raise NotImplementedError

    def postprocess_transitions(self, transitions):
        return transitions

    @staticmethod
    def compute_metrics(episodes):
        del episodes
        return {}

    def solve(self, env, epoch=None, init_state=None, time_limit=None):
        yield from super().solve(env, epoch, init_state, time_limit)

        self._epoch = epoch

        model_env = env

        if time_limit is not None:
            env = envs.TimeLimitWrapper(env, time_limit)

        if init_state is None:

            observation = env.reset()
        else:

            observation = env.restore_state(init_state)

        yield from self.reset(model_env, observation)

        for callback in self._callbacks:
            callback.on_episode_begin(env, observation, epoch)

        transitions = []
        done = False
        info = {}
        while not done:

            (action, agent_info) = yield from self.act(observation)
            (next_observation, reward, done, info) = env.step(action)

            for callback in self._callbacks:
                callback.on_real_step(
                    agent_info, action, next_observation, reward, done
                )

            transitions.append(data.Transition(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
                agent_info=agent_info,
            ))
            observation = next_observation

        for callback in self._callbacks:
            callback.on_episode_end()

        transitions = self.postprocess_transitions(transitions)

        return_ = sum(transition.reward for transition in transitions)
        solved = info['solved'] if 'solved' in info else None
        truncated = (info['TimeLimit.truncated']
                     if 'TimeLimit.truncated' in info else None)
        transition_batch = data.nested_stack(transitions)
        action_space_size = space.max_size(model_env.action_space)
        return data.Episode(
            transition_batch=transition_batch,
            return_=return_,
            solved=solved,
            truncated=truncated,
            action_space_size=action_space_size
        )


class AgentCallback:
    def __init__(self, agent):
        self._agent = agent

    def on_episode_begin(self, env, observation, epoch):
        pass

    def on_episode_end(self):
        pass

    def on_real_step(self, agent_info, action, observation, reward, done):
        pass

    def on_pass_begin(self):
        pass

    def on_pass_end(self):
        pass

    def on_model_step(self, agent_info, action, observation, reward, done):
        pass
