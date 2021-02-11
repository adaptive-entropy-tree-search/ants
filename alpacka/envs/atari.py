"""Atari environments."""

import copy

import gym
import numpy as np
from gym.wrappers import atari_preprocessing

from alpacka.envs import base

try:
    from gym.envs.atari import atari_env
except gym.error.DependencyNotInstalled:
    atari_env = None


class Atari(base.RestorableEnv):
    installed = atari_env is not None

    allowed_stochasticities = [
        base.Stochasticity.none,
        base.Stochasticity.episodic,
        base.Stochasticity.universal,
    ]

    class Renderer(base.EnvRenderer):

        def render_state(self, state_info):
            obs = np.array(state_info) * 255
            return np.broadcast_to(obs, obs.shape[:2] + (3,)).astype(np.uint8)

        def render_action(self, action):
            return atari_env.ACTION_MEANING[action].lower()

    def __init__(
            self,
            game='pong',
            stochasticity=base.Stochasticity.episodic,
            sticky_actions=None,
            env_kwargs=None,
            wrapper_kwargs=None,
    ):
        if atari_env is None:
            raise ImportError(
                'Could not import gym.envs.atari! HINT: Install gym[atari].'
            )

        if sticky_actions is None:
            sticky_actions = stochasticity is base.Stochasticity.universal

        default_repeat_prob = 0.0
        if stochasticity is base.Stochasticity.none:

            default_noop_max = 0
            assert not sticky_actions
        else:

            default_noop_max = 30
            if sticky_actions:
                default_repeat_prob = 0.25

        env_kwargs = {
            'obs_type': 'image',
            'frameskip': 1,
            'repeat_action_probability': default_repeat_prob,
            **(env_kwargs or {})
        }
        env = atari_env.AtariEnv(game, **env_kwargs)

        class Spec:
            id = 'NoFrameskip'

        env.spec = Spec

        wrapper_kwargs = {
            'noop_max': default_noop_max,
            'scale_obs': True,
            **(wrapper_kwargs or {})
        }
        self._env = atari_preprocessing.AtariPreprocessing(
            env, **wrapper_kwargs
        )
        self.stochasticity = stochasticity

        self.observation_space = copy.copy(self._env.observation_space)
        if len(self.observation_space.shape) == 2:
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low[..., None],
                high=self.observation_space.high[..., None],
            )
        self.action_space = self._env.action_space

    def reset(self):
        self._env.reset()
        return self._observation

    def step(self, action):
        (_, reward, done, info) = self._env.step(action)
        return (self._observation, reward, done, info)

    def close(self):
        self._env.close()

    def clone_state(self):
        if self.stochasticity is base.Stochasticity.universal:
            env_state = self._env.clone_state()
        else:
            env_state = self._env.clone_full_state()

        state = (env_state, copy.deepcopy(self._env.obs_buffer))

        if self.stochasticity is not base.Stochasticity.universal:
            state += (self._env.unwrapped.np_random,)

        return state

    def restore_state(self, state):
        (env_state, obs_buffer) = state[:2]

        if self.stochasticity is base.Stochasticity.universal:
            self._env.restore_state(env_state)
        else:
            self._env.restore_full_state(env_state)
            self._env.unwrapped.np_random = state[2]

        self._env.obs_buffer = copy.deepcopy(obs_buffer)
        return self._observation

    @property
    def state_info(self):
        return self._observation

    @property
    def _observation(self):
        obs = self._env._get_obs()
        if len(obs.shape) == 2:
            obs = obs[..., None]
        return obs.copy()
