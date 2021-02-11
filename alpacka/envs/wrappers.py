"""Environment wrappers."""

import collections
import copy
import functools
import pickle

import gym
import numpy as np
from gym import wrappers

from alpacka.envs import base


class ModelWrapper(gym.Wrapper):

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        return self.env.restore_state(state)


_TimeLimitWrapperState = collections.namedtuple(
    '_TimeLimitWrapperState',
    ['super_state', 'elapsed_steps']
)


class TimeLimitWrapper(wrappers.TimeLimit, ModelWrapper):

    def clone_state(self):
        assert self._elapsed_steps is not None, (
            'Environment must be reset before the first clone_state().'
        )

        return _TimeLimitWrapperState(
            super_state=super().clone_state(), elapsed_steps=self._elapsed_steps
        )

    def restore_state(self, state):
        try:
            self._elapsed_steps = state.elapsed_steps
            state = state.super_state
        except AttributeError:
            self._elapsed_steps = 0

        return super().restore_state(state)


_FrameStackWrapperState = collections.namedtuple(
    '_FrameStackWrapperState', ['super_state', 'stack']
)


class FrameStackWrapper(ModelWrapper):

    def __init__(self, env, n_frames=4, axis=-1, concatenate=False):
        super().__init__(env)
        self._n_frames = n_frames
        self._axis = axis
        self._concatenate = concatenate

        self._stack = collections.deque(maxlen=n_frames)

        def repeat_frame(x):
            return self._join_frames([x] * n_frames)

        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=repeat_frame(env.observation_space.low),
            high=repeat_frame(env.observation_space.high),
        )

    def _join_frames(self, frames):
        join_fn = np.concatenate if self._concatenate else np.stack
        return join_fn(frames, axis=self._axis)

    def _assert_stack_full(self):
        assert len(self._stack) == self._n_frames, (
            'Environment must be reset first.'
        )

    def _get_observation(self):
        self._assert_stack_full()
        return self._join_frames(self._stack)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for _ in range(self._n_frames - 1):
            self._stack.append(np.zeros_like(observation))
        self._stack.append(observation)
        return self._get_observation()

    def step(self, action):
        (observation, reward, done, info) = self.env.step(action)
        self._stack.append(observation)
        return (self._get_observation(), reward, done, info)

    def clone_state(self):
        self._assert_stack_full()
        return _FrameStackWrapperState(
            super_state=super().clone_state(), stack=self._stack.copy()
        )

    def restore_state(self, state):
        self._stack = state.stack.copy()
        super().restore_state(state.super_state)
        return self._get_observation()


_StateCachingWrapperTransition = collections.namedtuple(
    '_StateCachingWrapperTransition',
    ['observation', 'reward', 'done', 'info', 'next_state'],
)


class _StateCachingWrapperState(collections.namedtuple(
    '_StateCachingWrapperState', ['state', 'observation', 'hash']
)):

    def __new__(cls, state, observation):
        return super().__new__(
            cls,
            state=state,
            observation=observation,
            hash=hash(pickle.dumps(state)),
        )

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return self.hash


class StateCachingWrapper(ModelWrapper):

    def __init__(self, env, capacity=10000):
        assert env.stochasticity in (
            base.Stochasticity.none, base.Stochasticity.episodic
        )

        super().__init__(env)
        self._transition = functools.lru_cache(maxsize=capacity)(
            self._compute_transition
        )
        self._current_state = None

    def _compute_transition(self, state, action):
        self.env.restore_state(state.state)
        (observation, reward, done, info) = self.env.step(action)
        next_state = _StateCachingWrapperState(
            self.env.clone_state(), observation
        )
        return _StateCachingWrapperTransition(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            next_state=next_state,
        )

    def reset(self, **kwargs):
        self._transition.cache_clear()

        observation = self.env.reset(**kwargs)
        state = _StateCachingWrapperState(self.env.clone_state(), observation)
        self._current_state = state

        return copy.deepcopy(observation)

    def step(self, action):
        assert self._current_state is not None, (
            'Environment must be reset first.'
        )

        transition = self._transition(self._current_state, action)
        self._current_state = transition.next_state
        observation = self._current_state.observation
        return copy.deepcopy(
            (observation, transition.reward, transition.done, transition.info)
        )

    def restore_state(self, state):
        self._current_state = state
        return copy.deepcopy(state.observation)

    def clone_state(self):
        return self._current_state

    @property
    def state_info(self):
        self.env.restore_state(self._current_state.state)
        return self.env.state_info


def wrap(env_class, wrapper_classes):
    if not wrapper_classes:
        return env_class
    (first, *rest) = wrapper_classes
    return lambda: first(wrap(env_class, rest)())
