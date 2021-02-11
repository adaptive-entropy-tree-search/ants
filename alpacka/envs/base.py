"""Base classes related to environments."""

import enum

import gin
import gym


@gin.constants_from_enum
class Stochasticity(enum.Enum):
    unknown = 0

    none = 1

    episodic = 2

    universal = 3


class RestorableEnv(gym.Env):
    stochasticity = Stochasticity.unknown

    def clone_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError


class EnvRenderer:

    def __init__(self, env):
        del env

    def render_state(self, state_info):
        raise NotImplementedError

    def render_heatmap(self, heatmap, current_state_info):
        raise NotImplementedError

    def render_action(self, action):
        raise NotImplementedError
