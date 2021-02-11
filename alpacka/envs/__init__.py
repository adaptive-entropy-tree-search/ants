from alpacka.envs import atari
from alpacka.envs import cartpole
from alpacka.envs import wrappers
from alpacka.envs.base import *


def configure_env(env_class):
    return gin.external_configurable(
        env_class, module='alpacka.envs'
    )


native_envs = []

CartPole = configure_env(cartpole.CartPole)

native_envs.extend([CartPole])

Atari = configure_env(atari.Atari)

FrameStackWrapper = configure_env(wrappers.FrameStackWrapper)
TimeLimitWrapper = configure_env(wrappers.TimeLimitWrapper)
StateCachingWrapper = configure_env(wrappers.StateCachingWrapper)
wrap = configure_env(wrappers.wrap)
