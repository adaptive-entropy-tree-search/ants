"""CartPole env."""

import numpy as np
from gym.envs import classic_control

from alpacka.envs import base

try:
    import cv2
except ImportError:
    cv2 = None


class CartPole(classic_control.CartPoleEnv, base.RestorableEnv):
    stochasticity = base.Stochasticity.episodic

    def __init__(self, solved_at=500, reward_scale=1., **kwargs):
        super().__init__(**kwargs)

        self.solved_at = solved_at
        self.reward_scale = reward_scale

        self._step = None

    def reset(self):
        self._step = 0
        return super().reset()

    def step(self, action):
        (observation, reward, done, info) = super().step(action)
        info['solved'] = self._step >= self.solved_at
        self._step += 1
        return (observation, reward * self.reward_scale, done, info)

    def clone_state(self):
        return (tuple(self.state), self.steps_beyond_done, self._step)

    def restore_state(self, state):
        (state, self.steps_beyond_done, self._step) = state
        self.state = np.array(state)
        return self.state

    class Renderer(base.EnvRenderer):

        def __init__(self, env):
            super().__init__(env)

            self._screen_width = 600
            self._screen_height = 400
            self._world_width = 2 * 2.4
            self._scale = self._screen_width / self._world_width
            self._cart_y = self._screen_height - 100
            self._pole_width = 10
            self._pole_len = self._scale
            self._cart_width = 50
            self._cart_height = 30

        def render_state(self, state_info):
            if cv2 is None:
                raise ImportError('Could not import cv2!')

            position, _, angle, _ = state_info
            cart_x = int(position * self._scale + self._screen_width / 2.0)

            img = np.ones((self._screen_height, self._screen_width, 3)) * 255.

            img = cv2.line(
                img,
                pt1=(0, self._cart_y),
                pt2=(self._screen_width, self._cart_y),
                color=(0, 0, 0)
            )

            img = cv2.rectangle(
                img,
                pt1=(cart_x - self._cart_width // 2,
                     self._cart_y - self._cart_height // 2),
                pt2=(cart_x + self._cart_width // 2,
                     self._cart_y + self._cart_height // 2),
                color=(0, 0, 0),
                thickness=cv2.FILLED
            )

            img = cv2.line(
                img,
                pt1=(cart_x, self._cart_y),
                pt2=(int(cart_x + self._pole_len * np.sin(angle)),
                     int(self._cart_y - self._pole_len * np.cos(angle))),
                color=(204, 153, 102),
                thickness=self._pole_width
            )

            img = cv2.circle(
                img,
                center=(cart_x, self._cart_y),
                radius=self._pole_width // 2,
                color=(127, 127, 204),
                thickness=cv2.FILLED
            )

            return img.astype(np.uint8)

        def render_action(self, action):
            return ['left', 'right'][action]
