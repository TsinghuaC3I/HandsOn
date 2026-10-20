"""Gym view exposing the common policy boundary over task-native environments."""
import gymnasium as gym
import numpy as np

from ..eef52 import EEF52Control
from ..interfaces import STATE_DIM, ACTION_DIM, CONTROL_HZ, VIEWS
from .registry import TASKS


class EEF52PolicyWrapper(gym.Wrapper):
    """Measured state, canonical RGB names, and physical world EEF52 commands.

    This view never uses a task's flattened privileged observation as state.
    Image-disabled environments expose state only. Complete paper rendering,
    stable origins, budgets and outcome receipts belong to the evaluator.
    """
    def __init__(self, env, task):
        super().__init__(env)
        raw = env.unwrapped
        self.task_spec = TASKS[task]
        self.native = TASKS[task].evaluation_adapter().native_view(
            raw, launch=getattr(raw, '_sim', None) is not None)
        self.control = EEF52Control(self.native)
        if abs(self.native._control_dt - 1 / CONTROL_HZ) > 1e-12:
            raise ValueError('Native control rate differs from 30 Hz')
        if abs(self.native._physics_dt * self.native._n_substeps - 1 / CONTROL_HZ) > 1e-12:
            raise ValueError('Native physical action rate differs from 30 Hz')
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (ACTION_DIM,), dtype=np.float32)
        spaces = {'state': gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32)}
        self._image_keys = {}
        native_spaces = raw.observation_space.spaces.get('images')
        if native_spaces is not None:
            keys = native_spaces.spaces
            front = next((k for k in ('front', 'main', 'random_camera', 'ego') if k in keys), None)
            if front is None or any(k not in keys for k in VIEWS[1:]):
                raise ValueError('Native policy observation is missing one of the three camera views')
            self._image_keys = dict(front=front, wrist_left='wrist_left', wrist_right='wrist_right')
            spaces.update({key: keys[native] for key, native in self._image_keys.items()})
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, raw):
        state, _ = self.control.state()
        result = dict(state=state)
        if self._image_keys:
            images = raw.get('images')
            if images is None and all(key in raw for key in self._image_keys.values()):
                images = raw
            # Some task-native paths deliberately defer capture until after step.
            if images is None:
                front, left, right = self.native.render()
                images = {self._image_keys['front']: front, 'wrist_left': left, 'wrist_right': right}
            result.update({key: images[native] for key, native in self._image_keys.items()})
        return result

    def reset(self, *, seed=None, options=None):
        kwargs = dict(seed=seed)
        if options is not None:
            kwargs['options'] = options
        raw, info = self.env.reset(**kwargs)
        return self.observation(raw), info

    def step(self, action):
        # AS019's native camera capture is deferred until PhysX has returned,
        # matching its collection observation wrapper.
        deferred = bool(self.task_spec.bridge and self._image_keys)
        previous = getattr(self.native, 'image_obs', False)
        try:
            if deferred:
                self.native.image_obs = False
            raw, reward, terminated, truncated, info = self.control.step(action)
        finally:
            if deferred:
                self.native.image_obs = previous
        return self.observation(raw), reward, terminated, truncated, info
