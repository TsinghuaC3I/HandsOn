from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space


_RECORDING_PASSTHROUGH_PREFIXES = (
    "xy3_",
    "xy4_",
    "xy5_",
    "as019_",
    "bc017_",
    "hs014_",
    "gen1_",
    "c01_",
    "e01_",
    "gen5_",
    "newton_",
    "p003_",
    "pizza_",
    "plan_",
    "physx_",
    "ycb02_",
    "xy6_",
)


class HandsOnObsAdapter(gym.ObservationWrapper):
    """Adapt raw task observations into the flattened HandsOn collection format."""

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        ordered_spaces = OrderedDict(
            (key, self.env.observation_space["state"][key]) for key in self.proprio_keys
        )
        # Gymnasium sorts Mapping inputs to Dict spaces for legacy flattening
        # compatibility, even when the mapping is an OrderedDict.  Pass an
        # explicit sequence of pairs so the task-declared order is retained;
        # otherwise the flattened Box bounds no longer line up with the values
        # concatenated below (notably a bounded per-task obstacle mask).
        self.proprio_space = gym.spaces.Dict(list(ordered_spaces.items()))

        # Recording-only telemetry stays outside the flattened policy state.
        # AS019 and BC017 use task-prefixed fields for their public and hidden
        # replay boundaries, Plan1 uses top-level c01_* fields, and E01 uses
        # e01_* for its complete actual hand state.  The reusable Plan task
        # template uses plan_* for arm-planning outputs, while P003 uses p003_*
        # for final-frame guard telemetry.  The isolated Isaac Sim recorder
        # uses physx_* for PhysX readback, and B03 uses newton_* for Newton
        # readback.  Forward only fields declared by the raw observation space
        # so ad-hoc debug values cannot leak into a recording.  The old
        # one-dimensional/128-value filter silently dropped contact matrices.
        self.passthrough_spaces = OrderedDict(
            (key, space)
            for key, space in self.env.observation_space.spaces.items()
            if key.startswith(_RECORDING_PASSTHROUGH_PREFIXES)
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space.get("images", {})),
                **self.passthrough_spaces,
            }
        )

    def observation(self, obs):
        proprio_values = [np.asarray(obs["state"][key]).ravel() for key in self.proprio_keys]
        if proprio_values:
            # ``flatten_space`` preserves the raw state's dtype.  Casting to
            # float64 unconditionally made the ABC/C10 adapter violate its
            # float32 Box contract, so Gymnasium ``contains`` rejected every
            # otherwise finite reset observation.  Follow the declared policy
            # space while retaining float64 for the D-long-horizon spaces that
            # intentionally request it.
            state_dtype = self.observation_space["state"].dtype
            flat_state = np.concatenate(proprio_values, axis=0).astype(
                state_dtype,
                copy=False,
            )
        else:
            flat_state = np.zeros(
                (0,), dtype=self.observation_space["state"].dtype
            )
        adapted = {"state": flat_state, **obs.get("images", {})}
        for key in self.passthrough_spaces:
            if key in obs:
                adapted[key] = np.asarray(obs[key])
        return adapted

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
