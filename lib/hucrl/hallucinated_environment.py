from typing import Tuple, Callable

import numpy as np
from gym.spaces import Box
from rllib.environment import AbstractEnvironment


class HallucinatedEnvironmentWrapper:

    def __init__(self, base_env: AbstractEnvironment):

        self._base_env = base_env

        assert not self._base_env.discrete_action and not self._base_env.discrete_state

        self.hall_shape = (self._base_env.dim_state, )
        self.dim_action = self._base_env.dim_action + self._base_env.dim_state
        self.action_space = Box(
            low=np.concatenate((self._base_env.action_space.low, -np.ones(self.hall_shape))),
            high=np.concatenate((self._base_env.action_space.high, np.ones(self.hall_shape))),
            shape=(self.dim_action,),
            dtype=np.float32,
        )

    def __getattr__(self, name: str):
        return getattr(self._base_env, name)

    def __setattr__(self, name, value):
        setattr(self._base_env, name, value)

    @property
    def original_dim_action(self) -> Tuple[int]:
        return self._base_env.dim_action

    @property
    def hallucinated_dim_action(self) -> Tuple[int]:
        return self.hall_shape

    @property
    def unwrapped(self) -> AbstractEnvironment:
        return self._base_env

    def step(self, action: np.ndarray) -> Tuple[np.ndarray]:
        return self._base_env.step(action[: self.original_dim_action[0]])

    def reset(self) -> np.ndarray:
        return self._base_env.reset()

    def render(self):
        return self._base_env.render()

    def close(self):
        return self._base_env.close()

    def seed(self, seed=None):
        if hasattr(self._base_env, "seed") and isinstance(self._base_env.seed, Callable):
            return self._base_env.seed(seed)
        else:
            return None
