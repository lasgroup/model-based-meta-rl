"""Implementation of a meta-learning environment"""

from typing import Callable

import numpy as np
from lib.environments.wrappers.random_environment import RandomEnvironment


class MetaEnvironmentWrapper:
    """A wrapper which samples environment from the task distribution for meta-learning"""

    def __init__(self, base_env: RandomEnvironment, params):

        self._base_env = base_env
        self._counters = {"steps": 0, "episodes": 0, "trials": 0}
        self.num_env_instances = params.num_env_instances
        self.env_params = np.random.rand(self.num_env_instances, self._base_env.num_random_params)
        self.current_env = 0
        self._base_env.set_transition_params(self.env_params[self.current_env])

    def sample_new_env(self):
        """Samples new environment parameters from the task distribution"""
        self._counters["trials"] += 1
        self.current_env = np.random.randint(self.num_env_instances)
        self._base_env.set_transition_params(self.env_params[self.current_env])

    def sample_next_env(self):
        """Samples next environment parameters from the initialized environments"""
        self._counters["trials"] += 1
        self.current_env += 1
        self._base_env.set_transition_params(self.env_params[self.current_env % self.num_env_instances])

    def __getattr__(self, name: str):
        return getattr(self._base_env, name)

    def reset(self) -> np.ndarray:
        self._counters["steps"] = 0
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

    def step(self, action):
        next_state, reward, done, info = self._base_env.step(action)
        self._counters["steps"] += 1
        if done:
            self._counters["episodes"] += 1
        return next_state, reward, done, info
