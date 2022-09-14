"""Implementation of a meta-learning environment"""

from typing import Callable, Union
from dotmap import DotMap

import numpy as np

from rand_param_envs.base import RandomEnv
from lib.environments.wrappers.random_environment import RandomEnvironment


class MetaEnvironmentWrapper:
    """A wrapper which samples environment from the task distribution for meta-learning"""

    def __init__(
            self,
            base_env: Union[RandomEnvironment, RandomEnv],
            params: DotMap,
            training: bool = True
    ):

        self._base_env = base_env

        self._training = training
        self.num_train_env_instances = params.num_train_env_instances
        self.num_test_env_instances = params.num_test_env_instances

        if hasattr(base_env, 'num_random_params'):
            self.train_env_params = np.random.rand(self.num_train_env_instances, self._base_env.num_random_params)
            self.test_env_params = np.random.rand(self.num_test_env_instances, self._base_env.num_random_params)
        else:
            self.train_env_params = self._base_env.sample_tasks(self.num_train_env_instances)
            self.test_env_params = self._base_env.sample_tasks(self.num_test_env_instances)

        self.current_train_env = 0
        self.current_test_env = 0

        if self.training:
            params = self.train_env_params[self.current_train_env]
        else:
            params = self.test_env_params[self.current_test_env]
        self._base_env.set_task(params)

        self._counters = {"steps": 0, "episodes": 0, "train_trials": 0, "test_trials": 0}

    def sample_new_env(self):
        """Samples new environment parameters from the task distribution"""
        if self.training:
            self._counters["train_trials"] += 1
            self.current_train_env = np.random.randint(self.num_train_env_instances)
            params = self.train_env_params[self.current_train_env]
        else:
            self._counters["test_trials"] += 1
            self.current_test_env = np.random.randint(self.num_test_env_instances)
            params = self.test_env_params[self.current_test_env]
        self._base_env.set_task(params)

    def sample_next_env(self):
        """Samples next environment parameters from the initialized environments"""
        if self.training:
            self._counters["train_trials"] += 1
            self.current_train_env += 1
            params = self.train_env_params[self.current_train_env % self.num_train_env_instances]
        else:
            self._counters["test_trials"] += 1
            self.current_test_env += 1
            params = self.test_env_params[self.current_test_env % self.num_test_env_instances]
        self._base_env.set_task(params)

    def __getattr__(self, name: str):
        return getattr(self._base_env, name)

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

    def step(self, action):
        next_state, reward, done, info = self._base_env.step(action)
        self._counters["steps"] += 1
        if done:
            self._counters["episodes"] += 1
        return next_state, reward, done, info

    @property
    def training(self):
        return self._training

    def train(self, val=True):
        self._training = val

    def eval(self, val=True):
        self._training = not val