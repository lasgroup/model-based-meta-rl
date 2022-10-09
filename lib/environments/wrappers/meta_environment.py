"""Implementation of a meta-learning environment"""

from typing import Callable, Union, Tuple
from dotmap import DotMap

import numpy as np

from rand_param_envs.base import RandomEnv
from rllib.environment import AbstractEnvironment

from lib.environments.random_mujocoMB_envs.random_mujoco_env import RandomMujocoEnv
from lib.environments.wrappers.random_environment import RandomEnvironment


class MetaEnvironmentWrapper:
    """A wrapper which samples environment from the task distribution for meta-learning"""

    def __init__(
            self,
            base_env: Union[RandomEnvironment, RandomEnv, RandomMujocoEnv],
            params: DotMap,
            training: bool = True
    ):

        self._base_env = base_env

        self._training = training
        self.num_train_env_instances = params.num_train_env_instances
        self.num_test_env_instances = params.num_test_env_instances

        if hasattr(base_env, 'num_random_params'):
            self._train_env_params = np.random.rand(self.num_train_env_instances, self._base_env.num_random_params)
            self._test_env_params = np.random.rand(self.num_test_env_instances, self._base_env.num_random_params)
        else:
            self._train_env_params = self._base_env.sample_tasks(self.num_train_env_instances)
            self._test_env_params = self._base_env.sample_tasks(self.num_test_env_instances)

        self.current_train_env = 0
        self.current_test_env = 0

        if self.training:
            params = self._train_env_params[self.current_train_env]
        else:
            params = self._test_env_params[self.current_test_env]
        self._base_env.set_task(params)

        self._counters = {"steps": 0, "episodes": 0, "train_trials": 0, "test_trials": 0}

    @property
    def train_env_params(self):
        return self._train_env_params

    @property
    def test_env_params(self):
        return self._test_env_params

    @train_env_params.setter
    def train_env_params(self, val):
        assert isinstance(val, list)
        self._train_env_params = val
        self.num_train_env_instances = len(val)
        self.current_train_env = self.current_train_env % self.num_train_env_instances
        if self.training:
            self._base_env.set_task(self._train_env_params[self.current_train_env])

    @test_env_params.setter
    def test_env_params(self, val):
        assert isinstance(val, list)
        self._test_env_params = val
        self.num_test_env_instances = len(val)
        self.current_test_env = self.current_test_env % self.num_test_env_instances
        if not self.training:
            self._base_env.set_task(self._test_env_params[self.current_test_env])

    def sample_new_env(self):
        """Samples new environment parameters from the task distribution"""
        if self.training:
            self._counters["train_trials"] += 1
            self.current_train_env = np.random.randint(self.num_train_env_instances)
            params = self._train_env_params[self.current_train_env]
        else:
            self._counters["test_trials"] += 1
            self.current_test_env = np.random.randint(self.num_test_env_instances)
            params = self._test_env_params[self.current_test_env]
        self._base_env.set_task(params)

    def sample_next_env(self):
        """Samples next environment parameters from the initialized environments"""
        if self.training:
            self._counters["train_trials"] += 1
            self.current_train_env += 1
            params = self._train_env_params[self.current_train_env % self.num_train_env_instances]
        else:
            self._counters["test_trials"] += 1
            self.current_test_env += 1
            params = self._test_env_params[self.current_test_env % self.num_test_env_instances]
        self._base_env.set_task(params)

    def get_env(self, env_id=0):
        """Returns an environment with params set as env_id"""
        if self.training:
            self.current_train_env = env_id
            params = self._train_env_params[self.current_train_env]
        else:
            self.current_test_env = env_id
            params = self._test_env_params[self.current_test_env]
        self._base_env.set_task(params)
        return self._base_env

    def set_base_env(self, env):
        self._base_env = env
        if self.training:
            params = self._train_env_params[self.current_train_env]
        else:
            params = self._test_env_params[self.current_test_env]
        self._base_env.set_task(params)

    @property
    def num_random_params(self):
        return self._base_env.num_random_params

    @property
    def goal(self):
        return self._base_env.goal

    @property
    def original_dim_action(self) -> Tuple[int]:
        return self._base_env.dim_action

    @property
    def hallucinated_dim_action(self) -> Tuple[int]:
        return self._base_env.hall_shape

    @property
    def unwrapped(self) -> AbstractEnvironment:
        return self._base_env

    def set_task(self, random_samples):
        self._base_env.set_task(random_samples)

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