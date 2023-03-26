from typing import Callable, Iterable, Union

import torch
import numpy as np
from gym.spaces import Box

from rllib.model import AbstractModel
from rllib.util.rollout import step_model
from rllib.util.neural_networks.utilities import to_torch
from stable_baselines3.common.vec_env import VecEnv

from lib.datasets.utils import sample_states


class ModelBasedEnvironment(VecEnv):

    def __init__(
            self,
            dynamical_model: AbstractModel,
            reward_model: AbstractModel,
            action_scale: Union[float, np.ndarray],
            num_envs: int,
            sim_num_steps: int,
            initial_states_distribution: Union[Callable, None],
    ):

        observation_space, action_space = self.get_obs_and_action_space(dynamical_model)

        super().__init__(num_envs, observation_space, action_space)

        self.max_steps = sim_num_steps

        self.dynamical_model = dynamical_model
        self.reward_model = reward_model

        self.action_scale = self.get_action_scale(action_scale)
        self.initial_states_distribution = initial_states_distribution

        self.state = None
        self._time = 0
        self.return_vals = None

    def get_obs_and_action_space(self, dynamical_model):

        observation_space = Box(
            low=np.full(dynamical_model.dim_state, -np.inf),
            high=np.full(dynamical_model.dim_state, np.inf),
            shape=dynamical_model.dim_state,
            dtype=np.float32,
        )

        action_space = Box(
            low=-np.ones(dynamical_model.dim_action),
            high=np.ones(dynamical_model.dim_action),
            shape=dynamical_model.dim_action,
            dtype=np.float32,
        )

        return observation_space, action_space

    def get_action_scale(self, action_scale):

        if isinstance(action_scale, np.ndarray):
            action_scale = to_torch(action_scale)
        elif not isinstance(action_scale, torch.Tensor):
            action_scale = torch.full(
                self.dynamical_model.dim_action, action_scale, dtype=torch.get_default_dtype()
            )

        if len(action_scale) < self.dynamical_model.dim_action[0]:
            extra_dim = self.dynamical_model.dim_action[0] - len(action_scale)
            action_scale = torch.cat((action_scale, torch.ones(extra_dim)))

        return action_scale

    def reset_torch(self):
        states = self.initial_states_distribution().squeeze()
        states = sample_states(states, self.num_envs)
        self.state = states
        self._time = 0
        self.return_vals = None
        return states

    def reset(self):
        return self.reset_torch().detach().numpy()

    def step(self, action):

        info = [{"TimeLimit.truncated": False}] * self.num_envs
        action = self.action_scale.numpy() * action.clip(-1.0, 1.0)

        observation, next_state, _ = step_model(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            termination_model=None,
            state=self.state,
            action=torch.tensor(action, dtype=self.state.dtype, device=self.state.device),
            action_scale=self.action_scale,
            done=None,
            pi=None,
        )

        self.state = next_state

        done = np.full(self.num_envs, False)
        self._time += 1

        if self._time == self.max_steps:
            for i in range(self.num_envs):
                info[i]["TimeLimit.truncated"] = True
                info[i]["terminal_observation"] = next_state[i].detach().numpy()
            next_state = self.reset_torch()

        return next_state.detach().numpy(), observation.reward.detach().squeeze().numpy(), done, info

    def step_async(self, actions):
        self.return_vals = self.step(actions)

    def step_wait(self):
        return self.return_vals

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return [None] * self.num_envs

    def get_valid_indices(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if not isinstance(indices, Iterable):
            return [indices]
        return indices

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self.get_valid_indices(indices))

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return getattr(self, method_name)(*method_args, **method_kwargs)[self.get_valid_indices(indices)]

    def get_attr(self, attr_name, indices=None):
        return getattr(self, attr_name)[self.get_valid_indices(indices)]

    def get_images(self):
        raise NotImplementedError("Images are not implemented for the model-based environment.")

    def getattr_depth_check(self, name, already_found):
        return None

    def set_attr(self, attr_name, value, indices=None):
        for i in self.get_valid_indices(indices):
            getattr(self, name=attr_name)[i] = value

    def set_initial_distribution(self, sampler):
        self.initial_states_distribution = sampler
