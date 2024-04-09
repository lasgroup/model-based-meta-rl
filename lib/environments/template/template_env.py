from abc import abstractmethod
from typing import Tuple, Dict, Optional

import numpy
import torch
import numpy as np

from rllib.reward.state_action_reward import StateActionReward
from rllib.environment import AbstractEnvironment, GymEnvironment
from lib.environments.point_envs.envs.point_env_2d import PointEnv2D


class TemplateRewardModel(StateActionReward):
    """
    Template Reward Model Class. Implement the abstract methods from StateActionReward to create a custom reward model.

    """

    def __init__(self, *args, **kwargs):
        super(TemplateRewardModel, self).__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    def copy(self):
        """
        Return a copy of the reward model.
        """
        return self.__class__(*self._args, **self._kwargs)

    @abstractmethod
    def state_reward(self, state: torch.Tensor, next_state: Optional[torch.Tensor] = None) -> torch.tensor:
        """
        Return the reward given the state and the next state.
        """
        raise NotImplementedError


class TemplateEnv(AbstractEnvironment):
    """
    Template Environment Class. Implement the abstract methods from AbstractEnvironment to create a custom environment.

    For more details, refer to the documentation of the AbstractEnvironment class.

    For a simple example class, refer to the PointEnv2D class.
    """
    def __init__(self, *args, **kwargs):
        super(TemplateEnv, self).__init__(*args, **kwargs)
        self._reward_model = TemplateRewardModel()
        self._state = None
        self._time = 0

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        next_state, reward, done, info = self.step_(action)
        if self.num_states > 0 and done:  # Move to terminal state.
            next_state = self.num_states - 1
        self._time += 1
        if isinstance(reward, torch.Tensor):
            if reward.shape[-1] != self.dim_reward[0]:
                reward = reward.unsqueeze(-1).repeat_interleave(self.dim_reward[0], -1)
        else:
            reward = np.atleast_1d(reward)
            if reward.shape[-1] != self.dim_reward[0]:
                reward = np.tile(reward, (self.dim_reward[0], 1)).T
        return next_state, reward, done, info

    def reset(self) -> torch.Tensor:
        self._state = self.reset_()
        observation = self._state.clone()
        self._time = 0
        return observation

    @abstractmethod
    def step_(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment
        Returns:
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        raise NotImplementedError

    @abstractmethod
    def reset_(self) -> torch.Tensor:
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    @abstractmethod
    def done(self, obs: torch.Tensor) -> bool:
        """
        Check if the episode is done.
        Outputs
        -------
        done : a boolean, indicating whether the episode has ended
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Perform any necessary cleanup before closing the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        """
        raise NotImplementedError

    def reward(self, obs: torch.Tensor, act: torch.Tensor, obs_next: torch.Tensor) -> float:
        return self._reward_model(obs, act, obs_next)[0].item()

    def reward_model(self) -> TemplateRewardModel:
        return self._reward_model.copy()

    @property
    def goal(self):
        """
        Return the goal of the environment for goal-conditioned tasks.
        """
        raise NotImplementedError

    @property
    def time(self) -> float:
        return self._time

    @property
    def state(self) -> torch.Tensor:
        return self._state

    @state.setter
    def state(self, value: torch.Tensor):
        self._state = value


class TemplateGymEnv(GymEnvironment):
    """
    Template Environment Class for OpenAI Gym environments.
    Implement the abstract methods from AbstractEnvironment to create a custom environment.

    For more details, refer to the documentation of the AbstractEnvironment class.

    For a simple example class, refer to the GymEnvironment class.
    """
    def __init__(self, *args, **kwargs):
        super(TemplateGymEnv, self).__init__(*args, **kwargs)
        self._reward_model = TemplateRewardModel()
        self._state = None
        self._time = 0

