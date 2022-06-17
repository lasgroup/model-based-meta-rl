"""
Adapted from ProMP: Proximal Meta-Policy Search by Jonas Rothfuss
https://github.com/jonasrothfuss/ProMP
"""

from rllib.environment.abstract_environment import AbstractEnvironment

import torch
import numpy as np
from gym.spaces import Box
from rllib.reward.state_action_reward import StateActionReward


class PointEnv2D(AbstractEnvironment):

    def __init__(self):
        super(PointEnv2D, self).__init__(
            dim_state=(2,),
            dim_action=(2,),
            observation_space=Box(low=-np.inf, high=np.inf, shape=(2,)),
            action_space=Box(low=-0.1, high=0.1, shape=(2,)),
            dim_observation=(-1,),
            num_states=-1,
            num_actions=-1,
            num_observations=-1,
            dim_reward=(1,),
        )
        self._reward_model = PointEnv2DReward()
        self._time = 0

    def step(self, action):
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

    def step_(self, action):
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
        prev_state = self._state
        self._state = prev_state + torch.clip(action, -0.1, 0.1)
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = self._state.clone()
        return next_observation, reward, done, {}

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._state = 4 * (torch.rand(size=(2,)) - 0.5)
        observation = self._state.clone()
        self._time = 0
        return observation

    def done(self, obs):
        if obs.ndim == 1:
            return abs(obs[0]) < 0.01 and abs(obs[1]) < 0.01
        elif obs.ndim == 2:
            return torch.logical_and(torch.abs(obs[:, 0]) < 0.01, torch.abs(obs[:, 1]) < 0.01)

    def reward(self, obs, act, obs_next):
        return self._reward_model(obs_next, act)[0].item()

    def reward_model(self):
        return self._reward_model.copy()

    @property
    def time(self) -> float:
        return self._time

    @property
    def state(self) -> torch.Tensor:
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


class PointEnv2DReward(StateActionReward):
    """
    Reward model for the 2D point environment.
    """

    dim_action = (2,)

    def __init__(self, ctrl_cost_weight=0):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)

    def copy(self):
        """Get copy of reward model."""
        return PointEnv2DReward(
            ctrl_cost_weight=self.ctrl_cost_weight
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        if state.ndim == 1:
            return -torch.sqrt(state[0] ** 2 + state[1] ** 2)
        else:
            return -torch.sqrt(state[..., 0] ** 2 + state[..., 1] ** 2)
