from rllib.environment.abstract_environment import AbstractEnvironment

import torch
import numpy as np
from gym.spaces import Box
from rllib.reward.state_action_reward import StateActionReward


class RCCarDemoEnv(AbstractEnvironment):

    wait_time = 0.03
    max_steps = 500

    def __init__(self):
        super(RCCarDemoEnv, self).__init__(
            dim_state=(6,),
            dim_action=(2,),
            observation_space=Box(low=-np.inf, high=np.inf, shape=(2,)),
            action_space=Box(low=-1, high=1, shape=(2,)),
            dim_observation=(-1,),
            num_states=-1,
            num_actions=-1,
            num_observations=-1,
            dim_reward=(1,),
        )
        self._goal = np.zeros(3)
        self._reward_model = RCCarEnvReward(goal=self._goal)
        self._time = 0

        self._prev_time = None
        self._state = None
        self._pos = None
        self._vel = None

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
        prev_state = self._state
        self._state = self.get_state()
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = self._state.copy()
        return next_observation, reward, done, {}

    def reset(self):
        self._state = self.get_state()
        observation = self._state.copy()
        self._time = 0
        return observation

    def done(self, obs):
        if self._time >= self.max_steps:
            return True
        return False

    def get_state(self):
        pos = np.random.rand(3)
        vel = np.random.rand(3)
        return np.concatenate([pos, vel])

    def reward(self, obs, act, obs_next):
        return self._reward_model(obs, act, obs_next)[0].item()

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


class RCCarEnvReward(StateActionReward):
    """
    Reward model for the RC Car environment.
    """

    dim_action = (2,)

    def __init__(self, goal, ctrl_cost_weight=0, speed_cost_weight=5):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.goal = goal
        self.speed_cost_weight = speed_cost_weight

    def copy(self):
        """Get copy of reward model."""
        return RCCarEnvReward(
            goal=self.goal,
            ctrl_cost_weight=self.ctrl_cost_weight,
            speed_cost_weight=self.speed_cost_weight
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        dist_cost = torch.sqrt(torch.sum(torch.square(next_state[..., :3] - self.goal), dim=-1))
        speed_cost = torch.sqrt(torch.sum(torch.square(next_state[..., 3:]), dim=-1))
        return - dist_cost - self.speed_cost_weight * speed_cost
