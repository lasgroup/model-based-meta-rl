import gym
import torch
import numpy as np
from gym.spaces import Discrete, Box
from rllib.environment import AbstractEnvironment


class GymEnvironment(AbstractEnvironment):

    def __init__(self, env_name, params, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        self.env.seed(params.seed)

        if isinstance(self.env.action_space, Discrete):
            dim_action = 1
            num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            dim_action = self.env.action_space.shape
            num_actions = -1
        else:
            raise NotImplementedError

        if isinstance(self.env.observation_space, Discrete):
            dim_state = 1,
            num_states = self.env.observation_space.n
        elif isinstance(self.env.observation_space, Box):
            dim_state = self.env.observation_space.shape[0],
            num_states = -1
        else:
            raise NotImplementedError
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super(GymEnvironment, self).__init__(
            dim_state=dim_state,
            dim_action=dim_action,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            dim_observation=self.env.observation_space.shape,
            num_states=num_states,
            num_observations=num_states,
            num_actions=num_actions
        )
        self._time = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
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

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def reset(self):
        self._time = 0
        return self.env.reset()

    def close(self):
        return self.env.close()

    @property
    def goal(self):
        if hasattr(self.env, "goal"):
            return self.env.goal
        return None

    @property
    def state(self):
        if hasattr(self.env, "state"):
            return self.env.state
        elif hasattr(self.env, "s"):
            return self.env.s
        else:
            raise NotImplementedError

    @state.setter
    def state(self, value):
        if hasattr(self.env, "set_state"):
            if hasattr(self.env, "sim"):  # mujocopy environments.
                self.env.set_state(
                    value[: len(self.env.sim.data.qpos)],
                    value[len(self.env.sim.data.qpos):],
                )
            else:
                self.env.set_state(value)
        elif hasattr(self.env, "state"):
            self.env.state = value
        elif hasattr(self.env, "s"):
            self.env.s = value
        else:
            raise NotImplementedError

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time
