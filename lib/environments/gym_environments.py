import gym
from gym.spaces import Discrete, Box
from rllib.environment import AbstractEnvironment


class GymEnvironment(AbstractEnvironment):

    def __init__(self, env_name, *args, **kwargs):
        self.env = gym.make(env_name, render_mode="human", **kwargs)

        if isinstance(self.env.action_space, Discrete):
            dim_action = 1,
            num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            dim_action = self.env.action_space.shape[0]
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

        super().__init__(
            dim_state=dim_state,
            dim_action=dim_action,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            dim_observation=self.env.observation_space,
            num_states=num_states,
            num_actions=num_actions
        )

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()
