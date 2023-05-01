import torch
import numpy as np

from rllib.util.rollout import step_model

from lib.datasets.utils import sample_states
from lib.environments.wrappers.model_based_environment import ModelBasedEnvironment


class RCCarModelBasedEnvironment(ModelBasedEnvironment):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._goal = self.reward_model.goal
        self.pos_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(2),
            covariance_matrix=0.8*torch.eye(2)
        )
        self.theta_dist = torch.distributions.uniform.Uniform(-torch.pi, torch.pi)

    def reset_torch(self):
        states = self.initial_states_distribution().squeeze()
        states = sample_states(states, self.num_envs)
        states = self.sample_pose(states)
        self.state = states
        self._time = 0
        self.return_vals = None
        return states

    def sample_pose(self, states):
        pos_samples = self.pos_dist.rsample(states.shape[:-1])
        theta_samples = self.theta_dist.rsample(states.shape[:-1]).reshape(-1, 1)
        pose = torch.cat([pos_samples, theta_samples], dim=-1)
        states[..., :3] = pose
        return states

    def reset(self):
        return self.reset_torch().detach().numpy()

    def step(self, action):

        info = [{"TimeLimit.truncated": False}] * self.num_envs
        action = self.action_scale.numpy() * action.clip(-1.0, 1.0)

        with torch.no_grad():
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

    def local_transform(self, state):
        local_goal = self.transform2d(self._goal, state[..., :3], translate=True)
        local_vel = self.transform2d(state[..., 3:], state[..., :3], translate=False)
        return np.concatenate([local_goal, local_vel], axis=-1)

    @staticmethod
    def transform2d(pos, transform, translate=True):
        x, y, theta = transform[..., 0], transform[..., 1], transform[..., 2]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        if translate:
            dx, dy, dtheta = pos[..., 0] - x, pos[..., 1] - y, pos[..., 2] - theta
        else:
            dx, dy, dtheta = pos[..., 0], pos[..., 1], pos[..., 2]
        dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
        phi = np.arctan2(dy, dx) - theta
        d = np.sqrt(dx ** 2 + dy ** 2)
        transformed_pos = np.stack([d * np.cos(phi), d * np.sin(phi), dtheta], axis=-1)

        return transformed_pos
