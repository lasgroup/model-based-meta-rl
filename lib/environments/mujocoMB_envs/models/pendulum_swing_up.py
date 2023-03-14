import torch
from torch import nn

from rllib.environment.mujoco.pendulum_swing_up import PendulumSwingUpEnv


class PendulumSwingUpModel(nn.Module):

    def __init__(self, env: PendulumSwingUpEnv):
        super().__init__()
        self.max_speed = env.max_speed
        self.max_torque = env.max_torque
        self.dt = env.dt
        self.g = env.g
        self.m = env.m
        self.l = env.l

    def forward(self, state, action, next_state=None):
        action = torch.clamp(action, min=-self.max_torque, max=self.max_torque)[..., 0]

        theta = torch.arctan2(state[..., 1], state[..., 0])

        inertia = self.m * self.l ** 2 / 3.0
        omega_dot = -3 * self.g / (2 * self.l) * torch.sin(theta + torch.pi) + action / inertia

        new_omega = state[..., 2] + omega_dot * self.dt
        new_theta = theta + new_omega * self.dt

        new_omega = torch.clamp(new_omega, min=-self.max_speed, max=self.max_speed)

        new_state = torch.stack([torch.cos(new_theta), torch.sin(new_theta), new_omega], dim=-1)

        return new_state, torch.zeros_like(new_state)
