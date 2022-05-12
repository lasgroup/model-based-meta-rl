import torch
from torch import nn
from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance


# TODO: From H-UCRL
class PendulumReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0):
        super().__init__(dim_state=(2,), dim_action=(1,), model_kind="rewards")
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-0.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_cost + action_cost

        return cost.unsqueeze(-1), torch.zeros(1)


# TODO: From H-UCRL
class PendulumStateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat(
            (torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1
        )
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((angle, angular_velocity), dim=-1)
        return states_