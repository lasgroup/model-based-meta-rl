import torch
from rllib.dataset.transforms import AbstractTransform


class ActionStacking(AbstractTransform):

    def __init__(self, action_dim: int, action_stacking_dim: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.action_stacking_dim = action_stacking_dim

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        assert observation.state.shape[-1] == 6 + self.action_stacking_dim, \
            "Action stacking can only be used for RCCar environment!"
        if self.action_stacking_dim > 0:
            if (observation.next_state is not None) and (observation.next_state.shape == observation.state.shape):
                observation.next_state = observation.next_state[..., :-self.action_stacking_dim]
        return observation

    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        assert observation.state.shape[-1] == 6 + self.action_stacking_dim, \
            "Action stacking can only be used for RCCar environment!"

        if self.action_stacking_dim > 0:
            if (observation.next_state is not None) and (observation.next_state.shape[:-1] == observation.state.shape[:-1]):
                action_buffer = observation.state[..., -self.action_stacking_dim:].clone()
                action_buffer = self.update_buffer(action_buffer, observation.action)
                observation.next_state = torch.cat([observation.next_state, action_buffer], dim=-1)
            if ((observation.next_state_scale_tril is not None) and
                    (observation.next_state_scale_tril.shape[:-2] == observation.state.shape[:-1])):
                std = observation.next_state_scale_tril.diagonal(dim1=-2, dim2=-1)
                std = torch.cat([std, 1e-9 * torch.ones(*observation.state.shape[:-1], self.action_stacking_dim)], dim=-1)
                observation.next_state_scale_tril = torch.diag_embed(std)
        return observation

    def update_buffer(self, action_buffer, action):
        action_buffer[..., :-self.action_dim] = action_buffer[..., self.action_dim:].clone()
        action_buffer[..., -self.action_dim:] = action[..., :self.action_dim]
        return action_buffer
