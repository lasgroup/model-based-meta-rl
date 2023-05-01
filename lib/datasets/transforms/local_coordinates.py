"""Implementation of a Transformation that converts the global position and velocity into local coordinates."""

import torch
import torch.jit

from rllib.dataset.transforms.abstract_transform import AbstractTransform
from rllib.util.utilities import safe_cholesky


class LocalCoordinates(AbstractTransform):

    def __init__(self):
        super().__init__()

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        assert observation.state.shape[-1] == 6, "Local Transform can only be used for RCCar environment!"

        global_pose = observation.state[..., :3].clone().detach()
        observation.state = self.transform_state(observation.state, global_pose)
        if (observation.next_state is not None) and (observation.next_state.shape == observation.state.shape):
            observation.next_state = self.transform_state(observation.next_state, global_pose)
            observation.next_state = observation.next_state - observation.state
        if ((observation.next_state_scale_tril is not None) and
                (observation.next_state_scale_tril.shape[:-1] == observation.state.shape)):
            observation.next_state_scale_tril = self.transform_scale_tril(
                observation.next_state_scale_tril,
                global_pose
            )
        observation.transform_info = {"global_pose": global_pose}

        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        assert (hasattr(observation, "transform_info")) and ("global_pose" in observation.transform_info.keys()), \
            "Cannot find information for performing inverse. Please check calling functions!"

        transform = observation.transform_info["global_pose"]
        transform_inv = self.transform2d(torch.zeros_like(observation.state[..., :3]), transform)
        if (observation.next_state is not None) and (observation.next_state.shape == observation.state.shape):
            observation.next_state = observation.next_state + observation.state
            observation.next_state = self.transform_state(observation.next_state, transform_inv)
        if ((observation.next_state_scale_tril is not None) and
                (observation.next_state_scale_tril.shape[:-1] == observation.state.shape)):
            observation.next_state_scale_tril = self.transform_scale_tril(
                observation.next_state_scale_tril,
                transform_inv
            )
        observation.state = self.transform_state(observation.state, transform_inv)
        delattr(observation, "transform_info")

        return observation

    def transform_state(self, state, transform):
        transformed_pos = self.transform2d(state[..., :3], transform=transform, translate=True)
        transformed_vel = self.transform2d(state[..., 3:], transform=transform, translate=False)

        return torch.cat([transformed_pos, transformed_vel], dim=-1)

    def transform_scale_tril(self, scale_tril, transform):
        # covariance = scale_tril @ scale_tril.transpose(-1, -2)
        # std = torch.sqrt(covariance.diagonal(dim1=-2, dim2=-1))
        std = scale_tril.diagonal(dim1=-2, dim2=-1)
        transformed_pos_std = self.transform2d(std[..., :3], transform=transform, translate=False)
        transformed_vel_std = self.transform2d(std[..., 3:], transform=transform, translate=False)
        transformed_var = torch.square(torch.cat([transformed_pos_std, transformed_vel_std], dim=-1))
        return safe_cholesky(torch.diag_embed(transformed_var))

    @staticmethod
    def transform2d(pos, transform, translate=True):
        x, y, theta = transform[..., 0], transform[..., 1], transform[..., 2]
        if translate:
            dx, dy, dtheta = pos[..., 0] - x, pos[..., 1] - y, pos[..., 2] - theta
        else:
            dx, dy, dtheta = pos[..., 0], pos[..., 1], pos[..., 2]
        theta = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi
        phi = torch.atan2(dy, dx) - theta
        d = torch.sqrt(dx ** 2 + dy ** 2)
        transformed_pos = torch.stack([d * torch.cos(phi), d * torch.sin(phi), dtheta], dim=-1)

        return transformed_pos
