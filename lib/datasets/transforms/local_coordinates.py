"""Implementation of a Transformation that converts the global position and velocity into local coordinates."""
import inspect

import torch
import torch.jit

from rllib.dataset.transforms.abstract_transform import AbstractTransform
from rllib.util.utilities import safe_cholesky


class LocalCoordinates(AbstractTransform):

    def __init__(self, action_stacking_dim: int = 0):
        super().__init__()
        self.action_stacking_dim = action_stacking_dim

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""

        global_pose = observation.state[..., :3].clone().detach()
        observation.state = self.transform_state(observation.state, global_pose)
        if (observation.next_state is not None) and (observation.next_state.shape == observation.state.shape):
            observation.next_state = self.transform_state(observation.next_state, global_pose)
            observation.next_state[..., :6] = observation.next_state[..., :6] - observation.state[..., :6]
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
        transform_inv = self.transform2d(torch.zeros_like(observation.state[..., :3]), transform, inverse=True)
        if (observation.next_state is not None) and (observation.next_state.shape == observation.state.shape):
            observation.next_state[..., :6] = observation.next_state[..., :6] + observation.state[..., :6]
            observation.next_state = self.transform_state(observation.next_state, transform_inv, inverse=True)
        if ((observation.next_state_scale_tril is not None) and
                (observation.next_state_scale_tril.shape[:-1] == observation.state.shape)):
            observation.next_state_scale_tril = self.transform_scale_tril(
                observation.next_state_scale_tril,
                transform_inv
            )
        observation.state = self.transform_state(observation.state, transform_inv, inverse=True)
        delattr(observation, "transform_info")

        return observation

    def transform_state(self, state, transform, inverse=False):
        transformed_pos = self.transform2d(state[..., :3], transform=transform, translate=True, inverse=inverse)
        transformed_vel = self.transform2d(state[..., 3:6], transform=transform, translate=False, inverse=inverse)

        return torch.cat([transformed_pos, transformed_vel, state[..., 6:]], dim=-1)

    def transform_scale_tril(self, scale_tril, transform):
        # covariance = scale_tril @ scale_tril.transpose(-1, -2)
        # std = torch.sqrt(covariance.diagonal(dim1=-2, dim2=-1))
        std = scale_tril.diagonal(dim1=-2, dim2=-1)
        transformed_pos_std = self.transform2d(std[..., :3], transform=transform, translate=False)
        transformed_vel_std = self.transform2d(std[..., 3:6], transform=transform, translate=False)
        # std is always positive, so we take the absolute value
        transformed_var = torch.square(torch.cat([transformed_pos_std, transformed_vel_std, std[..., 6:]], dim=-1))
        return safe_cholesky(torch.diag_embed(transformed_var), jitter=10e-12)

    # def transform_scale_tril(self, scale_tril, transform):
    #     dimensions = scale_tril.shape[:-2]
    #     covariance = scale_tril @ scale_tril.transpose(-1, -2)
    #     C_pos = covariance[..., :2, :2]
    #     C_vel = covariance[..., 3:5, 3:5]
    #     c = torch.cos(transform[:, 2])
    #     s = torch.sin(transform[:, 2])
    #     R_1 = torch.stack((c, -s), dim=-1)
    #     R_2 = torch.stack((s, c), dim=-1)
    #     R = torch.stack((R_1, R_2), dim=-1)
    #     # R = torch.cat((c, -s, s, c), dim=-1).reshape(*dimensions, 2, 2)
    #     R_t = R.transpose(-1, -2)
    #     cov_pos = torch.matmul(R, torch.matmul(C_pos, R_t))
    #     cov_vel = torch.matmul(R, torch.matmul(C_vel, R_t))
    #     covariance[..., :2, :2] = cov_pos
    #     covariance[..., 3:5, 3:5] = cov_vel
    #     scale_tril_updated = safe_cholesky(covariance.clone(), jitter=10e-9)
    #     return scale_tril_updated

    @staticmethod
    def transform2d(pos, transform, translate=True, inverse=False):
        x, y, theta = transform[..., 0], transform[..., 1], transform[..., 2]
        if translate:
            dx, dy, dtheta = pos[..., 0] - x, pos[..., 1] - y, pos[..., 2] - theta
        else:
            dx, dy, dtheta = pos[..., 0], pos[..., 1], pos[..., 2]
        theta_norm = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi
        if translate and not inverse:
            if torch.any(torch.abs(dtheta.flatten()) > torch.pi):
                raise RuntimeError("Consecutive states too far from each other. Check data")
            # dtheta = ((dtheta + torch.pi) % (2 * torch.pi)) - torch.pi
        phi = torch.atan2(dy, dx) - theta_norm
        d = torch.sqrt(dx ** 2 + dy ** 2)
        transformed_pos = torch.stack([d * torch.cos(phi), d * torch.sin(phi), dtheta], dim=-1)

        return transformed_pos
