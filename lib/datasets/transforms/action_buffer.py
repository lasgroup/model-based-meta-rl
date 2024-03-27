import numpy as np
import torch
from rllib.dataset.transforms import AbstractTransform


class ActionBufferScaler(AbstractTransform):
    """Implementation of an Action Scaler.

    Given an action, it will scale it by dividing it by scale.

    Parameters
    ----------
    scale: float.

    """

    def __init__(self, scale, action_stacking_dim):
        super().__init__()
        if isinstance(scale, np.ndarray):
            assert np.all((scale == scale[0]))
            scale = scale[0]
        self.scale = scale
        self._action_stacking_dim = action_stacking_dim

    def forward(self, observation):
        """See `AbstractTransform.__call__'."""
        if self._action_stacking_dim > 0:
            action_buffer = observation.state[..., observation.state.shape[-1] - self._action_stacking_dim:]
            scaled_buffer = action_buffer / self.scale
            observation.state[..., observation.state.shape[-1] - self._action_stacking_dim:] = scaled_buffer
        return observation

    @torch.jit.export
    def inverse(self, observation):
        """See `AbstractTransform.inverse'."""
        if self._action_stacking_dim > 0:
            action_buffer = observation.state[..., observation.state.shape[-1] - self._action_stacking_dim:]
            scaled_buffer = action_buffer * self.scale
            observation.state[..., observation.state.shape[-1] - self._action_stacking_dim:] = scaled_buffer
        return observation
