import torch
import numpy as np


class RBFKernel(torch.nn.Module):
    """
    RBF kernel

    :math:`K(x, y) = exp(||x-v||^2 / (2h))

    """

    def __init__(self, bandwidth=None):
        super().__init__()
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        # Apply the median heuristic (PyTorch does not give true median)
        if self.bandwidth is None:
            np_dnorm2 = norm_sq.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(np_dnorm2.shape[0] + 1))
            return np.sqrt(h).item()
        else:
            return self.bandwidth

    def forward(self, x, y):
        dnorm2 = squared_norm(x, y)
        bandwidth = self._bandwidth(dnorm2)
        gamma = 1.0 / (1e-8 + 2 * bandwidth ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


def squared_norm(x, y):
    xx = x.matmul(x.t())
    xy = x.matmul(y.t())
    yy = y.matmul(y.t())
    return -2 * xy + xx.diag().unsqueeze(1) + yy.diag().unsqueeze(0)
