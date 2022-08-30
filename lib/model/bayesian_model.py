"""Dynamical Model parametrized with a (P or D) ensemble of Neural Networks."""

import numpy as np
import torch
import torch.jit

from rllib.model import NNModel
from rllib.model.utilities import PredictionStrategy

from lib.neural_networks.bayesian_neural_network import FeedForwardBNN


class BayesianNNModel(NNModel):
    """Bayesian Neural Network Model.

    Parameters
    ----------
    num_particles: int.
        Number of particles for the bayesian posterior.
    prediction_strategy: str, optional (default=moment_matching).
        String that indicates how to compute the predictions of the Bayesian NN.
    deterministic: bool, optional (default=False).
        Bool that indicates if the bayesian networks are probabilistic or deterministic.

    Other Parameters
    ----------------
    See NNModel.
    """
    def __init__(
        self,
        num_particles=10,
        prediction_strategy="moment_matching",
        deterministic=False,
        *args,
        **kwargs,
    ):
        super().__init__(deterministic=False, *args, **kwargs)
        self.num_particles = num_particles

        self.nn = torch.nn.ModuleList(
            [
                FeedForwardBNN(
                    num_particles=num_particles,
                    prediction_strategy=prediction_strategy,
                    deterministic=deterministic,
                    **model.kwargs,
                )
                for model in self.nn
            ]
        )

    def sample_posterior(self) -> None:
        """Set a posterior particle."""
        self.set_head(np.random.choice(self.num_particles))

    def scale(self, state, action):
        """Get epistemic variance at a state-action pair."""
        with PredictionStrategy(self, prediction_strategy="moment_matching"):
            _, scale = self.forward(state, action[..., : self.dim_action[0]])

        return scale

    def get_avg_log_likelihood(self, y_pred, y_target):
        return self.nn[0].get_avg_log_likelihood(y_pred, y_target)

    def get_prior_prob(self):
        return self.nn[0].get_prior_prob()

    def kernel(self, X1, X2):
        return self.nn[0].kernel(X1, X2)

    def forward_nn(self, state, action, next_state=None):
        """Get Next-State distribution."""
        state_action = self.state_actions_to_input_data(state, action)
        mean = self.nn[0].forward_nn(state_action)
        return mean

    @property
    def sqrt_mode(self):
        return self.nn[0].sqrt_mode

    @property
    def particles(self):
        return self.nn[0].particles

    @torch.jit.export
    def set_head(self, head_ptr: int):
        """Set posterior particle."""
        for nn in self.nn:
            nn.set_head(head_ptr)

    @torch.jit.export
    def get_head(self) -> int:
        """Get posterior particle."""
        return self.nn[0].get_head()

    @torch.jit.export
    def set_head_idx(self, head_ptr):
        """Set posterior particle"""
        for nn in self.nn:
            nn.set_head_idx(head_ptr)

    @torch.jit.export
    def get_head_idx(self):
        """Get posterior particle index."""
        return self.nn[0].get_head_idx()

    @torch.jit.export
    def set_prediction_strategy(self, prediction: str):
        """Set prediction strategy."""
        for nn in self.nn:
            nn.set_prediction_strategy(prediction)

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get prediction strategy."""
        return self.nn[0].get_prediction_strategy()

    @property
    def name(self):
        """Get Model name."""
        return f"{'Deterministic' if self.deterministic else 'Probabilistic'} Ensemble"

    def parameters(self, recurse: bool = True):
        for parameter in self.nn[0].parameters():
            yield parameter
