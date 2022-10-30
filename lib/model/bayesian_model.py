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
        num_particles=15,
        prediction_strategy="moment_matching",
        deterministic=False,
        likelihood_std=0.1,
        include_aleatoric_uncertainty=False,
        *args,
        **kwargs,
    ):
        super().__init__(deterministic=False, *args, **kwargs)
        self.num_particles = num_particles

        nn_kwargs = kwargs.copy()
        for model in self.nn:
            nn_kwargs.update(model.kwargs)
        self.nn = torch.nn.ModuleList(
            [
                FeedForwardBNN(
                    num_particles=num_particles,
                    prediction_strategy=prediction_strategy,
                    deterministic=deterministic,
                    likelihood_std=likelihood_std,
                    include_aleatoric_uncertainty=include_aleatoric_uncertainty,
                    **nn_kwargs,
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

    def get_model_config(self):
        config = {
            "dim_state": self.dim_state,
            "dim_action": self.dim_action,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
        }
        config.update(self.nn[0].kwargs)
        return config

    def named_parameters(self):
        return self.nn[0].named_parameters()

    def parameter_shapes(self):
        return self.nn[0].parameter_shapes()

    def parameters_as_vector(self):
        return self.nn[0].parameters_as_vector()

    def set_prior(self, prior):
        self.nn[0].set_prior(prior)

    def set_parameters_as_vector(self, value):
        self.nn[0].set_parameters_as_vector(value)

    @property
    def num_parameters(self):
        return self.nn[0].num_parameters

    @property
    def n_batched_models(self):
        return self.nn[0].n_batched_models

    def get_avg_log_likelihood(self, y_pred, y_target):
        return self.nn[0].get_avg_log_likelihood(y_pred, y_target)

    def get_prior_prob(self):
        return self.nn[0].get_prior_prob()

    def kernel(self, X1, X2):
        return self.nn[0].kernel(X1, X2)

    def forward_nn(self, state, action, next_state=None, nn_params=None):
        """Get Next-State distribution."""
        state_action = self.state_actions_to_input_data(state, action)
        mean = self.nn[0].forward_nn(state_action, nn_params)
        return mean

    def normalize_state_and_action(self, state, action):
        state_action = self.state_actions_to_input_data(state, action)
        normalized_state_action = self.nn[0].normalize_input(state_action)
        return normalized_state_action[..., :state.shape[-1]], normalized_state_action[..., -action.shape[-1]:]

    def normalize_target(self, output):
        return self.nn[0].normalize_target(output)

    def set_normalization_stats(self, normalization_stats):
        self.nn[0].set_normalization_stats(normalization_stats)

    def set_particles(self, particles):
        assert self.particles.shape == particles.shape
        self.nn[0].particles = particles

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
