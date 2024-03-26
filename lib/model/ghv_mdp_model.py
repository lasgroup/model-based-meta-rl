import torch

from typing import Any, Tuple
from rllib.model import EnsembleModel, NNModel
from rllib.util.neural_networks.neural_networks import Ensemble, HeteroGaussianNN


class GHVEnsemble(Ensemble):

    def __init__(
            self,
            in_dim,
            out_dim,
            latent_dim,
            num_learn_samples,
            prior_std,
            num_heads,
            prediction_strategy="moment_matching",
            deterministic=True,
            *args,
            **kwargs,
    ):
        self.original_in_dim = in_dim
        HeteroGaussianNN.__init__(self, (in_dim[0] + latent_dim,), (out_dim[0] * num_heads,), *args, **kwargs)

        self.kwargs.update(
            out_dim=out_dim,
            num_heads=num_heads,
            prediction_strategy=prediction_strategy,
        )

        self.latent_dim = latent_dim
        self.num_learn_samples = num_learn_samples
        self.prior_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(latent_dim),
            covariance_matrix=prior_std * torch.eye(latent_dim)
        )

        self.num_heads = num_heads
        self.head_ptr = 0
        self.head_indexes = torch.zeros(1).long()
        self.deterministic = deterministic
        self.prediction_strategy = prediction_strategy

        self.latent_mean = torch.nn.parameter.Parameter(torch.zeros(latent_dim), requires_grad=True)
        self.latent_std_diag = torch.nn.parameter.Parameter(torch.zeros(latent_dim), requires_grad=True)
        self.reset_latent_dist()

    def forward(self, x):
        z = self.sample_latent(x.shape[:-1])
        x_z = torch.cat((x, z), dim=-1)
        return super().forward(x_z)

    def sample_latent(self, sample_shape):
        distribution = torch.distributions.normal.Normal(
            loc=self.latent_mean,
            scale=torch.nn.functional.softplus(self.latent_std_diag)
        )
        return distribution.rsample(sample_shape)

    def reset_latent_dist(self):
        with torch.no_grad():
            self.latent_mean.copy_(torch.zeros(self.latent_dim))
            self.latent_std_diag.copy_(torch.log(torch.exp(torch.ones(self.latent_dim)) - 1))


class GHVEnsembleModel(EnsembleModel):

    def __init__(
            self,
            latent_dim=8,
            num_learn_samples=2,
            prior_std=1.0,
            num_heads=5,
            prediction_strategy="moment_matching",
            deterministic=False,
            *args,
            **kwargs,
    ):
        NNModel.__init__(self, deterministic=False, *args, **kwargs)
        self.num_heads = num_heads

        self.nn = torch.nn.ModuleList(
            [
                GHVEnsemble(
                    num_heads=num_heads,
                    num_learn_samples=2,
                    prior_std=1.0,
                    latent_dim=latent_dim,
                    prediction_strategy=prediction_strategy,
                    deterministic=deterministic,
                    **model.kwargs,
                )
                for model in self.nn
            ]
        )

    @property
    def latent_dist_params(self):
        return [self.nn[0].latent_mean, self.nn[0].latent_std_diag]

    @property
    def num_learn_samples(self):
        return self.nn[0].num_learn_samples

    @property
    def prior_dist(self):
        return self.nn[0].prior_dist

    def reset_latent_dist(self):
        self.nn[0].reset_latent_dist()
