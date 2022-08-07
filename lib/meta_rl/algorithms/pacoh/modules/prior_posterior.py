import math
import torch


class GaussianPrior(torch.nn.Module):

    def __init__(self, nn_param_size, nn_prior_std, likelihood_param_size=0,
                 likelihood_prior_mean=math.log(0.1), likelihood_prior_std=1.0):
        super().__init__()

        self.nn_param_size = nn_param_size
        self.likelihood_param_size = likelihood_param_size

        nn_prior_mean = torch.zeros(nn_param_size)
        nn_prior_std = torch.ones(nn_param_size) * nn_prior_std
        self.prior_dist_nn = torch.distributions.Independent(
            torch.distributions.Normal(nn_prior_mean, nn_prior_std),
            reinterpreted_batch_ndims=1
        )

        # mean and std of the Normal distribution over the log_std of the likelihood
        likelihood_prior_mean = torch.ones(likelihood_param_size) * likelihood_prior_mean
        likelihood_prior_std = torch.ones(likelihood_param_size) * likelihood_prior_std
        self.prior_dist_likelihood = torch.distributions.Independent(
            torch.distributions.Normal(likelihood_prior_mean, likelihood_prior_std),
            reinterpreted_batch_ndims=1
        )

    def sample(self, size):
        return torch.cat([self.prior_dist_nn.sample(size), self.prior_dist_likelihood.sample(size)], dim=-1)

    def log_prob(self, param_values, model_params_prior_weight=1.0):
        nn_params, likelihood_params = self._split_params(param_values)
        log_prob_nn = self.prior_dist_nn.log_prob(nn_params)
        log_prob_likelihood = self.prior_dist_likelihood.log_prob(likelihood_params)
        return model_params_prior_weight * log_prob_nn + log_prob_likelihood

    def _split_params(self, params):
        assert params.shape[-1] == self.nn_param_size + self.likelihood_param_size
        nn_params, likelihood_params = torch.split(params, [self.nn_param_size, self.likelihood_param_size], dim=-1)
        return nn_params, likelihood_params
