import math
from collections import OrderedDict

import torch

from lib.meta_rl.algorithms.pacoh.modules.vectorized_nn import VectorizedModel


class GaussianPriorPerVariable(VectorizedModel):

    def __init__(self, nn_model, priors_idx, add_init_noise=True, init_noise_std=0.1,
                 likelihood_param_size=0, likelihood_prior_mean=math.log(0.01), likelihood_prior_std=1.0,
                 name='gaussian_prior'):
        """
        Gaussian prior for a NN model
        Args:
            nn_model: Vectorized NN Module
            priors_idx: Index for prior parameters
            likelihood: likelihood object
            config (dict): configuration dict
            name (str): module name
        """
        super().__init__(input_dim=None, output_dim=None)
        self.name = name

        self.init_noise_std = init_noise_std

        self.base_variable_sizes = []
        self.base_variable_names = []

        # initialize prior associated with the model parameters
        self._init_model_prior_from_models(nn_model=nn_model, priors_idx=priors_idx, init_noise_std=init_noise_std)

        # initialize prior associated with the likelihood parameters
        self._init_likelihood_prior(likelihood_param_size, likelihood_prior_mean, likelihood_prior_std)

        # size of the model vs. likelihood parameters so that they can be separated if necessary
        self.split_sizes = [sum(v_s) for v_s in self.base_variable_sizes]

        # add noise to initialization
        if add_init_noise:
            self._add_noise_to_variables(init_noise_std)

    def sample(self, n_samples, use_init_std=False):
        """
        Generates n_samples samples from the prior distribution
        Args:
            n_samples (int): number of samples to draw
            use_init_std (bool): whether to use the init std

        Returns: Tensor of shape (n_samples, parameter_size)

        """
        mp = self._sample_param('model_parameters', n_samples, use_init_std)

        if self.learn_likelihood_variables:
            lp = self._sample_param('likelihood_parameters', n_samples, use_init_std)
            return torch.cat([mp, lp], dim=1)
        else:
            return mp

    def log_prob(self, parameters, model_params_prior_weight=1.0):
        split = torch.split(parameters, self.split_sizes, dim=-1)

        log_prob = model_params_prior_weight * self._log_prob('model_parameters', split[0])

        if self.learn_likelihood_variables:
            log_prob += self._log_prob('likelihood_parameters', split[1])

        return log_prob

    def _sample_param(self, param_name, n_samples, use_init_std):
        param_name = f'{self.name}/{param_name}'

        name = f'{param_name}_mean'
        mean = getattr(self, name)

        name = f'{param_name}_log_var'
        log_var = getattr(self, name)

        if use_init_std:
            std = torch.ones_like(log_var) * self.init_noise_std
        else:
            std = (0.5 * log_var).exp()

        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=std),
            reinterpreted_batch_ndims=1)

        return dist.rsample([n_samples])

    def _log_prob(self, param_name, param):
        param_name = f'{self.name}/{param_name}'

        name = f'{param_name}_mean'
        mean = getattr(self, name)

        name = f'{param_name}_log_var'
        log_var = getattr(self, name)
        std = (0.5 * log_var).exp()

        dist = torch.distributions.Independent(
            torch.distributions.Normal(mean, std),
            reinterpreted_batch_ndims=1)

        return dist.log_prob(param)

    def _init_model_prior_from_models(self, nn_model, priors_idx, init_noise_std=0.1):
        assert isinstance(priors_idx, torch.Tensor)

        # save variable names and sizes
        self.base_variable_names.append(list(nn_model.named_parameters().keys()))
        self.base_variable_sizes.append([shape[-1] for shape in nn_model.parameter_shapes().values()])

        means = []
        log_vars = []

        for param_name, params in nn_model.named_parameters().items():
            params = params[priors_idx]
            param_shape = params.shape[-1]

            init_std = params.std(dim=0)  # std of model initializations

            if 'weight' in param_name:
                # take the initialization of the first model as mean of the prior
                init_mean = params[0]
                # take half of the std across model initializations as std of the prior
                init_log_var = 2 * (0.5 * (init_std + 1e-8)).log()
            elif 'bias' in param_name:
                # sample prior mean of bias
                init_mean = torch.normal(mean=0.0, std=init_noise_std, size=[param_shape])
                # use init_std for the log_var
                init_log_var = torch.ones(param_shape) * 2 * math.log(0.5 * (init_noise_std + 1e-8))
            else:
                raise Exception("Unknown variable type")

            means.append(init_mean.reshape(-1,))
            log_vars.append(init_log_var.reshape(-1,))

        means = torch.cat(means, dim=0)
        log_vars = torch.cat(log_vars, dim=0)

        name = f'{self.name}/model_parameters_mean'
        setattr(self, name, torch.tensor(torch.squeeze(means), dtype=torch.float32, requires_grad=True))

        name = f'{self.name}/model_parameters_log_var'
        setattr(self, name, torch.tensor(torch.squeeze(log_vars), dtype=torch.float32, requires_grad=True))

    def _init_likelihood_prior(self, likelihood_param_size, likelihood_prior_mean, likelihood_prior_std):
        self.learn_likelihood_variables = False
        if likelihood_param_size > 0:
            self.learn_likelihood_variables = True
            self.base_variable_names.append(['std'])
            self.base_variable_sizes.append([likelihood_param_size])

            mean = torch.ones(likelihood_param_size, dtype=torch.float32) * likelihood_prior_mean
            name = f'{self.name}/likelihood_parameters_mean'
            setattr(self, name, torch.tensor(mean, dtype=torch.float32, requires_grad=True))

            log_var = torch.ones(likelihood_param_size, dtype=torch.float32) * torch.log(likelihood_prior_std)
            name = f'{self.name}/likelihood_parameters_log_var'
            setattr(self, name, torch.tensor(log_var, dtype=torch.float32, requires_grad=True))

    def _add_noise_to_variables(self, init_noise_std):
        # exclude mean and log_var for model parameters, since they have been initialized from the models
        # --> they are already noisy
        if self.learn_likelihood_variables:
            for v in [f'{self.name}/likelihood_parameters_mean', f'{self.name}/likelihood_parameters_log_var']:
                noise = torch.normal(mean=0.0, std=init_noise_std, size=getattr(self, v).shape)
                setattr(self, v, getattr(self, v) + noise)

    def parameter_shapes(self):
        param_shapes = OrderedDict(
            {
                f'{self.name}/model_parameters_mean': getattr(self, f'{self.name}/model_parameters_mean').shape,
                f'{self.name}/model_parameters_log_var': getattr(self, f'{self.name}/model_parameters_log_var').shape,
            }
        )
        if self.learn_likelihood_variables:
            param_shapes[f'{self.name}/likelihood_parameters_mean'] = getattr(self, f'{self.name}/likelihood_parameters_mean').shape
            param_shapes[f'{self.name}/likelihood_parameters_log_var'] = getattr(self, f'{self.name}/likelihood_parameters_log_var').shape

        return param_shapes

    def named_parameters(self):
        named_params = OrderedDict(
            {
                f'{self.name}/model_parameters_mean': getattr(self, f'{self.name}/model_parameters_mean'),
                f'{self.name}/model_parameters_log_var': getattr(self, f'{self.name}/model_parameters_log_var')
            }
        )
        if self.learn_likelihood_variables:
            named_params[f'{self.name}/likelihood_parameters_mean'] = getattr(self, f'{self.name}/likelihood_parameters_mean')
            named_params[f'{self.name}/likelihood_parameters_log_var'] = getattr(self, f'{self.name}/likelihood_parameters_log_var')

        return named_params


class BatchedGaussianPrior(VectorizedModel):

    def __init__(self, batched_model, n_batched_priors, likelihood_param_size=0, likelihood_prior_mean=math.log(0.1),
                 likelihood_prior_std=1.0, add_init_noise=True, init_noise_std=0.1, name='batched_Gaussian_prior'):
        """
        Batched Gaussian priors for the model
        Args:
            batched_model: a batched NN model for which to instantiate a prior
            likelihood: likelihood object
            config (dict): configuration for the prior
            name (str): name of the module
        """
        super().__init__(input_dim=None, output_dim=None)

        self.name = name
        self.n_batched_priors = n_batched_priors
        self.priors = []

        n_models = batched_model.n_batched_models
        self.n_models_per_prior = n_models // self.n_batched_priors
        assert self.n_models_per_prior > 0

        for i in range(self.n_batched_priors):
            params_idx = torch.arange(i*self.n_models_per_prior, (i+1)*self.n_models_per_prior)
            self.priors.append(GaussianPriorPerVariable(nn_model=batched_model, priors_idx=params_idx,
                                                        add_init_noise=add_init_noise, init_noise_std=init_noise_std,
                                                        likelihood_param_size=likelihood_param_size,
                                                        likelihood_prior_mean=likelihood_prior_mean,
                                                        likelihood_prior_std=likelihood_prior_std,
                                                        name=f'{self.name}_{i}'))

        self.split_sizes = self.priors[0].split_sizes
        self._variable_sizes = None

    def log_prob(self, parameters, model_params_prior_weight=1.0):
        # parameters should have dimensions [#priors, #samples_per_prior, #params]
        log_prob = torch.stack([self.priors[i].log_prob(parameters[i], model_params_prior_weight=model_params_prior_weight)
                             for i in range(self.n_batched_priors)])
        return log_prob

    def sample(self, n_samples, use_init_std=False):
        sample = torch.stack([self.priors[i].sample(n_samples, use_init_std)
                           for i in range(self.n_batched_priors)])
        return sample

    def sample_parametrized(self, n_samples, variables_vectorized):
        assert variables_vectorized.ndim == 3
        assert variables_vectorized.shape[1] == self.n_batched_priors
        for i in range(self.n_batched_priors):
            self.priors[i].set_parameters_as_vector(variables_vectorized[:, i].reshape(-1))
        samples = self.sample(n_samples)
        return samples

    def parameter_shapes(self):
        param_dict = OrderedDict()
        for i in range(len(self.priors)):
            for name, param in self.priors[i].parameter_shapes().items():
                param_dict[name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()
        for i in range(len(self.priors)):
            for name, param in self.priors[i].named_parameters().items():
                param_dict[name] = param

        return param_dict

    def get_parameters_stacked_per_prior(self):
        parameters = self.parameters_as_vector()
        assert parameters.ndim == 1
        return parameters.reshape((self.n_batched_priors, -1))


class GaussianPosterior(torch.nn.Module):
    def __init__(self, stacked_nn_init_params, likelihood_param_size=0):
        super().__init__()
        self.likelihood_param_size = likelihood_param_size

        # mean & std for nn params
        nn_param_size = stacked_nn_init_params.shape[-1]
        mean_nn_params = torch.zeros(nn_param_size)
        post_init_std = stacked_nn_init_params.ste(dim=0) + 1.0 / nn_param_size
        log_stddev_nn_params = post_init_std

        # mean & std for likelihood params
        mean_likelihood_params = -2 * torch.ones(likelihood_param_size)
        log_stddev_likelihood_params = torch.ones(likelihood_param_size)

        self.mean = torch.tensor(torch.cat([mean_nn_params, mean_likelihood_params], dim=0), requires_grad=True)
        self.log_std = torch.tensor(
            torch.cat([log_stddev_nn_params, log_stddev_likelihood_params], dim=0).log(), requires_grad=True)

    @property
    def stddev(self):
        return self.log_std.exp()

    @property
    def dist(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.mean, self.log_std.exp()), reinterpreted_batch_ndims=1)

    def sample(self, size):
        return self.dist.sample(size)

    def log_prob(self, param_values):
        return self.dist.log_prob(param_values)


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
