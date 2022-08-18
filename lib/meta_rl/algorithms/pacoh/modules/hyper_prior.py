import torch


class GaussianHyperPrior(torch.nn.Module):
    def __init__(self, batched_prior_module,
                 mean_mean=0.0, bias_mean_std=0.5, kernel_mean_std=0.5,
                 log_var_mean=-3.0, bias_log_var_std=0.5, kernel_log_var_std=0.5,
                 likelihood_log_var_mean_mean=-8,
                 likelihood_log_var_mean_std=1.0,
                 likelihood_log_var_log_var_mean=-4,
                 likelihood_log_var_log_var_std=0.2,
                 name='GaussianHyperPrior'):

        super().__init__()
        self.name = name

        # mean_mean (hyperprior_prior) implies mean of the mean of params(kernels and biases)
        self.mean_mean = mean_mean
        self.bias_mean_std = bias_mean_std
        self.kernel_mean_std = kernel_mean_std

        # log_var_mean (hyperprior_prior) implies log_var of the mean of params(kernels and biases)
        self.log_var_mean = log_var_mean
        self.bias_log_var_std = bias_log_var_std
        self.kernel_log_var_std = kernel_log_var_std

        self.likelihood_log_var_mean_mean = likelihood_log_var_mean_mean
        self.likelihood_log_var_mean_std = likelihood_log_var_mean_std
        self.likelihood_log_var_log_var_mean = likelihood_log_var_log_var_mean
        self.likelihood_log_var_log_var_std = likelihood_log_var_log_var_std

        self.prior_module_names = [prior.name for prior in batched_prior_module.priors]
        self.n_batched_priors = len(self.prior_module_names)

        prior_module = batched_prior_module.priors[0]
        variables = prior_module.named_parameters()

        self.batched_variable_sizes = list(batched_prior_module.parameter_shapes().values())
        self.batched_variable_names = list(batched_prior_module.named_parameters().keys())

        self.base_variable_sizes = prior_module.base_variable_sizes
        self.base_variable_names = prior_module.base_variable_names

        for name, var in variables.items():
            self.process_variable(name, var)

    def process_variable(self, name, var):
        if 'model_parameters' in name:
            sizes = self.base_variable_sizes[0]
            names = self.base_variable_names[0]

            if 'mean' in name:
                # prior for model_parameters_mean
                means = []
                stds = []

                for v_size, v_name in zip(sizes, names):
                    mean = torch.ones((1, v_size), dtype=torch.float32) * self.mean_mean

                    if 'bias' in v_name:
                        std = torch.ones((1, v_size), dtype=torch.float32) * self.bias_mean_std
                    elif 'weight' in v_name:
                        std = torch.ones((1, v_size), dtype=torch.float32) * self.kernel_mean_std
                    else:
                        raise Exception("Unexpected parameter")

                    means.append(mean)
                    stds.append(std)

                means = torch.cat(means, dim=1)
                means = means.type(torch.float32).squeeze()
                stds = torch.cat(stds, dim=1)
                stds = stds.type(torch.float32).squeeze()

                if means.ndim == 0:
                    means = means.unsqueeze(dim=0)

                dist = torch.distributions.Independent(
                    torch.distributions.Normal(means, stds), reinterpreted_batch_ndims=1)

                def log_prob(parameters):
                    return dist.log_prob(parameters)

            elif 'log_var' in name:
                # prior for model_parameters_mean
                means = []
                stds = []

                for v_size, v_name in zip(sizes, names):
                    mean = torch.ones((1, v_size), dtype=torch.float32) * self.log_var_mean

                    if 'bias' in v_name:
                        std = torch.ones((1, v_size), dtype=torch.float32) * self.bias_log_var_std
                    elif 'weight' in v_name:
                        std = torch.ones((1, v_size), dtype=torch.float32) * self.kernel_log_var_std
                    else:
                        raise Exception("Unexpected parameter")

                    means.append(mean)
                    stds.append(std)

                means = torch.cat(means, dim=1)
                means = means.type(torch.float32).squeeze()
                stds = torch.cat(stds, dim=1)
                stds = stds.type(torch.float32).squeeze()

                if means.ndim == 0:
                    means = means.unsqueeze(dim=0)

                dist = torch.distributions.Independent(
                    torch.distributions.Normal(means, stds), reinterpreted_batch_ndims=1)

                def log_prob(parameters):
                    return dist.log_prob(parameters)

            else:
                raise Exception("Unexpected variable name")

        elif 'likelihood_parameters' in name:
            sizes = self.base_variable_sizes[1]

            if 'mean' in name:
                # prior for likelihood_parameters_mean
                means = torch.ones(sizes[0], dtype=torch.float32) * self.likelihood_log_var_mean_mean
                stds = torch.ones(sizes[0], dtype=torch.float32) * self.likelihood_log_var_mean_std

                means = torch.cat(means, dim=1)
                means = means.type(torch.float32).squeeze()
                stds = torch.cat(stds, dim=1)
                stds = stds.type(torch.float32).squeeze()

                if means.ndim == 0:
                    means = means.unsqueeze(dim=0)

                dist = torch.distributions.Independent(
                    torch.distributions.Normal(means, stds), reinterpreted_batch_ndims=1)

                def log_prob(parameters):
                    return dist.log_prob(parameters)

            elif 'log_var' in name:
                # prior for likelihood_parameters_log_var
                means = torch.ones(sizes[0], dtype=torch.float32) * self.likelihood_log_var_log_var_mean
                stds = torch.ones(sizes[0], dtype=torch.float32) * self.likelihood_log_var_log_var_std
                means = torch.cat(means, dim=1)
                means = means.type(torch.float32).squeeze()
                stds = torch.cat(stds, dim=1)
                stds = stds.type(torch.float32).squeeze()

                if means.ndim == 0:
                    means = means.unsqueeze(dim=0)

                dist = torch.distributions.Independent(
                    torch.distributions.Normal(means, stds), reinterpreted_batch_ndims=1)

                def log_prob(parameters):
                    return dist.log_prob(parameters)

            else:
                raise Exception("Unexpected variable name")

        else:
            raise Exception("Unexpeted variable name")

        suffix = name.split('/')[1]
        for prior_module_name in self.prior_module_names:
            var_name = f'{prior_module_name}/{suffix}'
            setattr(self, var_name, log_prob)

    def log_prob_vectorized(self, params_vectorized, model_params_prior_weight=1.0):
        assert params_vectorized.shape[1] == self.n_batched_priors
        parameters = params_vectorized.reshape((-1,))
        param_split = torch.split(parameters, [int(size[0]) for size in self.batched_variable_sizes])
        log_probs = torch.cat(
            [getattr(self, vname)(v).unsqueeze(0) for v, vname in zip(param_split, self.batched_variable_names)], dim=0
        ).reshape((self.n_batched_priors, -1))
        prefactor = torch.tensor(
            [model_params_prior_weight if 'model' in var_name else 1.0 for var_name in self.batched_variable_names]
        ).reshape((self.n_batched_priors, -1))
        log_probs *= prefactor
        log_probs = log_probs.sum(dim=-1)
        return log_probs

    def log_prob(self, variables):
        log_probs = torch.cat([getattr(self, v.name)(v) for v in variables], dim=-1).reshape((self.n_batched_priors, -1))
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        return log_probs

