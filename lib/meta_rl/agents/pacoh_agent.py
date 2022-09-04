import copy
import math
import time
from collections import deque

import numpy as np
import torch

from rllib.dataset import stack_list_of_tuples

from lib.agents import MPCAgent
from lib.datasets import TrajectoryReplay
from lib.model.bayesian_model import BayesianNNModel
from lib.model.bayesian_model_learning import train_bayesian_nn_step
from lib.meta_rl.algorithms.pacoh.modules.kernels import RBFKernel
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from lib.meta_rl.algorithms.pacoh.modules.likelihood import GaussianLikelihood
from lib.meta_rl.algorithms.pacoh.modules.hyper_prior import GaussianHyperPrior
from lib.meta_rl.algorithms.pacoh.modules.prior_posterior import BatchedGaussianPrior


# TODO:
"""
Make it work with H-UCRL
Save and Load trajectory replay
Check whether meta_fit works
Check whether task_fit works
Plot trajectories and axis
hyperparam tuning:
    num_iter_meta_train
    num_iter_meta_test
    num_particles: test and train
    per_task_batch_size
    observation_queue size
"""


class PACOHAgent(MPCAgent):
    """
    Implementation of the Meta MBRL Agent based on PACOH.

    References
    ----------
    Rothfuss, J., Fortuin, V., Josifoski, M., Krause, A.
    PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees.
    """

    def __init__(
            self,
            meta_environment: MetaEnvironmentWrapper = None,
            max_memory: int = 1000000,
            meta_lr: float = 2e-3,
            num_iter_meta_train: int = 20000,
            num_iter_meta_test=100,
            n_samples_per_prior=10,
            num_hyper_posterior_particles=3,
            num_posterior_particles=5,
            hyper_prior_weight=1e-4,
            hyper_prior_nn_std=0.4,
            hyper_prior_log_var_mean=-3.0,
            hyper_prior_likelihood_log_var_mean_mean=-8,
            hyper_prior_likelihood_log_var_log_var_mean=-4.,
            meta_batch_size=10,
            per_task_batch_size=32,
            eval_num_context_samples=16,
            *args,
            **kwargs
    ):
        super().__init__(
            max_memory=max_memory,
            *args,
            **kwargs
        )

        self.meta_environment = meta_environment

        self.dataset = TrajectoryReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.transformations
        )
        self.observation_queue = deque([], eval_num_context_samples)

        self.hyper_prior_weight = hyper_prior_weight

        # data handling & settings
        self.n_batched_models_train = num_hyper_posterior_particles * n_samples_per_prior
        self.n_batched_models_test = num_hyper_posterior_particles * num_posterior_particles
        self.n_samples_per_prior = n_samples_per_prior
        self.n_batched_priors = num_hyper_posterior_particles

        # get config from bayesian model
        self.eval_model_config = self.dynamical_model.base_model.get_model_config()
        assert self.eval_model_config["num_particles"] == num_posterior_particles * self.n_batched_priors

        self.input_dim = self.eval_model_config['in_dim'][0]
        self.output_dim = self.eval_model_config['out_dim'][0]

        self.num_iter_meta_test = num_iter_meta_test
        self.num_iter_meta_train = num_iter_meta_train
        self.meta_batch_size = meta_batch_size
        self.per_task_batch_size = per_task_batch_size
        self.eval_num_context_samples = eval_num_context_samples

        self.mll_pre_factor = self._compute_mll_prefactor()

        # setup NN
        train_model_config = self.eval_model_config
        train_model_config.update({"num_particles": self.n_batched_models_train})
        self.meta_nn_model = BayesianNNModel(**train_model_config)
        # if isinstance(self.meta_nn_model, HallucinatedModel):
        #     self.meta_nn_model = HallucinatedModel(
        #     self.meta_nn_model,
        #     self.dynamical_model.transformations,
        #     beta=self.dynamical_model.beta
        # )
        # elif isinstance(self.meta_nn_model, TransformedModel):
        #     self.meta_nn_model = TransformedModel(
        #     self.meta_nn_model,
        #     self.dynamical_model.transformations
        # )

        # setup likelihood
        self.likelihood = GaussianLikelihood(self.output_dim, self.n_batched_models_train,
                                             std=self.eval_model_config["likelihood_std"],
                                             trainable=self.eval_model_config["learn_likelihood"])
        self.learn_likelihood = self.eval_model_config["learn_likelihood"]
        self.likelihood_std = self.eval_model_config["likelihood_std"]

        # setup prior
        self.nn_param_size = self.meta_nn_model.num_parameters
        self.likelihood_param_size = self.output_dim if self.learn_likelihood else 0

        self.prior_module = BatchedGaussianPrior(
            batched_model=self.meta_nn_model, n_batched_priors=self.n_batched_priors,
            likelihood_param_size=self.likelihood_param_size,
            likelihood_prior_mean=math.log(0.01),
            likelihood_prior_std=1.0,
            name='gaussian_prior'
        )

        self.hyper_prior_module = GaussianHyperPrior(
            self.prior_module,
            mean_mean=0.0, bias_mean_std=hyper_prior_nn_std,
            kernel_mean_std=hyper_prior_nn_std,
            log_var_mean=hyper_prior_log_var_mean,
            bias_log_var_std=hyper_prior_nn_std,
            kernel_log_var_std=hyper_prior_nn_std,
            likelihood_log_var_mean_mean=hyper_prior_likelihood_log_var_mean_mean,
            likelihood_log_var_mean_std=1.0,
            likelihood_log_var_log_var_mean=hyper_prior_likelihood_log_var_log_var_mean,
            likelihood_log_var_log_var_std=0.2
        )

        self.hyper_posterior_particles = self.prior_module.get_parameters_stacked_per_prior().detach().unsqueeze(dim=0)
        self.hyper_posterior_particles.requires_grad = True

        self.kernel = RBFKernel(bandwidth=self.eval_model_config["bandwidth"])

        optimizer_params = self.model_optimizer.defaults
        optimizer_params.update({'lr': meta_lr})
        self.meta_optimizer = type(self.model_optimizer)([self.hyper_posterior_particles], **optimizer_params)

    def set_meta_environment(self, meta_environment):
        self.meta_environment = meta_environment

    def observe(self, observation):
        transformed_observation = observation.clone()
        for transform in self.dynamical_model.transformations:
            transformed_observation = transform(transformed_observation)
        self.observation_queue.append(transformed_observation)
        super().observe(observation)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self.fit_task()
        return super().act(state)

    def start_episode(self):
        assert self.meta_environment is not None, "Meta training environment has not been set!"
        self.meta_environment.sample_next_env()
        self.dataset.start_episode()
        super().start_episode()

    def end_episode(self):
        self.dataset.end_episode()
        super().end_episode()
        self.observation_queue.clear()

    def train(self, val=True):
        """Set the agent in training mode"""
        self.meta_environment.train(val)
        super().train(val)

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.meta_environment.eval(val)
        self.meta_fit()
        super().eval(val)

    def fit_task(self):
        eval_model = BayesianNNModel(**self.eval_model_config, meta_learned_prior=self.prior_module)
        eval_model.train()
        if len(self.observation_queue) > 0:
            past_observations = stack_list_of_tuples([observation for observation in self.observation_queue.copy()])
            for num_iter in range(self.num_iter_meta_test):
                train_bayesian_nn_step(
                    model=eval_model,
                    observation=past_observations,
                    optimizer=self.model_optimizer
                )
            self.dynamical_model.base_model.set_parameters_as_vector(eval_model.parameters_as_vector())

    def get_meta_batch(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        task_batches = []
        n_train_samples = []
        task_ids = np.random.choice(dataset.num_episodes, self.meta_batch_size)
        for task_id in task_ids:
            observation, _, _ = dataset.sample_task_batch(self.per_task_batch_size, task_id)
            n_train_samples.append(dataset.trajectory_lengths[task_id])
            task_batches.append(observation)
        return task_batches, torch.tensor(n_train_samples, dtype=torch.float32)

    def meta_fit(self, log_period=500, eval_period=1000):
        """
        Fits the hyper-posterior (PACOH) particles with SVGD
        """
        print("Start meta-training -------------------- ")
        t = time.time()

        self.mll_pre_factor = self._compute_mll_prefactor()

        for it in range(self.num_iter_meta_train):

            meta_task_batch, n_train_samples = self.get_meta_batch()
            log_prob = self.step(meta_task_batch, n_train_samples)

            message = ''
            if it % log_period == 0 or it % eval_period == 0:
                avg_log_prob = log_prob.mean().numpy()
                message += '\nIter %d/%d - Time %.2f sec' % (it, self.num_iter_meta_train, time.time() - t)
                message += ' - Train-Loss: %.5f' % avg_log_prob

                # # run validation and print results
                # if it % eval_period == 0 and it > 0:
                #     eval_metrics_mean, eval_metrics_std = self.meta_eval_datasets(meta_val_data=self.dataset)
                #     for key in eval_metrics_mean:
                #         message += '- Val-%s: %.3f +- %.3f' % (key, eval_metrics_mean[key], eval_metrics_std[key])

                t = time.time()

            if len(message) > 0:
                print(message)

        loss = -log_prob.mean().numpy()
        return loss

    def step(self, meta_batch, n_train_samples):
        """
        performs one meta-training optimization step (SVGD step on the hyper-posterior)
        """
        # erase gradients
        self.meta_optimizer.zero_grad()

        # compute score (grads of log_prob)
        log_prob, grads_log_prob = self._pacoh_log_prob_and_grad(meta_batch, n_train_samples)
        score = grads_log_prob[0]

        # SVGD
        particles = self.hyper_posterior_particles.squeeze(0)
        K_XX, grad_K_XX = self._get_kernel_matrix_and_grad(particles)
        svgd_grads = -(K_XX.unsqueeze(0) @ score - grad_K_XX.unsqueeze(0)) / particles.shape[1]

        self.hyper_posterior_particles.grad = svgd_grads

        self.meta_optimizer.step()
        return log_prob.detach()

    def _sample_params_from_prior(self):
        param_sample = self.prior_module.sample_parametrized(self.n_samples_per_prior, self.hyper_posterior_particles)
        param_sample = param_sample.reshape((self.n_batched_models_train, param_sample.shape[-1]))
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(param_sample)
        return nn_params, likelihood_std

    def _pacoh_log_prob_and_grad(self, meta_batch, n_train_samples):
        log_likelihood = self._estimate_mll(self.hyper_posterior_particles, meta_batch, n_train_samples)
        log_prior_prob = self.hyper_prior_module.log_prob_vectorized(self.hyper_posterior_particles,
                                                                     model_params_prior_weight=self.hyper_prior_weight)
        log_prob = log_likelihood + log_prior_prob

        grads = torch.autograd.grad(log_prob.sum(), self.hyper_posterior_particles)
        return log_prob, grads

    def _estimate_mll(self, prior_samples, meta_batch, n_train_samples):
        param_sample = self.prior_module.sample_parametrized(self.n_samples_per_prior, prior_samples)
        log_likelihood = self._compute_likelihood_across_tasks(param_sample, meta_batch)

        # multiply by sqrt of m and apply logsumexp
        neg_log_likelihood = log_likelihood * (n_train_samples[:, None, None]).sqrt()
        mll = neg_log_likelihood.logsumexp(dim=-1) - torch.tensor(self.n_samples_per_prior, dtype=torch.float32).log()

        # sum over meta-batch size and adjust for number of tasks
        mll_sum = mll.sum(dim=0) * (self.dataset.num_episodes / self.meta_batch_size)

        return self.mll_pre_factor * mll_sum

    def _compute_likelihood_across_tasks(self, params, meta_batch):
        """
        Compute the average likelihood, i.e. the mean of the likelihood for the points in the batch (x,y)
        If you want an unbiased estimator of the dataset likelihood, set the prefactor to the number of points

        Returns: log_likelihood_across_tasks with shape (meta_batch_size, n_hyper_posterior_samples, n_prior_samples)
        """
        params = params.reshape((self.n_batched_models_train, params.shape[-1]))
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(params)

        # iterate over tasks
        log_likelihood_list = []
        for i in range(self.meta_batch_size):
            observation = meta_batch[i]
            state, action = observation.state, observation.action
            y = observation.next_state
            y = y.reshape((-1, y.shape[-1]))

            # NN forward pass
            pred_fn = self.get_forward_fn(nn_params)
            y_hat = pred_fn.forward_nn(
                state.reshape((-1, state.shape[-1])),
                action.reshape((-1, action.shape[-1]))
            )  # (k, b, d)

            # compute likelihood
            log_likelihood = self.likelihood.log_prob(y_hat, y, likelihood_std)
            log_likelihood = log_likelihood.reshape((self.n_batched_priors, self.n_samples_per_prior))
            log_likelihood_list.append(log_likelihood)

        log_likelihood_across_tasks = torch.stack(log_likelihood_list)
        return log_likelihood_across_tasks

    def get_forward_fn(self, params):
        reg_model = copy.deepcopy(self.meta_nn_model)
        reg_model.set_parameters_as_vector(params)
        return reg_model

    def _get_kernel_matrix_and_grad(self, X):
        X2 = X.detach().clone()
        K_XX = self.kernel(X, X2)
        K_grad = torch.autograd.grad(K_XX.sum(), X)
        return K_XX, K_grad[0]

    def _compute_mll_prefactor(self):
        dataset_sizes = torch.tensor(self.dataset.trajectory_lengths, dtype=torch.float32)
        harmonic_mean_dataset_size = 1. / (1. / dataset_sizes).mean()
        return 1 / ((harmonic_mean_dataset_size * self.dataset.num_episodes).sqrt() + 1)

    def _split_into_nn_params_and_likelihood_std(self, params):
        assert params.ndim == 2
        assert params.shape[-1] == (self.nn_param_size + self.likelihood_param_size)

        n_particles = params.shape[0]
        nn_params = params[:, :self.nn_param_size]
        if self.likelihood_param_size > 0:
            likelihood_std = params[:, -self.likelihood_param_size:].exp()
        else:
            likelihood_std = torch.ones((n_particles, self.output_dim)) * self.likelihood_std

        assert likelihood_std.shape == (n_particles, self.output_dim)
        return nn_params, likelihood_std
