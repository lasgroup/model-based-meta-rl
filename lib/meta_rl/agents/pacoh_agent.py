import os
import copy
import math
import time
from collections import deque
from functools import reduce

import numpy as np
import torch

from utils.utils import get_project_path
from rllib.dataset import stack_list_of_tuples
from rllib.util.training.utilities import get_model_validation_score

from lib.agents import MBPOAgent
from lib.datasets import TrajectoryReplay
from lib.model.bayesian_model import BayesianNNModel
from lib.model.bayesian_model_learning import train_bayesian_nn_step
from lib.meta_rl.algorithms.pacoh.modules.kernels import RBFKernel
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from lib.meta_rl.algorithms.pacoh.modules.likelihood import GaussianLikelihood
from lib.meta_rl.algorithms.pacoh.modules.hyper_prior import GaussianHyperPrior
from lib.meta_rl.algorithms.pacoh.modules.prior_posterior import BatchedGaussianPrior

"""
Add plot trajectories and axis
"""


class PACOHAgent(MBPOAgent):
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
            meta_lr: float = 8e-4,
            num_iter_meta_train: int = 20000,
            num_iter_meta_test: int = 2000,
            num_iter_eval_train=20,
            num_learn_eval_steps=400,
            n_samples_per_prior=10,
            num_hyper_posterior_particles=3,
            num_posterior_particles=5,
            hyper_prior_weight=1e-4,
            hyper_prior_nn_std=0.4,
            hyper_prior_log_var_mean=-3.0,
            hyper_prior_likelihood_log_var_mean_mean=-8,
            hyper_prior_likelihood_log_var_log_var_mean=-4.,
            meta_batch_size=4,
            per_task_batch_size=8,
            eval_num_context_samples=32,
            use_data_normalization=False,
            fit_at_step=False,
            early_stopping=True,
            env_name="",
            trajectory_replay_load_path=None,
            multiple_runs_id=0,
            *args,
            **kwargs
    ):
        super().__init__(
            max_memory=max_memory,
            *args,
            **kwargs
        )

        self.meta_environment = meta_environment
        self.env_name = env_name
        self.multiple_runs_id = multiple_runs_id
        self.fit_at_step = fit_at_step

        self.dataset = TrajectoryReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.transformations
        )
        if trajectory_replay_load_path is not None:
            self.load_trajectory_replay(trajectory_replay_load_path)
            self.save_data = False
        else:
            self.save_data = True

        self.observation_queue = deque([], 50000)

        self.use_data_normalization = use_data_normalization

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
        self.num_iter_eval_train = num_iter_eval_train
        self.num_learn_eval_steps = num_learn_eval_steps
        self.num_learn_steps_on_trial = 5 * self.num_learn_steps
        self.early_stopping = early_stopping
        self.meta_batch_size = meta_batch_size
        self.per_task_batch_size = per_task_batch_size
        self.eval_num_context_samples = eval_num_context_samples

        self.mll_pre_factor = self._compute_mll_prefactor()

        # setup NN
        train_model_config = self.eval_model_config.copy()
        train_model_config.update({"num_particles": self.n_batched_models_train})
        self.meta_nn_model = BayesianNNModel(**train_model_config)

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
        self.meta_optimizer = torch.optim.Adam([self.hyper_posterior_particles], lr=meta_lr)

    def set_meta_environment(self, meta_environment):
        self.meta_environment = meta_environment

    def observe(self, observation):
        transformed_observation = observation.clone()
        for transform in self.dynamical_model.transformations:
            transformed_observation = transform(transformed_observation)
        self.observation_queue.append(transformed_observation)
        super().observe(observation)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if (not self.training) and self.fit_at_step:
            self.fit_task(
                num_iter_eval_train=self.num_iter_eval_train,
                num_learn_eval_steps=self.num_learn_eval_steps
            )
        return super().act(state)

    def start_episode(self):
        assert self.meta_environment is not None, "Meta training environment has not been set!"
        self.meta_environment.sample_next_env()
        self.dataset.start_episode()
        super().start_episode()

    def end_episode(self):
        self.dataset.end_episode()
        if (not self.training) and (not self.fit_at_step):
            # TODO: If not training, train using dataset (instead of observation queue) and also learn policy
            self.fit_task(
                num_iter_eval_train=self.model_learn_num_iter,
                num_learn_eval_steps=self.num_learn_steps
            )
        super().end_episode()

    def train(self, val=True):
        """Set the agent in training mode"""
        self.meta_environment.train(val)
        super().train(val)

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.meta_environment.eval(val)
        # if self.save_data:
        #     self.save_trajectory_replay()
        self.meta_fit()
        super().eval(val)
        self.start_trial()

    def start_trial(self):
        self.observation_queue.clear()
        self.dynamical_model.base_model.set_prior(self.prior_module)
        self.dynamical_model.base_model.set_normalization_stats(self.get_normalization_stats())
        self.model_optimizer = type(self.model_optimizer)(
            [self.dynamical_model.base_model.particles], **self.model_optimizer.defaults
        )
        self.policy.reset_buffer()
        self.simulate_and_learn_policy(self.num_learn_steps_on_trial)

    def fit_task(self, num_iter_eval_train=None, num_learn_eval_steps=None):
        self.dynamical_model.train()
        if len(self.observation_queue) > 0:
            for num_iter in range(num_iter_eval_train):
                batch_idx = np.random.choice(len(self.observation_queue), self.eval_num_context_samples)
                eval_observations = stack_list_of_tuples([self.observation_queue[i] for i in batch_idx])
                eval_observations.action = eval_observations.action[..., : self.dynamical_model.dim_action[0]]
                train_bayesian_nn_step(
                    model=self.dynamical_model.base_model,
                    observation=eval_observations,
                    optimizer=self.model_optimizer
                )
            self.simulate_and_learn_policy(learn_steps=num_learn_eval_steps)

    def get_meta_batch(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        task_batches = []
        n_train_samples = []
        assert dataset.num_episodes > self.meta_batch_size, "Number of tasks should be larger than meta batch size"
        task_ids = np.random.choice(dataset.num_episodes, self.meta_batch_size, replace=False)
        for task_id in task_ids:
            observation, n_train_sample = self.get_meta_batch_task(task_id, dataset)
            task_batches.append(observation)
            n_train_samples.append(n_train_sample)
        return task_batches, torch.tensor(n_train_samples, dtype=torch.float32)

    def get_meta_batch_task(self, task_id, dataset=None, eval_samples=False, num_eval_samples=0):
        if dataset is None:
            dataset = self.dataset
        observation, _, _ = dataset.sample_task_batch(self.per_task_batch_size, task_id, eval_samples, num_eval_samples)
        return observation, dataset.trajectory_lengths[task_id]

    def meta_fit(self, log_period=500, eval_period=3000):
        """
        Fits the hyper-posterior (PACOH) particles with SVGD
        """
        print("Start meta-training -------------------- ")
        t = time.time()

        self.mll_pre_factor = self._compute_mll_prefactor()
        self._compute_normalization_stats()
        avg_log_probs = []

        for it in range(self.num_iter_meta_train):

            meta_task_batch, n_train_samples = self.get_meta_batch()
            log_prob = self.step(meta_task_batch, n_train_samples)
            avg_log_probs.append(log_prob.mean().numpy())

            message = ''
            if it % log_period == 0 or it % eval_period == 0:
                avg_log_prob = avg_log_probs[-1]
                message += '\nIter %d/%d - Time %.2f sec' % (it, self.num_iter_meta_train, time.time() - t)
                message += ' - Train-Loss: %.5f' % avg_log_prob

                # # run validation and print results
                # if it % eval_period == 0 and it > 0:
                #     dataset = self.val_dataset if hasattr(self, 'val_dataset') else self.dataset
                #     eval_metrics_mean, eval_metrics_std = self.meta_eval_dataset(meta_val_dataset=dataset)
                #     for key in eval_metrics_mean:
                #         message += '- Val-%s: %.3e +- %.3e' % (key, eval_metrics_mean[key], eval_metrics_std[key])

                t = time.time()

            if len(message) > 0:
                print(message)

            if self.early_stopping and len(avg_log_probs) > 100:
                running_avg_log_probs = np.mean(avg_log_probs[-100:])
                if running_avg_log_probs > -200:   # When average negative log-likelihood is high
                    print(f"\n---------Early stopping meta-training after {it} iterations---------")
                    break

        loss = -log_prob.mean().numpy()
        return loss

    def meta_eval_dataset(self, meta_val_dataset, num_eval_samples=10):
        task_ids = np.random.choice(meta_val_dataset.num_episodes, self.meta_batch_size, replace=False)
        mse, sharpness, calibration_score = [], [], []
        for task_id in task_ids:
            eval_model = BayesianNNModel(
                **self.eval_model_config,
                meta_learned_prior=self.prior_module,
                normalization_stats=self.get_normalization_stats()
            )
            eval_model_optimizer = type(self.model_optimizer)(
                [eval_model.particles],
                **self.model_optimizer.defaults
            )
            for num_iter in range(self.num_iter_meta_test):
                meta_task_batch, _ = self.get_meta_batch_task(task_id, meta_val_dataset, False, num_eval_samples)
                train_bayesian_nn_step(
                    model=eval_model,
                    observation=meta_task_batch,
                    optimizer=eval_model_optimizer
                )
            eval_obs, _ = self.get_meta_batch_task(task_id, meta_val_dataset, True, num_eval_samples)
            _, mse_, sharpness_, calibration_score_ = get_model_validation_score(eval_model, eval_obs)
            mse.append(mse_)
            sharpness.append(sharpness_)
            calibration_score.append(calibration_score_)
        return self._aggregate_eval_metrics_across_tasks(mse, sharpness, calibration_score)

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
        log_likelihood = self._compute_batched_likelihood_across_tasks(param_sample, meta_batch)

        # multiply by sqrt of m and apply logsumexp
        neg_log_likelihood = log_likelihood * (n_train_samples[:, None, None]).sqrt()
        # TODO: This should be plus ln(L)
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
            state, next_state = observation.state, observation.next_state
            action = observation.action[..., :self.dynamical_model.base_model.dim_action[0]]
            state, action, next_state = self._normalize_data(state, action, next_state)
            y = next_state.reshape((-1, next_state.shape[-1]))

            # NN forward pass
            pred_fn = self.get_forward_fn(nn_params)
            y_hat = pred_fn.forward_nn(
                state.reshape((-1, state.shape[-1])),
                action.reshape((-1, action.shape[-1])),
                nn_params=nn_params
            )  # (k, b, d)

            # compute likelihood
            log_likelihood = self.likelihood.log_prob(y_hat, y, likelihood_std)
            log_likelihood = log_likelihood.reshape((self.n_batched_priors, self.n_samples_per_prior))
            log_likelihood_list.append(log_likelihood)

        log_likelihood_across_tasks = torch.stack(log_likelihood_list)
        return log_likelihood_across_tasks

    def _compute_batched_likelihood_across_tasks(self, params, meta_batch):
        """
        Compute the average likelihood, i.e. the mean of the likelihood for the points in the batch (x,y)
        If you want an unbiased estimator of the dataset likelihood, set the prefactor to the number of points

        Returns: log_likelihood_across_tasks with shape (meta_batch_size, n_hyper_posterior_samples, n_prior_samples)
        """
        params = params.reshape((self.n_batched_models_train, params.shape[-1]))
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(params)

        # iterate over tasks
        log_likelihood_list = []
        state_list = []
        action_list = []
        next_state_list = []
        batch_size_cum_list = [0, ]

        for i in range(self.meta_batch_size):
            observation = meta_batch[i]
            state, next_state = observation.state, observation.next_state
            action = observation.action[..., :self.dynamical_model.base_model.dim_action[0]]
            state, action, next_state = self._normalize_data(state, action, next_state)

            state_list.append(state.reshape((-1, state.shape[-1])))
            action_list.append(action.reshape((-1, action.shape[-1])))
            next_state_list.append(next_state.reshape((-1, next_state.shape[-1])))
            batch_size_cum_list.append(batch_size_cum_list[-1] + reduce(lambda x, y: x * y, list(state.shape[:-1])))

        stacked_states = torch.cat(state_list, dim=0)
        stacked_actions = torch.cat(action_list, dim=0)
        stacked_next_states = torch.cat(next_state_list, dim=0)

        # NN forward pass
        pred_fn = self.get_forward_fn(nn_params)
        stacked_y_hat = pred_fn.forward_nn(
            stacked_states,
            stacked_actions,
            nn_params=nn_params
        )  # (k, b, d)

        for i in range(self.meta_batch_size):
            y = stacked_next_states[batch_size_cum_list[i]: batch_size_cum_list[i + 1]]
            y_hat = stacked_y_hat[:, batch_size_cum_list[i]: batch_size_cum_list[i + 1]]

            # compute likelihood
            log_likelihood = self.likelihood.log_prob(y_hat, y, likelihood_std)
            log_likelihood = log_likelihood.reshape((self.n_batched_priors, self.n_samples_per_prior))
            log_likelihood_list.append(log_likelihood)

        log_likelihood_across_tasks = torch.stack(log_likelihood_list)
        return log_likelihood_across_tasks

    def get_forward_fn(self, params):
        self.meta_nn_model.set_parameters_as_vector(params)
        return self.meta_nn_model

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

    @staticmethod
    def _aggregate_eval_metrics_across_tasks(mse, sharpness, calibration_score):
        eval_metrics_mean, eval_metrics_std = {}, {}
        for key in ['mse', 'sharpness', 'calibration_score']:
            eval_metrics_mean[key] = np.mean(eval(key)).item()
            eval_metrics_std[key] = np.std(eval(key)).item()
        return eval_metrics_mean, eval_metrics_std

    def _compute_normalization_stats(self):
        train_obs = self.dataset.all_data
        train_states, train_next_states = train_obs.state, train_obs.next_state
        train_actions = train_obs.action[..., :self.dynamical_model.base_model.dim_action[0]]

        self.state_mean = train_states.reshape((-1, train_states.shape[-1])).mean(dim=0)
        self.state_std = train_states.reshape((-1, train_states.shape[-1])).std(dim=0)
        self.action_mean = train_actions.reshape((-1, train_actions.shape[-1])).mean(dim=0)
        self.action_std = train_actions.reshape((-1, train_actions.shape[-1])).std(dim=0)
        self.next_state_mean = train_next_states.reshape((-1, train_next_states.shape[-1])).mean(dim=0)
        self.next_state_std = train_next_states.reshape((-1, train_next_states.shape[-1])).std(dim=0)

        if not self.use_data_normalization:
            self._set_unit_normalization_stats()

        assert self.state_mean.ndim == 1
        assert self.state_std.ndim == 1
        assert self.action_mean.ndim == 1
        assert self.action_std.ndim == 1
        assert self.next_state_mean.ndim == 1
        assert self.next_state_std.ndim == 1

    def _normalize_data(self, state, action, next_state):
        assert state.shape[-1] == self.state_mean.shape[-1]
        assert action.shape[-1] == self.action_mean.shape[-1]
        assert next_state.shape[-1] == self.next_state_mean.shape[-1]

        normalized_state = (state - self.state_mean) / self.state_std
        normalized_action = (action - self.action_mean) / self.action_std
        normalized_next_state = (next_state - self.next_state_mean) / self.next_state_std

        return normalized_state, normalized_action, normalized_next_state

    def get_normalization_stats(self):
        return {
            "x_mean": torch.cat((self.state_mean, self.action_mean), dim=-1),
            "x_std": torch.cat((self.state_std, self.action_std), dim=-1),
            "y_mean": self.next_state_mean,
            "y_std": self.next_state_std,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std
        }

    def _set_unit_normalization_stats(self):
        self.state_mean = torch.zeros_like(self.state_mean)
        self.state_std = torch.ones_like(self.state_std)
        self.action_mean = torch.zeros_like(self.action_mean)
        self.action_std = torch.ones_like(self.action_std)
        self.next_state_mean = torch.zeros_like(self.next_state_mean)
        self.next_state_std = torch.ones_like(self.next_state_std)

    def set_normalization_stats(self, normalization_stats):
        self.state_mean = normalization_stats['state_mean']
        self.state_std = normalization_stats['state_std']
        self.action_mean = normalization_stats['action_mean']
        self.action_std = normalization_stats['action_std']
        self.next_state_mean = normalization_stats['y_mean']
        self.next_state_std = normalization_stats['y_std']

    def save_trajectory_replay(self, params, base_path="experiments/meta_rl_experiments/experience_replay", mode="train"):
        model_kind = params.pacoh_training_model_kind
        num_tasks = params.num_train_env_instances
        action_cost = str(params.action_cost).replace(".", "")
        proj_rel_path = base_path + f"_{model_kind}_{num_tasks}tasks_{action_cost}acost"
        file_name = f"{self.env_name}_{params.exploration}_{mode}_{self.dataset.num_episodes}_{self.multiple_runs_id}.pkl"
        if not os.path.exists(os.path.join(get_project_path(), proj_rel_path)):
            os.makedirs(os.path.join(get_project_path(), proj_rel_path), exist_ok=True)
        save_path = os.path.join(get_project_path(), proj_rel_path, file_name)
        torch.save(self.dataset, save_path)

    def load_trajectory_replay(self, project_rel_path):
        load_path = os.path.join(get_project_path(), project_rel_path)
        self.dataset = torch.load(load_path)
        if os.path.exists(load_path.replace('train', 'test')):
            self.val_dataset = torch.load(load_path.replace('train', 'test'))
