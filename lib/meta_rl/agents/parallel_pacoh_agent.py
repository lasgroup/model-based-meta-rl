import copy

import ray
import numpy as np

import experiments.meta_rl_experiments.run_utils as run_utils

from lib.datasets.utilities import combine_datasets
from lib.meta_rl.agents.pacoh_agent import PACOHAgent
from utils.get_environments import get_wrapped_env, get_wrapped_meta_env
from experiments.meta_rl_experiments.remote_utils import rollout_agent, add_dataset, train_agent


class ParallelPACOHAgent(PACOHAgent):
    """
    Implementation for PACOH Agent for parallel meta-training data collection.

    References
    ----------
    Rothfuss, J., Fortuin, V., Josifoski, M., Krause, A.
    PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees.
    """

    def __init__(
            self,
            parallel_episodes_per_env: int = 1,
            num_episodes_per_rollout: int = 1,
            max_env_steps: int = 200,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.parallel_episodes_per_env = parallel_episodes_per_env
        self.num_episodes_per_rollout = num_episodes_per_rollout
        self.max_env_steps = max_env_steps

    def rollout(self, params, meta_environment=None, num_episodes=1, render=False):
        if meta_environment is None:
            meta_environment = self.meta_environment
        copy_agents_id = []
        for episode in range(num_episodes):
            env_copy, _, _ = get_wrapped_meta_env(
                params,
                meta_training_tasks=meta_environment.train_env_params,
                meta_test_tasks=meta_environment.test_env_params
            )
            env_copy.eval()
            env_copy.current_test_env = episode - 1
            agent_copy = self.get_copy(params, env_copy)
            agent_copy.training = False
            copy_agents_id.append(
                rollout_agent.remote(env_copy, agent_copy, self.max_env_steps, num_episodes=1, render=render)
            )
        copy_agents = ray.get(copy_agents_id)
        returns = [agent.logger.get("eval_return-0")[-1] for agent in copy_agents]

        return np.asarray(returns)

    def get_copy(self, params, meta_env):
        _, agent = run_utils.get_environment_and_meta_agent(params=params)
        agent.set_meta_environment(meta_env)
        agent.prior_module.sample_parametrized(
            self.n_samples_per_prior,
            self.hyper_posterior_particles.detach().clone()
        )
        agent.hyper_posterior_particles = self.hyper_posterior_particles.detach().clone()
        agent.set_normalization_stats(self.get_normalization_stats())
        return agent

    @staticmethod
    def train_agents(agents):
        results_id = []
        for agent in agents:
            results_id.append(train_agent.remote(agent))
        return ray.get(results_id)

    def store_rollout_data(self, datasets, agents):
        assert self.parallel_episodes_per_env * len(agents) == len(datasets)
        results_id = []
        for i, agent in enumerate(agents):
            env_dataset = combine_datasets(
                datasets[i * self.parallel_episodes_per_env:(i + 1) * self.parallel_episodes_per_env]
            )
            results_id.append(
                add_dataset.remote(agent, env_dataset)
            )
        return ray.get(results_id)

    def collect_meta_training_data(self, params, meta_environment, agents, num_train_episodes, max_env_steps=None):
        train_returns = []
        num_rollouts = num_train_episodes // (len(agents) * self.num_episodes_per_rollout)
        max_env_steps = self.max_env_steps if max_env_steps is None else max_env_steps
        tasks = meta_environment.train_env_params
        for rollout in range(num_rollouts):
            copy_agents_id = []
            for task, agent in zip(tasks, agents):
                for i in range(self.parallel_episodes_per_env):
                    env_copy, _, _ = get_wrapped_env(params, task)
                    agent_copy = copy.deepcopy(agent)
                    agent_copy.dataset.reset()
                    copy_agents_id.append(
                        rollout_agent.remote(env_copy, agent_copy, max_env_steps, self.num_episodes_per_rollout)
                    )
            copy_agents = ray.get(copy_agents_id)
            datasets = [agent.dataset for agent in copy_agents]
            agents = self.store_rollout_data(datasets, agents)
            agents = self.train_agents(agents)
            for agent in copy_agents:
                train_returns.append(agent.logger.get("train_return-0")[-1])
        self.store_meta_training_data(agents)
        return train_returns

    def store_meta_training_data(self, agents):
        self.dataset.reset()
        for agent in agents:
            self.dataset.start_episode()
            self.dataset.add_dataset(agent.dataset)
            self.dataset.end_episode()
