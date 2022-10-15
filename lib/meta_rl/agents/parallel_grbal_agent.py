import copy
from collections import deque

import ray
import numpy as np

from experiments.meta_rl_experiments import run_utils

from lib.meta_rl.agents import GrBALAgent
from utils.get_environments import get_wrapped_meta_env
from experiments.meta_rl_experiments.remote_utils import rollout_parallel_agent, auto_garbage_collect


class ParallelGrBALAgent(GrBALAgent):
    """
    Implementation of a parallel Gradient-based Adaptive Learner.

    References
    ----------
    Nagabandi, A., Clavera, I., Liu, S., Fearing, R., Abbeel, P., Levine, S., Finn, C.,
    Learning to Adapt in Dynamic Real World Environments through Meta Reinforcement Learning.
    """
    def __init__(
            self,
            num_parallel_agents: int = 1,
            num_episodes_per_rollout: int = 1,
            max_env_steps: int = 200,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_parallel_agents = num_parallel_agents
        self.num_episodes_per_rollout = num_episodes_per_rollout
        self.max_env_steps = max_env_steps

    def eval_rollout(self, params, meta_environment=None, num_episodes=1, render=False, use_early_termination=True):
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
            agent_copy = self.get_copy(params=copy.deepcopy(params), meta_env=env_copy)
            agent_copy.training = False
            copy_agents_id.append(
                rollout_parallel_agent.remote(
                    env_copy,
                    agent_copy,
                    self.max_env_steps,
                    num_episodes=1,
                    render=render,
                    use_early_termination=use_early_termination
                )
            )
        copy_agents = ray.get(copy_agents_id)
        for agent in copy_agents:
            self.logger.end_episode(**agent.logger[-1])
            self.update_counters(agent)
            print(self)
        returns = [agent.logger.get("eval_return-0")[-1] for agent in copy_agents]

        return np.asarray(returns)

    def training_rollout(self, meta_environment, parallel_agents, envs, num_train_episodes, max_env_steps=None, use_early_termination=True):
        assert len(parallel_agents) == self.num_parallel_agents
        train_returns = []
        total_episodes_per_rollout = self.num_parallel_agents * self.num_episodes_per_rollout
        num_rollouts = num_train_episodes // total_episodes_per_rollout
        max_env_steps = self.max_env_steps if max_env_steps is None else max_env_steps
        tasks = meta_environment.train_env_params * 2  # Brute-force implementation of a circular list
        curr_task_idx = 0
        for rollout in range(num_rollouts):
            parallel_agents_id = []
            for agent, env in zip(parallel_agents, envs):
                agent = self.get_copy(agent=agent)
                agent.dataset.reset()
                env.train_env_params = tasks[curr_task_idx:curr_task_idx+self.num_episodes_per_rollout]
                parallel_agents_id.append(
                    rollout_parallel_agent.remote(
                        env,
                        agent,
                        max_env_steps,
                        self.num_episodes_per_rollout,
                        use_early_termination=use_early_termination
                    )
                )
                curr_task_idx = (curr_task_idx + self.num_episodes_per_rollout) % (len(tasks) // 2)
            parallel_agents = ray.get(parallel_agents_id)
            self.store_rollout_data(parallel_agents)
            self.learn()
            self.log_parallel_agents(parallel_agents)
            for agent in parallel_agents:
                train_returns = train_returns + agent.logger.get("train_return-0")[-self.num_episodes_per_rollout:]
            auto_garbage_collect()
        return train_returns

    def get_copy(self, params=None, meta_env=None, agent=None):
        assert (params is not None or agent is not None) and (params is None or agent is None)
        if agent is None:
            assert meta_env is not None
            params.safe_log_dir = False
            params.save_statistics = False
            params.log_to_file = False
            params.use_wandb = False
            params.model_learn_num_iter = 0
            _, agent = run_utils.get_environment_and_meta_agent(params=params)
            agent.set_meta_environment(meta_env)
        else:
            assert isinstance(agent, GrBALAgent) or isinstance(agent, ParallelGrBALAgent)

        agent.observation_queue = deque([], self.past_segment_len)
        for obs in self.observation_queue.copy():
            agent.observation_queue.append(obs)
        agent.pre_update_model.load_state_dict(self.pre_update_model.state_dict())
        return agent

    def store_rollout_data(self, agents):
        for agent in agents:
            self.dataset.start_episode()
            self.dataset.add_dataset(agent.dataset)
            self.dataset.end_episode()

    def log_parallel_agents(self, rollout_agents):
        assert len(rollout_agents) == self.num_parallel_agents
        train_dict = {key: value[1] for key, value in self.logger.current.items()}
        for i, rollout_agent in enumerate(rollout_agents):
            for j in reversed(range(self.num_episodes_per_rollout)):
                episode_dict = rollout_agent.logger[-(j+1)]
                if i == (self.num_parallel_agents - 1) and j == 0:
                    episode_dict.update(train_dict)
                self.logger.end_episode(**episode_dict)
                self.update_counters(rollout_agent)
                print(self)

    def update_counters(self, agent):
        self.counters["total_episodes"] += 1
        self.counters["total_steps"] += agent.episode_steps[-1]
        if self.training:
            self.counters["train_episodes"] += 1
            self.counters["train_steps"] += agent.train_steps
        else:
            self.counters["eval_episodes"] += 1
