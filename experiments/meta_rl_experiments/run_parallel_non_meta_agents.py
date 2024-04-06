import os
import ray
import copy
import yaml
import time
import torch
import numpy as np
from rllib.util.training.utilities import Evaluate

import experiments.meta_rl_experiments.run_utils as run_utils

from utils.get_environments import get_wrapped_env, get_wrapped_meta_env
from experiments.meta_rl_experiments.run_utils import get_environment_and_meta_agent
from experiments.meta_rl_experiments.remote_utils import rollout_parallel_agent, update_counters, log_agents


def eval_rollout(base_agent, params, meta_environment=None, num_episodes=1, render=False, use_early_termination=True):
    if meta_environment is None:
        meta_environment = base_agent.meta_environment
    copy_agents_id = []
    for task in range(params.num_test_env_instances):
        env_copy, _, _ = get_wrapped_meta_env(
            params,
            meta_training_tasks=[meta_environment.test_env_params[task]],
            meta_test_tasks=[meta_environment.test_env_params[task]]
        )
        env_copy.eval()
        agent_copy = get_copy(base_agent, copy.deepcopy(params), env_copy)
        copy_agents_id.append(
            rollout_parallel_agent.remote(
                env_copy,
                agent_copy,
                params.max_steps,
                num_episodes=num_episodes,
                render=render,
                use_early_termination=use_early_termination
            )
        )
    copy_agents = ray.get(copy_agents_id)
    returns = np.zeros((len(copy_agents), num_episodes))
    for i, agent in enumerate(copy_agents):
        returns[i] = np.array(agent.logger.get("train_return-0"))
        for episode in reversed(range(num_episodes)):
            episode_dict = agent.logger[-episode-1]
            if i == len(copy_agents) - 1:
                episode_dict["env_avg_train_return-0"] = np.mean(returns[:, -episode-1], axis=0)
            print(base_agent)
        update_counters(base_agent, agent)
    log_agents(base_agent, copy_agents, num_episodes)

    return returns.flatten()


def get_copy(base_agent, params, meta_env):
    params.safe_log_dir = False
    params.save_statistics = False
    params.log_to_file = False
    params.use_wandb = False
    _, agent = get_environment_and_meta_agent(params=params)
    agent.set_meta_environment(meta_env)
    agent.dataset.reset()
    return agent


if __name__ == "__main__":

    start = time.time()

    params = run_utils.get_params()
    base_environment, base_agent = get_environment_and_meta_agent(copy.deepcopy(params))
    train_returns = 0.0

    base_agent.logger.save_hparams(params.toDict())
    base_agent.logger.export_to_json()  # Save statistics.

    metrics = dict()
    returns = eval_rollout(
        base_agent,
        copy.deepcopy(params),
        base_environment,
        num_episodes=params.num_test_episodes_per_env,
        render=params.render,
        use_early_termination=not params.skip_early_termination
    )
    eval_returns = np.mean(returns)
    print(f"Test Cumulative Rewards: {eval_returns}")

    metrics.update({"train_returns": train_returns, "test_returns": eval_returns})

    base_agent.logger.log_metrics(hparams=params.toDict(), metrics=metrics)

    print(f'---------------------------------\nTotal Run Time: {(time.time()-start)/60} min')
