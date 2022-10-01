import os
import ray
import copy
import yaml
import time
import torch
import numpy as np

from dotmap import DotMap
from rllib.util.training.utilities import Evaluate

from lib.environments import ENVIRONMENTS_PATH
from experiments.meta_rl_experiments.parser import get_argument_parser
from experiments.lib_environments.run_utils import get_environment_and_agent
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from experiments.meta_rl_experiments.run_utils import get_environment_and_meta_agent


def set_tasks(meta_envs):
    task_envs = []
    for i, env in enumerate(meta_envs):
        task_envs.append(env.get_env(env_id=i))
    return task_envs


def get_parallel_environments_and_agents(params):
    params.agent_name = 'mpc'
    params.safe_log_dir = False
    params.save_statistics = False
    params.use_wandb = False

    envs_agents = [(get_environment_and_agent(params)) for _ in range(params.num_train_env_instances)]

    task_envs = [MetaEnvironmentWrapper(env_agent[0], params) for env_agent in envs_agents]
    task_agents = [env_agent[1] for env_agent in envs_agents]

    task_envs = set_tasks(task_envs)

    return task_envs, task_agents


if __name__ == "__main__":

    start = time.time()

    parser = get_argument_parser()
    params = vars(parser.parse_args())
    with open(
        os.path.join(
            ENVIRONMENTS_PATH,
            params["env_group"],
            "config",
            params["env_config_file"],
        ),
        "r"
    ) as file:
        env_config = yaml.safe_load(file)
    params.update(env_config)
    params = DotMap(params)

    ray.init(num_cpus=params.num_cpu_cores)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(min(4, params.num_cpu_cores))

    assert params.agent_name == 'parallel_pacoh'
    meta_environment, meta_agent = get_environment_and_meta_agent(params)

    envs, agents = get_parallel_environments_and_agents(copy.deepcopy(params))

    if params.pacoh_collect_meta_data:
        train_returns = meta_agent.collect_meta_training_data(params, meta_environment, agents, params.train_episodes)

    meta_agent.logger.export_to_json()  # Save statistics.

    metrics = dict()
    with Evaluate(meta_agent):
        returns = meta_agent.rollout(
            params,
            meta_environment,
            num_episodes=params.test_episodes,
            render=params.render,
        )
        eval_returns = np.mean(returns)
        print(f"Test Cumulative Rewards: {eval_returns}")

    metrics.update({"train_returns": train_returns})
    metrics.update({"test_returns": eval_returns})

    meta_agent.logger.log_hparams(params.toDict(), metrics)

    print(f'---------------------------------\nTotal Run Time: {(time.time()-start)/60} min')
