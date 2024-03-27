import os
import ray
import copy
import yaml
import time
import torch
import numpy as np

from dotmap import DotMap
from rllib.util.training.utilities import Evaluate

from experiments import AGENT_CONFIG_PATH
from experiments.meta_rl_experiments.parser import get_argument_parser
from experiments.lib_environments.run_utils import get_environment_and_agent
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from experiments.meta_rl_experiments.run_utils import get_environment_and_meta_agent


def set_tasks(envs, meta_environment):
    task_envs = []
    for i, env in enumerate(envs):
        env.train_env_params = meta_environment.train_env_params
        env.test_env_params = meta_environment.test_env_params
        task_envs.append(env.get_env(env_id=i))
    return task_envs


def get_parallel_environments_and_agents(params):
    if params.agent_name == 'parallel_pacoh':
        params.agent_name = 'mbpo'
    elif params.agent_name == 'parallel_cem_pacoh':
        params.agent_name = 'mpc'
    params.safe_log_dir = False
    params.log_to_file = False
    params.save_statistics = False
    params.use_wandb = False
    params.model_kind = params.pacoh_training_model_kind

    envs_agents = [(get_environment_and_agent(params)) for _ in range(params.num_train_env_instances)]

    task_envs = [MetaEnvironmentWrapper(env_agent[0], params) for env_agent in envs_agents]
    task_agents = [env_agent[1] for env_agent in envs_agents]

    return task_envs, task_agents


if __name__ == "__main__":

    start = time.time()

    parser = get_argument_parser()
    params = vars(parser.parse_args())

    if params["agent_config_path"] == "":
        params["agent_config_path"] = AGENT_CONFIG_PATH
    with open(
        os.path.join(
            params["agent_config_path"],
            params["env_config_file"]
        ),
        "r"
    ) as file:
        env_config = yaml.safe_load(file)

    for config_set in ["training", "model", "policy", "mpc"]:
        params.update(env_config[config_set])

    agent_config = env_config[params["agent_name"].split('_')[-1]]
    params.update(agent_config)

    params = DotMap(params)

    ray.init(
        num_cpus=params.num_cpu_cores,
        object_store_memory=(1000 * 1e6 * params.num_cpu_cores)
    )

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(min(4, params.num_cpu_cores))

    assert params.agent_name in ['parallel_pacoh', 'parallel_cem_pacoh']
    meta_environment, meta_agent = get_environment_and_meta_agent(copy.deepcopy(params))

    meta_agent.logger.save_hparams(params.toDict())

    if params.collect_meta_data:
        envs, agents = get_parallel_environments_and_agents(copy.deepcopy(params))
        envs = set_tasks(envs, meta_environment)
        train_returns = meta_agent.training_rollout(
            copy.deepcopy(params), meta_environment, agents, params.train_episodes, not params.skip_early_termination
        )
        train_returns = np.mean(train_returns)
        meta_agent.save_trajectory_replay(params)
    else:
        train_returns = 0.0

    meta_agent.logger.export_to_json()  # Save statistics.

    metrics = dict()
    with Evaluate(meta_agent):
        returns = meta_agent.eval_rollout(
            copy.deepcopy(params),
            meta_environment,
            num_episodes=params.num_test_episodes_per_env,
            render=params.render,
            use_early_termination=not params.skip_early_termination
        )
        eval_returns = np.mean(returns)
        print(f"Test Cumulative Rewards: {eval_returns}")

    metrics.update({"train_returns": train_returns, "test_returns": eval_returns})

    meta_agent.logger.log_metrics(hparams=params.toDict(), metrics=metrics)

    print(f'---------------------------------\nTotal Run Time: {(time.time()-start)/60} min')
