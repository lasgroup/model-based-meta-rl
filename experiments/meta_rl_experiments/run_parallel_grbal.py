import os
import ray
import copy
import yaml
import time
import torch
import numpy as np

from dotmap import DotMap
from rllib.util.training.utilities import Evaluate

from experiments.meta_rl_experiments import AGENT_CONFIG_PATH
from lib.environments import ENVIRONMENTS_PATH
from experiments.meta_rl_experiments.parser import get_argument_parser
from experiments.meta_rl_experiments.run_utils import get_environment_and_meta_agent


def get_parallel_environments_and_agents(params):
    params.agent_name = 'grbal'
    params.safe_log_dir = False
    params.log_to_file = False
    params.save_statistics = False
    params.use_wandb = False
    params.model_learn_num_iter = 0

    envs_agents = [(get_environment_and_meta_agent(params)) for _ in range(params.grbal_num_parallel_agents)]

    task_envs = [env_agent[0] for env_agent in envs_agents]
    task_agents = [env_agent[1] for env_agent in envs_agents]

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

    if params["agent_config_path"] == "":
        params["agent_config_path"] = AGENT_CONFIG_PATH
    with open(
            os.path.join(
                params["agent_config_path"],
                params["agent_name"].split('_')[-1] + "_defaults.yaml"
            ),
            "r"
    ) as file:
        agent_config = yaml.safe_load(file)
    params.update(agent_config)

    params = DotMap(params)

    ray.init(num_cpus=params.num_cpu_cores)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(min(4, params.num_cpu_cores))

    assert params.agent_name == 'parallel_grbal'
    meta_environment, meta_agent = get_environment_and_meta_agent(params)

    envs, agents = get_parallel_environments_and_agents(copy.deepcopy(params))

    meta_agent.logger.save_hparams(params.toDict())

    if params.collect_meta_data:
        train_returns = meta_agent.training_rollout(
            meta_environment, agents, envs, params.train_episodes, use_early_termination=not params.skip_early_termination
        )
        train_returns = np.mean(train_returns)
    else:
        meta_agent.meta_fit()
        train_returns = []

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
