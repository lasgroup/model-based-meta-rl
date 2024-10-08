import copy
import time
import numpy as np

from rllib.util.training.utilities import Evaluate

from experiments.meta_rl_experiments import run_utils
from experiments.lib_environments.run_utils import get_environment_and_agent
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper


def set_tasks(envs, meta_environment):
    task_envs = []
    for i, env in enumerate(envs):
        env.train_env_params = meta_environment.train_env_params
        env.test_env_params = meta_environment.test_env_params
        task_envs.append(env.get_env(env_id=i))
    return task_envs


def get_parallel_environments_and_agents(params):
    if params.agent_name == 'parallel_mbpo_pacoh':
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

    params = run_utils.get_params()

    assert params.agent_name in ['parallel_mbpo_pacoh', 'parallel_cem_pacoh']
    meta_environment, meta_agent = run_utils.get_environment_and_meta_agent(copy.deepcopy(params))

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
