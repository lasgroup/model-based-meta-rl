import copy
import time
import numpy as np

from rllib.util.training.utilities import Evaluate

from experiments.meta_rl_experiments import run_utils


def get_parallel_environments_and_agents(params):
    params.agent_name = 'grbal'
    params.safe_log_dir = False
    params.log_to_file = False
    params.save_statistics = False
    params.use_wandb = False
    params.model_learn_num_iter = 0
    params.num_learn_steps = 0
    params.collect_meta_data = True

    envs_agents = [(run_utils.get_environment_and_meta_agent(params)) for _ in range(params.grbal_num_parallel_agents)]

    task_envs = [env_agent[0] for env_agent in envs_agents]
    task_agents = [env_agent[1] for env_agent in envs_agents]

    return task_envs, task_agents


if __name__ == "__main__":

    start = time.time()

    params = run_utils.get_params()

    assert params.agent_name == 'parallel_grbal'
    meta_environment, meta_agent = run_utils.get_environment_and_meta_agent(params)

    meta_agent.logger.save_hparams(params.toDict())

    if params.collect_meta_data:
        envs, agents = get_parallel_environments_and_agents(copy.deepcopy(params))
        train_returns = meta_agent.training_rollout(
            meta_environment, agents, envs, params.train_episodes, use_early_termination=not params.skip_early_termination
        )
        train_returns = np.mean(train_returns)
    else:
        meta_agent.meta_learn()
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
