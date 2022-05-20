import gpytorch
import numpy as np
from rllib.util.training.agent_training import train_agent, evaluate_agent


def train_and_evaluate_agent(environment, agent, params):
    """

    :param environment:
    :param agent:
    :param params:
    :return:
    """
    agent.logger.save_hparams(params.toDict())
    with gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_var(), (
            gpytorch.settings.fast_pred_samples()
    ), (gpytorch.settings.memory_efficient()):
        train_agent(
            agent,
            environment,
            num_episodes=params.train_episodes,
            max_steps=params.environment_max_steps,
            plot_flag=False,
            callback_frequency=1,
            print_frequency=1,
            save_milestones=None,
            render=False,
            callbacks=None,
        )

        agent.logger.export_to_json()  # Save statistics.

        # %% Test agent.
        metrics = dict()
        evaluate_agent(
            agent,
            environment,
            num_episodes=params.test_episodes,
            max_steps=params.environment_max_steps,
            render=False,
        )

        returns = np.mean(agent.logger.get("environment_return")[-params.test_episodes:])
        metrics.update({"test/test_env_returns": returns})
        returns = np.mean(agent.logger.get("environment_return")[: -params.test_episodes])
        metrics.update({"test/train_env_returns": returns})

        agent.logger.log_hparams(params.toDict(), metrics)
