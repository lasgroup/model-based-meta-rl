from typing import Union

import sys
import argparse

from utils.logger import Logger
import utils.get_agents as agents
from utils.get_environments import get_environment
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper

from rllib.model import AbstractModel
from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.abstract_environment import AbstractEnvironment
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction, StateNormalizer, RewardNormalizer, \
    NextStateNormalizer


def get_environment_and_meta_agent(params: argparse.Namespace) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param params: environment arguments
    :return: RL environment and agent
    """
    environment, reward_model, termination_model = get_environment(params)

    # TODO: Add more transformations
    transformations = [
        MeanFunction(DeltaState()),
        ActionScaler(scale=environment.action_scale),
    ]

    if params.agent_name == "rl2":
        agent, comment = agents.get_rl2_agent(
            environment=environment,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "grbal":
        agent, comment = agents.get_grbal_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            termination_model=termination_model,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "pacoh":
        agent, comment = agents.get_pacoh_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            termination_model=termination_model,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "parallel_pacoh":
        agent, comment = agents.get_parallel_pacoh_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            termination_model=termination_model,
            params=params,
            input_transform=None
        )
    else:
        raise NotImplementedError

    name = f"{params.env_config_file.replace('-', '_').replace('.yaml', '').replace('mujoco', '')}" \
           f"_{params.agent_name}" \
           f"_{params.exploration}"
    agent.logger = Logger(
        name=name,
        comment=comment,
        safe_log_dir=params.safe_log_dir,
        log_dir=params.log_dir,
        save_statistics=params.save_statistics,
        use_wandb=params.use_wandb,
        offline_mode=params.offline_logger
    )
    if params.log_to_file:
        sys.stdout = agent.logger

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)
    environment = MetaEnvironmentWrapper(environment, params)

    agent.set_meta_environment(environment)

    return environment, agent
