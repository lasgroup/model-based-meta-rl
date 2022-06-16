import argparse
import sys

import torch
import numpy as np
from utils.logger import Logger

from lib.environments.point_envs.point_env_2d import PointEnv2D
from utils.get_agents import get_mpc_agent, get_mpc_policy_agent, get_ppo_agent
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction, StateNormalizer, RewardNormalizer, \
    NextStateNormalizer
from rllib.environment.abstract_environment import AbstractEnvironment


def get_environment_and_agent(params: argparse.Namespace) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param params: environment arguments
    :return: RL environment and agent
    """
    environment = PointEnv2D()

    reward_model = environment.reward_model()

    # TODO: Add more transformations
    transformations = [
        MeanFunction(DeltaState()),
        ActionScaler(scale=environment.action_scale),
    ]

    if params.agent_name == "mpc":
        agent, comment = get_mpc_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "mpc_policy":
        agent, comment = get_mpc_policy_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "ppo":
        agent, comment = get_ppo_agent(
            environment=environment,
            params=params,
            input_transform=None
        )
    else:
        raise NotImplementedError

    name = f"PointEnv2D" \
           f"_{params.agent_name}" \
           f"_{params.exploration}"
    agent.logger = Logger(
        name=name,
        comment=comment,
        log_dir=params.log_dir,
        save_statistics=params.save_statistics,
        use_wandb=params.use_wandb,
        offline_mode=params.offline_logger
    )
    if params.log_to_file:
        sys.stdout = agent.logger

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)

    return environment, agent
