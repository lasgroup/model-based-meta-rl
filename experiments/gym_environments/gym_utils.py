import argparse
import sys

import torch
import numpy as np
from utils.logger import Logger

from lib.environments.gym_environment import GymEnvironment
from utils.get_agents import get_mpc_agent, get_mpc_policy_agent, get_ppo_agent
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper

from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction, StateNormalizer, RewardNormalizer, \
    NextStateNormalizer
from rllib.environment.system_environment import AbstractEnvironment


def get_environment_and_agent(params: argparse.Namespace) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param params: environment arguments
    :return: RL environment and agent
    """
    environment = GymEnvironment(
        env_name=params["name"],
        params=params,
        ctrl_cost_weight=params["action_cost"]
    )

    reward_model = environment.env.reward_model()

    transformations = [
        MeanFunction(DeltaState()),
        StateNormalizer(dim=environment.dim_state),
        ActionScaler(scale=environment.action_scale),
        RewardNormalizer(dim=environment.dim_reward),
        NextStateNormalizer(dim=environment.dim_state),
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

    agent.logger = Logger(
        name=params.env_config_file.replace("-", "_").replace(".yaml", "").replace("mujoco", "") + params.agent_name,
        comment=comment,
        save_statistics=params.save_statistics,
        use_wandb=True
    )
    if params.log_to_file:
        sys.stdout = agent.logger

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)

    return environment, agent
