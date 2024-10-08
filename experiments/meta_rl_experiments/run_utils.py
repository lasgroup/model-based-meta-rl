import os
import ray
import sys
import copy
import yaml
import time
import torch
import dotmap
import argparse
import numpy as np

from typing import Union
from dotmap import DotMap

from utils.logger import Logger
import utils.get_agents as agents

from experiments import AGENT_CONFIG_PATH
from utils.get_environments import get_environment
from experiments.meta_rl_experiments.parser import get_argument_parser
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from lib.datasets.transforms import LocalCoordinates, ActionBufferScaler, ActionStacking

from rllib.model import AbstractModel
from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.abstract_environment import AbstractEnvironment
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction, StateNormalizer, RewardNormalizer, \
    NextStateNormalizer


def get_environment_and_meta_agent(params: dotmap.DotMap) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param params: environment arguments
    :return: RL environment and agent
    """
    environment, reward_model, termination_model = get_environment(params)

    transformations = [
        MeanFunction(DeltaState()),
        ActionScaler(scale=environment.action_scale),
    ]

    if "rccar" in params.env_config_file:
        transformations = [
            LocalCoordinates(action_stacking_dim=environment.action_stacking_dim),
            ActionScaler(scale=environment.action_scale),
            ActionBufferScaler(scale=environment.action_scale, action_stacking_dim=environment.action_stacking_dim),
            ActionStacking(action_stacking_dim=environment.action_stacking_dim, action_dim=environment.dim_action[0])
        ]

    try:
        agent_callable = eval(f"agents.get_{params.agent_name}_agent")
        agent, comment = agent_callable(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            termination_model=termination_model,
            params=params,
            input_transform=None
        )
    except AttributeError:
        raise ValueError(f"Agent {params.agent_name} is not a valid agent.")

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
        offline_mode=params.offline_logger,
        project=params.wandb_project
    )
    if params.log_to_file:
        sys.stdout = agent.logger

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)

    if params.env_load_params_from_file:
        env_params_dir = os.path.join(
            "experiments/meta_rl_experiments/random_env_params",
            f"{params.env_config_file.replace('-', '_').replace('.yaml', '')}"
        )
    else:
        env_params_dir = None
    environment = MetaEnvironmentWrapper(environment, params, env_params_dir=env_params_dir)

    agent.set_meta_environment(environment)

    return environment, agent


def get_params():
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

    return params
