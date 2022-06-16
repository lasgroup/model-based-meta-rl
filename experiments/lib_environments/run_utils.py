import sys
import argparse

from utils.logger import Logger
from lib.environments.wrappers.gym_environment import GymEnvironment
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper
from utils.get_agents import get_mpc_agent, get_mpc_policy_agent, get_ppo_agent

from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.abstract_environment import AbstractEnvironment
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction, StateNormalizer, RewardNormalizer, \
    NextStateNormalizer


def get_environment_and_agent(params: argparse.Namespace) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param params: environment arguments
    :return: RL environment and agent
    """
    if params.env_group == "mujocoMB_env":
        environment = GymEnvironment(
            env_name=params["name"],
            params=params,
            ctrl_cost_weight=params["action_cost"] if params["use_action_cost"] else 0.0
        )
        reward_model = environment.env.reward_model()

    if params.env_group == "point_envs":
        from lib.environments.point_envs import PointEnv2D
        environment = PointEnv2D()
        reward_model = environment.reward_model()

    else:
        raise NotImplementedError

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

    name = f"{params.env_config_file.replace('-', '_').replace('.yaml', '').replace('mujoco', '')}" \
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
