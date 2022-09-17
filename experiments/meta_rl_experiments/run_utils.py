import sys
import argparse

from utils.logger import Logger
from utils.get_agents import get_rl2_agent, get_grbal_agent, get_pacoh_agent
from lib.environments.wrappers.gym_environment import GymEnvironment
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from lib.environments.wrappers.random_gym_environment import RandomGymEnvironment

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
    if params.env_group == "mujocoMB_envs":
        environment = GymEnvironment(
            env_name=params.name,
            params=params,
            ctrl_cost_weight=params.action_cost if params.use_action_cost else 0.0
        )
        reward_model = environment.env.reward_model()
        if params.use_exact_termination_model:
            termination_model = environment.env._termination_model().copy()
        else:
            termination_model = environment.env.termination_model()

    elif params.env_group == "gym_envs":
        import lib.environments.gym_envs
        environment = GymEnvironment(
            env_name=params.name,
            params=params
        )
        reward_model = eval(f"lib.environments.gym_envs.{params.reward_model}")()
        termination_model = None

    elif params.env_group == "point_envs":
        from lib.environments.point_envs import RandomPointEnv2D
        environment = RandomPointEnv2D()
        reward_model = environment.reward_model()
        termination_model = None

    elif params.env_group == "random_mujocoMB_envs":
        environment = RandomGymEnvironment(
            env_name=params.name,
            params=params,
            ctrl_cost_weight=params.action_cost if params.use_action_cost else 0.0
        )
        reward_model = environment.env.reward_model()
        if params.use_exact_termination_model:
            termination_model = environment.env._termination_model().copy()
        else:
            termination_model = environment.env.termination_model()

    else:
        raise NotImplementedError

    # TODO: Add more transformations
    transformations = [
        MeanFunction(DeltaState()),
        ActionScaler(scale=environment.action_scale),
    ]

    if params.agent_name == "rl2":
        agent, comment = get_rl2_agent(
            environment=environment,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "grbal":
        agent, comment = get_grbal_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            termination_model=termination_model,
            params=params,
            input_transform=None
        )
    elif params.agent_name == "pacoh":
        agent, comment = get_pacoh_agent(
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
