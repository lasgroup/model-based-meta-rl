from typing import Union, Callable, Iterable

import torch
import argparse
from torch import nn, optim

from lib.agents.mpc_agent import MPCAgent
from utils.get_learners import get_model, get_value_function, get_mpc_policy

from rllib.model import AbstractModel
from rllib.dataset.transforms import AbstractTransform
from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.abstract_environment import AbstractEnvironment


def get_mpc_agent(
        environment: AbstractEnvironment,
        reward: AbstractModel,
        transformations: Iterable[AbstractTransform],
        args: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> AbstractAgent:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param reward: Reward model
    :param transformations: State and action transformations
    :param args: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        transformations=transformations,
        input_transform=input_transform,
        args=args
    )

    # Define model optimizer
    model_optimizer = optim.Adam(
        dynamical_model.parameters(),
        lr=args.model_opt_lr,
        weight_decay=args.model_opt_weight_decay,
    )

    # Define value function.
    value_function = get_value_function(dim_state, args, input_transform)

    terminal_reward = value_function if args.mpc_terminal_reward else None

    # Define policy
    policy = get_mpc_policy(
        dynamical_model=dynamical_model,
        reward=reward,
        args=args,
        action_scale=environment.action_scale,
        terminal_reward=terminal_reward,
        termination_model=termination_model
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {args.exploration.capitalize()} {args.action_cost}"

    agent = MPCAgent(
        policy,
        model_optimizer=model_optimizer,
        exploration_scheme=args.exploration,
        initial_distribution=initial_distribution,
        gamma=args.gamma,
        comment=comment,
    )

    return agent
