from typing import Union, Callable, Iterable, Tuple

import torch
import argparse

from rllib.agent import PPOAgent, SACAgent
from rllib.algorithms.ppo import PPO
from torch import nn, optim
from torch.nn.modules import loss

from lib.agents.mpc_agent import MPCAgent
from lib.agents.mpc_policy_agent import MPCPolicyAgent
from utils.get_learners import get_model, get_value_function, get_mpc_policy, get_nn_policy

from rllib.model import AbstractModel
from rllib.dataset.transforms import AbstractTransform
from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.abstract_environment import AbstractEnvironment


def get_mpc_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[MPCAgent, str]:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    model_optimizer = optim.Adam(
        dynamical_model.parameters(),
        lr=params.model_opt_lr,
        weight_decay=params.model_opt_weight_decay,
    )

    # Define value function.
    # TODO: Use as terminal reward  and train value function in ModelBasedAgent
    value_function = get_value_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    terminal_reward = value_function if params.mpc_terminal_reward else None

    # Define policy
    policy = get_mpc_policy(
        dynamical_model=dynamical_model,
        reward=reward_model,
        params=params,
        action_scale=environment.action_scale,
        terminal_reward=terminal_reward,
        termination_model=termination_model
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    agent = MPCAgent(
        policy,
        model_optimizer=model_optimizer,
        exploration_scheme=params.exploration,
        initial_distribution=initial_distribution,
        gamma=params.gamma,
        comment=comment,
    )

    return agent, comment


def get_mpc_policy_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        termination_model: Callable = None,
        input_transform: nn.Module = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[MPCPolicyAgent, str]:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param termination_model: Early termination check
    :param input_transform: Input transformation
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    model_optimizer = optim.Adam(
        dynamical_model.parameters(),
        lr=params.model_opt_lr,
        weight_decay=params.model_opt_weight_decay,
    )

    # Define value function.
    # TODO: Use as terminal reward  and train value function in ModelBasedAgent
    value_function = get_value_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define policy
    policy = get_nn_policy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    # TODO write own PPO with base Abstract Algorithm
    # Define actor-critic algorithm
    algorithm = PPO(
        policy=policy,
        critic=value_function,
        criterion=loss.MSELoss(reduction="mean"),
        gamma=params.gamma,
        epsilon=0.2,
        clamp_value=False
    )

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(value_function.parameters()),
        lr=params.ppo_opt_lr,
        weight_decay=params.ppo_opt_weight_decay,
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    agent = MPCPolicyAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        algorithm=algorithm,
        model_optimizer=model_optimizer,
        policy_optimizer=policy_optimizer,
        exploration_scheme=params.exploration,
        initial_distribution=initial_distribution,
        gamma=params.gamma,
        comment=comment,
    )

    return agent, comment


def get_ppo_agent(
        environment: AbstractEnvironment,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> Tuple[PPOAgent, str]:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define value function.
    # TODO: Use as terminal reward and train value function in ModelBasedAgent
    value_function = get_value_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define policy
    policy = get_nn_policy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(value_function.parameters()),
        lr=params.ppo_opt_lr,
        weight_decay=params.ppo_opt_weight_decay,
    )

    # Define Agent
    agent_name = PPOAgent.name
    comment = f"{agent_name} {params.exploration.capitalize()}"

    agent = PPOAgent(
        policy=policy,
        critic=value_function,
        optimizer=policy_optimizer,
        num_iter=128,
        batch_size=32,
        epsilon=0.2,
        gamma=params.gamma,
        comment=comment
    )

    return agent, comment
