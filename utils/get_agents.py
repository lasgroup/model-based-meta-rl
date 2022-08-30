from typing import Union, Callable, Iterable, Tuple

import torch
import argparse

from rllib.agent import PPOAgent, SACAgent
from rllib.algorithms.ppo import PPO
from rllib.algorithms.sac import SAC
from rllib.dataset import ExperienceReplay
from torch import nn, optim
from torch.nn.modules import loss

from lib.agents import MPCAgent, MPCPolicyAgent
from lib.meta_rl.agents import RLSquaredAgent, GrBALAgent
from utils.get_learners import get_model, get_value_function, get_q_function, get_mpc_policy, get_nn_policy, \
    get_recurrent_value_function, get_rnn_policy

from rllib.model import AbstractModel
from rllib.dataset.transforms import AbstractTransform
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
    try:
        model_optimizer = optim.AdamW(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = model_optimizer = optim.AdamW(
            dynamical_model.base_model.parameters(),
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
    # TODO: Use as terminal reward  and train value function in ModelBasedAgent
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
        use_validation_set=params.use_validation_set,
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
    model_optimizer = optim.AdamW(
        dynamical_model.parameters(),
        lr=params.model_opt_lr,
        weight_decay=params.model_opt_weight_decay,
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

    # Define actor-critic algorithm
    if params.mpc_policy == "ppo":
        # Define value function.
        critic = get_value_function(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            params=params,
            input_transform=input_transform
        )

        algorithm = PPO(
            policy=policy,
            critic=critic,
            criterion=loss.MSELoss(reduction="mean"),
            gamma=params.gamma,
            epsilon=0.2,
            clamp_value=False
        )
    elif params.mpc_policy == "sac":
        # Define q function.
        critic = get_q_function(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            params=params,
            input_transform=input_transform
        )

        algorithm = SAC(
            policy=policy,
            critic=critic,
            criterion=loss.MSELoss(reduction="mean"),
            gamma=params.gamma
        )
    else:
        raise NotImplementedError

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(critic.parameters()),
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
        use_validation_set=params.use_validation_set,
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
    Get an Proximal Policy Optimization (PPO) agent
    :param environment: RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: An PPO based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define value function.
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
    comment = f"{params.agent_name} {params.exploration.capitalize()}"

    agent = PPOAgent(
        policy=policy,
        critic=value_function,
        optimizer=policy_optimizer,
        num_iter=128,
        batch_size=32,
        epsilon=0.2,
        eta=params.ppo_eta,  # Controls agent exploration, higher value leads to more exploration
        gamma=params.gamma,
        comment=comment
    )

    return agent, comment


def get_sac_agent(
        environment: AbstractEnvironment,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> Tuple[SACAgent, str]:
    """
    Get a Soft Actor-Critic (SAC) agent
    :param environment: RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: A SAC agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define q function.
    q_function = get_q_function(
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
        list(policy.parameters()) + list(q_function.parameters()),
        lr=params.sac_opt_lr,
        weight_decay=params.sac_opt_weight_decay,
    )

    memory = ExperienceReplay(max_len=params.sac_memory_len)

    # Define Agent
    comment = f"{params.agent_name} {params.exploration.capitalize()}"

    agent = SACAgent(
        policy=policy,
        critic=q_function,
        optimizer=policy_optimizer,
        memory=memory,
        num_iter=128,
        gamma=params.gamma,
        comment=comment
    )

    return agent, comment


def get_rl2_agent(
        environment: AbstractEnvironment,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> Tuple[PPOAgent, str]:
    """
    Get an Reinforcement Learning Squared (RL^2) agent
    :param environment: Meta RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: An RL^2 based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define value function.
    value_function = get_recurrent_value_function(
        dim_state=(dim_state[0] + dim_action[0] + 2, ),
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define policy
    policy = get_rnn_policy(
        dim_state=(dim_state[0] + dim_action[0] + 2, ),
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
    comment = f"{params.agent_name} {params.exploration.capitalize()}"

    agent = RLSquaredAgent(
        policy=policy,
        critic=value_function,
        optimizer=policy_optimizer,
        trial_len=params.rl2_trial_len,
        num_iter=64,
        batch_size=32,
        epsilon=0.2,
        eta=params.ppo_eta,  # Controls agent exploration, higher value leads to more exploration
        gamma=params.gamma,
        comment=comment
    )

    return agent, comment


def get_grbal_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[GrBALAgent, str]:
    """
    Get a Gradient-based Adaptive Learner agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: An GrBAL agent
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
    model_optimizer = optim.AdamW(
        dynamical_model.parameters(),
        lr=params.model_opt_lr,
        weight_decay=params.model_opt_weight_decay,
    )

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
    # TODO: Use as terminal reward and train value function in ModelBasedAgent
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

    agent = GrBALAgent(
        mpc_policy=policy,
        model_optimizer=model_optimizer,
        past_segment_len=params.grbal_past_segment_len,
        future_segment_len=params.grbal_future_segment_len,
        model_learn_batch_size=params.model_learn_batch_size,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        gamma=params.gamma,
        comment=comment,
    )

    return agent, comment
