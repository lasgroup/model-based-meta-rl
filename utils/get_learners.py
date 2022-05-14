from typing import Union, SupportsFloat, Iterable, Callable

import argparse
import numpy as np
from torch import nn

from rllib.policy import MPCPolicy
from rllib.dataset.transforms import AbstractTransform
from rllib.algorithms.mpc import CEMShooting
from rllib.model import AbstractModel, EnsembleModel, TransformedModel
from rllib.value_function import AbstractValueFunction, NNValueFunction


def get_model(
        dim_state: int,
        dim_action: int,
        args: argparse.Namespace,
        transformations: Iterable[AbstractTransform] = None,
        input_transform: nn.Module = None
) -> AbstractModel:
    """
    Returns a learnable dynamics model for the environment
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param args: model parameters
    :param transformations: State and action transformations
    :param input_transform: Input transformation
    :return: Dynamics model for the environment
    """
    transformations = list() if not transformations else transformations

    if args.model_kind in ["ProbabilisticEnsemble", "DeterministicEnsemble"]:
        model = EnsembleModel(
            dim_state=dim_state,
            dim_action=dim_action,
            num_heads=args.model_num_heads,
            layers=args.model_layers,
            biased_head=not args.model_unbiased_head,
            non_linearity=args.model_non_linearity,
            input_transform=input_transform,
            deterministic=args.model_kind == "DeterministicEnsemble",
        )
    else:
        raise NotImplementedError

    args.update({"model": model.__class__.__name__})

    # TODO: Add wrapper for hallucinated model
    dynamical_model = TransformedModel(model, transformations)

    return dynamical_model


def get_value_function(
        dim_state: int,
        args: argparse.Namespace,
        input_transform: nn.Module = None
) -> AbstractValueFunction:
    """
    Returns a learnable state-value function
    :param dim_state: State dimensionality
    :param args: Value function parameters
    :param input_transform: Input transformation
    :return: A learnable value function for the state
    """
    value_function = NNValueFunction(
        dim_state=dim_state,
        layers=args.value_function_layers,
        biased_head=not args.value_function_unbiased_head,
        non_linearity=args.value_function_non_linearity,
        input_transform=input_transform,
        tau=args.value_function_tau,
    )

    args.update({"value_function": value_function.__class__.__name__})

    return value_function


def get_mpc_policy(
        dynamical_model: AbstractModel,
        reward: AbstractModel,
        args: argparse.Namespace,
        action_scale: Union[SupportsFloat, np.ndarray],
        terminal_reward: Union[AbstractValueFunction, None] = None,
        termination_model: Union[Callable, None] = None
) -> MPCPolicy:
    """
    Returns a learnable/fixed MPC policy
    :param dynamical_model: Dynamics model
    :param reward: Reward model
    :param args: Policy arguments
    :param action_scale: Action scale
    :param terminal_reward: Terminal reward function
    :param termination_model: Early termination check
    :return: MPC Policy
    """
    if args.mpc_solver == "cem":
        solver = CEMShooting(
            dynamical_model=dynamical_model,
            reward_model=reward,
            horizon=args.mpc_horizon,
            gamma=args.gamma,
            scale=1 / 8,
            action_scale=action_scale,
            num_iter=args.mpc_num_iter,
            num_particles=args.mpc_num_particles,
            num_elites=args.mpc_num_elites,
            alpha=args.mpc_alpha,
            terminal_reward=terminal_reward,
            termination_model=termination_model,
            warm_start=not args.mpc_not_warm_start,
            default_action=args.mpc_default_action,
            num_cpu=1,
        )
    else:
        raise NotImplementedError

    policy = MPCPolicy(solver)

    return policy
