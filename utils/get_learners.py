from typing import Union, SupportsFloat, Iterable, Callable

import argparse
import numpy as np
from torch import nn

from rllib.policy import MPCPolicy, NNPolicy
from rllib.dataset.transforms import AbstractTransform
from rllib.algorithms.mpc import CEMShooting
from rllib.model import AbstractModel, EnsembleModel, TransformedModel
from rllib.value_function import AbstractValueFunction, NNValueFunction

from lib.hucrl.hallucinated_model import HallucinatedModel


def get_model(
        dim_state: int,
        dim_action: int,
        params: argparse.Namespace,
        transformations: Iterable[AbstractTransform] = None,
        input_transform: nn.Module = None
) -> AbstractModel:
    """
    Returns a learnable dynamics model for the environment
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param params: model parameters
    :param transformations: State and action transformations
    :param input_transform: Input transformation
    :return: Dynamics model for the environment
    """
    transformations = list() if not transformations else transformations

    if params.model_kind in ["ProbabilisticEnsemble", "DeterministicEnsemble"]:
        model = EnsembleModel(
            dim_state=dim_state,
            dim_action=dim_action,
            num_heads=params.model_num_heads,
            layers=params.model_layers,
            biased_head=not params.model_unbiased_head,
            non_linearity=params.model_non_linearity,
            input_transform=input_transform,
            deterministic=params.model_kind == "DeterministicEnsemble",
        )
    else:
        raise NotImplementedError

    params.update({"model": model.__class__.__name__})

    if params.exploration == "optimistic":
        dynamical_model = HallucinatedModel(model, transformations, beta=params.beta)
    else:
        dynamical_model = TransformedModel(model, transformations)

    return dynamical_model


def get_value_function(
        dim_state: int,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> AbstractValueFunction:
    """
    Returns a learnable state-value function
    :param dim_state: State dimensionality
    :param params: Value function parameters
    :param input_transform: Input transformation
    :return: A learnable value function for the state
    """
    value_function = NNValueFunction(
        dim_state=dim_state,
        layers=params.value_function_layers,
        biased_head=not params.value_function_unbiased_head,
        non_linearity=params.value_function_non_linearity,
        input_transform=input_transform,
        tau=params.value_function_tau,
    )

    params.update({"value_function": value_function.__class__.__name__})

    return value_function


def get_nn_policy(
        dim_state: tuple,
        dim_action: tuple,
        input_transform: nn.Module,
        action_scale: Union[SupportsFloat, np.ndarray],
        params: argparse.Namespace,
) -> NNPolicy:
    """
    Returns a learnable NN policy
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param input_transform: Input transformation
    :param params: Policy arguments
    :param action_scale: Action scale
    :return: NN Policy
    """

    if params.exploration == "optimistic":
        dim_action = (dim_action[0] + dim_state[0],)

    policy = NNPolicy(
        dim_state=dim_state,
        dim_action=dim_action,
        input_transform=input_transform,
        action_scale=action_scale,
        tau=params.policy_tau,
        layers=params.policy_layers,
        biased_head=not params.policy_unbiased_head,
        non_linearity=params.policy_non_linearity,
        deterministic=params.policy_deterministic,
        squashed_output=True
    )

    params.update({"policy": policy.__class__.__name__})

    return policy


def get_mpc_policy(
        dynamical_model: AbstractModel,
        reward: AbstractModel,
        params: argparse.Namespace,
        action_scale: Union[SupportsFloat, np.ndarray],
        terminal_reward: Union[AbstractValueFunction, None] = None,
        termination_model: Union[Callable, None] = None
) -> MPCPolicy:
    """
    Returns a learnable/fixed MPC policy
    :param dynamical_model: Dynamics model
    :param reward: Reward model
    :param params: Policy arguments
    :param action_scale: Action scale
    :param terminal_reward: Terminal reward function
    :param termination_model: Early termination check
    :return: MPC Policy
    """
    if params.mpc_solver == "cem":
        solver = CEMShooting(
            dynamical_model=dynamical_model,
            reward_model=reward,
            horizon=params.mpc_horizon,
            gamma=params.gamma,
            scale=1 / 8,
            action_scale=action_scale,
            num_iter=params.mpc_num_iter,
            num_elites=params.mpc_num_elites,
            alpha=params.mpc_alpha,
            terminal_reward=terminal_reward,
            termination_model=termination_model,
            warm_start=not params.mpc_not_warm_start,
            default_action=params.mpc_default_action,
            num_cpu=1,
        )
    else:
        raise NotImplementedError

    policy = MPCPolicy(solver)

    return policy
