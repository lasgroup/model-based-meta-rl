from typing import Union, SupportsFloat, Iterable, Callable

import argparse
import numpy as np
from torch import nn

from rllib.policy import MPCPolicy, NNPolicy
from rllib.dataset.transforms import AbstractTransform
from rllib.algorithms.mpc import CEMShooting
from rllib.model import AbstractModel, EnsembleModel, TransformedModel, NNModel
from rllib.value_function import AbstractValueFunction, NNValueFunction, AbstractQFunction, NNEnsembleQFunction, \
    NNQFunction

from lib.model.bayesian_model import BayesianNNModel
from lib.policies.rnn_policy import RNNPolicy
from lib.solvers.icem_shooting import ICEMShooting
from lib.hucrl.hallucinated_model import HallucinatedModel
from lib.value_function.rnn_value_function import RNNValueFunction


def get_model(
        dim_state: tuple,
        dim_action: tuple,
        num_states: int,
        num_actions: int,
        params: argparse.Namespace,
        transformations: Iterable[AbstractTransform] = None,
        input_transform: nn.Module = None
) -> AbstractModel:
    """
    Returns a learnable dynamics model for the environment
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param num_states: Number of states
    :param num_actions: Number of actions
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
            num_states=num_states,
            num_actions=num_actions,
            num_heads=params.model_num_heads,
            layers=params.model_layers,
            biased_head=not params.model_unbiased_head,
            non_linearity=params.model_non_linearity,
            input_transform=input_transform,
            deterministic=params.model_kind == "DeterministicEnsemble",
        )
    elif params.model_kind in ["ProbabilisticNN", "DeterministicNN"]:
        model = NNModel(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            layers=params.model_layers,
            biased_head=not params.model_unbiased_head,
            non_linearity=params.model_non_linearity,
            input_transform=input_transform,
            heteroscedastic=params.model_heteroscedastic,
            deterministic=params.model_kind == "DeterministicNN",
        )
    elif params.model_kind in ["BayesianNN"]:
        model = BayesianNNModel(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            layers=params.model_layers,
            biased_head=not params.model_unbiased_head,
            non_linearity=params.model_non_linearity,
            input_transform=input_transform,
            deterministic=False,
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
        dim_state: tuple,
        dim_action: tuple,
        num_states: int,
        num_actions: int,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> AbstractValueFunction:
    """
    Returns a learnable state-value function
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param num_states: Number of states
    :param num_actions: Number of actions
    :param params: Value function parameters
    :param input_transform: Input transformation
    :return: A learnable value function for the state
    """
    value_function = NNValueFunction(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        layers=params.value_function_layers,
        biased_head=not params.value_function_unbiased_head,
        non_linearity=params.value_function_non_linearity,
        input_transform=input_transform,
        tau=params.value_function_tau,
    )

    params.update({"value_function": value_function.__class__.__name__})

    return value_function


def get_recurrent_value_function(
        dim_state: tuple,
        dim_action: tuple,
        num_states: int,
        num_actions: int,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> RNNValueFunction:
    """
    Returns a learnable recurrent state-value function
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param num_states: Number of states
    :param num_actions: Number of actions
    :param params: Value function parameters
    :param input_transform: Input transformation
    :return: A learnable value function for the state
    """
    value_function = RNNValueFunction(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        embedding_layers=(200, 200),
        layers=params.value_function_layers,
        biased_head=not params.value_function_unbiased_head,
        non_linearity="ReLU",
        input_transform=input_transform,
        tau=params.value_function_tau,
    )

    params.update({"value_function": value_function.__class__.__name__})

    return value_function


def get_q_function(
        dim_state: tuple,
        dim_action: tuple,
        num_states: int,
        num_actions: int,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> NNQFunction:
    """
    Returns a learnable Q value function
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param num_states: Number of states
    :param num_actions: Number of actions
    :param params: Q function parameters (Same as value function parameters)
    :param input_transform: Input transformation
    :return: A learnable q function for the state-action pair
    """
    if params.exploration == "optimistic":
        dim_action = (dim_action[0] + dim_state[0],)

    q_function = NNEnsembleQFunction(
        num_heads=params.value_function_num_heads,
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        layers=params.value_function_layers,
        biased_head=not params.value_function_unbiased_head,
        non_linearity=params.value_function_non_linearity,
        input_transform=input_transform,
        tau=params.value_function_tau,
    )

    params.update({"q_function": q_function.__class__.__name__})

    return q_function


def get_nn_policy(
        dim_state: tuple,
        dim_action: tuple,
        num_states: int,
        num_actions: int,
        input_transform: nn.Module,
        action_scale: Union[SupportsFloat, np.ndarray],
        params: argparse.Namespace,
) -> NNPolicy:
    """
    Returns a learnable NN policy
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param num_states: Number of states
    :param num_actions: Number of actions
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
        num_states=num_states,
        num_actions=num_actions,
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


def get_rnn_policy(
        dim_state: tuple,
        dim_action: tuple,
        num_states: int,
        num_actions: int,
        input_transform: nn.Module,
        action_scale: Union[SupportsFloat, np.ndarray],
        params: argparse.Namespace,
) -> RNNPolicy:
    """
    Returns a learnable RNN policy
    :param dim_state: State dimensionality
    :param dim_action: Action dimensionality
    :param num_states: Number of states
    :param num_actions: Number of actions
    :param input_transform: Input transformation
    :param params: Policy arguments
    :param action_scale: Action scale
    :return: RNN Policy
    """

    if params.exploration == "optimistic":
        dim_action = (dim_action[0] + dim_state[0],)

    policy = RNNPolicy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=action_scale,
        tau=params.policy_tau,
        embedding_layers=(200, 200),
        layers=params.policy_layers,
        biased_head=not params.policy_unbiased_head,
        non_linearity="ReLU",
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
            gamma=params.gamma,
            scale=1 / 8,
            action_scale=action_scale,
            num_model_steps=params.mpc_horizon,
            num_iter=params.mpc_num_iter,
            num_elites=params.mpc_num_elites,
            alpha=params.mpc_alpha,
            terminal_reward=terminal_reward,
            termination_model=termination_model,
            warm_start=not params.mpc_not_warm_start,
            default_action=params.mpc_default_action,
            num_cpu=1,
        )
    elif params.mpc_solver == "icem":
        solver = ICEMShooting(
            dynamical_model=dynamical_model,
            reward_model=reward,
            gamma=params.gamma,
            scale=0.5,
            action_scale=action_scale,
            num_model_steps=params.mpc_horizon,
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
