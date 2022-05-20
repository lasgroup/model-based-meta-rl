from typing import Callable

import torch
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.model import AbstractModel

from lib.agents.model_based_agent import ModelBasedAgent


class MPCPolicyAgent(ModelBasedAgent):
    """Agent with MPC policy for Model-based RL."""

    def __init__(
            self,
            dynamical_model: AbstractModel,
            reward_model: AbstractModel,
            termination_model: Callable,
            algorithm: AbstractAlgorithm,
            model_optimizer: torch.optim.Optimizer = None,
            policy_optimizer: torch.optim.Optimizer = None,
            initial_distribution: torch.distributions.Distribution = None,
            exploration_scheme: str = "thompson_sampling",
            gamma: float = 1.0,
            tensorboard=False,
            comment="",
    ):

        self.algorithm = algorithm

        super().__init__(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            model_optimizer=model_optimizer,
            algorithm=self.algorithm,
            value_function=self.algorithm.critic_target,
            policy=self.algorithm.policy,
            plan_horizon=8,  # Calling the mpc policy already plans.
            plan_samples=16,
            plan_elites=1,
            model_learn_num_iter=50,
            model_learn_batch_size=32,
            bootstrap=True,
            policy_opt_num_iter=10,
            policy_opt_batch_size=4,
            policy_opt_gradient_steps=50,
            policy_opt_target_update_frequency=4,
            policy_update_frequency=1,
            optimizer=policy_optimizer,
            max_memory=10000,
            sim_num_steps=32,
            sim_refresh_interval=400,
            initial_distribution=initial_distribution,
            exploration_scheme=exploration_scheme,
            gamma=gamma,
            exploration_steps=0,
            exploration_episodes=0,
            tensorboard=tensorboard,
            comment=comment,
        )
