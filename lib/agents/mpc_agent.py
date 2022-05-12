from typing import Callable

import torch
from rllib.agent import ModelBasedAgent
from rllib.policy import MPCPolicy


class MPCAgent(ModelBasedAgent):
    """Agent with MPC policy for Model-based RL."""

    def __init__(
            self,
            mpc_policy: MPCPolicy,
            model_optimizer: torch.optim.Optimizer = None,
            initial_distribution: torch.distributions.Distribution = None,
            exploration_scheme: str = "thompson_sampling",
            gamma: float = 1.0,
            tensorboard=False,
            comment="",
    ):

        self.algorithm = None

        super().__init__(
            dynamical_model=mpc_policy.solver.dynamical_model,
            reward_model=mpc_policy.solver.reward_model,
            termination_model=mpc_policy.solver.termination_model,
            model_optimizer=model_optimizer,
            algorithm=self.algorithm,
            value_function=mpc_policy.solver.terminal_reward,
            policy=mpc_policy,
            plan_horizon=0,  # Calling the mpc policy already plans.
            plan_samples=0,
            plan_elites=0,
            model_learn_num_iter=50,
            model_learn_batch_size=32,
            bootstrap=True,
            policy_opt_num_iter=0,
            max_memory=10000,
            initial_distribution=initial_distribution,
            exploration_scheme=exploration_scheme,
            gamma=gamma,
            exploration_steps=0,
            exploration_episodes=0,
            tensorboard=tensorboard,
            comment=comment,
        )
