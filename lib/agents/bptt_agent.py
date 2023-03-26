import contextlib
from dataclasses import asdict
from typing import Callable, Union

import torch
from torch import nn
from tqdm import tqdm
from gym.utils import colorize

from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.rollout import rollout_model
from rllib.value_function import AbstractValueFunction
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.dataset import average_dataclass, stack_list_of_tuples

from lib.algorithms.bptt import BPTT
from lib.datasets.utils import sample_states
from lib.agents.model_based_agent import ModelBasedAgent


class BPTTAgent(ModelBasedAgent):
    """Agent with BPTT policy for Model-based RL."""

    def __init__(
            self,
            dynamical_model: AbstractModel,
            reward_model: AbstractModel,
            termination_model: Callable,
            policy: AbstractPolicy,
            critic: AbstractValueFunction,
            model_optimizer: torch.optim.Optimizer = None,
            policy_optimizer: torch.optim.Optimizer = None,
            initial_distribution: torch.distributions.Distribution = None,
            true_model: Union[nn.Module, None] = None,
            exploration_scheme: str = "greedy",
            use_validation_set: bool = False,
            model_learn_num_iter: int = 50,
            policy_opt_gradient_steps: int = 500,
            sim_num_steps: int = 32,
            max_memory: int = 10000,
            td_lambda: float = 0.95,
            gamma: float = 0.99,
            tensorboard=False,
            comment="",
    ):

        algorithm = BPTT(
            gamma=gamma,
            td_lambda=td_lambda,
            policy=policy,
            critic=critic
        )

        super().__init__(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            model_optimizer=model_optimizer,
            algorithm=algorithm,
            value_function=critic,
            policy=policy,
            use_validation_set=use_validation_set,
            plan_horizon=0,  # TODO: Calling the mpc policy already plans.
            plan_samples=16,
            plan_elites=1,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=32,
            bootstrap=True,
            policy_opt_num_iter=1,
            policy_opt_batch_size=4,   # 4
            policy_opt_gradient_steps=policy_opt_gradient_steps,
            policy_opt_target_update_frequency=4,
            policy_update_frequency=1,
            optimizer=policy_optimizer,
            max_memory=max_memory,
            sim_num_steps=sim_num_steps,
            sim_refresh_interval=400,
            initial_distribution=initial_distribution,
            exploration_scheme=exploration_scheme,
            gamma=gamma,
            exploration_steps=0,
            exploration_episodes=0,
            tensorboard=tensorboard,
            comment=comment,
        )

        self.true_model = true_model

    def simulate_and_learn_policy(self):
        """

        :return:
        """
        print(colorize("\nOptimizing policy with simulated data from the model", "yellow"))

        self.dynamical_model.eval()

        for i in tqdm(range(self.policy_opt_num_iter)):
            self.learn_policy()

    def simulate_model(self, initial_states=None):
        """

        :return:
        """
        if initial_states is None:
            initial_states = self.sample_initial_states()

        if self.true_model is not None:
            dynamical_model = self.true_model
        else:
            dynamical_model = self.dynamical_model

        trajectory = rollout_model(
            dynamical_model=dynamical_model,
            reward_model=self.reward_model,
            policy=self.algorithm.policy,
            initial_state=initial_states,
            max_steps=self.sim_num_steps,
            termination_model=self.termination_model,
            detach_state=True,
        )
        self.sim_trajectory = stack_list_of_tuples(trajectory)

    def learn_policy(self):

        self.algorithm.reset()
        initial_states = self.sample_initial_states()

        for _ in range(self.policy_opt_gradient_steps):

            def closure():
                states = sample_states(initial_states, self.policy_opt_batch_size)
                self.simulate_model(initial_states=states)
                self._log_simulated_trajectory()
                self.optimizer.zero_grad()
                losses = self.algorithm(self.sim_trajectory)
                losses.combined_loss.backward()
                return losses

            if self.train_steps % self.policy_update_frequency == 0:
                cm = contextlib.nullcontext()
            else:
                cm = DisableGradient(self.policy)

            with cm:
                losses = self.optimizer.step(closure)

            self.logger.update(**asdict(average_dataclass(losses)))
            self.logger.update(**self.algorithm.info())

            self.counters["train_steps"] += 1

            if self.early_stop(losses, **self.algorithm.info()):
                break

        self.algorithm.reset()
        self.algorithm.update()
