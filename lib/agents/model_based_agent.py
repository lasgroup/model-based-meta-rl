import argparse
import contextlib
from dataclasses import asdict
from typing import Callable

import gpytorch
import torch
from gym.utils import colorize

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset import BootstrapExperienceReplay, StateExperienceReplay, Observation, stack_list_of_tuples, \
    average_dataclass
from rllib.model import TransformedModel, AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.rollout import rollout_model
from rllib.util.training.model_learning import train_model
from rllib.util.utilities import tensor_to_distribution
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.value_estimation import mb_return
from rllib.value_function import AbstractValueFunction
from tqdm import tqdm

from utils.utils import get_logger_layout


class ModelBasedAgent(AbstractAgent):
    """
    Agent for Model based RL
    """

    # TODO: Check if default values make sense.
    def __init__(
            self,
            dynamical_model: AbstractModel,
            reward_model: AbstractModel,
            termination_model: Callable,
            model_optimizer: torch.optim.Optimizer,
            algorithm: AbstractAlgorithm,
            value_function: AbstractValueFunction,
            policy: AbstractPolicy,
            model_learn_num_iter: int = 0,
            model_learn_batch_size: int = 32,
            plan_horizon: int = 8,
            plan_samples: int = 16,
            plan_elites: int = 1,
            bootstrap: bool = True,
            max_memory: int = 10000,
            policy_opt_num_iter: int = 0,
            policy_opt_batch_size: int = None,
            policy_opt_gradient_steps: int = 0,
            policy_opt_target_update_frequency: int = 1,
            policy_update_frequency: int = 1,
            optimizer: torch.optim.Optimizer = None,
            sim_num_steps: int = 20,
            sim_initial_states_num_trajectories: int = 8,
            sim_initial_dist_num_trajectories: int = 8,
            sim_memory_num_trajectories: int = 0,
            sim_max_memory: int = 10000,
            sim_refresh_interval: int = 1,
            sim_num_subsample: int = 1,
            initial_distribution: torch.distributions.Distribution = None,
            exploration_scheme: str = None,
            gamma=1.0,
            exploration_steps=0,
            exploration_episodes=0,
            tensorboard=False,
            comment=""
    ):
        self.algorithm = algorithm
        super().__init__(
            train_frequency=0,
            num_rollouts=0,
            policy_update_frequency=policy_update_frequency,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment
        )

        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        self.algorithm = algorithm
        self.model_optimizer = model_optimizer
        self.value_function = value_function

        self.model_learn_num_iter = model_learn_num_iter
        self.model_learn_batch_size = model_learn_batch_size

        self.policy = policy

        self.plan_horizon = plan_horizon
        self.plan_samples = plan_samples
        self.plan_elites = plan_elites

        if hasattr(dynamical_model.base_model, "num_heads"):
            num_heads = dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        self.dataset = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=dynamical_model.transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
        )
        self.sim_dataset = StateExperienceReplay(
            max_len=sim_max_memory, dim_state=dynamical_model.dim_state
        )
        self.initial_states = StateExperienceReplay(
            max_len=sim_max_memory, dim_state=self.dynamical_model.dim_state
        )

        self.policy_opt_num_iter = policy_opt_num_iter
        if policy_opt_batch_size is None:
            policy_opt_batch_size = self.model_learn_batch_size
        self.policy_opt_batch_size = policy_opt_batch_size
        self.policy_opt_gradient_steps = policy_opt_gradient_steps
        self.policy_opt_target_update_frequency = policy_opt_target_update_frequency
        self.optimizer = optimizer

        self.sim_trajectory = None
        self.sim_num_steps = sim_num_steps
        self.sim_initial_states_num_trajectories = sim_initial_states_num_trajectories
        self.sim_initial_dist_num_trajectories = sim_initial_dist_num_trajectories
        self.sim_memory_num_trajectories = sim_memory_num_trajectories
        self.sim_refresh_interval = sim_refresh_interval
        self.sim_num_subsample = sim_num_subsample
        self.initial_distribution = initial_distribution
        self.new_episode = True
        self.exploration_scheme = exploration_scheme

        if exploration_scheme == "thompson":
            self.dynamical_model.set_prediction_strategy("posterior")

        logger_layout = get_logger_layout(num_heads)
        if self.logger.writer is not None:
            self.logger.writer.add_custom_scalars(logger_layout)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state:
        :return:
        """
        if self.plan_horizon == 0:
            action = super().act(state)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            state = state.float()
            self.pi = tensor_to_distribution(self.policy(state), **self.policy.dist_params)
            action = self.plan(state).detach()

        action_scale = self.policy.action_scale.detach()

        return action.clip(-action_scale, +action_scale)

    def observe(self, observation: Observation):
        """

        :param observation:
        :return:
        """
        super().observe(observation)
        self.dataset.append(observation)
        if self.new_episode:
            self.initial_states.append(observation.state.unsqueeze(0))
            self.new_episode = False

    def start_episode(self):
        """

        :return:
        """
        super().start_episode()
        self.new_episode = True
        if self.exploration_scheme == "thompson":
            self.dynamical_model.sample_posterior()

    def end_episode(self):
        if self.training:
            self.learn()
        super().end_episode()

    def plan(self, state: torch.Tensor) -> torch.Tensor:
        """

        :param state:
        :return:
        """
        self.dynamical_model.eval()

        value, trajectory = mb_return(
            state,
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=self.policy,
            num_model_steps=self.plan_horizon,
            gamma=self.gamma,
            num_particles=self.plan_samples,
            value_function=self.value_function,
            reward_transformer=self.algorithm.reward_transformer,
            termination_model=self.termination_model,
        )

        idx = torch.topk(value.squeeze(), k=self.plan_elites, largest=True)[1]

        # Return first action and the mean over the elite samples.
        return trajectory.action[idx, 0].mean(0)

    def learn(self):
        """

        :return:
        """
        self.learn_model()

        if (
                self.total_steps < self.exploration_steps or
                self.total_episodes < self.exploration_episodes
        ):
            return
        else:
            self.simulate_and_learn_policy()

    def learn_model(self):
        """

        :return:
        """
        if self.model_learn_num_iter > 0:
            print(colorize("Training Dynamics Model", "yellow"))

            train_model(
                self.dynamical_model.base_model,
                train_set=self.dataset,
                max_iter=self.model_learn_num_iter,
                optimizer=self.model_optimizer,
                logger=self.logger,
                batch_size=self.model_learn_batch_size,
                epsilon=-1.0,
            )

    def simulate_and_learn_policy(self):
        """

        :return:
        """
        print(colorize("\nOptimizing policy with simulated data from the model", "yellow"))

        self.dynamical_model.eval()
        self.sim_dataset.reset()

        with DisableGradient(self.dynamical_model), gpytorch.settings.fast_pred_var():
            for i in tqdm(range(self.policy_opt_num_iter)):
                with torch.no_grad():
                    self.simulate_model()

                self._log_simulated_trajectory()

                self.learn_policy()

                if (
                    self.sim_refresh_interval > 0 and
                        (i + 1) % self.sim_refresh_interval == 0
                ):
                    self.sim_dataset.reset()

    def simulate_model(self):
        """

        :return:
        """
        initial_states = self.initial_states.sample_batch(
            self.sim_initial_states_num_trajectories
        )

        if self.sim_initial_states_num_trajectories > 0:
            initial_states_ = self.initial_distribution.sample(
                (self.sim_initial_dist_num_trajectories, )
            )
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        if self.sim_memory_num_trajectories > 0:
            obs, *_ = self.dataset.sample_batch(self.sim_memory_num_trajectories)
            for transform in self.dataset.transformations:
                obs = transform.inverse(obs)
            initial_states_ = obs.state[:, 0, :]
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        initial_states = initial_states.unsqueeze(0)

        self.policy.reset()

        trajectory = rollout_model(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=self.algorithm.policy,
            initial_state=initial_states,
            max_steps=self.sim_num_steps,
            termination_model=self.termination_model,
        )
        self.sim_trajectory = stack_list_of_tuples(trajectory)

        states = self.sim_trajectory.state.reshape(-1, *self.dynamical_model.dim_state)
        self.sim_dataset.append(states[:: self.sim_num_subsample])

    def learn_policy(self):

        self.algorithm.reset()
        for _ in range(self.policy_opt_gradient_steps):

            def closure():
                states = self.sim_dataset.sample_batch(self.policy_opt_batch_size)
                # TODO: can be done by sampling from sim_trajectory
                with torch.no_grad():
                    trajectory = rollout_model(
                        dynamical_model=self.dynamical_model,
                        reward_model=self.reward_model,
                        policy=self.algorithm.old_policy,
                        initial_state=states,
                        max_steps=self.sim_num_steps,
                        termination_model=self.termination_model,
                    )
                    # trajectory = stack_list_of_tuples(trajectory)
                self.optimizer.zero_grad()
                losses = self.algorithm(trajectory)
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
            if self.train_steps % self.policy_opt_target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()

            if self.early_stop(losses, **self.algorithm.info()):
                break

        self.algorithm.reset()

    def _log_simulated_trajectory(self):
        average_return = self.sim_trajectory.reward.sum(0).mean().item()
        average_scale = (
            torch.diagonal(self.sim_trajectory.next_state_scale_tril, dim1=-1, dim2=-2)
                .square()
                .sum(-1)
                .sum(0)
                .mean()
                .sqrt()
                .item()
        ) if self.sim_trajectory.next_state_scale_tril.ndim > 1 else torch.nan
        self.logger.update(sim_entropy=self.sim_trajectory.entropy.mean().item())
        self.logger.update(sim_return=average_return)
        self.logger.update(sim_scale=average_scale)
        self.logger.update(sim_max_state=self.sim_trajectory.state.abs().max().item())
        self.logger.update(sim_max_action=self.sim_trajectory.action.abs().max().item())
        try:
            r_ctrl = self.reward_model.reward_ctrl.mean().detach().item()
            r_state = self.reward_model.reward_state.mean().detach().item()
            self.logger.update(sim_reward_ctrl=r_ctrl)
            self.logger.update(sim_reward_state=r_state)
        except AttributeError:
            pass
        try:
            r_o = self.reward_model.reward_dist_to_obj
            r_g = self.reward_model.reward_dist_to_goal
            self.logger.update(sim_reward_dist_to_obj=r_o.mean().detach().item())
            self.logger.update(sim_reward_dist_to_goal=r_g.mean().detach().item())
        except AttributeError:
            pass
