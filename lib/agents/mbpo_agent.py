from typing import Callable

import torch

from rllib.model import AbstractModel
from stable_baselines3.common.base_class import BaseAlgorithm
from wandb.integration.sb3 import WandbCallback

from lib.agents.model_based_agent import ModelBasedAgent
from lib.environments.wrappers.model_based_environment import ModelBasedEnvironment


class MBPOAgent(ModelBasedAgent):
    """Agent with SB3 policy for Model-based RL."""

    def __init__(
            self,
            dynamical_model: AbstractModel,
            reward_model: AbstractModel,
            termination_model: Callable,
            model_based_env: ModelBasedEnvironment,
            policy: BaseAlgorithm,
            model_optimizer: torch.optim.Optimizer = None,
            initial_distribution: torch.distributions.Distribution = None,
            exploration_scheme: str = "greedy",
            use_validation_set: bool = False,
            model_learn_num_iter: int = 50,
            policy_opt_gradient_steps: int = 500,
            sim_num_steps: int = 32,
            max_memory: int = 10000,
            gamma: float = 0.99,
            tensorboard=False,
            comment="",
    ):

        super().__init__(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            termination_model=termination_model,
            model_optimizer=model_optimizer,
            policy=policy,
            value_function=None,
            algorithm=None,
            use_validation_set=use_validation_set,
            plan_horizon=0,
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
            optimizer=None,
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

        self.model_based_env = model_based_env
        self.model_based_env.set_initial_distribution(self.sample_initial_states)
        self.pi = None

        self.num_learn_steps = self.policy.train_freq * self.policy_opt_gradient_steps / self.policy.gradient_steps

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state:
        :return:
        """
        if self.training:
            action, _ = self.policy.predict(state, deterministic=False)
        else:
            action, _ = self.policy.predict(state, deterministic=True)

        action = action.clip(-1.0, 1.0)
        action_scale = self.policy.action_scale.detach()

        action = action_scale * action

        action = action.detach().to("cpu").numpy()

        return action.clip(-action_scale, +action_scale)

    def simulate_and_learn_policy(self):

        if self.logger.use_wandb:
            callback = WandbCallback()
        else:
            callback = None

        self.policy.learn(
            total_timesteps=self.num_learn_steps,
            log_interval=5,
            callback=callback
        )

        # self.policy.reset_buffer()  # Clear buffer after every

    def save(self, filename, directory=None):
        pass
