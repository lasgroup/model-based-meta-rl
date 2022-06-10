import argparse
import sys

import torch
import numpy as np
from torch import nn

from utils.logger import Logger
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper
from utils.get_agents import get_mpc_agent, get_mpc_policy_agent, get_ppo_agent

from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance
from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.systems import InvertedPendulum
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction
from rllib.environment.system_environment import SystemEnvironment, AbstractEnvironment


# TODO: From H-UCRL
class PendulumReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0):
        super().__init__(dim_state=(2,), dim_action=(1,), model_kind="rewards")
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-0.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_cost + action_cost

        return cost.unsqueeze(-1), torch.zeros(1)


class PendulumStateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat(
            (torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1
        )
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((angle, angular_velocity), dim=-1)
        return states_


def get_environment_and_agent(params: argparse.Namespace) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param params: environment arguments
    :return: RL environment and agent
    """
    initial_state = torch.tensor([np.pi, 0.0])

    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])
    )

    reward_model = PendulumReward(action_cost=params.action_cost)

    environment = SystemEnvironment(
        InvertedPendulum(
            mass=params.pendulum_mass,
            length=params.pendulum_length,
            friction=params.pendulum_friction,
            step_size=params.pendulum_step_size
        ),
        reward=reward_model,
        initial_state=initial_state,
        termination_model=None,
    )

    transformations = [
        ActionScaler(scale=environment.action_scale),
        MeanFunction(DeltaState())
    ]

    input_transform = PendulumStateTransform()

    if params.agent_name == "mpc":
        agent, comment = get_mpc_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            params=params,
            input_transform=input_transform,
            initial_distribution=exploratory_distribution
        )
    elif params.agent_name == "mpc_policy":
        agent, comment = get_mpc_policy_agent(
            environment=environment,
            reward_model=reward_model,
            transformations=transformations,
            params=params,
            input_transform=input_transform,
            initial_distribution=exploratory_distribution
        )
    elif params.agent_name == "ppo":
        agent, comment = get_ppo_agent(
            environment=environment,
            params=params,
            input_transform=input_transform
        )
    else:
        raise NotImplementedError

    agent.logger = Logger(
        name=f"inverted_pendulum_{params.agent_name}_{params.exploration}",
        comment=comment,
        save_statistics=params.save_statistics,
        results_dir="results",
        use_wandb=True,
        offline_mode=params.offline_logger
    )
    if params.log_to_file:
        sys.stdout = agent.logger

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)

    return environment, agent
