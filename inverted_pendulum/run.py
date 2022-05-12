import gpytorch.settings
import torch
import numpy as np
import argparse

from dotmap import DotMap
from rllib.model import AbstractModel
from rllib.util.training.agent_training import train_agent

from inverted_pendulum import PendulumReward, PendulumStateTransform
from utils import get_mpc_agent, train_and_evaluate_agent

from rllib.agent.abstract_agent import AbstractAgent
from rllib.environment.systems import InvertedPendulum
from rllib.environment.system_environment import SystemEnvironment, AbstractEnvironment
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction


def argument_parser() -> argparse.ArgumentParser:
    """
    Returns parser for the inverted pendulum environment
    :return: parser: argument parser
    """
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment_max_steps", type=int, default=400)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=1)

    # Environment parameters
    parser.add_argument("--pendulum-mass", type=float, default=1)
    parser.add_argument("--pendulum-length", type=float, default=1)
    parser.add_argument("--pendulum-friction", type=float, default=0.005)
    parser.add_argument("--pendulum-step-size", type=float, default=0.0125)

    # Reward parameters
    parser.add_argument("--action-cost", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Model parameters
    parser.add_argument("--model-kind", type=str, default="ProbabilisticEnsemble")
    parser.add_argument("--model-num-heads", type=int, default=5)
    parser.add_argument("--model-layers", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--model-non-linearity", type=str, default="ReLU")
    parser.add_argument("--model-unbiased-head", action="store_false")  # TODO: Should be store_true
    parser.add_argument("--model-opt-lr", type=float, default=1e-3)
    parser.add_argument("--model-opt-weight-decay", type=float, default=0)
    parser.add_argument("--model-learn-num-iter", type=int, default=50)
    parser.add_argument("--model-learn-batch-size", type=int, default=32)

    # Simulation and replay buffer parameters
    parser.add_argument("--not-bootstrap", action="store_true")
    parser.add_argument("--max-memory", type=int, default=10000)
    parser.add_argument("--sim-max-memory", type=int, default=100000)
    parser.add_argument("--sim-num-steps", type=int, default=400)
    parser.add_argument("--sim-initial-dist-num-trajectories", type=int, default=4)
    parser.add_argument("--sim-initial-states-num-trajectories", type=int, default=4)
    parser.add_argument("--sim-memory-num-trajectories", type=int, default=0)

    # Value function parameters
    parser.add_argument("--value-function-layers", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--value-function-unbiased-head", action="store_false")  # TODO: Should be store_true
    parser.add_argument("--value-function-non-linearity", type=str, default="ReLU")
    parser.add_argument("--value-function-tau", type=float, default=0)

    # Agent parameters
    parser.add_argument("--agent-name", type=str, default="mpc")
    parser.add_argument("--exploration", type=str, default="greedy")

    # MPC parameters
    parser.add_argument("--mpc-solver", type=str, choices=["cem"], default="cem")
    parser.add_argument("--mpc-num-iter", type=int, default=5)
    parser.add_argument("--mpc-num-particles", type=int, default=16)
    parser.add_argument("--mpc-num-elites", type=int, default=1)
    parser.add_argument("--mpc-alpha", type=float, default=0)
    parser.add_argument("--mpc-not-warm-start", type=bool, default=False)
    parser.add_argument("--mpc-default-action", type=str,
                        choices=["zero", "constant", "mean"], default="zero")

    # Planning parameters

    return parser


def get_environment_and_agent(args: argparse.Namespace) -> (AbstractEnvironment, AbstractAgent):
    """
    Creates an environment and agent with the given parameters
    :param args: environment arguments
    :return: RL environment and agent
    """
    initial_state = torch.tensor([np.pi, 0.0])

    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])
    )

    reward_model = PendulumReward(action_cost=args.action_cost)

    environment = SystemEnvironment(
        InvertedPendulum(
            mass=args.pendulum_mass,
            length=args.pendulum_length,
            friction=args.pendulum_friction,
            step_size=args.pendulum_step_size
        ),
        reward=reward_model,
        initial_state=initial_state,
        termination_model=None,
    )

    # TODO: Check where these transformations are used.
    transformations = [
        ActionScaler(scale=environment.action_scale),
        MeanFunction(DeltaState())
    ]

    input_transform = PendulumStateTransform()

    if args.agent_name == "mpc":
        agent = get_mpc_agent(
            environment=environment,
            reward=reward_model,
            transformations=transformations,
            args=args,
            input_transform=input_transform,
            initial_distribution=exploratory_distribution
        )
    else:
        raise NotImplementedError

    return environment, agent


if __name__ == "__main__":

    parser = argument_parser()
    args = DotMap(vars(parser.parse_args()))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    environment, agent = get_environment_and_agent(args)

    train_and_evaluate_agent(environment, agent, args)

