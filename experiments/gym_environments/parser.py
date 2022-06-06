import argparse


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Returns parser for the inverted pendulum environment
    :return: parser: argument parser
    """
    parser = argparse.ArgumentParser()

    # Agent parameters
    parser.add_argument(
        "--agent-name",
        type=str,
        default="mpc_policy",
        choices=["mpc", "mpc_policy", "ppo"]
    )
    parser.add_argument(
        "--exploration",
        type=str,
        default="optimistic",
        choices=["optimistic", "greedy", "thompson"]
    )
    parser.add_argument("--beta", type=float, default=1.0)

    # Training Parameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=2)

    # Logger Parameters
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-to-file", type=bool, default=True)
    parser.add_argument("--save-statistics", action="store_true")
    parser.add_argument("--offline_logger", type=bool, default=False)

    # Environment parameters
    parser.add_argument(
        "--env-config-file",
        type=str,
        default="cart-pole-mujoco.yaml",
        help="Choose one of the pre-defined environment config files"
    )
    parser.add_argument("--env-config-path", type=str, default="config/envs")
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--max-steps", type=int, default=400)

    # Reward parameters
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
    parser.add_argument("--sim-max-memory", type=int, default=10000)
    parser.add_argument("--sim-num-steps", type=int, default=400)
    parser.add_argument("--sim-initial-dist-num-trajectories", type=int, default=8)
    parser.add_argument("--sim-initial-states-num-trajectories", type=int, default=8)
    parser.add_argument("--sim-memory-num-trajectories", type=int, default=0)

    # Value function parameters
    parser.add_argument("--value-function-layers", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--value-function-unbiased-head", action="store_false")  # TODO: Should be store_true
    parser.add_argument("--value-function-non-linearity", type=str, default="ReLU")
    parser.add_argument("--value-function-tau", type=float, default=0)

    # Value function parameters
    parser.add_argument("--policy-layers", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--policy-unbiased-head", action="store_false")  # TODO: Should be store_true
    parser.add_argument("--policy-non-linearity", type=str, default="ReLU")
    parser.add_argument("--policy-tau", type=float, default=0)
    parser.add_argument("--policy-deterministic", action="store_true")

    # MPC parameters
    parser.add_argument("--mpc-solver", type=str, choices=["cem"], default="cem")
    parser.add_argument("--mpc-num-iter", type=int, default=5)
    parser.add_argument("--mpc-num-particles", type=int, default=16)
    parser.add_argument("--mpc-num-elites", type=int, default=1)
    parser.add_argument("--mpc-alpha", type=float, default=0)
    parser.add_argument("--mpc-not-warm-start", type=bool, default=False)
    parser.add_argument("--mpc-default-action", type=str,
                        choices=["zero", "constant", "mean"], default="zero")

    # PPO parameters
    parser.add_argument("--ppo-opt-lr", type=float, default=1e-4)
    parser.add_argument("--ppo-opt-weight-decay", type=float, default=0)

    # Planning parameters

    return parser
