import argparse


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Returns parser for the inverted pendulum environment
    :return: parser: argument parser
    """
    parser = argparse.ArgumentParser()

    # Environment parameters
    parser.add_argument(
        "--env-config-file",
        type=str,
        default="point_env_2d.yaml",
        help="Choose one of the pre-defined environment config files"
    )
    parser.add_argument(
        "--env-group",
        type=str,
        default="point_envs",
        choices=["gym_envs", "mujocoMB_envs", "point_envs"]
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)

    # Agent parameters
    parser.add_argument(
        "--agent-name",
        type=str,
        default="mpc",
        choices=["mpc", "mpc_policy", "ppo", "sac"]
    )
    # TODO: Check where exploration is used
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
    parser.add_argument("--test-episodes", type=int, default=5)

    # Reward parameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--use-action-cost", action="store_true")

    # Logger Parameters
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-to-file", type=bool, default=True)
    parser.add_argument("--save-statistics", action="store_true")
    parser.add_argument("--offline-logger", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")

    # Model parameters
    parser.add_argument(
        "--model-kind",
        type=str,
        default="BayesianNN",
        choices=["ProbabilisticEnsemble", "BayesianNN"]
    )
    parser.add_argument("--model-num-heads", type=int, default=5)
    parser.add_argument("--model-layers", type=int, nargs="*", default=[200, 200, 200, 200])
    parser.add_argument("--model-unbiased-head", action="store_true")
    parser.add_argument("--model-heteroscedastic", type=bool, default=True)
    parser.add_argument("--model-non-linearity", type=str, default="Tanh")
    parser.add_argument("--model-opt-lr", type=float, default=3e-4)
    parser.add_argument("--model-opt-weight-decay", type=float, default=0.01)
    parser.add_argument("--model-learn-num-iter", type=int, default=50)
    parser.add_argument("--model-learn-batch-size", type=int, default=32)
    parser.add_argument("--use-validation-set", action="store_true")

    # Simulation and replay buffer parameters
    parser.add_argument("--not-bootstrap", action="store_true")
    parser.add_argument("--max-memory", type=int, default=10000)
    parser.add_argument("--sim-max-memory", type=int, default=10000)
    parser.add_argument("--sim-num-steps", type=int, default=16)
    parser.add_argument("--sim-initial-dist-num-trajectories", type=int, default=8)
    parser.add_argument("--sim-initial-states-num-trajectories", type=int, default=8)
    parser.add_argument("--sim-memory-num-trajectories", type=int, default=0)

    # Value function parameters
    parser.add_argument("--value-function-layers", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--value-function-unbiased-head", action="store_true")
    parser.add_argument("--value-function-non-linearity", type=str, default="Tanh")
    parser.add_argument("--value-function-tau", type=float, default=0)
    parser.add_argument("--value-function-num-heads", type=int, default=2)

    # Value function parameters
    parser.add_argument("--policy-layers", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--policy-unbiased-head", action="store_true")
    parser.add_argument("--policy-non-linearity", type=str, default="Tanh")
    parser.add_argument("--policy-tau", type=float, default=0)
    parser.add_argument("--policy-deterministic", action="store_true")

    # MPC parameters
    parser.add_argument("--mpc-solver", type=str, choices=["cem", "icem"], default="cem")
    parser.add_argument("--mpc-policy", type=str, choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--mpc-num-iter", type=int, default=5)
    parser.add_argument("--mpc-num-particles", type=int, default=400)
    parser.add_argument("--mpc-num-elites", type=int, default=10)
    parser.add_argument("--mpc-horizon", type=int, default=16)
    parser.add_argument("--mpc-alpha", type=float, default=0.1)
    parser.add_argument("--mpc-terminal-reward", type=bool, default=False)
    parser.add_argument("--mpc-not-warm-start", type=bool, default=False)
    parser.add_argument("--mpc-default-action", type=str,
                        choices=["zero", "constant", "mean"], default="constant")

    # PPO parameters
    parser.add_argument("--ppo-opt-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-opt-weight-decay", type=float, default=0)
    parser.add_argument("--ppo-eta", type=float, default=0.01)

    # SAC parameters
    parser.add_argument("--sac-opt-lr", type=float, default=3e-4)
    parser.add_argument("--sac-opt-weight-decay", type=float, default=0)
    parser.add_argument("--sac-memory-len", type=int, default=100000)

    # Planning parameters

    return parser
