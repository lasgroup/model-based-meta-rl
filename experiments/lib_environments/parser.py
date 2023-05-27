import argparse


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Returns parser for the inverted pendulum environment
    :return: parser: argument parser
    """
    parser = argparse.ArgumentParser()

    # Compute parameters
    parser.add_argument("--num-cpu-cores", type=int, default=4)

    # Environment parameters
    parser.add_argument(
        "--env-config-file",
        type=str,
        default="rccar_env.yaml",
        help="Choose one of the pre-defined environment config files"
    )
    parser.add_argument(
        "--env-group",
        type=str,
        default="rccar_envs",
        choices=["gym_envs", "mujocoMB_envs", "point_envs", "rccar_envs"]
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--skip-early-termination", action="store_true")

    # Agent parameters
    parser.add_argument(
        "--agent-name",
        type=str,
        default="mbpo",
        choices=["mpc", "mpc_policy", "bptt", "ppo", "sac", "mbpo"]
    )
    parser.add_argument(
        "--exploration",
        type=str,
        default="greedy",
        choices=["optimistic", "greedy", "thompson"]
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--agent-config-path", type=str, default="")

    # Training Parameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--eval-frequency", type=int, default=0)

    # Reward parameters
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--use-action-cost", action="store_true")
    parser.add_argument("--use-exact-termination-model", action="store_true")

    # Logger Parameters
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-to-file", type=bool, default=True)
    parser.add_argument("--safe-log-dir", type=bool, default=True)
    parser.add_argument("--save-statistics", action="store_true")
    parser.add_argument("--offline-logger", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--exp-name", type=str, default="test_exp")
    parser.add_argument("--wandb-project", type=str, default="MBRL")

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
    parser.add_argument("--model-opt-lr", type=float, default=1e-3)
    parser.add_argument("--model-opt-weight-decay", type=float, default=0.0)
    parser.add_argument("--model-learn-num-iter", type=int, default=50)
    parser.add_argument("--model-learn-batch-size", type=int, default=32)
    parser.add_argument("--model-prediction-strategy", type=str, default="moment_matching",
                        choices=["moment_matching", "sample_multiple_head"])
    parser.add_argument("--use-validation-set", action="store_true")
    parser.add_argument("--model-include-aleatoric-uncertainty", type=bool, default=True)

    # Simulation and replay buffer parameters
    parser.add_argument("--not-bootstrap", action="store_true")
    parser.add_argument("--max-memory", type=int, default=500000)
    parser.add_argument("--sim-max-memory", type=int, default=500000)
    parser.add_argument("--sim-num-steps", type=int, default=16)
    parser.add_argument("--sim-initial-dist-num-trajectories", type=int, default=8)
    parser.add_argument("--sim-initial-states-num-trajectories", type=int, default=8)
    parser.add_argument("--sim-memory-num-trajectories", type=int, default=0)
    parser.add_argument("--sim-n-envs", type=int, default=32)

    # Value function parameters
    parser.add_argument("--value-function-layers", type=int, nargs="*", default=[400, 400, 400, 400])
    parser.add_argument("--value-function-unbiased-head", action="store_true")
    parser.add_argument("--value-function-non-linearity", type=str, default="Swish")
    parser.add_argument("--value-function-tau", type=float, default=0)
    parser.add_argument("--value-function-num-heads", type=int, default=2)

    # Policy parameters
    parser.add_argument("--policy-layers", type=int, nargs="*", default=[400, 400, 400])
    parser.add_argument("--policy-opt-gradient-steps", type=int, default=500)
    parser.add_argument("--policy-unbiased-head", action="store_true")
    parser.add_argument("--policy-non-linearity", type=str, default="Swish")
    parser.add_argument("--policy-tau", type=float, default=0.005)
    parser.add_argument("--policy-deterministic", action="store_true")
    parser.add_argument("--policy-grad-steps", type=int, default=1)
    parser.add_argument("--policy-train-freq", type=int, default=1)

    # MPC parameters
    parser.add_argument("--mpc-solver", type=str, choices=["cem", "icem", "pets"], default="icem")
    parser.add_argument("--mpc-policy", type=str, choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--mpc-num-iter", type=int, default=5)
    parser.add_argument("--mpc-num-particles", type=int, default=1000)
    parser.add_argument("--mpc-num-elites", type=int, default=50)
    parser.add_argument("--mpc-pets-trajectory-samples", type=int, default=5)
    parser.add_argument("--mpc-horizon", type=int, default=16)
    parser.add_argument("--mpc-alpha", type=float, default=0.1)
    parser.add_argument("--mpc-noise-beta", type=float, default=2.0)
    parser.add_argument("--mpc-terminal-reward", type=bool, default=False)
    parser.add_argument("--mpc-not-warm-start", type=bool, default=False)
    parser.add_argument("--mpc-default-action", type=str,
                        choices=["zero", "constant", "mean"], default="constant")

    # PPO parameters
    parser.add_argument("--ppo-opt-lr", type=float, default=5e-5)
    parser.add_argument("--ppo-opt-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ppo-eta", type=float, default=0.01)
    parser.add_argument("--ppo-clip-gradient-val", type=float, default=2.0)

    # SAC parameters
    parser.add_argument("--sac-opt-lr", type=float, default=3e-4)
    parser.add_argument("--sac-opt-weight-decay", type=float, default=0)
    parser.add_argument("--sac-memory-len", type=int, default=100000)
    parser.add_argument("--sac-use-sde", type=bool, default=False)
    parser.add_argument("--sac-ent-coef", type=str, default="auto")

    # Planning parameters

    return parser
