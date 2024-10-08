import argparse


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Returns parser for the inverted pendulum environment
    :return: parser: argument parser
    """
    parser = argparse.ArgumentParser()

    # Compute parameters
    parser.add_argument("--num-cpu-cores", type=int, default=5)

    # Environment parameters
    parser.add_argument(
        "--env-config-file",
        type=str,
        default="random-sim-rccar-env.yaml",
        help="Choose one of the pre-defined environment config files"
    )
    parser.add_argument(
        "--env-group",
        type=str,
        default="random_mujocoMB_envs",
        choices=["gym_envs", "mujocoMB_envs", "random_mujocoMB_envs", "point_envs"]
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--skip-early-termination", action="store_true")
    parser.add_argument("--env-random-scale-limit", type=float, default=3.0)

    # MetaRL Agent parameters
    parser.add_argument(
        "--agent-name",
        type=str,
        default="parallel_mbpo_pacoh",
        choices=["rl2", "grbal", "parallel_grbal", "ghvmdp", "parallel_ghvmdp", "mbpo_pacoh", "parallel_mbpo_pacoh",
                 "cem_pacoh", "parallel_cem_pacoh", "mbpo", "mpc"]
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
    parser.add_argument("--multiple-runs-id", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--eval-frequency", type=int, default=0)
    parser.add_argument("--env-load-params-from-file", type=bool, default=False)

    # Reward parameters
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--use-action-cost", action="store_true")
    parser.add_argument("--use-exact-termination-model", action="store_true")

    # Meta Learning Parameters
    parser.add_argument("--collect-meta-data", action="store_true")
    parser.add_argument("--num-train-env-instances", type=int, default=5)
    parser.add_argument("--num-test-env-instances", type=int, default=40)
    parser.add_argument("--num-test-episodes-per-env", type=int, default=40)

    # Logger Parameters
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-to-file", type=bool, default=True)
    parser.add_argument("--safe-log-dir", type=bool, default=True)
    parser.add_argument("--save-statistics", action="store_true")
    parser.add_argument("--offline-logger", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--exp-name", type=str, default="test_exp")
    parser.add_argument("--wandb-project", type=str, default="Meta-MBRL")

    # Model parameters
    parser.add_argument("--model-kind", type=str, default="ProbabilisticNN")
    parser.add_argument("--model-num-heads", type=int, default=5)
    parser.add_argument("--model-layers", type=int, nargs="*", default=[200, 200, 200, 200])
    parser.add_argument("--model-unbiased-head", action="store_true")
    parser.add_argument("--model-heteroscedastic", type=bool, default=True)
    parser.add_argument("--model-non-linearity", type=str, default="Swish")
    parser.add_argument("--model-opt-lr", type=float, default=1e-3)
    parser.add_argument("--model-opt-weight-decay", type=float, default=0.0)
    parser.add_argument("--model-learn-num-iter", type=int, default=50)
    parser.add_argument("--model-learn-batch-size", type=int, default=500)
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
    parser.add_argument("--recurrent-value-function-embedding-layers", type=int, nargs="*", default=[256, 256])

    # Policy function parameters
    parser.add_argument("--policy-layers", type=int, nargs="*", default=[400, 400, 400])
    parser.add_argument("--policy-opt-gradient-steps", type=int, default=500)
    parser.add_argument("--policy-unbiased-head", action="store_true")
    parser.add_argument("--policy-non-linearity", type=str, default="Swish")
    parser.add_argument("--policy-tau", type=float, default=0.005)
    parser.add_argument("--policy-deterministic", action="store_true")
    parser.add_argument("--policy-grad-steps", type=int, default=1)
    parser.add_argument("--policy-train-freq", type=int, default=1)
    parser.add_argument("--recurrent-policy-embedding-layers", type=int, nargs="*", default=[256, 256])

    # MPC parameters
    parser.add_argument("--mpc-solver", type=str, choices=["cem", "icem", "pets"], default="icem")
    parser.add_argument("--mpc-policy", type=str, choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--mpc-num-iter", type=int, default=5)
    parser.add_argument("--mpc-num-particles", type=int, default=1000)
    parser.add_argument("--mpc-num-elites", type=int, default=50)
    parser.add_argument("--mpc-horizon", type=int, default=16)
    parser.add_argument("--mpc-pets-trajectory-samples", type=int, default=5)
    parser.add_argument("--mpc-alpha", type=float, default=0.1)
    parser.add_argument("--mpc-noise-beta", type=float, default=2.0)
    parser.add_argument("--mpc-terminal-reward", type=bool, default=False)
    parser.add_argument("--mpc-not-warm-start", type=bool, default=False)
    parser.add_argument("--mpc-default-action", type=str,
                        choices=["zero", "constant", "mean"], default="constant")

    # PACOH parameters
    parser.add_argument("--pacoh-num-iter-meta-train", type=int, default=2000)
    parser.add_argument("--pacoh-num-iter-eval-train", type=int, default=50)
    parser.add_argument("--pacoh-num-hyper-posterior-particles", type=int, default=2)
    parser.add_argument("--pacoh-n-samples-per-prior", type=int, default=3)
    parser.add_argument("--pacoh-num-posterior-particles", type=int, default=2)
    parser.add_argument("--pacoh-optimistic-evaluation", action="store_true")
    parser.add_argument("--pacoh-likelihood-std", type=float, default=0.1)
    parser.add_argument("--pacoh-training-model-kind", type=str, default="ProbabilisticEnsemble")

    parser.add_argument("--parallel-episodes-per-env", type=int, default=1)
    parser.add_argument("--num-episodes-per-rollout", type=int, default=1)

    # RL2 parameters
    parser.add_argument("--rl2-trial-len", type=int, default=2)

    # GrBAL parameters
    parser.add_argument("--grbal-past-segment-len", type=int, default=32)
    parser.add_argument("--grbal-future-segment-len", type=int, default=32)
    parser.add_argument("--grbal-num-parallel-agents", type=int, default=8)
    parser.add_argument("--grbal-inner-lr", type=float, default=0.003)
    parser.add_argument("--grbal-num-iter-meta-train", type=int, default=2000)

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
