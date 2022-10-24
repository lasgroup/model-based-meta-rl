"""Adapated from https://github.com/lasgroup/simulation_transfer by Jonas Rothfuss"""
from collections import OrderedDict

from experiments.experiment_utils import generate_base_command, generate_run_commands, hash_dict, RESULT_DIR

from datetime import datetime
import argparse
import numpy as np
import copy
import os

search_configs = OrderedDict({
    # random search
    "env-config-file": ['random-ant.yaml', 'random-half-cheetah.yaml'],
    "agent-name": ['parallel_pacoh'],
    "exploration": ['optimistic']
})


def main(args):
    rds = np.random.RandomState(args.seed)
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)

    command_list = []
    possible_configs = np.argwhere(np.ones([len(search_configs[param]) for param in search_configs.keys()]))
    for idx in possible_configs:
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        for key in ["seed", "num_seeds_per_hparam", "dry", "long"]:
            flags.pop(key)
        for i, param in enumerate(search_configs.keys()):
            flags[param] = search_configs[param][idx[i]]

        exp_name = f"{flags['env-config-file'].replace('-', '_').replace('.yaml', '').replace('mujoco', '')}" \
                   f"_{flags['agent-name']}" \
                   f"_{flags['exploration']}"
        current_time = datetime.now().strftime("%b%d_%H_%M_%S")
        exp_path = os.path.join(exp_base_path, f"{exp_name}_{current_time}")
        flags['agent-config-path'] = os.path.abspath(os.path.join('slurm_outputs', args.exp_name, 'configs'))

        if flags['agent-name'] == "ppo" and flags['exploration'] == "optimistic":
            continue

        if flags['agent-name'] == 'parallel_pacoh':
            run_file_path = os.path.abspath('experiments/meta_rl_experiments/run_parallel_pacoh.py')
        elif flags['agent-name'] == 'parallel_grbal':
            run_file_path = os.path.abspath('experiments/meta_rl_experiments/run_parallel_grbal.py')
        else:
            run_file_path = os.path.abspath('experiments/meta_rl_experiments/run.py')

        for j in range(args.num_seeds_per_hparam):
            seed = init_seeds[j]
            flags_ = dict(**flags, **{'seed': seed})
            flags_hash = hash_dict(flags_)
            flags_['log-dir'] = os.path.join(exp_path, flags_hash)
            flags_['multiple-runs-id'] = j
            cmd = generate_base_command(run_file_path, flags=flags_)
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(
        command_list,
        exp_name=args.exp_name,
        mode='euler_slurm',
        num_cpus=args.num_cpu_cores,
        promt=True,
        dry=args.dry,
        long=args.long
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument("--exp-name", type=str, default="pacoh_defaults")
    parser.add_argument("--num-seeds-per-hparam", type=int, default=5)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--num-cpu-cores", type=int, default=20)

    # Training Parameters
    parser.add_argument(
        "--env-group",
        type=str,
        default="random_mujocoMB_envs",
        choices=["gym_envs", "mujocoMB_envs", "random_mujocoMB_envs", "point_envs"]
    )
    parser.add_argument("--train-episodes", type=int, default=1200)
    parser.add_argument("--test-episodes", type=int, default=40)
    parser.add_argument("--eval-frequency", type=int, default=0)
    parser.add_argument("--skip-early-termination", action="store_true")

    # Meta Learning Parameters
    parser.add_argument("--num-train-env-instances", type=int, default=40)
    parser.add_argument("--num-test-env-instances", type=int, default=40)
    parser.add_argument("--num-test-episodes-per-env", type=int, default=40)
    parser.add_argument("--env-random-scale-limit", type=float, default=3.0)

    parser.add_argument("--use-action-cost", action="store_true")

    parser.add_argument("--save-statistics", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--offline-logger", action="store_true")

    args = parser.parse_args()
    main(args)

# python experiments/launch_experiments.py --exp-name initial_test --long --use-wandb --offline-logger --skip-early-termination --use-action-cost --num-cpu-cores 20 --dry
