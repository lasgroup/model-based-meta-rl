"""Adapated from https://github.com/lasgroup/simulation_transfer by Jonas Rothfuss"""
from collections import OrderedDict

from experiments.experiment_utils import generate_base_command, generate_run_commands, hash_dict, RESULT_DIR

from datetime import datetime
import argparse
import numpy as np
import copy
import os

from utils.utils import get_project_path


def resolve_agent_name(agent_name):
    AGENT_DICT = {
        "pacoh_rl": ['parallel_cem_pacoh', 'optimistic'],
        "pacoh_rl_greedy": ['parallel_cem_pacoh', 'greedy'],
        "pacoh_rl_sac": ['parallel_mbpo_pacoh', 'greedy'],
        "grbal": ['parallel_grbal', 'greedy'],
        "hucrl": ['mpc', 'optimistic'],
        "pets": ['mpc', 'greedy'],
        "mbpo": ['mbpo', 'greedy'],
    }
    return AGENT_DICT[agent_name]


def main(args):

    rds = np.random.RandomState(args.seed)
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)

    command_list = []
    internal_agent_name, exploration = resolve_agent_name(args.agent_name)

    # transfer flags from the args
    flags = copy.deepcopy(args.__dict__)
    for key in ["seed", "num_seeds", "dry", "long"]:
        flags.pop(key)
    flags["agent_name"] = internal_agent_name
    flags["exploration"] = exploration
    flags["pacoh_optimistic_evaluation"] = exploration == "optimistic" and "pacoh" in internal_agent_name
    print(flags)

    exp_name = f"{flags['env_config_file'].replace('-', '_').replace('.yaml', '').replace('mujoco', '')}" \
               f"_{flags['agent_name']}" \
               f"_{flags['exploration']}"
    current_time = datetime.now().strftime("%b%d_%H_%M_%S")
    exp_path = os.path.join(exp_base_path, f"{exp_name}_{current_time}")

    if flags['agent_name'] in ['parallel_mbpo_pacoh', 'parallel_cem_pacoh']:
        run_file_path = os.path.join(get_project_path(), 'experiments/meta_rl_experiments/run_parallel_pacoh.py')
    elif flags['agent_name'] in ['parallel_grbal']:
        run_file_path = os.path.join(get_project_path(), 'experiments/meta_rl_experiments/run_parallel_grbal.py')
    else:
        run_file_path = os.path.join(get_project_path(), 'experiments/meta_rl_experiments/run_parallel_non_meta_agents.py')

    for j in range(args.num_seeds):
        seed = init_seeds[j]
        flags_ = dict(**flags, **{'seed': seed})
        flags_hash = hash_dict(flags_)
        if flags_['exploration'] == 'greedy':
            flags_['exp_name'] = flags_['exp_name'].replace('opt', 'gre')
        flags_['log-dir'] = os.path.join(exp_path, flags_hash)
        flags_['multiple-runs-id'] = j
        cmd = generate_base_command(run_file_path, flags=flags_)
        command_list.append(cmd)

    # submit jobs
    generate_run_commands(
        command_list,
        exp_name=args.exp_name,
        mode='local',
        num_cpus=args.num_cpu_cores,
        promt=True,
        dry=args.dry,
        long=args.long
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-name", type=str, default="pacoh_defaults")
    parser.add_argument("--num-seeds", type=int, default=1)

    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--num-cpu-cores", type=int, default=4)

    # Training Parameters
    parser.add_argument("--env-config-file", type=str, default="random-sparse-pendulum.yaml")
    parser.add_argument(
        "--agent-name",
        type=str,
        default="pacoh_rl",
        choices=["pacoh_rl", "pacoh_rl_greedy", "pacoh_rl_sac", "grbal", "hucrl", "pets", "mbpo"]
    )
    parser.add_argument("--collect-meta-data", type=bool, default=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()
    main(args)

# python experiments/launch_experiments.py --exp-name initial_test --long --use-wandb --skip-early-termination --use-action-cost --num-cpu-cores 20 --dry
