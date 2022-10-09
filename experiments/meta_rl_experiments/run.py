import os

import yaml
import torch
import numpy as np

from dotmap import DotMap

from lib.environments import ENVIRONMENTS_PATH
from experiments.meta_rl_experiments import AGENT_CONFIG_PATH
from experiments.meta_rl_experiments.run_utils import get_environment_and_meta_agent
from experiments.meta_rl_experiments.parser import get_argument_parser
from utils.train_and_evaluate import train_and_evaluate_agent


if __name__ == "__main__":

    parser = get_argument_parser()
    params = vars(parser.parse_args())
    with open(
        os.path.join(
            ENVIRONMENTS_PATH,
            params["env_group"],
            "config",
            params["env_config_file"],
        ),
        "r"
    ) as file:
        env_config = yaml.safe_load(file)
    params.update(env_config)

    if params.agent_config_path == "":
        params.agent_config_path = AGENT_CONFIG_PATH
    with open(
        os.path.join(
            params.agent_config_path,
            params["agent_name"].split('_')[-1] + "_defaults.yaml"
        ),
        "r"
    ) as file:
        agent_config = yaml.safe_load(file)
    params.update(agent_config)

    params = DotMap(params)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(min(4, params.num_cpu_cores))

    environment, agent = get_environment_and_meta_agent(params)

    train_and_evaluate_agent(environment, agent, params)

