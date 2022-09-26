import os

import yaml
import torch
import numpy as np

from dotmap import DotMap

from lib.environments import ENVIRONMENTS_PATH
from experiments.lib_environments.run_utils import get_environment_and_agent
from experiments.lib_environments.parser import get_argument_parser
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
    params = DotMap(params)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_cpu_cores)

    environment, agent = get_environment_and_agent(params)

    train_and_evaluate_agent(environment, agent, params)

