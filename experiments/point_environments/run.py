import os

import yaml
import torch
import numpy as np

from dotmap import DotMap

from experiments.point_environments.point_utils import get_environment_and_agent
from experiments.point_environments.parser import get_argument_parser
from utils.train_and_evaluate import train_and_evaluate_agent


if __name__ == "__main__":

    parser = get_argument_parser()
    params = vars(parser.parse_args())
    params = DotMap(params)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(1)

    environment, agent = get_environment_and_agent(params)

    train_and_evaluate_agent(environment, agent, params)

