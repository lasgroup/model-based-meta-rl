import torch
import numpy as np

from dotmap import DotMap

from inverted_pendulum.inverted_pendulum_utils import get_environment_and_agent
from inverted_pendulum.parser import get_argument_parser
from utils.train_and_evaluate import train_and_evaluate_agent


if __name__ == "__main__":

    parser = get_argument_parser()
    params = DotMap(vars(parser.parse_args()))

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(1)

    environment, agent = get_environment_and_agent(params)

    train_and_evaluate_agent(environment, agent, params)

