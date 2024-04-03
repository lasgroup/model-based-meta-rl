import os

import yaml
import torch
import numpy as np

from dotmap import DotMap

from experiments import AGENT_CONFIG_PATH
from experiments.meta_rl_experiments import run_utils
from experiments.meta_rl_experiments.run_utils import get_environment_and_meta_agent
from experiments.meta_rl_experiments.parser import get_argument_parser
from utils.train_and_evaluate import train_and_evaluate_agent


if __name__ == "__main__":

    params = run_utils.get_params()

    environment, agent = get_environment_and_meta_agent(params)

    train_and_evaluate_agent(environment, agent, params)

