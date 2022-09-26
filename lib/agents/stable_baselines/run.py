import os
import yaml

import gym
import torch
import numpy as np

from dotmap import DotMap

from lib.environments import ENVIRONMENTS_PATH
from experiments.lib_environments.parser import get_argument_parser
from experiments.lib_environments.run_utils import get_environment_and_agent

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure

from utils.logger import Logger

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda: False

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

    environment, _ = get_environment_and_agent(params)

    params.agent_name = f"sb3_{params.agent_name}"
    # set up logger

    sb3_logger = configure(params.log_dir, ["stdout", "csv", "tensorboard"])

    name = f"{params.env_config_file.replace('-', '_').replace('.yaml', '').replace('mujoco', '')}" \
           f"_{params.agent_name}" \
           f"_{params.exploration}"
    rllib_logger = Logger(
            name=name,
            comment=f"{params.agent_name} {params.exploration.capitalize()}",
            log_dir=params.log_dir,
            save_statistics=params.save_statistics,
            use_wandb=params.use_wandb,
            offline_mode=params.offline_logger
        )

    if params.agent_name == "sb3_ppo":
        model = PPO("MlpPolicy", environment, verbose=1)
    elif params.agent_name == "sb3_sac":
        model = SAC("MlpPolicy", environment, verbose=1)
    else:
        raise NotImplementedError
    model.set_logger(sb3_logger)
    model.learn(total_timesteps=400000)

    returns = []
    for eps in range(20):

        obs = environment.reset()
        eps_return = 0

        for t in range(params.max_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = environment.step(action)
            eps_return += rewards
            # environment.render()

        print(eps_return)
        returns.append(eps_return)
        rllib_logger.end_episode(**{"eval-return-0": eps_return})

    metrics = dict()
    returns = np.mean(np.array(returns))
    metrics.update({"test_returns": returns})
    rllib_logger.log_hparams(params.toDict(), metrics)
