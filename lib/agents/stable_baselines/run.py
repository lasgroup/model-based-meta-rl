import os
import yaml

import gym
import torch
import numpy as np

from dotmap import DotMap

from experiments.gym_environments.gym_utils import get_environment_and_agent
from experiments.gym_environments.parser import get_argument_parser
from utils.train_and_evaluate import train_and_evaluate_agent

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

# del model  # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

env = make_vec_env("CartPole-v1", n_envs=1)
obs = env.reset()
for i in range(400):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

parser = get_argument_parser()
params = vars(parser.parse_args())
with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        params["env_config_path"],
        params["env_config_file"],
    ),
    "r"
) as file:
    env_config = yaml.safe_load(file)
params.update(env_config)
params = DotMap(params)

torch.manual_seed(params.seed)
np.random.seed(params.seed)
torch.set_num_threads(1)

environment, _ = get_environment_and_agent(params)

model = PPO("MlpPolicy", environment, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole_self")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole_self")
obs = environment.reset()
for i in range(400):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = environment.step(action)
    environment.render()

