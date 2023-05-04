import os
from msvcrt import getch

import yaml
import torch
import numpy as np

from dotmap import DotMap
from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.util.neural_networks.utilities import to_torch
from rllib.util.rollout import step_model, step_env
from rllib.util.utilities import sample_model

from lib.algorithms import SB3_SAC
from experiments import AGENT_CONFIG_PATH
from experiments.lib_environments.parser import get_argument_parser
from experiments.lib_environments.run_utils import get_environment_and_agent
from experiments.rccar_experiments.run import setup_config, observe_transitions

from experiments.rccar_experiments.rccar_utils import save_as_csv, load_data, DATA_DIR, eval_model_trajectories


if __name__ == "__main__":

    params = setup_config()
    params.use_validation_set = True

    environment, agent = get_environment_and_agent(params)

    agent.train()
    # train_and_evaluate_agent(environment, agent, params)

    files = [file for file in sorted(os.listdir(DATA_DIR)) if "sampled" in file]
    val_files = files[-1:]
    train_files = files[:-1]

    states_list, actions_list, next_states_list = load_data(train_files)

    agent.start_episode()
    for states, actions, next_states in zip(states_list, actions_list, next_states_list):
        observe_transitions(agent, states, actions, next_states)

    agent.end_episode()

    # agent.simulate_and_learn_policy()

    torch.save(agent.dynamical_model.base_model.nn[0].particles, "rccar_model.pt")
    agent.policy.save("rccar_policy")
    # agent.policy.save_replay_buffer("rccar_replay_buffer")

    # agent.dynamical_model.base_model.nn[0].particles = torch.load("rccar_model.pt")
    # agent.policy = SB3_SAC.load("rccar_policy")
    # agent.policy.load_replay_buffer("rccar_replay_buffer")

    # AbstractAgent.end_episode(agent)
    # print(agent)

    agent.eval()
    agent.dynamical_model.eval()

    agent.set_goal(environment.goal)
    state = environment.reset()

    while True:
        print(f"Starting state: {state} \nPress y to confirm...")
        if getch() == "y":
            break

    agent.start_episode()

    done = False
    time_step = 0
    max_steps = 200

    states_list = []
    actions_list = []

    input("Press Enter to start episode...")

    while not done:

        action = agent.act(state)

        states_list.append(state.reshape(-1))
        actions_list.append(action.reshape(-1))

        obs, state, done, info = step_env(
            environment=environment,
            state=state,
            action=action,
            action_scale=agent.policy.action_scale,
            pi=agent.pi,
            render=False,
        )

        time_step += 1
        if max_steps <= time_step:
            break

    save_as_csv(states_list, actions_list)
