import os

import yaml
import torch
import numpy as np

from dotmap import DotMap
from rllib.agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.util.neural_networks.utilities import to_torch
from rllib.util.rollout import step_model
from rllib.util.utilities import sample_model

from experiments import AGENT_CONFIG_PATH
from experiments.lib_environments.run_utils import get_environment_and_agent
from experiments.lib_environments.parser import get_argument_parser
from utils.train_and_evaluate import train_and_evaluate_agent

from experiments.rccar_experiments.rccar_utils import save_as_csv, load_data, DATA_DIR, eval_model_trajectories


def observe_transitions(agent, states, actions, next_states):

    for state, action, next_state in zip(states, actions, next_states):
        done = False
        action = to_torch(action)
        entropy, log_prob_action = 0.0, 1.0
        reward = sample_model(agent.reward_model, state, action, next_state)
        obs = Observation(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            entropy=entropy,
            log_prob_action=log_prob_action,
        ).to_torch()
        agent.observe(obs)


def setup_config():
    parser = get_argument_parser()
    params = vars(parser.parse_args())

    if params["agent_config_path"] == "":
        params["agent_config_path"] = AGENT_CONFIG_PATH
    with open(
            os.path.join(
                params["agent_config_path"],
                params["env_config_file"]
            ),
            "r"
    ) as file:
        env_config = yaml.safe_load(file)

    train_config = env_config["training"]
    agent_config = env_config[params["agent_name"].split('_')[-1]]

    params.update(train_config)
    params.update(agent_config)

    params = DotMap(params)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(min(4, params.num_cpu_cores))

    return params


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

    # agent.end_episode()

    # torch.save(agent.dynamical_model.base_model.nn[0].particles, "rccar_model.pt")
    agent.dynamical_model.base_model.nn[0].particles = torch.load("rccar_model.pt")

    agent.simulate_and_learn_policy()
    AbstractAgent.end_episode(agent)
    print(agent)

    agent.eval()
    agent.dynamical_model.eval()

    horizon = 10
    eval_model_trajectories(val_files, agent, horizon)

    horizon = 32
    eval_model_trajectories(val_files, agent, horizon)

    state = torch.tensor([-1.2, -1.0, 1.57, 0.0, 0.0, 0.0]).reshape((1, -1))
    states_list = []
    actions_list = []
    for i in range(1000):
        action = agent.act(state)
        with torch.no_grad():
            observation, next_state, _ = step_model(
                dynamical_model=agent.dynamical_model,
                reward_model=agent.reward_model,
                termination_model=None,
                state=state,
                action=torch.tensor(action, dtype=state.dtype, device=state.device),
                action_scale=agent.policy.action_scale,
                done=None,
                pi=None,
            )
        states_list.append(state.reshape(-1))
        actions_list.append(action.reshape(-1))
        state = next_state

    save_as_csv(states_list, actions_list)
