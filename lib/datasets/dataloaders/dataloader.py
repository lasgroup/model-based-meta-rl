import os
import torch
import numpy as np

from dotmap import DotMap
from typing import List, Tuple
from rllib.agent import AbstractAgent
from rllib.util.utilities import sample_model
from rllib.dataset.datatypes import Observation

import experiments.meta_rl_experiments.run_utils as run_utils
from utils.utils import get_project_path, get_dataset_path

"""
This script contains the functions to load and parse custom meta-training data to be used for training the meta-learner.
The script will store the data in the required format and save it to an appropriate path. The dataset can be loaded back
again  automatically by the meta-learner during training. 

The function `get_meta_training_data` should be implemented by the user to load the meta-training data. The function 
`dummy_meta_training_data` provides an example of how to load a pre-existing dataset and return the data in the required
format. It parses the `ReplayBuffer` format used to store the meta-training data in this framework. This function can 
also be used to convert the meta-training data collected using this repository to a custom format.
"""


def parse_meta_training_data(params: DotMap) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Use this function to load and parse the meta-training data to be used for training the meta-learner.
    The returned data should be a list of episodes, where each episode is a list of `(s, a, r, s')` tuples.
    The episodes need not be ordered by task. However, all transitions from an episode should belong to a single task.
    For a simple example, look at the implementation of the method `dummy_meta_training_data`.
    Each tuple should contain the following elements:
        - `s`: the state observation, shape (dim_state,)
        - `a`: the action taken by the agent, shape (dim_action,)
        - `r`: the reward received by the agent, shape (1,)
        - `s'`: the next state observation, shape (dim_state,)
    Args:
        params: DotMap containing the experiment parameters
    Output:
        meta_data: List of episodes, where each episode is a list of `(s, a, r, s')` tuples
    """
    raise NotImplementedError


def parse_replay_buffer_data(params: DotMap) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Example function to load and parse the meta-training data to be used for training the meta-learner.
    This function loads a pre-existing dataset and returns the data in the required format.
    """
    load_path = os.path.join(get_project_path(), get_dataset_path(params))
    dataset = torch.load(load_path)
    meta_training_data = []
    for episode_start, episode_len in zip(dataset.trajectory_starts, dataset.trajectory_lengths):
        episode = []
        observations = dataset.memory[episode_start:episode_start + episode_len]
        for i in range(len(observations) - 1):
            state = observations[i].state.numpy()
            action = observations[i].action.numpy()
            reward = observations[i].reward.numpy()
            next_state = observations[i].next_state.numpy()
            episode.append((state, action, reward, next_state))
        meta_training_data.append(episode)
    return meta_training_data


def observe_meta_train_data(agent, meta_train_data: List[List]):
    agent.train()
    for task_episode in meta_train_data:
        agent.start_episode()
        observe_transitions(agent, task_episode)
        agent.dataset.end_episode()
        AbstractAgent.end_episode(agent)


def observe_transitions(agent, transitions):
    for i, transition in enumerate(transitions):
        state, action, reward, next_state = transition
        done = (i == len(transitions) - 1)  # Assume the last transition is terminal
        action = torch.tensor(action)
        entropy, log_prob_action = 0.0, 1.0
        # If reward is not provided, sample it from the reward model
        if reward is None:
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


if __name__ == "__main__":

    params = run_utils.get_params()
    assert "pacoh" in params.agent_name, "The script for saving meta-training data uses PACOH agent. " \
                                         "Please set the agent_name to a PACOH agent in parser.py."

    _, agent = run_utils.get_environment_and_meta_agent(params)
    meta_data = parse_replay_buffer_data(params)
    # Uncomment the following line to use custom meta-training data
    # meta_data = parse_meta_training_data(params)
    observe_meta_train_data(agent, meta_data)
    agent.save_trajectory_replay(params)
