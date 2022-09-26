from dataclasses import asdict

import numpy as np
from rllib.dataset import stack_list_of_tuples
from rllib.dataset.datatypes import Observation


def sample_transitions(trajectories, num_samples=1):
    """
    Sample a batch of transitions from observations
    :param trajectories: Stacked_observations
    :param num_samples: Number of samples
    :return: Sampled transitions
    """

    if isinstance(trajectories, tuple):
        trajectories = stack_list_of_tuples(trajectories)

    random_batch = np.random.choice(trajectories.shape[-1], num_samples)

    new_obs = Observation(
        state=trajectories.state[:, :, random_batch, :],
        action=trajectories.action[:, :, random_batch, :],
        reward=trajectories.reward[:, :, random_batch, :],
        next_state=trajectories.next_state[:, :, random_batch, :],
        done=trajectories.done[:, :, random_batch],
        next_action=trajectories.next_action,
        log_prob_action=trajectories.log_prob_action[:, :, random_batch],
        entropy=trajectories.entropy[:, :, random_batch],
        state_scale_tril=trajectories.state_scale_tril,
        next_state_scale_tril=trajectories.next_state_scale_tril,
        reward_scale_tril=trajectories.reward_scale_tril,
    )

    return new_obs


def get_trajectory_segment(trajectory, start_index=0, end_index=None):
    segment_ = {key: val[start_index:end_index, ...] if val.ndim > 0 else val for key, val in asdict(trajectory).items()}
    return Observation(**segment_)


def combine_datasets(dataset_list):
    combined_dataset = type(dataset_list[0])(
        max_len=dataset_list[0].max_len,
        transformations=dataset_list[0].transformations,
        num_memory_steps=dataset_list[0].num_memory_steps
    )
    for data in dataset_list:
        combined_dataset.add_dataset(data)
    return combined_dataset
