"""Implementation of a Trajectory Replay Buffer."""

import numpy as np
import torch

from torch.utils.data import default_collate
from rllib.dataset import ExperienceReplay
from rllib.dataset.datatypes import Observation


class TrajectoryReplay(ExperienceReplay):
    """An Experience Replay Buffer that stores trajectories of varied lengths.

    The replay buffer stores transitions and accesses them IID.
    It also stores information when a single trajectory starts and ends.
    It erases the older samples once the buffer is full, like on a queue.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_starts = []
        self.trajectory_lengths = []

    def start_episode(self):
        self.trajectory_starts.append(self.ptr)

    def end_episode(self):
        trajectory_length = self.ptr - self.trajectory_starts[-1]
        if trajectory_length < 0:
            trajectory_length += self.max_len
        self.trajectory_lengths.append(trajectory_length)
        super().end_episode()

    def sample_segment(self, segment_len):
        """Samples a segment of given length from a random trajectory."""
        if max(self.trajectory_lengths) < segment_len:
            return None
        valid_trajectories = (np.array(self.trajectory_lengths) >= segment_len)
        idx_trajectory = np.random.choice(
            np.arange(len(self.trajectory_starts)),
            p=valid_trajectories/np.sum(valid_trajectories)
        )
        idx = np.random.choice(
            self.trajectory_lengths[idx_trajectory] - segment_len + 1
        ) + self.trajectory_starts[idx_trajectory]
        assert idx - self.trajectory_starts[idx_trajectory] + segment_len <= self.trajectory_lengths[idx_trajectory]
        observation = self._get_consecutive_observations(idx, segment_len)
        if self.raw:
            return observation
        for transform in self.transformations:
            observation = transform(observation)
        return observation

    def sample_task_batch(self, batch_size, task_id=None, eval_samples=False, num_eval_samples=0):
        if task_id is None:
            task_id = np.random.randint(len(self.trajectory_starts))
        if eval_samples:
            indices = self.trajectory_lengths[task_id] + self.trajectory_starts[task_id] - np.arange(num_eval_samples) - 1
        else:
            indices = np.random.choice(
                self.trajectory_lengths[task_id]-num_eval_samples,
                batch_size,
                replace=(batch_size > self.trajectory_lengths[task_id]-num_eval_samples)
            ) + self.trajectory_starts[task_id]
        if self.num_memory_steps == 0:
            obs = self._get_observation(indices)
            return obs, torch.tensor(indices), self.weights[indices]
        else:
            obs, idx, weight = default_collate([self[i] for i in indices])
            return Observation(**obs), idx, weight

    def reset(self):
        self.trajectory_starts = []
        self.trajectory_lengths = []
        super().reset()

    @property
    def num_episodes(self):
        return len(self.trajectory_starts)
