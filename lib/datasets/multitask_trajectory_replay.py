"""Implementation of Trajectory Replay Buffer for multiple tasks."""

import numpy as np
from rllib.dataset import stack_list_of_tuples

from lib.datasets import TrajectoryReplay


class MultiTaskTrajectoryReplay:
    """A Trajectory Replay Buffer Dataset for Multiple Tasks.

    The multi-task trajectory replay buffer stores trajectories for multiple tasks.
    It initializes new tasks if no previous trajectory from the task is stored
    On sampling from a task, it returns a trajectory IID.
    It erases the older samples once the buffer is full, like on a queue.

    Parameters
    ----------
    max_len: int.
        size of trajectory replay buffer for individual tasks.
    task_ids = list.
        list of task_ids to initialize the buffer.
    transformations: list of transforms.AbstractTransform, optional.
        A sequence of transformations to apply to the dataset, each of which is a
        callable that takes an observation as input and returns a modified observation.
        If they have an `update` method it will be called whenever a new trajectory
        is added to the dataset.

    Methods
    -------
    append(observation, task_id) -> None:
        append an observation to a task dataset.
    is_full(task_id): bool
        check if task buffer is full.
    all_data(task_id):
        Get all the transformed data for the task.
    sample_segment(segment_len):
        Get a segment of trajectory from randomly selected task.
    sample_segment_from_task(task_id, segment_len):
        Get a segment of trajectory from a given task.
    reset():
        Reset the memory to zero.
    get_observation(idx, task_id):
        Get the observation at a given index from the task.

    References
    ----------
    Lin, L. J. (1992).
    Self-improving reactive agents based on reinforcement learning, planning and
    teaching. Machine learning.
    """

    def __init__(self, max_len=10000, task_ids=None, transformations=None):
        self.max_len = max_len
        self.task_ids = task_ids if task_ids is not None else []

        self.transformations = transformations

        self._memory_list = {task_id: TrajectoryReplay(max_len, transformations) for task_id in task_ids}

    @property
    def num_tasks(self):
        """The number of initialized tasks."""
        return len(self.task_ids)

    def _initialize_task(self, task_id):
        """Initializes the memory buffer for a new task."""
        self.task_ids.append(task_id)
        self._memory_list[task_id] = TrajectoryReplay(self.max_len, self.transformations)

    def append(self, trajectory, task_id):
        """Appends a trajectory to the task."""
        if task_id not in self.task_ids:
            self._initialize_task(task_id)
        self._memory_list[task_id].append(trajectory)

    def sample_segment(self, segment_len):
        """Samples a trajectory segment from a randomly selected task."""
        task = np.random.randint(self.num_tasks)
        return self.sample_segment_from_task(task, segment_len)

    def sample_segment_from_task(self, task_id, segment_len):
        """Samples a trajectory segment from the task."""
        return self._memory_list[task_id].sample_segment(segment_len)

    def sample_batch(self, batch_size):
        """Samples a batch of transitions from a randomly selected task."""
        tasks = np.random.randint(self.num_tasks, size=batch_size)
        samples = stack_list_of_tuples([self.sample_batch_from_task(task, 1) for task in tasks])
        return samples

    def sample_batch_from_task(self, task_id, batch_size):
        """Samples a batch of transitions from the task."""
        return self._memory_list[task_id].sample_batch(batch_size)

    def get_observation(self, idx, task_id):
        """Returns an observation from the task."""
        return self._memory_list[task_id][idx]

    def is_full(self, task_id):
        """Flag that checks if memory in buffer for a task is full."""
        return self._memory_list[task_id].is_full

    def all_data(self, task_id):
        """Get all the data for a task."""
        return self._memory_list[task_id].all_data

    def size(self, task_id):
        """Get the buffer size of the task."""
        return self._memory_list[task_id].ptr

    def remove_task(self, task_id):
        """Remove a task from memory."""
        self._memory_list.pop(task_id)
        self.task_ids.remove(task_id)

    def reset(self):
        """Empty the buffer memory."""
        self._memory_list = dict()
        self.task_ids = []
