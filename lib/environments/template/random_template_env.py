from __future__ import annotations

import numpy as np

from typing import Dict, List

from .template_env import TemplateEnv
from lib.environments.point_envs.envs.random_point_env_2d import RandomPointEnv2D


class RandomTemplateEnv(TemplateEnv):
    """
    Random Template Environment. Re-implement the methods to create a custom environment with random parameters.

    For more details, refer to the documentation of the TemplateEnv class.

    For a simple example class, refer to the RandomPointEnv2D class.
    """

    RAND_PARAMS: List[str] = ['physics_param1', 'physics_param2', 'physics_param3', 'physics_param4']

    def __init__(self, rand_params: List[str] = None, random_scale_limit: int = 3.0, *args, **kwargs):
        TemplateEnv.__init__(self, *args, **kwargs)
        if rand_params is None:
            rand_params = self.RAND_PARAMS
        self.rand_params = rand_params
        self.random_scale_limit = random_scale_limit
        self.cur_params = None

    def sample_tasks(self, num_tasks: int) -> List[Dict]:
        """
        Generates randomized parameter sets for the environment.
        Implement this function to generate random tasks.

        Args:
            num_tasks (int) : number of different meta-tasks to be sampled.

        Returns:
            tasks (list) : an (num_tasks) length list of task parameters.
        """
        param_sets = []
        for _ in range(num_tasks):
            new_params = {}
            for param_key in self.rand_params:
                new_params[param_key] = np.random.uniform(-1, 1)
            param_sets.append(new_params)
        return param_sets

    def set_task(self, task: Dict):
        """
        Set the specified task to the environment.
        Implement this function to set the task parameters in the environment.

        Args:
            task : task parameters to be set in the environment.
        """
        for param_key, param_value in task.items():
            assert param_key in self.rand_params, f"Invalid task parameter {param_key}"
            if isinstance(param_value, np.ndarray):
                assert param_value.shape == getattr(self, param_key).shape, \
                    'shapes of new parameter value and old one must match'
            setattr(self, param_key, param_value)

        self.cur_params = task  # Store the current task parameters

    def get_task(self) -> Dict:
        """
        Get the current task of the environment.
        Implement this function to get the current task parameters.

        Returns:
            task : current task parameters of the environment.
        """
        return self.cur_params

