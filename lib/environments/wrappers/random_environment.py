"""Abstract class for an environment with random parameters for transition dynamics"""

from abc import ABC
from rllib.environment import AbstractEnvironment


class RandomEnvironment(AbstractEnvironment, ABC):

    num_random_params = None

    def __init__(self, *args, **kwargs):
        super(RandomEnvironment, self).__init__(*args, **kwargs)
        self.random_scale_limit = 1.0

    def set_task(self, random_samples):
        pass
