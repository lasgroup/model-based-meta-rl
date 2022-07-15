"""Abstract class for an environment with random parameters for transition dynamics"""

from abc import ABC
from rllib.environment import AbstractEnvironment


class RandomEnvironment(AbstractEnvironment, ABC):

    num_random_params = None

    def __init__(self, *args, **kwargs):
        super(RandomEnvironment, self).__init__(*args, **kwargs)

    def set_transition_params(self, random_samples):
        pass
