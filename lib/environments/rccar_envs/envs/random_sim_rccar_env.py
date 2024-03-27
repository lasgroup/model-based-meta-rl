import numpy as np

from lib.environments.rccar_envs.envs.dynamics_model import CarParams
from lib.environments.rccar_envs.envs.sim_rccar_env import RCCarSimEnv


class RandomRCCarSimEnv(RCCarSimEnv):

    RAND_PARAMS = ['c_m_1', 'c_m_2', 'c_d', 'd_r', 'd_f']

    def __init__(self, random_scale_limit=3.0, *args, **kwargs):
        super(RandomRCCarSimEnv, self).__init__(*args, **kwargs)
        self.random_scale_limit = random_scale_limit

    def set_task(self, random_samples):
        self._dynamics_params = CarParams(**random_samples)

    def sample_tasks(self, n_tasks):
        param_sets = []
        for _ in range(n_tasks):
            new_params = {}
            for param_name in self.RAND_PARAMS:
                coef = np.array(1.5) ** np.random.uniform(-self.random_scale_limit, self.random_scale_limit, size=1)
                new_params[param_name] = coef.item() * self._default_car_model_params[param_name]
            param_sets.append(new_params)
        return param_sets
