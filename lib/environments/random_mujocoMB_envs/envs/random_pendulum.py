from rllib.environment.mujoco.pendulum_swing_up import PendulumSwingUpEnv

from lib.environments.wrappers.random_mujoco_env import RandomMujocoEnv


class RandomMBPendulumEnv(PendulumSwingUpEnv, RandomMujocoEnv):
    """Random Pendulum Swing-up Environment"""

    RAND_PARAMS = ['g', 'l', 'm']
    RAND_PARAMS_EXTENDED = RAND_PARAMS

    def __init__(self, ctrl_cost_weight=0.1, random_scale_limit=3.0, rand_params=RAND_PARAMS, sparse=False, *args, **kwargs):
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        PendulumSwingUpEnv.__init__(self, ctrl_cost_weight=ctrl_cost_weight, sparse=sparse)
        RandomMujocoEnv.__init__(self, random_scale_limit, rand_params)

    def set_task(self, task):
        for param, param_val in task.items():
            assert isinstance(param_val, float), 'shapes of new parameter value and old one must match'
            setattr(self, param, param_val)
        self.cur_params = task
