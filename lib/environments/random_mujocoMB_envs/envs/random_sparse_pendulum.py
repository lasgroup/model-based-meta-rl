from lib.environments.random_mujocoMB_envs.envs.random_pendulum import RandomMBPendulumEnv


class RandomMBSparsePendulumEnv(RandomMBPendulumEnv):
    """Random Sparse Pendulum Swing-up Environment"""

    RAND_PARAMS = ['g', 'l', 'm']
    RAND_PARAMS_EXTENDED = RAND_PARAMS

    def __init__(self, ctrl_cost_weight=0.1, random_scale_limit=3.0, rand_params=RAND_PARAMS, *args, **kwargs):
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        RandomMBPendulumEnv.__init__(
            self,
            ctrl_cost_weight=ctrl_cost_weight,
            random_scale_limit=random_scale_limit,
            rand_params=rand_params,
            sparse=True,
            *args,
            **kwargs
        )
