from rllib.environment.mujoco.pusher import MBPusherEnv

from lib.environments.wrappers.random_mujoco_env import RandomMujocoEnv


class RandomMBPusherEnv(MBPusherEnv, RandomMujocoEnv):
    """Random Pusher Environment"""

    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, ctrl_cost_weight=0.1, random_scale_limit=3.0, rand_params=RAND_PARAMS, *args, **kwargs):
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        MBPusherEnv.__init__(self, ctrl_cost_weight=ctrl_cost_weight)
        RandomMujocoEnv.__init__(self, random_scale_limit, rand_params)
