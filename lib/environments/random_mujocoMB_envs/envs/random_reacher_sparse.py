from rllib.environment.mujoco.reacher_2d import MBReacherEnv
from rllib.reward.state_action_reward import StateActionReward
from lib.environments.random_mujocoMB_envs.envs.utils import tolerance
from lib.environments.wrappers.random_mujoco_env import RandomMujocoEnv


class SparseReacherReward(StateActionReward):
    """Reacher Reward model."""

    dim_action = (2,)

    def __init__(self, goal, ctrl_cost_weight=1.0):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight, goal=goal)

    def copy(self):
        """Copy reward model."""
        return SparseReacherReward(
            ctrl_cost_weight=self.ctrl_cost_weight, goal=self.goal
        )

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        dist_to_target = (state[..., -3:] ** 2).sum(-1)
        # Magic from
        # https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/mujoco/assets/reacher.xml#L4
        return tolerance(dist_to_target, (0, 0.009), margin=0)


class RandomMBReacherSparseEnv(MBReacherEnv, RandomMujocoEnv):
    """Random Reacher Environment"""

    RAND_PARAMS = ["body_mass", "dof_damping", "body_inertia", "geom_friction"]
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ["geom_size"]

    def __init__(
        self,
        ctrl_cost_weight=0.1,
        random_scale_limit=3.0,
        rand_params=RAND_PARAMS,
        *args,
        **kwargs
    ):
        assert set(rand_params) <= set(
            self.RAND_PARAMS_EXTENDED
        ), "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        MBReacherEnv.__init__(self, ctrl_cost_weight=ctrl_cost_weight)
        RandomMujocoEnv.__init__(self, random_scale_limit, rand_params)
        self._reward_model = SparseReacherReward(
            ctrl_cost_weight=ctrl_cost_weight, goal=None
        )
