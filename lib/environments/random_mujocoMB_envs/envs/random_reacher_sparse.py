import torch
from rllib.environment.mujoco.reacher_2d import MBReacherEnv
from rllib.reward.state_action_reward import StateActionReward
from lib.environments.wrappers.random_mujoco_env import RandomMujocoEnv


# https://github.com/deepmind/dm_control/blob/main/dm_control/utils/rewards.py#L93
_DEFAULT_VALUE_AT_MARGIN = 0.1


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, gaussian(d, value_at_margin))

    return value


def gaussian(x, value_at_1):
    scale = torch.sqrt(-2 * torch.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


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
