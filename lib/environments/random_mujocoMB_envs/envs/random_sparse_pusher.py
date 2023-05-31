from rllib.environment.mujoco.pusher import MBPusherEnv
from rllib.reward.state_action_reward import StateActionReward
import torch
from lib.environments.random_mujocoMB_envs.envs.utils import tolerance

from lib.environments.wrappers.random_mujoco_env import RandomMujocoEnv


class SparsePusherReward(StateActionReward):
    """Pusher Reward model."""

    dim_action = (7,)

    def __init__(self, goal, ctrl_cost_weight=0.1):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight, goal=goal)

    def copy(self):
        """Copy reward model."""
        return SparsePusherReward(
            ctrl_cost_weight=self.ctrl_cost_weight, goal=self.goal
        )

    def state_reward(self, state, next_state=None):
        """Compute State reward."""
        goal = state[..., -3:]
        end_effector = state[..., -6:-3]
        pluck = state[..., -9:-6]

        dist_to_ball = torch.linalg.norm(pluck - end_effector, dim=-1)
        dist_to_goal = torch.linalg.norm(pluck - goal, dim=-1)

        reach_reward = tolerance(dist_to_ball, (0, 0.05), 0.3)
        fetch_reward = tolerance(dist_to_goal, (0, 0.05 + 0.1), 0.1)
        reach_then_fetch = reach_reward * (0.5 + 0.5 * fetch_reward)
        # print(reach_then_fetch, reach_reward, fetch_reward)
        return reach_then_fetch


class RandomMBSparsePusherEnv(MBPusherEnv, RandomMujocoEnv):
    """Random Pusher Environment"""

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
        MBPusherEnv.__init__(self, ctrl_cost_weight=ctrl_cost_weight)
        RandomMujocoEnv.__init__(self, random_scale_limit, rand_params)
        self._reward_model = SparsePusherReward(
            ctrl_cost_weight=ctrl_cost_weight, goal=None
        )


if __name__ == "__main__":
    env = RandomMBSparsePusherEnv()
    env.reset()
    while True:
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()
        print("resetting...")
        env.reset()
