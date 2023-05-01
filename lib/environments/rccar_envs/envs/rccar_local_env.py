import torch
import numpy as np

from rllib.reward.state_action_reward import StateActionReward
from lib.environments.rccar_envs.envs.rccar_env import RCCarEnv


class RCCarLocalEnv(RCCarEnv):

    def __init__(self):
        super(RCCarLocalEnv, self).__init__()
        self._reward_model = RCCarLocalEnvReward(goal=self._goal)

    def get_state(self):
        local_goal = self.transform2d(self._goal, self._pos, translate=True)
        vel = np.array([self._vel.linear.x, self._vel.linear.y, self._vel.angular.z])
        local_vel = self.transform2d(vel, self._pos, translate=False)
        return np.concatenate([local_goal, local_vel])

    @staticmethod
    def transform2d(pos, transform, translate=True):
        x, y, theta = transform.x, transform.y, transform.theta
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        if translate:
            dx, dy, dtheta = pos[0] - x, pos[1] - y, pos[2] - theta
        else:
            dx, dy, dtheta = pos[0], pos[1], pos[2]
        dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
        phi = np.atan2(dy, dx) - theta
        d = np.sqrt(dx ** 2 + dy ** 2)
        transformed_pos = [d * np.cos(phi), d * np.sin(phi), dtheta]
        return np.array(transformed_pos)


class RCCarLocalEnvReward(StateActionReward):
    """
    Reward model for the RC Car environment.
    """

    dim_action = (2,)

    def __init__(self, goal, ctrl_cost_weight=0, speed_cost_weight=5):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.goal = goal
        self.speed_cost_weight = speed_cost_weight

    def copy(self):
        """Get copy of reward model."""
        return RCCarLocalEnvReward(
            goal=self.goal,
            ctrl_cost_weight=self.ctrl_cost_weight,
            speed_cost_weight=self.speed_cost_weight
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        dist_cost = torch.sqrt(torch.sum(torch.square(next_state[..., :3])))
        speed_cost = torch.sqrt(torch.sum(torch.square(next_state[..., 3:])))
        return - dist_cost - self.speed_cost_weight * speed_cost
