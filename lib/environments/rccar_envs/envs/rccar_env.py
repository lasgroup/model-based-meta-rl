from rllib.environment.abstract_environment import AbstractEnvironment

import rospy
import time
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D, Twist, Vector3

import torch
import numpy as np
from gym.spaces import Box
from rllib.reward.state_action_reward import StateActionReward
from torch import nn


class RCCarEnv(AbstractEnvironment):
    wait_time = 0.03
    max_steps = 500

    def __init__(self, ctrl_cost_weight=0.005):
        super(RCCarEnv, self).__init__(
            dim_state=(6,),
            dim_action=(2,),
            observation_space=Box(low=-np.inf, high=np.inf, shape=(2,)),
            action_space=Box(low=-1.0, high=1.0, shape=(2,)),
            dim_observation=(-1,),
            num_states=-1,
            num_actions=-1,
            num_observations=-1,
            dim_reward=(1,),
        )
        self._goal = np.array([0.0, 0.0, -1.57])
        self.zero_action = np.zeros(self.dim_action)

        self._reward_model = RCCarEnvReward(goal=self._goal, ctrl_cost_weight=ctrl_cost_weight)
        self._time = 0

        self._prev_time = None
        self._state = None
        self._pos = None
        self._vel = None

        self._init_ros()

    def _init_ros(self):
        rospy.init_node('RCEnvironment', anonymous=True)
        self.action_pub = rospy.Publisher('rc_control', Vector3, queue_size=1)
        self.reset_pub = rospy.Publisher('reset_wait', Bool, queue_size=10, latch=True)
        self.pos_sub = rospy.Subscriber('rc_pos', Pose2D, callback=self.set_pos)
        self.vel_sub = rospy.Subscriber('rc_vel', Twist, callback=self.set_vel)

    def set_pos(self, pos):
        self._pos = Pose2D(pos.x, -pos.y, pos.theta)

    def set_vel(self, vel):
        self._vel = Twist()
        self._vel.linear.x = vel.linear.x
        self._vel.linear.y = -1 * vel.linear.y
        self._vel.angular.z = vel.angular.z

    def step(self, action):
        next_state, reward, done, info = self.step_(action)
        if self.num_states > 0 and done:  # Move to terminal state.
            next_state = self.num_states - 1
        self._time += 1
        if isinstance(reward, torch.Tensor):
            if reward.shape[-1] != self.dim_reward[0]:
                reward = reward.unsqueeze(-1).repeat_interleave(self.dim_reward[0], -1)
        else:
            reward = np.atleast_1d(reward)
            if reward.shape[-1] != self.dim_reward[0]:
                reward = np.tile(reward, (self.dim_reward[0], 1)).T
        return next_state, reward, done, info

    def step_(self, action):
        prev_state = self._state
        self.apply_action(action)
        self.wait_for_action()
        self._prev_time = time.time()
        self._state = self.get_state()
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = self._state.copy()
        return next_observation, reward, done, {}

    def reset(self):
        self._prev_time = time.time()
        self._state = self.get_state()
        observation = self._state.copy()
        self._time = 0
        return observation

    def done(self, obs):
        dist = np.sqrt(np.square(obs[:2] - self._goal[:2]).sum())
        ang_dev = np.abs(obs[2] - self._goal[2])
        speed = np.sqrt(np.square(obs[3:]).sum())
        if (dist < 0.1 and ang_dev < 0.2 and speed < 0.1) or self._time >= self.max_steps:
            self.apply_action(self.zero_action)
            return True
        return False

    def apply_action(self, action):
        command = Vector3(action[0], 0.0, action[1])
        self.action_pub.publish(command)

    def wait_for_action(self):
        while time.time() < self._prev_time + self.wait_time:
            time.sleep(0.0001)

    def get_state(self):
        pos = np.array([self._pos.x, self._pos.y, self._pos.theta])
        pos[2] = ((pos[2] + np.pi) % (2 * np.pi)) - np.pi
        vel = np.array([self._vel.linear.x, self._vel.linear.y, self._vel.angular.z])
        return np.concatenate([pos, vel])

    def reward(self, obs, act, obs_next):
        return self._reward_model(obs, act, obs_next)[0].item()

    def reward_model(self):
        return self._reward_model.copy()

    @property
    def time(self) -> float:
        return self._time

    @property
    def state(self) -> torch.Tensor:
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


class RCCarEnvReward(StateActionReward):
    """
    Reward model for the RC Car environment.
    """

    dim_action = (2,)

    def __init__(self, goal, ctrl_cost_weight=0.005, speed_cost_weight=0):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.goal = goal
        self.speed_cost_weight = speed_cost_weight
        self.tolerance = ToleranceReward(
            lower_bound=0.0,
            upper_bound=0.05,
            margin_coef=20,
            sigmoid_="long_tail",
            value_at_margin=0.2,
        )
        self.theta_tolerance = ToleranceReward(
            lower_bound=0.0,
            upper_bound=0.1,
            margin_coef=32,
            sigmoid_="long_tail",
            value_at_margin=0.06,
        )

    def copy(self):
        """Get copy of reward model."""
        return RCCarEnvReward(
            goal=self.goal,
            ctrl_cost_weight=self.ctrl_cost_weight,
            speed_cost_weight=self.speed_cost_weight
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        pos_diff = next_state[..., :2] - self.goal[:2]
        theta_diff = next_state[..., 2] - self.goal[2]
        pos_dist = torch.sqrt(torch.sum(torch.square(pos_diff), dim=-1))
        theta_dist = torch.abs(((theta_diff + torch.pi) % (2 * torch.pi)) - torch.pi)
        ret = self.tolerance(pos_dist.type(torch.double)) + 0.5 * self.theta_tolerance(theta_dist.type(torch.double))
        return ret


class ToleranceReward(nn.Module):

    def __init__(
            self,
            lower_bound,
            upper_bound,
            margin_coef,
            value_at_margin,
            sigmoid_="long_tail",
    ):
        super(ToleranceReward, self).__init__()

        self.bounds = [lower_bound, upper_bound]
        self.margin = margin_coef * (upper_bound - lower_bound)
        self.value_at_margin = value_at_margin

        if lower_bound > upper_bound:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin_coef < 0:
            raise ValueError('`margin` must be non-negative.')

        assert sigmoid_ == "long_tail", "Other sigmoid functions have not been implemented."

    def forward(self, x: torch.Tensor):
        lower, upper = self.bounds
        in_bounds = torch.logical_and(lower <= x, x <= upper)
        if self.margin == 0:
            value = torch.where(in_bounds, 1.0, 0.0)
        else:
            d = torch.where(x < lower, lower - x, x - upper) / self.margin
            value = torch.where(in_bounds, 1.0, self.long_tail(d))

        return value

    def long_tail(self, x):
        scale = np.sqrt(1 / self.value_at_margin - 1)
        return 1 / ((x * scale) ** 2 + 1)
