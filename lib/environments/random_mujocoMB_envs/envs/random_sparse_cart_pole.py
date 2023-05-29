from typing import Any
import numpy as np
from pynput import keyboard
from rllib.environment.mujoco.cart_pole import MBCartPoleEnv
from rllib.reward.state_action_reward import StateActionReward
import torch
from lib.environments.random_mujocoMB_envs.envs.utils import tolerance

from lib.environments.wrappers.random_mujoco_env import RandomMujocoEnv


class SparseCartPoleReward(StateActionReward):
    dim_action = (1,)

    def __init__(self, ctrl_cost_weight, pendulum_length=0.6):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.pendulum_length = pendulum_length
        self.rail_length = 2.5

    def copy(self):
        """Get copy of reward model."""
        return SparseCartPoleReward(
            ctrl_cost_weight=self.ctrl_cost_weight, pendulum_length=self.pendulum_length
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        x0, theta = state[..., 0], state[..., 1]
        cart_in_bounds = tolerance(x0, (-self.rail_length, self.rail_length))
        angle_in_bounds = tolerance(-torch.cos(theta), (0.995, 1)).prod()
        return cart_in_bounds * angle_in_bounds


class RandomMBSparseCartPoleEnv(MBCartPoleEnv, RandomMujocoEnv):
    """Random Cart-Pole Environment"""

    RAND_PARAMS = ["body_mass", "dof_damping", "body_inertia", "geom_friction"]
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ["geom_size"]

    def __init__(
        self,
        ctrl_cost_weight=0.01,
        random_scale_limit=3.0,
        rand_params=RAND_PARAMS,
        *args,
        **kwargs
    ):
        assert set(rand_params) <= set(
            self.RAND_PARAMS_EXTENDED
        ), "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        MBCartPoleEnv.__init__(self, ctrl_cost_weight=ctrl_cost_weight)
        RandomMujocoEnv.__init__(self, random_scale_limit, rand_params)
        self._reward_model = SparseCartPoleReward(ctrl_cost_weight=ctrl_cost_weight)
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    def reset_model(self):
        """Reset the model."""
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()


class Controller:
    def __init__(self):
        self.action = np.zeros((1,))
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            k = key.char
        except:
            k = key.name
        if k == "j":
            self.action = np.ones_like(self.action) * -3
        elif k == "l":
            self.action = np.ones_like(self.action) * 3

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.action


if __name__ == "__main__":
    env = RandomMBSparseCartPoleEnv()
    env.reset()
    controller = Controller()
    while True:
        for _ in range(500):
            action = controller()
            env.step(action)
            env.render()
        print("resetting...")
        env.reset()
