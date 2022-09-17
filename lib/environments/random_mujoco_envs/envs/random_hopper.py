import numpy as np

from rand_param_envs.gym import utils
from rand_param_envs.base import RandomEnv

from rllib.reward.state_action_reward import StateActionReward


class LocomotionRewardPos(StateActionReward):
    r"""A locomotion reward model is used for locomotion robots.

    The reward function is computed as:
    r(s, a) = velocity + healthy + action_reward.

    The action reward is computed from the state-action reward.
    The velocity is the velocity_pos component of the state.
    """

    def __init__(
        self,
        dim_action,
        ctrl_cost_weight,
        forward_reward_weight=1.0,
        healthy_reward=0.0,
        velocity_pos=0
    ):
        self.dim_action = dim_action
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.velocity_pos = velocity_pos

    def copy(self):
        """Get copy of locomotion reward."""
        return LocomotionRewardPos(
            dim_action=self.dim_action,
            ctrl_cost_weight=self.ctrl_cost_weight,
            forward_reward_weight=self.forward_reward_weight,
            healthy_reward=self.healthy_reward,
            velocity_pos=self.velocity_pos
        )

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        return self.forward_reward_weight * state[..., self.velocity_pos] + self.healthy_reward


class RandomHopper(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        self._reward_model = LocomotionRewardPos(
            dim_action=(3,),
            ctrl_cost_weight=0.001,
            healthy_reward=1.0,
            forward_reward_weight=1.0,
            velocity_pos=5
        )

    def _step(self, a):
        obs = self._get_obs()
        reward = self._reward_model(obs, a)[0].item()
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def reward_model(self):
        """Get reward model."""
        return self._reward_model.copy()
