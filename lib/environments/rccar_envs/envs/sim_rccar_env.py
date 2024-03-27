import numpy as np

import torch
from gym.spaces import Box
from rllib.environment import AbstractEnvironment

from lib.environments.rccar_envs.envs.dynamics_model import RaceCar, CarParams
from lib.environments.rccar_envs.envs.utils import encode_angles, decode_angles, plot_rc_trajectory
from typing import Dict, Any, Tuple
from lib.environments.rccar_envs.envs.rccar_env import RCCarEnvReward, reached_goal, constraint_violation


class RCCarSimEnv(AbstractEnvironment):
    max_steps: int = 200
    _dt: float = 1 / 30.
    dim_action: Tuple[int] = (2,)
    _goal: np.array = np.array([0.0, 0.0, - np.pi / 2.])
    _init_pose: np.array = np.array([-1.04, -1.42, np.pi / 2.])
    _angle_idx: int = 2
    _obs_noise_stds: np.array = 0.05 * np.exp(np.array([-3.3170326, -3.7336411, -2.7081904,
                                                        -2.7841284, -2.7067015, -1.4446207]))
    _max_vel: np.array = np.array([8.0, 8.0, 15.0])

    _default_car_model_params_bicycle: Dict = {
        'use_blend': 0.0,
        'm': 1.65,
        'l_f': 0.13,
        'l_r': 0.17,
        'angle_offset': 0.02791893,
        'b_f': 2.58,
        'b_r': 3.39,
        'blend_ratio_lb': 0.4472136,
        'blend_ratio_ub': 0.5477226,
        'c_d': -1.8698378e-36,
        'c_f': 1.2,
        'c_m_1': 10.431917,
        'c_m_2': 1.5003588,
        'c_r': 1.27,
        'd_f': 0.02,
        'd_r': 0.017,
        'i_com': 2.78e-05,
        'steering_limit': 0.19989373
    }

    _default_car_model_params_blend: Dict = {
        'use_blend': 1.0,
        'm': 1.65,
        'l_f': 0.13,
        'l_r': 0.17,
        'angle_offset': 0.00731506,
        'b_f': 2.5134025,
        'b_r': 3.8303657,
        'blend_ratio_lb': -0.00057009,
        'blend_ratio_ub': -0.07274915,
        'c_d': -6.9619144e-37,
        'c_f': 1.2525784,
        'c_m_1': 10.93334,
        'c_m_2': 1.0498677,
        'c_r': 1.2915123,
        'd_f': 0.43698108,
        'd_r': 0.43703166,
        'i_com': 0.06707229,
        'steering_limit': 0.5739077
    }

    def __init__(self, ctrl_cost_weight: float = 0.005, encode_angle: bool = False, use_obs_noise: bool = True,
                 use_tire_model: bool = True, action_delay: float = 0.0, clip_velocities: bool = True, slow_task=False,
                 action_stacking: bool = True, car_model_params: Dict = None, env_params=None, seed: int = 230492394):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight for the control/action penalty
            use_obs_noise: whether to use observation noise
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """

        if env_params is not None:
            encode_angle = env_params.get('encode_angle', encode_angle)
            use_obs_noise = env_params.get('use_obs_noise', use_obs_noise)
            use_tire_model = env_params.get('use_tire_model', use_tire_model)
            action_delay = env_params.get('action_delay', action_delay)
            clip_velocities = env_params.get('clip_velocities', clip_velocities)
            action_stacking = env_params.get('action_stacking', action_stacking)

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if action_delay % self._dt == 0.0:
            self._act_delay_interpolation_weights = np.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = np.array([weight_first, 1.0 - weight_first])
        self.action_delay_buffer_size = int(np.ceil(action_delay / self._dt)) + 1
        self._action_buffer = np.zeros((self.action_delay_buffer_size, self.dim_action[0]))

        self.action_stacking = action_stacking and action_delay > 0.0
        self.action_stacking_dim = self.dim_action[0] * (self.action_delay_buffer_size - 1) if self.action_stacking else 0

        super(RCCarSimEnv, self).__init__(
            dim_state=(7 + self.action_stacking_dim,) if encode_angle else (6 + self.action_stacking_dim,),
            dim_action=(2,),
            observation_space=Box(low=-np.inf, high=np.inf, shape=(2,)),
            action_space=Box(low=-1.0, high=1.0, shape=(2,)),
            dim_observation=(-1,),
            num_states=-1,
            num_actions=-1,
            num_observations=-1,
            dim_reward=(1,),
        )

        self.encode_angle: bool = encode_angle
        self.use_obs_noise: bool = use_obs_noise
        self.rng = np.random.RandomState(seed=seed)

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=False)
        self._slow_task = slow_task

        # choose default params depending on whether we use the tire model or only the bicycle model
        self.use_tire_model = use_tire_model
        if use_tire_model:
            self._default_car_model_params = self._default_car_model_params_blend
        else:
            self._default_car_model_params = self._default_car_model_params_bicycle

        # update default params with user-defined params
        if car_model_params is None:
            _car_model_params = self._default_car_model_params
        else:
            _car_model_params = self._default_car_model_params
            _car_model_params.update(car_model_params)
        self._dynamics_params = CarParams(**_car_model_params)

        self.clip_velocities: bool = clip_velocities
        self._max_vel_norm: float = np.linalg.norm(self._max_vel[:2], axis=-1).item()

        # initialize reward model
        self._reward_model = RCCarEnvReward(goal=np.array(self._goal), ctrl_cost_weight=ctrl_cost_weight)

        # initialize time and state
        self._time: int = 0
        self._state: np.array = np.zeros(self.dim_state)

    def reset(self) -> np.array:
        """ Resets the environment to a random initial state close to the initial pose """

        # sample random initial state
        init_pos = self._init_pose[:2] + self.rng.uniform(size=(2,), low=-0.10, high=0.10)
        init_theta = self._init_pose[2:] + self.rng.uniform(size=(1,), low=-0.10 * np.pi, high=0.10 * np.pi)
        init_vel = np.zeros((3,)) + np.array([0.005, 0.005, 0.02]) * self.rng.normal(size=(3,))
        init_state = np.concatenate([init_pos, init_theta, init_vel])

        self._state = init_state
        self._action_buffer = np.zeros((self.action_delay_buffer_size, self.dim_action[0]))
        self._time = 0
        return np.array(self._state_to_obs(self._state))

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        """ Performs one step in the environment

         Args:
             action: action of shape (2,) --> (throttle, steering)
         """

        assert action.shape[-1:] == self.dim_action
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        # assert np.all(-1 <= action) and np.all(action <= 1), "action must be in [-1, 1]"

        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action_dyn = self._get_delayed_action(action).copy()
        else:
            # use action directly
            action_dyn = action.copy()

        if self._slow_task:
            action_dyn[0] = 0.17 * action_dyn[0] + (action_dyn[0] < 0.0) * 0.14 * action_dyn[0]

        # compute next state
        action_flipped = np.flip(action_dyn, axis=-1)  # flip because dynamics model expects (steering, throttle)
        self._state = self._dynamics_model.next_step(self._state, action_flipped, self._dynamics_params)
        if self.clip_velocities:
            self._state = self._get_clipped_velocity(self._state)
        self._time += 1
        obs = np.array(self._state_to_obs(self._state))

        # compute reward
        reward = self._reward_model(state=obs, action=np.array(action), next_state=obs)[0]

        # check if done
        done = self._done(obs)

        reached_goal_state = reached_goal(state=obs, goal=np.array(self._goal))
        if done and reached_goal_state.item():
            reward += 1.5 * (self.max_steps - self._time)

        # info dict
        info = {'time': self._time,
                'state': np.array(self._state),  # internal state (without obs noise and angle encoding)
                'reward': reward,
                'action_buffer': self._action_buffer.copy()}

        # return observation, reward, done, info
        return obs, reward, done, info

    def _done(self, obs: np.array) -> bool:
        return reached_goal(obs, np.array(self._goal)).item() or constraint_violation(obs).item() \
            or (self._time >= self.max_steps)

    def _state_to_obs(self, state: np.array) -> np.array:
        """ Adds observation noise to the state """
        assert state.shape[-1] == 6

        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * self.rng.normal(size=self._state.shape)
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (obs.shape[-1] == 6 and not self.encode_angle)

        if self.action_stacking:
            obs = np.concatenate([obs, self._action_buffer[1:].copy().flatten()], axis=-1)

        return obs

    def _get_delayed_action(self, action: np.array) -> np.array:
        # push action to action buffer
        self._action_buffer = np.concatenate([self._action_buffer[1:], action[None, :]], axis=0)

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = np.sum(self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None], axis=0)
        assert delayed_action.shape == self.dim_action
        return delayed_action

    def _get_clipped_velocity(self, state: np.array) -> np.array:
        """ Clips the velocity to the maximum velocity """
        assert state.shape[-1] == 6
        clipped_state = state.copy()
        # if np.linalg.norm(state[3:5], axis=-1).item() > self._max_vel_norm:
        #     clipped_state[3:5] = self._max_vel_norm * state[3:5] / np.linalg.norm(state[3:5], axis=-1)
        # clipped_state[5] = np.clip(state[5], a_min=-self._max_vel[2], a_max=self._max_vel[2])
        clipped_state[3:6] = np.clip(state[3:6], a_min=-self._max_vel, a_max=self._max_vel)
        return clipped_state

    def reward_model(self):
        return self._reward_model.copy()

    @property
    def time(self) -> float:
        return self._time

    @property
    def goal(self):
        return self._goal

    @property
    def state(self):
        """Return current state of environment."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


if __name__ == '__main__':
    ENCODE_ANGLE = False
    env = RCCarSimEnv(encode_angle=ENCODE_ANGLE,
                      action_delay=0.07,  # 70 ms action delay
                      use_obs_noise=True, seed=45645)

    s = env.reset()
    traj = [s]
    rewards = []
    for i in range(50):
        t = i / 30.
        a = np.array([0.8 / (t + 1), - 1 * np.cos(1.0 * t)])
        s, r, _, _ = env.step(a)
        traj.append(s)
        rewards.append(r)

    traj = np.stack(traj)

    plot_rc_trajectory(traj, encode_angle=ENCODE_ANGLE)

    from matplotlib import pyplot as plt

    plt.plot(np.arange(len(rewards)) / 30., rewards)
    plt.show()
