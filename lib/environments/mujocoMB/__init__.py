import os
ENV_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/envs")

from rllib.environment.mujoco.ant import MBAntEnv
from rllib.environment.mujoco.cart_pole import MBCartPoleEnv
from rllib.environment.mujoco.half_cheetah import MBHalfCheetahEnv
from rllib.environment.mujoco.hopper import MBHopperEnv
from rllib.environment.mujoco.inverted_double_pendulum import MBInvertedDoublePendulumEnv
from rllib.environment.mujoco.inverted_pendulum import MBInvertedPendulumEnv
from rllib.environment.mujoco.pusher import MBPusherEnv
from rllib.environment.mujoco.reacher_2d import MBReacherEnv
from rllib.environment.mujoco.reacher_3d import MBReacher3DEnv
from rllib.environment.mujoco.walker_2d import MBWalker2dEnv