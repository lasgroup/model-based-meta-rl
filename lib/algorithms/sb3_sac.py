import argparse

import wandb

from stable_baselines3 import SAC
from rllib.util.neural_networks.utilities import parse_nonlinearity

from lib.environments.wrappers.model_based_environment import ModelBasedEnvironment


class SB3_SAC(SAC):

    def __init__(
            self,
            learned_env: ModelBasedEnvironment,
            params: argparse.Namespace,
    ):

        policy_kwargs = dict(
            activation_fn=parse_nonlinearity(params.policy_non_linearity),
            net_arch=dict(pi=params.policy_layers, qf=params.value_function_layers)
        )

        super(SB3_SAC, self).__init__(
            policy="MlpPolicy",
            env=learned_env,
            learning_rate=params.ppo_opt_lr,
            policy_kwargs=policy_kwargs
        )

        self.action_scale = learned_env.action_scale

    def reset_buffer(self):
        self.replay_buffer.reset()

    def reset(self):
        pass

    def update(self):
        pass

    def set_goal(self, goal=None):
        pass
