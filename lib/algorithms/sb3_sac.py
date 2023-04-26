import argparse

import numpy as np
import wandb

from stable_baselines3 import SAC
from rllib.util.neural_networks.utilities import parse_nonlinearity

from lib.environments.wrappers.model_based_environment import ModelBasedEnvironment
from lib.hucrl.hallucinated_model import HallucinatedModel


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

        dim_action = list(learned_env.dynamical_model.dim_action)
        if hasattr(learned_env.dynamical_model, "h_dim_action"):
            assert isinstance(learned_env.dynamical_model, HallucinatedModel)
            dim_action[0] -= learned_env.dynamical_model.h_dim_action[0]

        super(SB3_SAC, self).__init__(
            device="cpu",
            policy="MlpPolicy",
            env=learned_env,
            learning_rate=params.ppo_opt_lr,
            gamma=params.gamma,
            tau=params.policy_tau,
            gradient_steps=params.policy_grad_steps,
            train_freq=params.policy_train_freq,
            use_sde=params.sac_use_sde,
            ent_coef=params.sac_ent_coef,
            target_entropy=-np.prod(dim_action).astype(np.float32),
            policy_kwargs=policy_kwargs
        )

        self.learned_env = learned_env
        self.action_scale = learned_env.action_scale
        self.num_steps_per_update = params.policy_train_freq / float(params.policy_grad_steps)

    def reset_buffer(self):
        self.replay_buffer.reset()

    def reset(self):
        pass

    def update(self):
        pass

    def set_goal(self, goal=None):
        pass
