from gym.utils import colorize

import torch

from lib.agents import MPCAgent
from lib.model.ghv_model_learning import ghv_model_loss, train_model
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper


class GHVMDPAgent(MPCAgent):

    def __init__(
            self,
            meta_environment: MetaEnvironmentWrapper = None,
            max_memory: int = 1000000,
            num_iter_eval_train: int = 20,
            fit_at_step=True,
            env_name="",
            trajectory_replay_load_path=None,
            multiple_runs_id=0,
            *args,
            **kwargs
    ):

        super().__init__(
            max_memory=max_memory,
            *args,
            **kwargs
        )

        self.meta_environment = meta_environment
        self.env_name = env_name
        self.multiple_runs_id = multiple_runs_id
        self.fit_at_step = fit_at_step
        self.num_iter_eval_train = num_iter_eval_train

        self.last_observation = None

        if trajectory_replay_load_path is not None:
            self.load_trajectory_replay(trajectory_replay_load_path)
            self.save_data = False
        else:
            self.save_data = True

        self.latent_optimizer = type(self.model_optimizer)(
            self.dynamical_model.base_model.latent_dist_params,
            **self.model_optimizer.defaults
        )

    def set_meta_environment(self, meta_environment):
        self.meta_environment = meta_environment

    def observe(self, observation):
        last_observation = observation.clone()
        # for transform in self.dynamical_model.transformations:
        #     last_observation = transform(last_observation)
        self.last_observation = last_observation
        return super().observe(observation)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        if (not self.training) and self.fit_at_step:
            self.fit_task(
                num_iter_eval_train=self.num_iter_eval_train
            )
        return super().act(state)

    def start_episode(self):
        assert self.meta_environment is not None, "Meta training environment has not been set!"
        self.meta_environment.sample_next_env()
        super().start_episode()

    def train(self, val=True):
        """Set the agent in training mode"""
        self.meta_environment.train(val)
        super().train(val)

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.meta_environment.eval(val)
        super().eval(val)
        self.start_trial()

    def start_trial(self):
        self.dynamical_model.base_model.reset_latent_dist()

    def fit_task(self, num_iter_eval_train=None):
        self.dynamical_model.train()
        model = self.dynamical_model.base_model
        if self.last_observation is not None:
            for num_iter in range(num_iter_eval_train):
                self.latent_optimizer.zero_grad()
                loss = ghv_model_loss(model, self.last_observation, dynamical_model=None).mean()
                loss.backward()
                self.latent_optimizer.step()

    def learn_model(self):
        if self.model_learn_num_iter > 0:
            print(colorize("Training Dynamics Model", "yellow"))

            train_fn = train_model
            if len(self.dataset) >= self.model_learn_batch_size:
                val_set = self.val_dataset if len(self.val_dataset) > self.model_learn_batch_size else None
                train_fn(
                    self.dynamical_model.base_model,
                    train_set=self.dataset,
                    validation_set=val_set,
                    max_iter=self.model_learn_num_iter,
                    optimizer=self.model_optimizer,
                    logger=self.logger,
                    batch_size=self.model_learn_batch_size,
                    epsilon=-1.0,
                )

    def meta_learn(self):
        pass
