"""Implementation of Gradient-based Adaptive Learner Algorithm"""
import os
import time
from collections import deque

import torch
from gym.utils import colorize

from rllib.dataset import stack_list_of_tuples
from rllib.util.neural_networks.utilities import deep_copy_module
from rllib.util.training.utilities import model_loss, get_model_validation_score

from utils.utils import get_project_path
from lib.agents.mpc_agent import MPCAgent
from lib.datasets import TrajectoryReplay
from lib.meta_rl.algorithms.maml import MAML
from lib.datasets.utilities import get_trajectory_segment
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper


class GrBALAgent(MPCAgent):
    """
    Implementation of the Gradient-based Adaptive Learner.

    References
    ----------
    Nagabandi, A., Clavera, I., Liu, S., Fearing, R., Abbeel, P., Levine, S., Finn, C.,
    Learning to Adapt in Dynamic Real World Environments through Meta Reinforcement Learning.
    """

    def __init__(
            self,
            meta_environment: MetaEnvironmentWrapper = None,
            past_segment_len: int = 16,
            future_segment_len: int = 16,
            inner_lr: float = 0.01,
            model_learn_batch_size: int = 500,
            max_memory: int = 1000000,
            num_iter_meta_train: int = 0,
            trajectory_replay_load_path=None,
            multiple_runs_id=0,
            clip_gradient_val=2.0,
            *args,
            **kwargs
    ):
        super().__init__(
            max_memory=max_memory,
            model_learn_batch_size=model_learn_batch_size,
            *args,
            **kwargs
        )

        self.multiple_runs_id = multiple_runs_id

        self.meta_environment = meta_environment

        self.pre_update_model = MAML(deep_copy_module(self.dynamical_model.base_model), lr=inner_lr)
        self.meta_optimizer = type(self.model_optimizer)(
            self.pre_update_model.parameters(),
            **self.model_optimizer.defaults
        )
        self.clip_gradient_val = clip_gradient_val
        self.num_iter_meta_train = num_iter_meta_train

        self.past_segment_len = past_segment_len
        self.future_segment_len = future_segment_len
        self.observation_queue = deque([], self.past_segment_len)

        self.dataset = TrajectoryReplay(
            max_len=max_memory,
            transformations=self.dynamical_model.transformations
        )
        if trajectory_replay_load_path is not None:
            self.load_trajectory_replay(trajectory_replay_load_path)

    def set_meta_environment(self, meta_environment):
        self.meta_environment = meta_environment

    def observe(self, observation):
        transformed_observation = observation.clone()
        for transform in self.dynamical_model.transformations:
            transformed_observation = transform(transformed_observation)
        self.observation_queue.append(transformed_observation)
        super().observe(observation)

    def act(self, state):
        if len(self.observation_queue) > 0:
            past_observations = stack_list_of_tuples([observation for observation in self.observation_queue.copy()])
            current_model = self.pre_update_model.clone()
            past_loss = model_loss(current_model, past_observations, dynamical_model=None).mean()
            current_model.adapt(past_loss)
        else:
            current_model = self.pre_update_model.clone()
        self.dynamical_model.base_model.load_state_dict(current_model.module.state_dict())
        return super().act(state)

    def start_episode(self):
        assert self.meta_environment is not None, "Meta training environment has not been set!"
        self.meta_environment.sample_next_env()
        self.dataset.start_episode()
        super().start_episode()

    def end_episode(self):
        self.dataset.end_episode()
        super().end_episode()

    def learn_model(self, epsilon=-1.0, non_decrease_iter=float("inf")):
        """
        Performs the GrBAL learning procedure for the dynamics model
        """
        if self.model_learn_num_iter > 0:
            print(colorize("Training Dynamics Model", "yellow"))

        self.dynamical_model.base_model.train()
        self.pre_update_model.train()
        for num_iter in range(self.model_learn_num_iter):
            self._maml_train_model_step()

        self._maml_validate_model_step()

    def meta_fit(self, log_period=500, eval_period=3000):
        """
        Fits the MAML pre-update model
        """
        print("Start meta-training -------------------- ")
        t = time.time()

        self.dynamical_model.base_model.train()
        self.pre_update_model.train()

        for it in range(self.num_iter_meta_train):

            loss = self._maml_train_model_step()

            message = ''
            if it % log_period == 0 or it % eval_period == 0:
                avg_log_prob = loss.mean().numpy()
                message += '\nIter %d/%d - Time %.2f sec' % (it, self.num_iter_meta_train, time.time() - t)
                message += ' - Train-Loss: %.5f' % avg_log_prob

                t = time.time()

            if len(message) > 0:
                print(message)

        loss = loss.mean().numpy()
        return loss

    def _train_model_step(self):
        self.model_optimizer.zero_grad()
        if max(self.dataset.trajectory_lengths) >= (self.past_segment_len + self.future_segment_len):
            loss = torch.zeros(1)
            for batch in range(self.model_learn_batch_size):
                trajectory = self.dataset.sample_segment(self.past_segment_len + self.future_segment_len)
                loss += model_loss(self.dynamical_model.base_model, trajectory, dynamical_model=None).mean()
            loss.backward()
            self.model_optimizer.step()
            self.logger.update(**{f"{self.dynamical_model.model_kind[:3]}-loss": loss.item()})

    def _maml_train_model_step(self):
        self.meta_optimizer.zero_grad()
        loss = torch.zeros(1)
        if max(self.dataset.trajectory_lengths) >= (self.past_segment_len + self.future_segment_len):
            for batch in range(self.model_learn_batch_size):
                trajectory = self.dataset.sample_segment(self.past_segment_len + self.future_segment_len)
                task_model = self.pre_update_model.clone()
                error = model_loss(
                    task_model,
                    get_trajectory_segment(trajectory, 0, self.past_segment_len),
                    dynamical_model=None
                ).mean()
                task_model.adapt(error)
                loss += model_loss(
                    task_model,
                    get_trajectory_segment(trajectory, -self.future_segment_len, None),
                    dynamical_model=None
                ).mean() / self.model_learn_batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pre_update_model.parameters(), self.clip_gradient_val)
            self.meta_optimizer.step()
            self.logger.update(**{f"{self.dynamical_model.model_kind[:3]}-loss": loss.item()})
        return loss.detach()

    def _maml_validate_model_step(self):
        mse, sharpness, calibration_score = [], [], []
        if max(self.dataset.trajectory_lengths) >= (self.past_segment_len + self.future_segment_len):
            for batch in range(self.model_learn_batch_size):
                self.meta_optimizer.zero_grad()
                trajectory = self.dataset.sample_segment(self.past_segment_len + self.future_segment_len)
                task_model = self.pre_update_model.clone()
                error = model_loss(
                    task_model,
                    get_trajectory_segment(trajectory, 0, self.past_segment_len),
                    dynamical_model=None
                ).mean()
                task_model.adapt(error)
                _, mse_, sharpness_, calibration_score_ = get_model_validation_score(
                    task_model,
                    get_trajectory_segment(trajectory, -self.future_segment_len, None)
                )
                mse.append(mse_)
                sharpness.append(sharpness_)
                calibration_score.append(calibration_score_)

            self.logger.update(
                **{
                    f"{self.dynamical_model.model_kind[:3]}-val-mse": sum(mse) / self.model_learn_batch_size,
                    f"{self.dynamical_model.model_kind[:3]}-sharp": sum(sharpness) / self.model_learn_batch_size,
                    f"{self.dynamical_model.model_kind[:3]}-calib": sum(calibration_score) / self.model_learn_batch_size,
                }
            )

    def load_trajectory_replay(self, project_rel_path):
        load_path = os.path.join(get_project_path(), project_rel_path)
        self.dataset = torch.load(load_path)
        if os.path.exists(load_path.replace('train', 'test')):
            self.val_dataset = torch.load(load_path.replace('train', 'test'))

    def train(self, val=True):
        """Set the agent in training mode"""
        self.meta_environment.train(val)
        super().train(val)

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.meta_environment.eval(val)
        super().eval(val)
