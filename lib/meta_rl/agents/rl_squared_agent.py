"""Implementation of RL^2 Algorithm"""

import torch
from rllib.agent import PPOAgent, AbstractAgent
from rllib.dataset.datatypes import Observation

from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper


class RLSquaredAgent(PPOAgent):
    """
    Implementation of the RL^2 Algorithm.

    References
    ----------
    Duan, Y., Schulman, J., Chen, Xi., Bartlett, P., Sutskever, I., Abbeel, P. (2017)
    RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning. ICLR.
    """

    def __init__(
            self,
            meta_environment: MetaEnvironmentWrapper = None,
            trial_len: int = 2,
            num_iter: int = 20,
            num_rollouts: int = 1,
            *args,
            **kwargs
    ):
        super().__init__(num_iter=num_iter, num_rollouts=num_rollouts, *args, **kwargs)

        self.counters = {
            "total_trials": 0,
            "train_trials": 0,
            "eval_trials": 0,
            "total_episodes": 0,
            "total_steps": 0,
            "train_steps": 0,
            "train_episodes": 0,
            "eval_episodes": 0,
        }

        self.meta_environment = meta_environment
        self.trial_len = trial_len

    def __str__(self):
        """Generate string to parse the agent."""
        comment = self.comment if len(self.comment) else self.policy.__class__.__name__
        opening = "=" * 88
        str_ = (
            f"\n{opening}\n{self.name} & {comment}\n"
            f"Total trials {self.total_trials}\n"
            f"Train trials {self.train_trials}\n"
            f"Eval trials {self.eval_trials}\n"
            f"Total episodes {self.total_episodes}\n"
            f"Train episodes {self.train_episodes}\n"
            f"Eval episodes {self.eval_episodes}\n"
            f"Total steps {self.total_steps}\n"
            f"Train steps {self.train_steps}\n"
            f"{self.logger}{opening}\n"
        )
        return str_

    def set_meta_environment(self, meta_environment):
        self.meta_environment = meta_environment

    def act(self, state):
        if len(self.trajectories[-1]) > 0:
            previous_observation = self.trajectories[-1][-1]
        else:
            previous_observation = Observation(state)
            previous_observation.action = torch.zeros(self.policy.dim_action)
            previous_observation.reward = torch.zeros(1)
            previous_observation.done = torch.zeros(1)
        extended_state = self.get_input_from_observation(state, previous_observation)
        return super().act(extended_state)

    def get_input_from_observation(self, state, previous_observation):
        extended_state = torch.hstack(
            (
                state,
                previous_observation.action,
                previous_observation.reward,
                previous_observation.done
            )
        )
        return extended_state

    def observe(self, observation):
        if len(self.trajectories[-1]) > 0:
            previous_observation = self.trajectories[-1][-1]
        else:
            previous_observation = Observation(observation.state)
            previous_observation.action = torch.zeros(self.policy.dim_action)
            previous_observation.reward = torch.zeros(1)
            previous_observation.done = torch.zeros(1)
        observation.state = self.get_input_from_observation(observation.state, previous_observation)
        observation.next_state = self.get_input_from_observation(observation.next_state, observation)
        super().observe(observation)

    def start_episode(self):
        assert self.meta_environment is not None, "Meta training environment has not been set!"

        if self.counters["total_episodes"] % self.trial_len == 0:
            self.start_trial()
        AbstractAgent.start_episode(self)

    def start_trial(self):
        self.meta_environment.sample_next_env()
        self.trajectories.append([])

    def end_episode(self):
        if (self.counters["total_episodes"] + 1) % self.trial_len == 0:
            self.end_trial()
        AbstractAgent.end_episode(self)

    def end_trial(self):
        self.counters["total_trials"] += 1
        if self.training:
            self.counters["train_trials"] += 1
        else:
            self.counters["eval_trials"] += 1

        end_trial_dict = dict()
        rewards = torch.stack([obs.reward for obs in self.trajectories[-1]])
        end_trial_dict.update(**{f"mean_trial_reward": torch.mean(rewards, dim=0).detach().item()})
        self.logger.update(**end_trial_dict)

        if self.train_at_end_episode:
            self.learn()
            self.trajectories = list()
        if self.num_rollouts == 0:
            self.trajectories = list()

        self.policy.reset()
        self.algorithm.critic.reset()
        self.algorithm.old_policy.reset()
        self.algorithm.critic_target.reset()
        self.algorithm.policy_target.reset()

    def train(self, val=True):
        """Set the agent in training mode"""
        self.meta_environment.train(val)
        super().train(val)

    def eval(self, val=True):
        """Set the agent in evaluation mode."""
        self.meta_environment.eval(val)
        super().eval(val)


    @property
    def train_trials(self):
        """Return number of training episodes."""
        return self.counters["train_trials"]

    @property
    def eval_trials(self):
        """Return number of evaluation episodes."""
        return self.counters["eval_trials"]

    @property
    def total_trials(self):
        """Return number of total episodes."""
        return self.counters["total_trials"]
