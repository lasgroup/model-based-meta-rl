from torch.nn.functional import mse_loss

from rllib.dataset.datatypes import Loss, Observation
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm

from lib.algorithms.utils import lambda_values, discount_sequence


class BPTT(AbstractAlgorithm):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def actor_loss(self, trajectory: Observation) -> Loss:
        state, action, reward, next_state, *r = trajectory
        reward_values = self.critic(next_state)  # TODO: Can also use a probabilistic critic
        reward_lambdas = lambda_values(
            reward_values,
            reward,
            self.gamma,
            self.td_lambda
        )
        discount = discount_sequence(self.gamma, next_state.shape[0])
        loss = - reward_lambdas * discount

        return Loss(policy_loss=loss).reduce(self.criterion.reduction)

    def critic_loss(self, trajectory: Observation) -> Loss:
        state, action, reward, next_state, *r = trajectory
        reward_values = self.critic_target(next_state)
        reward_lambdas = lambda_values(
            reward_values,
            reward,
            self.gamma,
            self.td_lambda
        )
        values = self.critic(next_state)
        loss = mse_loss(values, reward_lambdas)

        return Loss(critic_loss=loss).reduce(self.criterion.reduction)

    def regularization_loss(self, observation, num_trajectories=1):
        _ = super().regularization_loss(observation, num_trajectories)
        return Loss()
