import torch

from rllib.reward.state_action_reward import StateActionReward


class PendulumReward(StateActionReward):
    """
    Reward model for the mountain car gym environment.
    """
    dim_action = (1,)

    def __init__(self, ctrl_cost_weight=0.001):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        theta = torch.atan2(state[..., 1], state[..., 0])
        reward = - torch.square(theta) - 0.1 * torch.square(state[..., 2])
        return reward
