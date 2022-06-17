import torch

from rllib.reward.state_action_reward import StateActionReward


class MountainCarContinuousReward(StateActionReward):
    """
    Reward model for the mountain car gym environment.
    """
    dim_action = (1,)

    def __init__(self, ctrl_cost_weight=0.1):
        super().__init__(ctrl_cost_weight=ctrl_cost_weight)

    def state_reward(self, state, next_state=None):
        """Get reward that corresponds to the states."""
        reward = torch.zeros(state.shape[:-1])
        goal = torch.logical_and(
            next_state[..., 0] >= 0.45 * torch.ones(state.shape[:-1]),
            state[..., 0] < 0.45 * torch.ones(state.shape[:-1])
        )
        reward[goal] = 100
        return reward
