"""MPC Algorithms."""

import torch

from rllib.util.neural_networks.utilities import repeat_along_dimension

from lib.solvers.icem_shooting import ICEMShooting


class PETSShooting(ICEMShooting):
    r"""Cross Entropy Method for solving the MPC problem by Trajectory Sampling(PETS).

    The sampling distribution is adapted by fitting a Multivariate Gaussian to the
    best `num_elites' samples (action sequences) for `num_iter' times.

    Parameters
    ----------
    trajectory_samples_per_action: int, optional.
        Number of trajectories to sample for each action.

    References
    ----------
    Chua, K., Calandra, R., McAllister, R., Levine, S., (2018)
    Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models, NeurIPS
    """

    def __init__(
            self,
            trajectory_samples_per_action: int = 10,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.trajectory_samples_per_action = trajectory_samples_per_action

    def forward(self, state):
        """Return action that solves the MPC problem."""
        self.dynamical_model.eval()
        batch_shape = state.shape[:-1]
        self.initialize_actions(batch_shape)

        for i in range(self.num_iter):
            self.iteration = i
            action_sequence = self.get_candidate_action_sequence()
            repeated_state = repeat_along_dimension(state, number=self.trajectory_samples_per_action, dim=-2)
            repeated_state = repeat_along_dimension(repeated_state, number=action_sequence.shape[-2], dim=-2)
            action_sequence_rep = repeat_along_dimension(action_sequence, number=self.trajectory_samples_per_action,
                                                         dim=-3)
            returns = self.evaluate_action_sequence(action_sequence_rep, repeated_state)
            returns = returns.mean(dim=-3)
            self.elite_actions = self.get_best_action(action_sequence, returns)
            self.update_sequence_generation(self.elite_actions)

        if self.clamp:
            return self.elite_actions[..., 0, :].clamp(-1.0, 1.0)

        return self.elite_actions[..., 0, :]
