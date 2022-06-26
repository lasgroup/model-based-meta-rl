"""MPC Algorithms."""

import torch
import colorednoise
from torch.distributions import MultivariateNormal

from rllib.algorithms.mpc import CEMShooting
from rllib.util.neural_networks.utilities import repeat_along_dimension


class ICEMShooting(CEMShooting):
    r"""Improved Cross Entropy Method solves the MPC problem by adaptively sampling.

    The sampling distribution is adapted by fitting a Multivariate Gaussian to the
    best `num_elites' samples (action sequences) for `num_iter' times.

    Parameters
    ----------
    dynamical_model: state transition model.
    reward_model: reward model.
    num_model_steps: int.
        Horizon to solve planning problem.
    gamma: float, optional.
        Discount factor.
    num_iter: int, optional.
        Number of iterations of CEM method.
    num_particles: int, optional.
        Number of particles for shooting method.
    num_elites: int, optional.
        Number of elite samples to keep between iterations.
    alpha: float, optional. (default = 0.)
        Low pass filter of mean and covariance update.
    termination: Callable, optional.
        Termination condition.
    terminal_reward: terminal reward model, optional.
    warm_start: bool, optional.
        Whether or not to start the optimization with a warm start.
    default_action: str, optional.
         Default action behavior.

    References
    ----------
    P. Cristina, S. Shambhuraj, B. Sebastian, A. Jan, J. S., M. R, M. Georg (2020)
    Sample-efficient Cross-Entropy Method for Real-time Planning. Conference on Robot Learning
    """

    def __init__(
            self,
            noise_beta=2.0,
            elite_fraction=0.3,
            num_elites=10,
            decay_factor=1.25,
            alpha=0.1,
            scale=0.5,
            default_action="constant",
            *args,
            **kwargs
    ):
        super().__init__(default_action=default_action, num_elites=num_elites, alpha=alpha, scale=scale, *args, **kwargs)
        self.noise_beta = noise_beta
        self.elite_actions = None
        self.elite_fraction = elite_fraction
        self.decay_factor = decay_factor
        self.iteration = 0

    def sample_from_distribution(self, num_samples):
        if self.noise_beta > 0:
            samples = torch.tensor(colorednoise.powerlaw_psd_gaussian(
                self.noise_beta,
                size=(num_samples, *self.mean.shape[1:], self.mean.shape[0])
            ), dtype=torch.float32)
            samples = samples.permute((-1, *torch.arange(1, samples.ndim - 2), 0, -2))
        else:
            samples = torch.randn(*self.mean.shape[:-1], num_samples, self.mean.shape[-1])
        scaled_samples = self.mean.unsqueeze(-2) + (self.covariance.unsqueeze(-3) @ samples.unsqueeze(-1)).squeeze(-1)
        return scaled_samples

    def get_shifted_elites(self):
        shifted_elites = self.elite_actions[1:, ...]
        final_action = repeat_along_dimension(self.mean[-1:, ...], number=self.num_elites, dim=-2)
        shifted_elites = torch.cat((shifted_elites, final_action), dim=0)
        num_elite_samples = max(1, int(self.num_elites * self.elite_fraction))
        sample_idx = torch.multinomial(torch.ones(shifted_elites.shape[-2]), num_elite_samples)
        return shifted_elites[..., sample_idx, :]

    def get_elite_samples(self):
        num_elite_samples = max(1, int(self.num_elites * self.elite_fraction))
        sample_idx = torch.multinomial(torch.ones(self.num_elites), num_elite_samples)
        return self.elite_actions[..., sample_idx, :]

    def get_candidate_action_sequence(self):
        """Get candidate actions by sampling from a multivariate normal."""
        num_samples = max(int(self.num_particles * pow(self.decay_factor, -self.iteration)), 2 * self.num_elites)
        action_sequence = self.sample_from_distribution(num_samples)  # (num_samples, horizon, batch_size, dim_action)
        if self.iteration == 0 and self.elite_actions is not None:
            shifted_elites = self.get_shifted_elites()
            action_sequence = torch.cat((action_sequence, shifted_elites), dim=-2)
        elif self.iteration > 0:
            elite_samples = self.get_elite_samples()
            action_sequence = torch.cat((action_sequence, elite_samples), dim=-2)
        if self.iteration == self.num_iter - 1:
            action_sequence = torch.cat((action_sequence, self.mean.unsqueeze(-2)), dim=-2)
        if self.clamp:
            return action_sequence.clamp(-1.0, 1.0)
        return action_sequence

    def forward(self, state):
        """Return action that solves the MPC problem."""
        self.dynamical_model.eval()
        batch_shape = state.shape[:-1]
        self.initialize_actions(batch_shape)

        for i in range(self.num_iter):
            self.iteration = i
            action_sequence = self.get_candidate_action_sequence()
            repeated_state = repeat_along_dimension(state, number=action_sequence.shape[-2], dim=-2)
            returns = self.evaluate_action_sequence(action_sequence, repeated_state)
            self.elite_actions = self.get_best_action(action_sequence, returns)
            self.update_sequence_generation(self.elite_actions)

        if self.clamp:
            return self.elite_actions[..., 0, :].clamp(-1.0, 1.0)

        return self.elite_actions[..., 0, :]

    def reset(self, warm_action=None):
        """Reset warm action."""
        self.mean = warm_action
        self.elite_actions = None
        self.iteration = 0
