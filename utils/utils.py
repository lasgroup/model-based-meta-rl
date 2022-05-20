import numpy as np
from rllib.dataset import Observation, stack_list_of_tuples


def get_logger_layout(num_heads: int) -> dict:

    layout = {
        "Model Training": {
            "average": [
                "Multiline",
                [f"average/model-{i}" for i in range(num_heads)]
                + ["average/model_loss"],
            ]
        },
        "Policy Training": {
            "average": [
                "Multiline",
                ["average/value_loss", "average/policy_loss", "average/eta_loss"],
            ]
        },
        "Returns": {
            "average": [
                "Multiline",
                ["average/environment_return", "average/model_return"],
            ]
        },
    }

    return layout


def sample_trajectories(trajectories, num_samples=1):
    """
    Sample a batch of trajectories from observations
    :param trajectories: Stacked_observations
    :param num_samples: Number of samples
    :return: An MPC based age
    """

    if isinstance(trajectories, tuple):
        trajectories = stack_list_of_tuples(trajectories)

    random_batch = np.random.choice(trajectories.shape[-1], num_samples)

    new_obs = Observation(
        state=trajectories.state[:, :, random_batch, :],
        action=trajectories.action[:, :, random_batch, :],
        reward=trajectories.reward[:, :, random_batch, :],
        next_state=trajectories.next_state[:, :, random_batch, :],
        done=trajectories.done[:, :, random_batch],
        next_action=trajectories.next_action,
        log_prob_action=trajectories.log_prob_action[:, :, random_batch],
        entropy=trajectories.entropy[:, :, random_batch],
        state_scale_tril=trajectories.state_scale_tril,
        next_state_scale_tril=trajectories.next_state_scale_tril,
        reward_scale_tril=trajectories.reward_scale_tril,
    )

    return new_obs
