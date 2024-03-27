import numpy as np


def encode_angles(state: np.array, angle_idx: int) -> np.array:
    """ Encodes the angle (theta) as sin(theta) and cos(theta) """
    assert angle_idx < state.shape[-1] - 1
    theta = state[..., angle_idx:angle_idx + 1]
    state_encoded = np.concatenate([state[..., :angle_idx], np.sin(theta), np.cos(theta),
                                     state[..., angle_idx + 1:]], axis=-1)
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def decode_angles(state: np.array, angle_idx: int) -> np.array:
    """ Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = np.arctan2(state[..., angle_idx:angle_idx + 1],
                        state[..., angle_idx + 1:angle_idx + 2])
    state_decoded = np.concatenate([state[..., :angle_idx], theta, state[..., angle_idx + 2:]], axis=-1)
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def project_angle(theta: np.array) -> np.array:
    # make sure angles are in [-pi, pi]
    return (theta + np.pi) % (2 * np.pi) - np.pi


def angle_diff(theta1: np.array, theta2: np.array) -> np.array:
    # Compute the difference
    diff = theta1 - theta2
    # Normalize to [-pi, pi] range
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def plot_rc_trajectory(traj: np.array, show: bool = True, encode_angle: bool = False):
    """ Plots the trajectory of the RC car """
    if encode_angle:
        traj = decode_angles(traj, 2)

    import matplotlib.pyplot as plt
    scale_factor = 1.5
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(scale_factor * 12, scale_factor * 8))
    size = 3
    axes[0][0].set_xlim(-size, size)
    axes[0][0].set_ylim(-size, size)
    axes[0][0].scatter(0, 0)
    # axes[0][0].plot(traj[:, 0], traj[:, 1])
    axes[0][0].set_title('x-y')
    # Plot the velocity of the car as vectors
    total_vel = np.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    axes[0][0].quiver(traj[0:-1:3, 0], traj[0:-1:3, 1], traj[0:-1:3, 3], traj[0:-1:3, 4],
                      total_vel[0:-1:3], cmap='jet', scale=20,
                      headlength=2, headaxislength=2, headwidth=2, linewidth=0.2)

    t = np.arange(traj.shape[0]) / 30.
    # theta
    axes[0][1].plot(t, traj[:, 2])
    axes[0][1].set_xlabel('time')
    axes[0][1].set_ylabel('theta')
    axes[0][1].set_title('theta')

    # angular velocity
    axes[0][2].plot(t, traj[:, -1])
    axes[0][2].set_xlabel('time')
    axes[0][2].set_ylabel('angular velocity')
    axes[0][2].set_title('angular velocity')

    axes[1][0].plot(t, total_vel)
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('total velocity')
    axes[1][0].set_title('velocity')

    # vel x
    axes[1][1].plot(t, traj[:, 3])
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('velocity x')
    axes[1][1].set_title('velocity x')

    axes[1][2].plot(t, traj[:, 4])
    axes[1][2].set_xlabel('time')
    axes[1][2].set_ylabel('velocity y')
    axes[1][2].set_title('velocity y')

    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes

