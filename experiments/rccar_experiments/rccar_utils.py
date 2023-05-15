import os

import numpy as np
import torch
from rllib.util.rollout import step_model

DATA_DIR = "/home/arjun/Desktop/thesis/car-filter/data/recordings/"
SKIP_ROWS = 100
MAX_ROWS = -100
TIME_INT = 0.03


def split_controls(controls):
    splits = [0]
    last_control = controls[0]
    for i in range(len(controls)):
        if abs(controls[i, 0] - last_control[0]) > 0.001 or abs(controls[i, 1] - last_control[1]) > 0.001:
            splits.append(i)
            last_control = controls[i]

    avg_split = 7
    splits_new = [splits[0]]
    for i in range(1, len(splits)):
        diff = splits[i] - splits[i-1]
        if diff > 1.5 * avg_split:
            for j in range(0, diff // avg_split):
                if not (splits[i] - splits[i-1] - avg_split * (j + 1)) < 0.5 * avg_split:
                    splits_new.append(splits[i-1] + avg_split * (j + 1))
        splits_new.append(splits[i])

    return splits_new


def load_data(files, dir=None):
    states_list, actions_list, next_states_list = [], [], []
    for file in files:
        filepath = os.path.join(dir or DATA_DIR, file)
        states, actions, next_states = get_state_action_from_file(filepath)
        states_list.append(states)
        actions_list.append(actions)
        next_states_list.append(next_states)
    return states_list, actions_list, next_states_list


def get_state_action_from_file(filepath):
    data = np.loadtxt(
        fname=filepath,
        delimiter=',',
        skiprows=SKIP_ROWS,
    )

    controls = data[1:MAX_ROWS, 5:7]
    pose = data[1:MAX_ROWS, 7:10] * np.array([1.0, -1.0, 1.0])
    smoothed_vel = data[1:MAX_ROWS, 13:16] * np.array([1.0, -1.0, 1.0])

    # splits = split_controls(controls)

    pose_vel = np.concatenate([pose, smoothed_vel], axis=-1)
    states = pose_vel[:-1]
    actions = controls[:-1]
    next_states = pose_vel[1:]

    return states, actions, next_states


def save_as_csv(states, actions):
    with open("policy_out.csv", "w") as f:
        f.write(f"iFrame, midExpo, receiveMotive, receiveRecorder, matchedControl, "
                f"throttle, steer, "
                f"pos x, pos y, theta, "
                f"vel x, vel y, omega, "
                f"s vel x, s vel y, s omega\n")
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            out = f"{i}, {int(i * TIME_INT * 10**7)}, {0}, {0}, {0}, " \
                  f"{action[0]}, {action[1]}, " \
                  f"{state[0]}, {-1.0 * state[1]}, {state[2]}, " \
                  f"{state[3]}, {-1.0 * state[4]}, {state[5]}, " \
                  f"{state[3]}, {-1.0 * state[4]}, {state[5]}\n"
            f.write(out)


def eval_model_trajectories(files, agent, horizon=32):
    states_list, actions_list, next_states_list = load_data(files)
    states = states_list[0]
    actions = actions_list[0]
    next_states = next_states_list[0]

    num_states = len(states)
    num_traj = num_states // horizon

    actual_trajs = []
    model_trajs = []

    abs_errors = torch.zeros((num_traj, 6))

    for traj in range(num_traj):
        model_states = []
        traj_states = states[traj * horizon:(traj + 1) * horizon]
        traj_actions = actions[traj * horizon:(traj + 1) * horizon]
        state = torch.tensor(traj_states[0], dtype=torch.float).reshape((1, -1))
        for i in range(horizon):
            action = traj_actions[i].reshape((1, -1))
            with torch.no_grad():
                observation, next_state, _ = step_model(
                    dynamical_model=agent.dynamical_model,
                    reward_model=agent.reward_model,
                    termination_model=None,
                    state=state,
                    action=torch.tensor(action, dtype=state.dtype, device=state.device),
                    action_scale=agent.policy.action_scale,
                    done=None,
                    pi=None,
                )
            model_states.append(state.reshape(-1))
            state = next_state

        actual_trajs.append(traj_states)
        model_trajs.append(model_states)
        abs_errors[traj] = torch.abs(model_states[-1] - traj_states[-1])

    print(
        f"Prediction horizon:     {horizon}\n"
        f"Mean x error:           {abs_errors[:, 0].mean():.5f}\n"
        f"Mean y error:           {abs_errors[:, 1].mean():.5f}\n"
        f"Mean theta error:       {abs_errors[:, 2].mean():.5f}\n"
        f"Mean vel_x error:       {abs_errors[:, 3].mean():.5f}\n"
        f"Mean vel_y error:       {abs_errors[:, 4].mean():.5f}\n"
        f"Mean vel_theta error:   {abs_errors[:, 5].mean():.5f}\n"
    )
