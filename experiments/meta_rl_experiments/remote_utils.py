import gc
import os
import psutil

import ray
import torch
import numpy as np

from rllib.util.rollout import rollout_episode


@ray.remote
def rollout_parallel_agent(environment, agent, max_env_steps, num_episodes=1, render=False, use_early_termination=True):
    torch.set_num_threads(2)
    if hasattr(agent.policy, "replay_buffer"):
        serialize_replay_buffer(agent.policy.replay_buffer)
    for episode in range(num_episodes):
        rollout_episode(
            environment=environment,
            agent=agent,
            max_steps=max_env_steps,
            render=render,
            callback_frequency=0,
            callbacks=None,
            use_early_termination=use_early_termination
        )
        print(f"-------\nCompleted episode {episode + 1} out of {num_episodes} for agent on pid={os.getpid()}\n-------\n")
    return agent


@ray.remote
def add_dataset(agent, dataset):
    if hasattr(agent.policy, "replay_buffer"):
        serialize_replay_buffer(agent.policy.replay_buffer)
    agent.dataset.add_dataset(dataset)
    return agent


@ray.remote
def train_agent(agent):
    if hasattr(agent.policy, "replay_buffer"):
        serialize_replay_buffer(agent.policy.replay_buffer)
    agent.learn()
    return agent


def serialize_replay_buffer(buffer):
    buffer.observations = buffer.observations.copy()
    buffer.next_observations = buffer.next_observations.copy()
    buffer.actions = buffer.actions.copy()
    buffer.rewards = buffer.rewards.copy()
    buffer.dones = buffer.dones.copy()
    buffer.timeouts = buffer.timeouts.copy()


def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


def log_agents(base_agent, parallel_agents, num_episodes):
    for episode in range(num_episodes):
        episode_dict = {}
        for key in parallel_agents[0].logger.statistics[episode].keys():
            ret_array = np.zeros((len(parallel_agents)))
            for i, agent in enumerate(parallel_agents):
                ret_array[i] = agent.logger.statistics[episode][key]
            episode_dict[key] = np.mean(ret_array, axis=0)
        base_agent.logger.end_episode(prefix="env_average", **episode_dict)


def update_counters(base_agent, agent):
    base_agent.counters["total_episodes"] += 1
    base_agent.counters["total_steps"] += agent.total_steps
    if base_agent.training:
        base_agent.counters["train_episodes"] += 1
        base_agent.counters["train_steps"] += agent.train_steps
    else:
        base_agent.counters["eval_episodes"] += 1
