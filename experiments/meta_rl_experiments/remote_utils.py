import gc
import psutil

import ray

from rllib.util.rollout import rollout_episode


@ray.remote
def rollout_parallel_agent(environment, agent, max_env_steps, num_episodes=1, render=False, use_early_termination=True):
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
    return agent


@ray.remote
def add_dataset(agent, dataset):
    agent.dataset.add_dataset(dataset)
    return agent


@ray.remote
def train_agent(agent):
    agent.learn()
    return agent


def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
