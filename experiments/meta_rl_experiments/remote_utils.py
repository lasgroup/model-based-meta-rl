import ray

from rllib.util.rollout import rollout_episode


@ray.remote
def rollout_agent(environment, agent, max_env_steps, num_episodes=1, print_frequency=1, render=False, use_early_termination=True):
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
        if print_frequency and episode % print_frequency == 0:
            print(agent)
    return agent


@ray.remote
def add_dataset(agent, dataset):
    agent.dataset.add_dataset(dataset)
    return agent


@ray.remote
def train_agent(agent):
    agent.learn_model()
    return agent
