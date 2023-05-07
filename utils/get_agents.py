from typing import Tuple

import torch

from torch import optim
from torch.nn.modules import loss

from rllib.algorithms.ppo import PPO
from rllib.algorithms.sac import SAC
from rllib.agent import PPOAgent, SACAgent
from rllib.dataset import ExperienceReplay
from rllib.environment.abstract_environment import AbstractEnvironment

from lib.algorithms.sb3_sac import SB3_SAC
import lib.meta_rl.agents.parallel_pacoh_agent
import lib.meta_rl.agents.parallel_grbal_agent
from lib.agents import MPCAgent, MPCPolicyAgent, MBPOAgent, BPTTAgent
from lib.meta_rl.agents import RLSquaredAgent, GrBALAgent, PACOHAgent
from lib.environments.wrappers.model_based_environment import ModelBasedEnvironment
from lib.environments.wrappers.rccar_model_based_environment import RCCarModelBasedEnvironment

from utils.get_learners import *
from utils.utils import get_dataset_path


def get_mpc_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[MPCAgent, str]:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    # Define value function.
    # TODO: Use as terminal reward  and train value function in ModelBasedAgent
    value_function = get_value_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )
    # TODO: Use as terminal reward  and train value function in ModelBasedAgent
    terminal_reward = value_function if params.mpc_terminal_reward else None

    # Define policy
    policy = get_mpc_policy(
        dynamical_model=dynamical_model,
        reward=reward_model,
        params=params,
        action_scale=environment.action_scale,
        terminal_reward=terminal_reward,
        termination_model=termination_model
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    agent = MPCAgent(
        policy,
        model_optimizer=model_optimizer,
        exploration_scheme=params.exploration,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        model_learn_num_iter=params.model_learn_num_iter,
        max_memory=params.max_memory,
        gamma=params.gamma,
        comment=comment,
    )

    return agent, comment


def get_mpc_policy_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        termination_model: Callable = None,
        input_transform: nn.Module = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[MPCPolicyAgent, str]:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param termination_model: Early termination check
    :param input_transform: Input transformation
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    # Define policy
    policy = get_nn_policy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    # Define actor-critic algorithm
    if params.mpc_policy == "ppo":
        # Define value function.
        critic = get_value_function(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            params=params,
            input_transform=input_transform
        )

        algorithm = PPO(
            policy=policy,
            critic=critic,
            criterion=loss.MSELoss(reduction="mean"),
            gamma=0.99,
            epsilon=0.2,
            clamp_value=False
        )
    elif params.mpc_policy == "sac":
        # Define q function.
        critic = get_q_function(
            dim_state=dim_state,
            dim_action=dim_action,
            num_states=num_states,
            num_actions=num_actions,
            params=params,
            input_transform=input_transform
        )

        algorithm = SAC(
            policy=policy,
            critic=critic,
            criterion=loss.MSELoss(reduction="mean"),
            gamma=0.99
        )
    else:
        raise NotImplementedError

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(critic.parameters()),
        lr=params.ppo_opt_lr,
        weight_decay=params.ppo_opt_weight_decay,
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    agent = MPCPolicyAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        algorithm=algorithm,
        model_optimizer=model_optimizer,
        policy_optimizer=policy_optimizer,
        exploration_scheme=params.exploration,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        model_learn_num_iter=params.model_learn_num_iter,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        max_memory=params.max_memory,
        gamma=params.gamma,
        comment=comment,
    )

    return agent, comment


def get_bptt_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        termination_model: Callable = None,
        input_transform: nn.Module = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[BPTTAgent, str]:
    """
    Get an MPC based agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param termination_model: Early termination check
    :param input_transform: Input transformation
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # true_model = get_env_model(params.env_model_name, environment)

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    # Define policy
    policy = get_nn_policy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    critic = get_value_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(critic.parameters()),
        lr=params.ppo_opt_lr,
        weight_decay=params.ppo_opt_weight_decay,
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    agent = BPTTAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        policy=policy,
        critic=critic,
        true_model=None,
        model_optimizer=model_optimizer,
        policy_optimizer=policy_optimizer,
        exploration_scheme=params.exploration,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        model_learn_num_iter=params.model_learn_num_iter,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        max_memory=params.max_memory,
        gamma=params.gamma,
        td_lambda=0.95,
        comment=comment,
    )

    return agent, comment


def get_mbpo_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        termination_model: Callable = None,
        input_transform: nn.Module = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[MBPOAgent, str]:
    """
    Get an MBPO agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param termination_model: Early termination check
    :param input_transform: Input transformation
    :param initial_distribution: Distribution for initial exploration
    :return: An MPC based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    model_based_env = ModelBasedEnvironment(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        action_scale=environment.action_scale,
        num_envs=params.sim_n_envs,
        sim_num_steps=params.sim_num_steps,
        initial_states_distribution=None,
    )

    if "rccar" in params.env_config_file:
        model_based_env = RCCarModelBasedEnvironment(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            action_scale=environment.action_scale,
            num_envs=params.sim_n_envs,
            sim_num_steps=params.sim_num_steps,
            initial_states_distribution=None,
        )

    policy = SB3_SAC(
        learned_env=model_based_env,
        params=params
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    agent = MBPOAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        model_based_env=model_based_env,
        policy=policy,
        model_optimizer=model_optimizer,
        exploration_scheme=params.exploration,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        model_learn_num_iter=params.model_learn_num_iter,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        max_memory=params.max_memory,
        gamma=params.gamma,
        comment=comment,
    )

    return agent, comment


def get_ppo_agent(
        environment: AbstractEnvironment,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> Tuple[PPOAgent, str]:
    """
    Get an Proximal Policy Optimization (PPO) agent
    :param environment: RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: An PPO based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define value function.
    value_function = get_value_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define policy
    policy = get_nn_policy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(value_function.parameters()),
        lr=params.ppo_opt_lr,
        weight_decay=params.ppo_opt_weight_decay,
    )

    # Define Agent
    comment = f"{params.agent_name} {params.exploration.capitalize()}"

    agent = PPOAgent(
        policy=policy,
        critic=value_function,
        optimizer=policy_optimizer,
        num_iter=128,
        batch_size=32,
        epsilon=0.2,
        eta=params.ppo_eta,  # Controls agent exploration, higher value leads to more exploration
        clip_gradient_val=params.ppo_clip_gradient_val,
        gamma=0.99,
        comment=comment
    )

    return agent, comment


def get_sac_agent(
        environment: AbstractEnvironment,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> Tuple[SACAgent, str]:
    """
    Get a Soft Actor-Critic (SAC) agent
    :param environment: RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: A SAC agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    # Define q function.
    q_function = get_q_function(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define policy
    policy = get_nn_policy(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(q_function.parameters()),
        lr=params.sac_opt_lr,
        weight_decay=params.sac_opt_weight_decay,
    )

    memory = ExperienceReplay(max_len=params.sac_memory_len)

    # Define Agent
    comment = f"{params.agent_name} {params.exploration.capitalize()}"

    agent = SACAgent(
        policy=policy,
        critic=q_function,
        optimizer=policy_optimizer,
        memory=memory,
        num_iter=128,
        gamma=0.99,
        comment=comment
    )

    return agent, comment


def get_rl2_agent(
        environment: AbstractEnvironment,
        params: argparse.Namespace,
        input_transform: nn.Module = None
) -> Tuple[PPOAgent, str]:
    """
    Get an Reinforcement Learning Squared (RL^2) agent
    :param environment: Meta RL environment
    :param params: Agent arguments
    :param input_transform: Input transformation
    :return: An RL^2 based agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    params.exploration = "greedy"

    # Define value function.
    value_function = get_recurrent_value_function(
        dim_state=(dim_state[0] + dim_action[0] + 2,),
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        params=params,
        input_transform=input_transform
    )

    # Define policy
    policy = get_rnn_policy(
        dim_state=(dim_state[0] + dim_action[0] + 2,),
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        input_transform=input_transform,
        action_scale=environment.action_scale,
        params=params,
    )

    # Define model optimizer
    policy_optimizer = optim.Adam(
        list(policy.parameters()) + list(value_function.parameters()),
        lr=params.ppo_opt_lr,
        weight_decay=params.ppo_opt_weight_decay,
    )

    # Define Agent
    comment = f"{params.agent_name} {params.exploration.capitalize()}"

    agent = RLSquaredAgent(
        policy=policy,
        critic=value_function,
        optimizer=policy_optimizer,
        trial_len=params.rl2_trial_len,
        num_iter=64,
        batch_size=32,
        epsilon=0.2,
        eta=params.ppo_eta,  # Controls agent exploration, higher value leads to more exploration
        clip_gradient_val=params.ppo_clip_gradient_val,
        gamma=0.99,
        comment=comment
    )

    return agent, comment


def get_grbal_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[GrBALAgent, str]:
    """
    Get a Gradient-based Adaptive Learner agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: An GrBAL agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    params.exploration = "greedy"

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    model_based_env = ModelBasedEnvironment(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        action_scale=environment.action_scale,
        num_envs=params.sim_n_envs,
        sim_num_steps=params.sim_num_steps,
        initial_states_distribution=None,
    )

    policy = SB3_SAC(
        learned_env=model_based_env,
        params=params
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    if not params.collect_meta_data:
        trajectory_load_path = get_dataset_path(params)
        params.train_episodes = 0
    else:
        trajectory_load_path = None

    agent = GrBALAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        model_based_env=model_based_env,
        policy=policy,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        gamma=params.gamma,
        model_optimizer=model_optimizer,
        inner_lr=params.grbal_inner_lr,
        past_segment_len=params.grbal_past_segment_len,
        exploration_scheme=params.exploration,
        future_segment_len=params.grbal_future_segment_len,
        model_learn_num_iter=params.model_learn_num_iter,
        model_learn_batch_size=params.model_learn_batch_size,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        multiple_runs_id=params.multiple_runs_id,
        trajectory_replay_load_path=trajectory_load_path,
        num_iter_meta_train=params.grbal_num_iter_meta_train,
        comment=comment,
    )

    return agent, comment


def get_parallel_grbal_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[GrBALAgent, str]:
    """
    Get a Parallel Gradient-based Adaptive Learner agent
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: A parallel GrBAL agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    params.exploration = "greedy"

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    model_based_env = ModelBasedEnvironment(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        action_scale=environment.action_scale,
        num_envs=params.sim_n_envs,
        sim_num_steps=params.sim_num_steps,
        initial_states_distribution=None,
    )

    policy = SB3_SAC(
        learned_env=model_based_env,
        params=params
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    if not params.collect_meta_data:
        trajectory_load_path = get_dataset_path(params)
        params.train_episodes = 0
    else:
        trajectory_load_path = None

    agent = lib.meta_rl.agents.parallel_grbal_agent.ParallelGrBALAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        model_based_env=model_based_env,
        policy=policy,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        gamma=params.gamma,
        model_optimizer=model_optimizer,
        inner_lr=params.grbal_inner_lr,
        past_segment_len=params.grbal_past_segment_len,
        future_segment_len=params.grbal_future_segment_len,
        exploration_scheme=params.exploration,
        model_learn_batch_size=params.model_learn_batch_size,
        model_learn_num_iter=params.model_learn_num_iter,
        use_validation_set=params.use_validation_set,
        initial_distribution=initial_distribution,
        num_parallel_agents=params.grbal_num_parallel_agents,
        num_episodes_per_rollout=params.num_episodes_per_rollout,
        max_env_steps=params.max_steps,
        multiple_runs_id=params.multiple_runs_id,
        trajectory_replay_load_path=trajectory_load_path,
        num_iter_meta_train=params.grbal_num_iter_meta_train,
        comment=comment,
    )

    return agent, comment


def get_pacoh_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[PACOHAgent, str]:
    """
    Get a meta-RL agent based on PACOH
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: A PACOH agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    params.model_kind = "BayesianNN"
    if params.pacoh_optimistic_evaluation != (params.exploration == "optimistic"):
        raise AssertionError(
            "Only Parallel PACOH Agent supports different exploration modes for training and evaluation."
        )

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    model_based_env = ModelBasedEnvironment(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        action_scale=environment.action_scale,
        num_envs=params.sim_n_envs,
        sim_num_steps=params.sim_num_steps,
        initial_states_distribution=None,
    )

    policy = SB3_SAC(
        learned_env=model_based_env,
        params=params
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    if not params.collect_meta_data:
        trajectory_load_path = get_dataset_path(params)
        params.train_episodes = 0
    else:
        trajectory_load_path = None

    agent = PACOHAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        model_based_env=model_based_env,
        policy=policy,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        gamma=params.gamma,
        model_optimizer=model_optimizer,
        initial_distribution=initial_distribution,
        exploration_scheme=params.exploration,
        model_learn_num_iter=params.model_learn_num_iter,
        model_learn_batch_size=params.model_learn_batch_size,
        num_iter_meta_train=params.pacoh_num_iter_meta_train,
        num_iter_eval_train=params.pacoh_num_iter_eval_train,
        n_samples_per_prior=params.pacoh_n_samples_per_prior,
        num_hyper_posterior_particles=params.pacoh_num_hyper_posterior_particles,
        num_posterior_particles=params.pacoh_num_posterior_particles,
        env_name=params.env_config_file.replace('-', '_').replace('.yaml', ''),
        trajectory_replay_load_path=trajectory_load_path,
        multiple_runs_id=params.multiple_runs_id,
        comment=comment,
    )

    return agent, comment


def get_parallel_pacoh_agent(
        environment: AbstractEnvironment,
        reward_model: AbstractModel,
        transformations: Iterable[AbstractTransform],
        params: argparse.Namespace,
        input_transform: nn.Module = None,
        termination_model: Union[Callable, None] = None,
        initial_distribution: torch.distributions.Distribution = None
) -> Tuple[PACOHAgent, str]:
    """
    Get a meta-RL agent based on PACOH
    :param environment: RL environment
    :param reward_model: Reward model
    :param transformations: State and action transformations
    :param params: Agent arguments
    :param input_transform: Input transformation
    :param termination_model: Early termination check
    :param initial_distribution: Distribution for initial exploration
    :return: A PACOH agent
    """
    dim_state = environment.dim_state
    dim_action = environment.dim_action
    num_states = environment.num_states
    num_actions = environment.num_actions

    params.model_kind = "BayesianNN"
    params.exploration = "optimistic" if params.pacoh_optimistic_evaluation else "greedy"

    # Define dynamics model
    dynamical_model = get_model(
        dim_state=dim_state,
        dim_action=dim_action,
        num_states=num_states,
        num_actions=num_actions,
        transformations=transformations,
        input_transform=input_transform,
        params=params
    )

    # Define model optimizer
    try:
        model_optimizer = optim.Adam(
            dynamical_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )
    except ValueError:
        model_optimizer = optim.Adam(
            dynamical_model.base_model.parameters(),
            lr=params.model_opt_lr,
            weight_decay=params.model_opt_weight_decay,
        )

    model_based_env = ModelBasedEnvironment(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        action_scale=environment.action_scale,
        num_envs=params.sim_n_envs,
        sim_num_steps=params.sim_num_steps,
        initial_states_distribution=None,
    )

    policy = SB3_SAC(
        learned_env=model_based_env,
        params=params
    )

    # Define Agent
    model_name = dynamical_model.base_model.name
    comment = f"{model_name} {params.exploration.capitalize()}"

    if not params.collect_meta_data:
        trajectory_load_path = get_dataset_path(params)
        params.train_episodes = 0
    else:
        trajectory_load_path = None

    agent = lib.meta_rl.agents.parallel_pacoh_agent.ParallelPACOHAgent(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        termination_model=termination_model,
        model_based_env=model_based_env,
        policy=policy,
        policy_opt_gradient_steps=params.policy_opt_gradient_steps,
        sim_num_steps=params.sim_num_steps,
        gamma=params.gamma,
        model_optimizer=model_optimizer,
        initial_distribution=initial_distribution,
        exploration_scheme=params.exploration,
        num_iter_meta_train=params.pacoh_num_iter_meta_train,
        num_iter_eval_train=params.pacoh_num_iter_eval_train,
        n_samples_per_prior=params.pacoh_n_samples_per_prior,
        num_hyper_posterior_particles=params.pacoh_num_hyper_posterior_particles,
        num_posterior_particles=params.pacoh_num_posterior_particles,
        env_name=params.env_config_file.replace('-', '_').replace('.yaml', ''),
        trajectory_replay_load_path=trajectory_load_path,
        model_learn_num_iter=params.model_learn_num_iter,
        model_learn_batch_size=params.model_learn_batch_size,
        parallel_episodes_per_env=params.parallel_episodes_per_env,
        num_episodes_per_rollout=params.num_episodes_per_rollout,
        max_env_steps=params.max_steps,
        multiple_runs_id=params.multiple_runs_id,
        comment=comment,
    )

    return agent, comment
