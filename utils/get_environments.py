from lib.environments.wrappers.gym_environment import GymEnvironment
from lib.hucrl.hallucinated_environment import HallucinatedEnvironmentWrapper
from lib.environments.wrappers.meta_environment import MetaEnvironmentWrapper
from lib.environments.wrappers.random_gym_environment import RandomGymEnvironment


def get_environment(params):
    if params.env_group == "mujocoMB_envs":
        environment = GymEnvironment(
            env_name=params.name,
            params=params,
            ctrl_cost_weight=params.action_cost if params.use_action_cost else 0.0
        )
        reward_model = environment.env.reward_model()
        if params.use_exact_termination_model:
            termination_model = environment.env._termination_model().copy()
        else:
            termination_model = environment.env.termination_model()

    elif params.env_group == "gym_envs":
        import lib.environments.gym_envs
        environment = GymEnvironment(
            env_name=params.name,
            params=params
        )
        reward_model = eval(f"lib.environments.gym_envs.{params.reward_model}")()
        termination_model = None

    elif params.env_group == "point_envs":
        from lib.environments.point_envs import PointEnv2D, RandomPointEnv2D
        if 'random' in params.env_config_file:
            environment = RandomPointEnv2D()
        else:
            environment = PointEnv2D()
        reward_model = environment.reward_model()
        termination_model = None

    elif params.env_group == "random_mujocoMB_envs":
        environment = RandomGymEnvironment(
            env_name=params.name,
            params=params,
            ctrl_cost_weight=params.action_cost if params.use_action_cost else 0.0
        )
        reward_model = environment.env.reward_model()
        if params.use_exact_termination_model:
            termination_model = environment.env._termination_model().copy()
        else:
            termination_model = environment.env.termination_model()

    else:
        raise NotImplementedError

    return environment, reward_model, termination_model


def get_wrapped_env(params, task=None):

    environment, reward_model, termination_model = get_environment(params)

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)
    if task is not None:
        environment.set_task(task)

    return environment, reward_model, termination_model


def get_wrapped_meta_env(params, meta_training_tasks=None, meta_test_tasks=None):

    environment, reward_model, termination_model = get_environment(params)

    if params.exploration == "optimistic":
        environment = HallucinatedEnvironmentWrapper(environment)
    environment = MetaEnvironmentWrapper(environment, params)

    if meta_training_tasks is not None:
        environment.train_env_params = meta_training_tasks
    if meta_test_tasks is not None:
        environment.test_env_params = meta_test_tasks

    return environment, reward_model, termination_model
