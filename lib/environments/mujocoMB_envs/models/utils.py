from lib.environments.mujocoMB_envs.models import *
from lib.environments.wrappers.gym_environment import GymEnvironment


def get_env_model(model_name: str, environment: GymEnvironment):
    if not isinstance(model_name, str):
        raise NotImplementedError
    model = eval(model_name)(environment.env)
    return model
