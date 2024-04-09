from experiments.meta_rl_experiments import run_utils
from utils.train_and_evaluate import train_and_evaluate_agent


if __name__ == "__main__":

    params = run_utils.get_params()

    environment, agent = run_utils.get_environment_and_meta_agent(params)

    train_and_evaluate_agent(environment, agent, params)

