import os
import pathlib


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


def get_project_path():
    return pathlib.Path(__file__).absolute().parent.parent.resolve()


def get_dataset_path(params):
    return os.path.join(
        f"experiments/meta_rl_experiments/experience_replay_{params.num_train_env_instances}tasks_{str(params.action_cost).replace('.', '')}acost",
        f"{params.env_config_file.replace('-', '_').replace('.yaml', '')}_{params.exploration}_train_{params.train_episodes}_{params.multiple_runs_id}.pkl"
    )

