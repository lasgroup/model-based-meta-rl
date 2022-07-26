import numpy as np
from rllib.dataset import Observation, stack_list_of_tuples


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
