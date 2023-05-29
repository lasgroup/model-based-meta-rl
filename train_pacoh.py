import argparse
import os.path

import ray
import torch
import numpy as np
from rllib.dataset.datatypes import Observation

from lib.datasets.transforms.local_coordinates import LocalCoordinates
from utils.utils import get_project_path
from experiments.rccar_experiments.rccar_utils import load_data
from lib.meta_rl.algorithms.pacoh.bnn.bnn_svgd import BayesianNeuralNetworkSVGD
from lib.meta_rl.algorithms.pacoh.pacoh_nn_regression import PACOH_NN_Regression

meta_data_dir = os.path.join(
    ".", "meta_training_data"
)

task_names = [task for task in os.listdir(meta_data_dir) if "gm" in task]


def load_meta_data():
    meta_data = []
    for task in task_names:
        task_dir = os.path.join(meta_data_dir, task)
        files = [file for file in sorted(os.listdir(task_dir)) if "sampled" in file]
        states_list, actions_list, next_states_list = load_data(files, dir=task_dir)
        processed_data = process_data(states_list, actions_list, next_states_list)
        meta_data.append(processed_data)

    return meta_data


def apply_transforms(state, action, next_state):
    none = torch.tensor(0)
    transformations = [LocalCoordinates()]
    obs = Observation(
        torch.tensor(state),
        torch.tensor(action),
        none,
        torch.tensor(next_state),
        none,
        none,
        none,
        none,
        none,
        none,
    )
    for transformation in transformations:
        obs = transformation(obs)
    return (
        obs.state.detach().numpy(),
        obs.action.detach().numpy(),
        obs.next_state.detach().numpy(),
    )


def process_data(states_list, actions_list, next_states_list):
    
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    next_states = np.concatenate(next_states_list, axis=0)
    states, actions, next_states = apply_transforms(states, actions, next_states)
    states_actions = np.concatenate([states, actions], axis=-1)
    return states_actions, next_states


def eval_pacoh(evaluator):
    pacoh_eval_mean_per_task_ids = [
        eval_pacoh_per_task.remote(evaluator, task_num)
        for task_num in range(evaluator.n_tasks)
    ]
    pacoh_eval_mean_per_task = ray.get(pacoh_eval_mean_per_task_ids)
    (
        metrics_mean,
        metrics_std,
    ) = PACOH_NN_Regression._aggregate_eval_metrics_across_tasks(
        pacoh_eval_mean_per_task
    )
    return metrics_mean, metrics_std


@ray.remote
def eval_pacoh_per_task(evaluator, task_num):
    context_data, test_data = evaluator.setup_test_data(task_num)
    pacoh_model = evaluator.setup_pacoh_model(task_num)
    pacoh_task_eval, _ = pacoh_model.meta_eval_datasets([context_data + test_data])
    return pacoh_task_eval


class EvalNetwork:
    def __init__(
        self,
        meta_data,
        n_samples_context,
    ):
        self.meta_data = meta_data
        self.n_tasks = len(meta_data)
        self.n_samples_context = n_samples_context
        # self.pacoh_models = self.get_meta_trained_pacoh_models()
        self.pacoh_models = []

    def setup_test_data(self, task_num):
        task_data = self.meta_data[task_num]
        context_ids = np.random.choice(
            len(task_data[0]), size=self.n_samples_context, replace=False
        )
        test_ids = np.full((len(task_data[0]),), True, dtype=np.bool_)
        test_ids[context_ids] = False
        context_data = [task_data[0][context_ids], task_data[1][context_ids]]
        test_data = [task_data[0][test_ids], task_data[1][test_ids]]
        return context_data, test_data

    def get_meta_trained_pacoh_models(self):
        pacoh_models = []
        for task in range(self.n_tasks):
            pacoh_models.append(self.setup_pacoh_model(task))
        return pacoh_models

    def setup_pacoh_model(self, task_id):
        meta_train_data = [
            self.meta_data[i] for i in range(self.n_tasks) if i != task_id
        ]
        pacoh_model = PACOH_NN_Regression(
            meta_train_data,
            hidden_layer_sizes=(200, 200, 200, 200),
            num_iter_meta_train=60000,
            num_iter_meta_test=10 * self.n_samples_context,
            batch_size=min(8, self.n_samples_context),
            lr=1e-3,
            bandwidth=1.0,
        )
        pacoh_model.meta_fit(None)
        return pacoh_model

    def setup_bnn(self, context_data):
        bnn = BayesianNeuralNetworkSVGD(
            context_data[0],
            context_data[1],
            hidden_layer_sizes=(200, 200, 200, 200),
            n_particles=5,
            batch_size=min(8, self.n_samples_context),
            bandwidth=1.0,
        )
        return bnn

    def eval_bnn(self):
        bnn_eval_mean_per_task = []
        for task_num in range(self.n_tasks):
            context_data, test_data = self.setup_test_data(task_num)
            bnn = self.setup_bnn(context_data)
            bnn.fit(None, None, num_iter_fit=10 * self.n_samples_context)
            bnn_task_eval = bnn.eval(test_data[0], test_data[1])
            bnn_eval_mean_per_task.append(bnn_task_eval)
        (
            metrics_mean,
            metrics_std,
        ) = PACOH_NN_Regression._aggregate_eval_metrics_across_tasks(
            bnn_eval_mean_per_task
        )
        return metrics_mean, metrics_std

    def eval_pacoh(self):
        pacoh_eval_mean_per_task = []
        for task_num in range(self.n_tasks):
            context_data, test_data = self.setup_test_data(task_num)
            pacoh_model = self.pacoh_models[task_num]
            pacoh_task_eval, _ = pacoh_model.meta_eval_datasets(
                [context_data + test_data]
            )
            pacoh_eval_mean_per_task.append(pacoh_task_eval)
        (
            metrics_mean,
            metrics_std,
        ) = PACOH_NN_Regression._aggregate_eval_metrics_across_tasks(
            pacoh_eval_mean_per_task
        )
        return metrics_mean, metrics_std

    def eval_bnn_multitask(self):
        bnn_eval_mean_per_task = []
        for task_num in range(self.n_tasks):
            context_data, test_data = self.setup_test_data(task_num)
            random_task = np.random.randint(self.n_tasks)
            train_data = self.meta_data[random_task]
            bnn = self.setup_bnn(train_data)
            bnn.fit(None, None, num_iter_fit=20)
            bnn_task_eval = bnn.eval(test_data[0], test_data[1])
            bnn_eval_mean_per_task.append(bnn_task_eval)
        (
            metrics_mean,
            metrics_std,
        ) = PACOH_NN_Regression._aggregate_eval_metrics_across_tasks(
            bnn_eval_mean_per_task
        )
        return metrics_mean, metrics_std


if __name__ == "__main__":
    from torch.nn import Linear
    from utils.utils import torch_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples-context", type=int, default=10)
    parser.add_argument("--num-cpus", type=int, default=2)

    params = parser.parse_args()

    meta_data = load_meta_data()

    torch.set_num_threads(4)

    x, y = meta_data[0]
    n_samples_context = 5000
    x_train, y_train = x[:n_samples_context], y[:n_samples_context]
    x_test, y_test = x[n_samples_context:], y[n_samples_context:]
    normalization_stats = {
        "x_mean": np.zeros_like(x_train[0], dtype=np.float32),
        "x_std": np.ones_like(x_train[0], dtype=np.float32),
        "y_mean": np.zeros_like(y_train[0], dtype=np.float32),
        "y_std": np.ones_like(y_train[0], dtype=np.float32),
    }
    model = BayesianNeuralNetworkSVGD(
        x_train,
        y_train,
        hidden_layer_sizes=[200] * 4,
        n_particles=5,
        batch_size=min(8, n_samples_context),
        bandwidth=1.0,
        normalization_stats=normalization_stats,
    )
    model.fit(None, None, num_iter_fit=5000, log_period=500)

    y_raw, pred_dist = model.predict(x_test)
    var_per_dim = np.mean(y_test**2, axis=0) ** 0.5
    rmse_per_dim = np.mean((pred_dist.mean.cpu().numpy() - y_test) ** 2, axis=0) ** 0.5
    print(rmse_per_dim)
    for name, rmse, var in zip(
        ["x", "y", "theta", "vel_x", "vel_y", "vel_theta"], rmse_per_dim, var_per_dim
    ):
        print(f'{name}:{" "*(12 - len(name))} {rmse:.2f} | {var:.2f}')
