import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt

from torch import nn
from dotmap import DotMap

from lib.meta_rl.algorithms.maml import MAML
from lib.meta_rl.meta_evaluation.demo_task import DemoMetaLearningDataset


def get_fc_nn(layers, in_dim=1, out_dim=1, activation="tanh"):
    activation_layer = nn.Tanh if activation == "tanh" else nn.ReLU
    modules = [nn.Linear(in_dim, layers[0]), activation_layer()]
    for i in range(1, len(layers)):
        modules.append(nn.Linear(layers[i-1], layers[i]))
        modules.append(activation_layer())
    modules.append(nn.Linear(layers[-1], out_dim))
    return nn.Sequential(*modules)


def plot_model(model, pattern, ax=None):
    x = torch.linspace(-5, 5, steps=200).unsqueeze(-1)
    y = model(x).clone().detach()
    return plot_data(x, y, pattern, ax)


def plot_data(x, y, pattern, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, pattern)
    ax.set(xlim=[-5, 5], ylim=[-5.5, 5.5])
    return ax


def log_dict(logs):
    for key, val in logs.items():
        if isinstance(val, list):
            print(f"{key} :  {val[-1]}\n")
        else:
            print(f"{key} :  {val}\n")
    print("\n----------------\n")


def train_and_evaluate(model, dataset, params):

    loss_func = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.module.parameters(), lr=params.outer_lr, weight_decay=params.weight_decay)
    ax = None

    for i in range(params.num_train_iter):
        task_samples_x, task_samples_y = dataset.get_samples(params.num_task_samples, params.batch_size)
        training_dict = {"Epoch": i, "pre_update_error": 0, "post_update_error": 0}
        optimizer.zero_grad()
        loss = torch.zeros(1)
        for task in range(params.num_task_samples):
            task_model = model.clone()
            error = loss_func(
                task_model(task_samples_x[task, :params.num_context_samples]),
                task_samples_y[task, :params.num_context_samples]
            )
            task_model.adapt(error)
            training_dict["pre_update_error"] += error.clone().detach() / params.num_task_samples
            loss += loss_func(
                task_model(task_samples_x[task, params.num_context_samples:]),
                task_samples_y[task, params.num_context_samples:]
            ) / params.num_task_samples

        training_dict["post_update_error"] = loss.clone().detach().item()
        loss.backward()
        optimizer.step()
        log_dict(training_dict)

    eval_dict = {"Task": [], "pre_update_error": [], "post_update_error": []}
    for gradient_steps in range(1, params.num_gradient_steps+1):
        for task in range(params.num_test_tasks):
            if gradient_steps == params.num_gradient_steps:
                ax = plot_model(model, 'g')
            optimizer.zero_grad()
            eval_dict["Task"].append(task)
            task_context_x, task_context_y, task_eval_x, task_eval_y = dataset.get_test_data_from_task(task)
            task_model = model.clone()
            if gradient_steps == params.num_gradient_steps:
                ax = plot_data(task_eval_x.squeeze(0), task_eval_y.squeeze(0), 'yo', ax=ax)
                ax = plot_data(task_context_x.squeeze(0), task_context_y.squeeze(0), 'rx', ax=ax)
            for g in range(gradient_steps):
                error = loss_func(
                    task_model(task_context_x),
                    task_context_y
                )
                task_model.adapt(error)
            eval_dict["pre_update_error"].append(error.clone().detach())
            eval_dict["post_update_error"].append(
                loss_func(
                    task_model(task_eval_x),
                    task_eval_y
                ).clone().detach()
            )
            # log_dict(eval_dict)
            if gradient_steps == params.num_gradient_steps:
                plot_model(task_model, 'b', ax)
                plt.show()

        print(f"Avg. pre-update error for {gradient_steps} gradient_steps: {np.mean(eval_dict['pre_update_error'])}")
        print(f"Avg. post-update error for {gradient_steps} gradient_steps: {np.mean(eval_dict['post_update_error'])}")


if __name__ == "__main__":

    with open("meta_evaluation_config.yaml", "r") as file:
        params = yaml.safe_load(file)
    params = DotMap(params)

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(1)

    model = get_fc_nn([40, 40], in_dim=1, out_dim=1, activation="relu")
    model = MAML(model, lr=params.inner_lr)

    dataset = DemoMetaLearningDataset(params)

    train_and_evaluate(model, dataset, params)
