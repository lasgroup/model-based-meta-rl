import torch
import numpy as np

PHASE_RANGE = [-np.pi/2, np.pi/2]
AMP_RANGE = [0.1, 5.0]
INPUT_RANGE = [-5.0, 5.0]


def _func(x: torch.Tensor, phase=None, amplitude=None):
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if phase is None:
        phase = torch.zeros((x.shape[0]))
    if amplitude is None:
        amplitude = torch.ones((x.shape[0]))
    return amplitude.unsqueeze(-1) * torch.sin(x + phase.unsqueeze(-1))


class DemoMetaLearningDataset:

    def __init__(self, num_train_tasks=500, num_test_tasks=20, num_samples=10000,
                 num_context_samples=10, observation_noise_std=0.1):

        self.num_train_tasks = num_train_tasks
        self.num_test_tasks = num_test_tasks

        self.num_samples = num_samples
        self.num_context_samples = num_context_samples

        self.observation_noise_std = observation_noise_std

        self.train_task_params = torch.rand(num_train_tasks, 2)
        self.test_task_params = torch.rand(num_test_tasks, 2)

        self.train_data = self.generate_data(self.num_train_tasks, self.num_samples, self.train_task_params)
        self.test_data = self.generate_data(self.num_test_tasks, self.num_samples, self.test_task_params)

    def generate_data(self, tasks, num_samples, params):
        amplitude_samples = (AMP_RANGE[1] - AMP_RANGE[0]) * params[:, 0] + AMP_RANGE[0]
        phase_samples = (PHASE_RANGE[1] - PHASE_RANGE[0]) * params[:, 1] + PHASE_RANGE[0]
        input_samples = (INPUT_RANGE[1] - INPUT_RANGE[0]) * torch.rand(tasks, num_samples) + INPUT_RANGE[0]
        output = _func(input_samples, phase_samples, amplitude_samples)
        output = output + torch.randn(output.shape) * self.observation_noise_std
        return torch.stack((input_samples, output), dim=-1)

    def get_samples_from_task(self, batch_size, task_id):
        samples = torch.multinomial(torch.ones(self.num_samples), batch_size)
        return self.train_data[task_id, samples, :-1], self.train_data[task_id, samples, -1:]

    def get_samples(self, num_tasks, batch_size):
        task_samples = torch.multinomial(torch.ones(self.num_train_tasks), num_tasks)
        samples_x, samples_y = [], []
        for task in task_samples:
            x, y = self.get_samples_from_task(batch_size, task)
            samples_x.append(x)
            samples_y.append(y)
        return torch.stack(samples_x, dim=0), torch.stack(samples_y, dim=0)

    def get_test_data_from_task(self, task_id):
        task_data = self.test_data[task_id].unsqueeze(0)
        return task_data[:, :self.num_context_samples, :-1], task_data[:, :self.num_context_samples, -1:], \
               task_data[:, self.num_context_samples:, :-1], task_data[:, self.num_context_samples:, -1:]
