import math

import numpy as np
import torch

SEED = 245
SHUFFLING_RANDOM_SEED = 123
TASK_SHUFFLING_RANDOM_SEED = 456


class MetaDatasetSampler:

    def __init__(self, data, per_task_batch_size, meta_batch_size=-1, n_batched_models=None, tiled=False,
                 flatten_data=False,  standardization=True, random_seed=SEED):
        """
        Encapsulates the meta-data with its statistics (mean and std) and provides batching functionality

        Args:
            data: list of tuples of ndarrays. Either [(train_x_1, train_t_1), ..., (train_x_n, train_t_n)] or
                    [(context_x, context_y, test_x, test_y), ( ... ), ...]
            per_task_batch_size (int): number of samples in each batch sampled from a task
            meta_batch_size: number of tasks sampled in each meta-training step
            n_batched_models: number of batched models for tiling. Is ignored when tiled = False
            tiled (bool): weather the returned batches should be tiled/stacked so that the first dimension
                        corresponds to the number of batched models
            standardization (bool): whether to compute standardization statistics. If False, the data statistics are
                                    set to zero mean and unit std --> standardization has no effect
        """
        self.n_tasks = len(data)

        torch.manual_seed(random_seed)

        if meta_batch_size > self.n_tasks:
            print(f"NOTE: The requested meta batch size `{meta_batch_size}` is bigger the number "
                  f"of training tasks `{self.n_tasks}`. Reverting to using all of the tasks in each batch")
            meta_batch_size = -1

        if meta_batch_size == -1:
            meta_batch_size = self.n_tasks

        self.meta_batch_size = meta_batch_size
        self.tasks = []

        for task_data in data:
            if len(task_data) == 2:
                self.tasks.append(DatasetSampler(train_data=task_data, val_data=None,
                                                 batch_size=per_task_batch_size, n_batched_models=n_batched_models,
                                                 tiled=tiled, flatten_data=flatten_data))
            elif len(task_data) == 4:
                context_x, context_y, test_x, test_y = task_data
                train_data, val_data = (context_x, context_y), (test_x, test_y)
                self.tasks.append(DatasetSampler(train_data=train_data, val_data=val_data,
                                                 batch_size=per_task_batch_size, n_batched_models=n_batched_models,
                                                 tiled=tiled, flatten_data=flatten_data))
            else:
                raise Exception("Unexpected data shape")

        if per_task_batch_size == -1:
            # set to number of samples per task
            n_samples_per_task = [task_tuple[0].shape[0] for task_tuple in data]
            assert len(set(n_samples_per_task)) == 1, "n_samples differ across tasks --> per_task_batch_size must be set > 0"
            per_task_batch_size = n_samples_per_task[0]
        self.per_task_batch_size = per_task_batch_size

        self.input_dim = self.tasks[0].input_dim
        self.output_dim = self.tasks[0].output_dim
        self.n_train_samples = [task.n_train_samples for task in self.tasks]

        # Standardization of inputs and outputs
        self.standardization = standardization
        if standardization:
            self.x_mean, self.y_mean, self.x_std, self.y_std = self._compute_global_standardization_stats()
        else:
            self.x_mean, self.y_mean, self.x_std, self.y_std = self._get_zero_mean_unit_std_stats()

        self._update_task_standardization_stats()

        # Task batching
        tasks_ids = torch.arange(self.n_tasks, dtype=torch.int32)

        self.task_id_loader = torch.utils.data.DataLoader(
            tasks_ids,
            shuffle=True,
            batch_size=1,
        )
        self.task_id_sampler = iter(self.task_id_loader)

        self.steps_per_epoch = math.ceil(self.n_tasks / self.meta_batch_size) * self.tasks[0].steps_per_epoch

    def get_meta_batch(self, shuffle=True):
        meta_batch_x, meta_batch_y = [], []
        n_train_samples, batch_size, task_ids = [], [], []

        for task_id in range(self.meta_batch_size):
            if shuffle:
                try:
                    task_id = next(self.task_id_sampler)[0]
                except StopIteration:
                    self.task_id_sampler = iter(self.task_id_loader)
                    task_id = next(self.task_id_sampler)[0]

            task = self.tasks[task_id]

            # Get data sample
            x, y = task.get_batch()
            meta_batch_x.append(x)
            meta_batch_y.append(y)

            # Add additional data
            n_train_samples.append(task.n_train_samples)
            batch_size.append(task.batch_size)
            task_ids.append(task_id)

        meta_batch_x = torch.stack(meta_batch_x, dim=0)
        meta_batch_y = torch.stack(meta_batch_y, dim=0)

        n_train_samples = torch.tensor(n_train_samples, dtype=torch.float32)
        batch_size = torch.tensor(batch_size, dtype=torch.float32)
        task_ids = torch.tensor(task_ids, dtype=torch.float32)

        return meta_batch_x, meta_batch_y, n_train_samples, batch_size, task_ids

    def copy_standardization_stats(self, obj):
        """
        Copies the standardization stats of an object to self
        """
        assert all([hasattr(obj, stats_var) for stats_var in ['x_mean', 'y_mean', 'x_std', 'y_std']])
        self.x_mean = obj.x_mean
        self.y_mean = obj.y_mean
        self.x_std = obj.x_std
        self.y_std = obj.y_std

    def plot_data(self, tasks_to_include, is_training, plot_val_data=True, ax=None):
        import matplotlib.pyplot as plt

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if tasks_to_include is None:
            tasks_to_include = range(self.n_tasks)

        for i in tasks_to_include:
            task = self.tasks[i]
            if is_training:
                task.plot_data(True, False, ax, f'Task {i+1}-', context=False)
            else:
                task.plot_data(True, plot_val_data, ax, f'Task {i+1}-')

        if new_axis:
            plt.legend()
            plt.show()

    def plot_prediction_functions(self, model, ax=None, plot_pred_std=False, plot_pred_lines=False,
                                  sample_functions=True, sample_from_prior=False, plot_data=False, title=None):
        assert(plot_pred_std or plot_pred_lines)
        import matplotlib.pyplot as plt
        x_min = torch.tensor([task.x_min for task in self.tasks]).min()
        x_max = torch.tensor([task.x_max for task in self.tasks]).max()

        flat_x = torch.linspace(x_min, x_max, 100)
        x = flat_x.reshape((100, self.output_dim))
        y_pred, pred_dist = model._predict(x, sample_functions=sample_functions, sample_from_prior=sample_from_prior)

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if plot_pred_std:
            std = pred_dist.stddev.reshape((1, -1))[0]
            mean = pred_dist.mean.reshape((1, -1))[0]
            top = mean + std
            bottom = mean - std

            ax.fill_between(flat_x, top, bottom, alpha=0.2)

        if plot_pred_lines:
            for i in range(y_pred.shape[0]):
                ax.plot(flat_x, y_pred[i].reshape((1, -1))[0], color='green', alpha=0.3, linewidth=1)

        if plot_data:
            self.plot_data(None, True, ax=ax)

        if new_axis:
            if title is None:
                ax.set_title("Prior functions sample" if sample_from_prior else "Posterior functions sample")
            else:
                ax.set_title(title)
            plt.show()

    def set_original_shape(self, original_shape):
        self.original_shape = original_shape

        # call setter in all tasks / dataset samplers
        for task in self.tasks:
            task.set_original_shape(original_shape)

    def _compute_global_standardization_stats(self):
        X_stack = torch.cat([task.train_data[0] for task in self.tasks], dim=0)
        Y_stack = torch.cat([task.train_data[1] for task in self.tasks], dim=0)

        x_mean = X_stack.mean(dim=0, keepdim=True)
        y_mean = Y_stack.mean(dim=0, keepdim=True)
        x_std = X_stack.std(dim=0, keepdim=True) + 1e-8
        y_std = Y_stack.std(dim=0, keepdim=True) + 1e-8

        assert x_mean.ndim == 2
        assert y_mean.ndim == 2
        assert x_std.ndim == 2
        assert y_std.ndim == 2

        return x_mean, y_mean, x_std, y_std

    def _get_zero_mean_unit_std_stats(self):
        x_mean = torch.zeros((1, self.input_dim), dtype=torch.float32)
        y_mean = torch.zeros((1, self.output_dim), dtype=torch.float32)
        x_std = torch.ones((1, self.input_dim), dtype=torch.float32)
        y_std = torch.ones((1, self.output_dim), dtype=torch.float32)
        return x_mean, y_mean, x_std, y_std

    def _update_task_standardization_stats(self):
        for task in self.tasks:
            task.copy_standardization_stats(self)

    def get_standardization_stats(self):
        return self.x_mean, self.y_mean, self.x_std, self.y_std


class DatasetSampler:
    def __init__(self, train_data, val_data, batch_size, n_batched_models=None, tiled=False, flatten_data=False,
                 x_mean=None, x_std=None, y_mean=None, y_std=None):

        self.train_data = self._handle_input_dimensionality(train_data[0], train_data[1])
        self._set_data_ranges_for_plotting()

        self.tiled = tiled
        self.flatten_data = flatten_data
        self.n_batched_models = n_batched_models
        self.n_train_samples = self.train_data[0].shape[0]
        assert not tiled or n_batched_models is not None

        if batch_size > self.n_train_samples:
            print(f"NOTE: The requested batch size `{batch_size}` is bigger"
                  f"than the number of training samples `{self.n_train_samples}`")

        if batch_size == -1:
            batch_size = self.n_train_samples

        self.batch_size = batch_size
        self.steps_per_epoch = max(1, math.ceil(self.n_train_samples / self.batch_size))

        self.input_dim = self.train_data[0].shape[1]
        self.output_dim = self.train_data[1].shape[1]

        self._set_standardization_values(x_mean, x_std, y_mean, y_std)

        train_dataset = torch.utils.data.DataLoader(
            torch.cat((self.train_data[0], self.train_data[1]), dim=-1),
            shuffle=True,
            batch_size=batch_size,
            drop_last=True
        )
        self.train_data_loader = train_dataset
        self.train_data_sampler = iter(self.train_data_loader)

        if val_data is not None:
            self.val_data = self._handle_input_dimensionality(val_data[0], val_data[1])
            x_min_val, x_max_val = torch.floor(self.val_data[0].min()), torch.ceil(self.val_data[0].max())

            x_min_val = x_min_val.type(torch.float32)
            x_max_val = x_max_val.type(torch.float32)

            self.x_min = min([x_min_val, self.x_min])
            self.x_max = max([x_max_val, self.x_max])

            self.n_val_samples = self.val_data[0].shape[0]

            val_dataset = torch.utils.data.DataLoader(
                torch.cat((self.val_data[0], self.val_data[1]), dim=-1),
                shuffle=True,
                batch_size=batch_size,
                drop_last=True
            )
            self.val_data_loader = val_dataset
            self.val_data_sampler = iter(self.val_data_loader)

    def get_batch(self):
        try:
            batch = next(self.train_data_sampler)
        except StopIteration:
            self.train_data_sampler = iter(self.train_data_loader)
            batch = next(self.train_data_sampler)
        x, y = batch[..., :self.input_dim], batch[..., -self.output_dim:]
        x, y = self.process_batch(x, y)
        return x, y

    def process_batch(self, x, y):
        """
        Standardizes, reshapes and tiles both x and y if needed
        Args:
            x (tf.Tensor): input batch
            y (tf.Tensor): target batch

        Returns: processed (x, y)

        """
        x, y = self._standardize(x, y)
        x = self._reshape(x)

        if self.tiled:
            x, y = self._tile_batch(x, y)

        return x, y

    def process_eval_batch(self, x, y):
        """
        Standardizes, reshapes only x and tiles both x and y if needed
        Args:
            x (tf.Tensor): input batch
            y (tf.Tensor): target batch

        Returns: (x, y)
        """
        x, _ = self._standardize(x, None)
        x = self._reshape(x)

        if self.tiled:
            x, _ = self._tile_batch(x, y)

        return x, y

    def _tile_batch(self, x, y=None):
        tile_multiplies_x = [1, self.n_batched_models] + list(torch.ones_like(x.shape))
        x = x.unsqueeze(dim=0)
        x = torch.tile(x, tile_multiplies_x)

        if y is not None:
            tile_multiplies_y = [1, self.n_batched_models] + list(torch.ones_like(y.shape))
            y = y.unsqueeze(dim=0)
            y = torch.tile(y, tile_multiplies_y)

        return x, y

    def _reshape(self, x):
        """
        brings the data back into the original shape
        """
        if getattr(self, 'original_shape', None) is not None and not self.flatten_data:
            new_shape = list(x.shape[:-1]) + self.original_shape
            return x.reshape(new_shape)
        return x

    def copy_standardization_stats(self, obj):
        """
        Copies the standardization stats of an object to self
        """
        assert all([hasattr(obj, stats_var) for stats_var in ['x_mean', 'y_mean', 'x_std', 'y_std']])
        self.x_mean = obj.x_mean
        self.y_mean = obj.y_mean
        self.x_std = obj.x_std
        self.y_std = obj.y_std

    def plot_data(self, train, val, ax=None, label_prefix="", context=True):
        import matplotlib.pyplot as plt
        assert (train or val)

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if val:
            x, y = self.val_data
            ax.scatter(x, y, label=f'{label_prefix}test', alpha=0.1)

        if train:
            x, y = self.train_data
            ax.scatter(x, y, color='black', label=f'{label_prefix}context' if context else f'{label_prefix}train',
                       marker='x' if context else 'o', alpha=1 if context else 0.5)

        if new_axis:
            plt.legend()
            plt.show()

    def plot_prediction_functions(self, model, ax=None, plot_pred_std=False, plot_pred_lines=False,
                                  sample_functions=True, sample_from_prior=False, plot_data=True, title=None):
        assert(plot_pred_std or plot_pred_lines)
        import matplotlib.pyplot as plt
        flat_x = torch.linspace(self.x_min, self.x_max, self.n_val_samples)
        x = flat_x.reshape(self.val_data[0].shape)
        y_pred, pred_dist = model._predict(x, sample_functions=sample_functions, sample_from_prior=sample_from_prior)

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if plot_pred_std:
            std = 2 * pred_dist.stddev.reshape((1, -1))[0]
            mean = pred_dist.mean.reshape((1, -1))[0]
            top = mean + std
            bottom = mean - std

            ax.fill_between(flat_x, top, bottom, alpha=0.2)

        if plot_pred_lines:
            for i in range(y_pred.shape[0]):
                ax.plot(flat_x, y_pred[i].reshape((1, -1))[0], color='green', alpha=0.3, linewidth=1)

        if plot_data:
            self.plot_data(True, True, ax=ax)

        if new_axis:
            if title is None:
                ax.set_title("Prior functions sample" if sample_from_prior else "Posterior functions sample")
            else:
                ax.set_title(title)
            plt.show()

    @staticmethod
    def _handle_input_dimensionality(x, y):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)
        if y.ndim == 1:
            y = y.unsqueeze(dim=-1)

        assert x.shape[0] == y.shape[0]
        assert x.ndim == 2
        assert y.ndim == 2

        return x, y

    def _set_standardization_values(self, x_mean, x_std, y_mean, y_std):
        if x_mean is None:
            x_mean = torch.zeros(self.input_dim, dtype=torch.float32)
            y_mean = torch.zeros(self.output_dim, dtype=torch.float32)
            x_std = torch.ones(self.input_dim, dtype=torch.float32)
            y_std = torch.ones(self.output_dim, dtype=torch.float32)

        if  x_mean.ndim == 1:
            x_mean = x_mean.unsqueeze(dim=0)
            y_mean = y_mean.unsqueeze(dim=0)
            x_std = x_std.unsqueeze(dim=0)
            y_std = y_std.unsqueeze(dim=0)

        assert x_mean.ndim == 2
        assert y_mean.ndim == 2
        assert x_std.ndim == 2
        assert y_std.ndim == 2

        self.x_mean, self.y_mean = x_mean, y_mean
        self.x_std, self.y_std = x_std, y_std

    def _set_data_ranges_for_plotting(self):
        x_min, x_max = torch.floor(self.train_data[0].min()), torch.ceil(self.train_data[0].max())
        self.x_min = x_min.type(dtype=torch.float32)
        self.x_max = x_max.type(dtype=torch.float32)

    def set_original_shape(self, original_shape):
        self.original_shape = original_shape

    def _standardize(self, x, y=None):
        x = (x - self.x_mean) / self.x_std

        if y is not None:
            y = (y - self.y_mean) / self.y_std

        return x, y


def _split_into_batches(array, max_batch_size):
    import math
    n_elements = len(array)
    if max_batch_size == -1:
        max_batch_size = n_elements
    n_batches = math.ceil(n_elements / float(max_batch_size))
    remainder = n_elements % n_batches
    batches = []
    idx = 0
    for i in range(n_batches):
        if i < remainder:
            batch_size = n_elements // n_batches + 1
        else:
            batch_size = n_elements // n_batches
        batches.append(array[idx:idx + batch_size])
        idx += batch_size

    assert all([len(batch) <= max_batch_size for batch in batches])
    assert np.sum([len(batch) for batch in batches]) == n_elements
    return batches
