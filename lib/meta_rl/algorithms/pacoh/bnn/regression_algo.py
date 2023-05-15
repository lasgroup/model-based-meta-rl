import torch

from tqdm import trange
from lib.meta_rl.algorithms.pacoh.modules.affine_transform import AffineTransform


class RegressionModel:
    likelihood = None
    batch_size = None
    nn_param_size = None
    likelihood_param_size = None
    likelihood_std = None

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        train_dataloader = self._get_dataloader(self.x_train, self.y_train, self.batch_size)
        train_batch_sampler = iter(train_dataloader)
        loss_list = []
        pbar = trange(num_iter_fit)
        for i in pbar:
            try:
                batch = next(train_batch_sampler)
            except StopIteration:
                train_batch_sampler = iter(train_dataloader)
                batch = next(train_batch_sampler)
            x_batch, y_batch = batch[..., :self.input_dim], batch[..., -self.output_dim:]
            loss = self.step(x_batch, y_batch)
            loss_list.append(loss)

            if i % log_period == 0:
                loss = torch.stack(loss_list, dim=0).mean().item()
                loss_list = []
                message = dict(loss=loss)
                if x_val is not None and y_val is not None:
                    metric_dict = self.eval(x_val, y_val)
                    message.update(metric_dict)
                pbar.set_postfix(message)

    def eval(self, x, y):
        x, y = self._handle_input_data(x, y, convert_to_tensor=True)
        _, pred_dist = self.predict(x)
        return self.likelihood.calculate_eval_metrics(pred_dist, y)

    def plot_predictions(self, x_plot, show=False, plot_train_data=True, ax=None):
        from matplotlib import pyplot as plt
        assert self.input_dim == 1 and self.output_dim == 1
        y_pred, pred_dist = self.predict(x_plot)
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # plot predictive mean and confidence interval
        ax.plot(x_plot, pred_dist.mean)
        lcb, ucb = pred_dist.mean - 2 * pred_dist.stddev, pred_dist.mean + 2 * pred_dist.stddev
        ax.fill_between(x_plot, lcb.numpy().flatten(), ucb.numpy().flatten(), alpha=0.2)

        for i in range(y_pred.shape[0]):
            ax.plot(x_plot, y_pred[i], color='green', alpha=0.2, linewidth=1.0)

        if plot_train_data:
            # unnormalize training data & plot it
            x_train = self.x_train * self.x_std + self.x_mean
            y_train = self.y_train * self.y_std + self.y_mean
            ax.scatter(x_train, y_train)

        if show:
            plt.show()

    def _process_train_data(self, x_train, y_train, normalization_stats=None):
        self.x_train, self.y_train = self._handle_input_data(x_train, y_train, convert_to_tensor=True)
        self.input_dim, self.output_dim = self.x_train.shape[-1], self.y_train.shape[-1]
        self.num_train_samples = self.x_train.shape[0]
        self._compute_normalization_stats(self.x_train, self.y_train, normalization_stats)
        self.x_train, self.y_train = self._normalize_data(self.x_train, self.y_train)

    def _compute_normalization_stats(self, x_train, y_train, normalization_stats=None):
        if normalization_stats is None:
            if x_train.shape[0] > 1:
                self.x_mean = x_train.mean(dim=0)
                self.x_std = x_train.std(dim=0)
                self.y_mean = y_train.mean(dim=0)
                self.y_std = y_train.std(dim=0)
            else:
                self.x_mean = torch.zeros(x_train.shape[-1])
                self.x_std = torch.ones(x_train.shape[-1])
                self.y_mean = torch.zeros(y_train.shape[-1])
                self.y_std = torch.ones(y_train.shape[-1])
        else:
            self.x_mean = normalization_stats['x_mean']
            self.x_std = normalization_stats['x_std']
            self.y_mean = normalization_stats['y_mean']
            self.y_std = normalization_stats['y_std']
        self.affine_pred_dist_transform = AffineTransform(
            normalization_mean=self.y_mean,
            normalization_std=self.y_std
        )

    def _set_normalization_stats(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.affine_pred_dist_transform = AffineTransform(
            normalization_mean=self.y_mean,
            normalization_std=self.y_std
        )

    def _get_dataloader(self, x, y, batch_size):
        x, y = self._handle_input_data(x, y, convert_to_tensor=True)
        num_train_points = x.shape[0]

        if batch_size == -1:
            batch_size = num_train_points
        elif batch_size > 0:
            pass
        else:
            raise AssertionError('batch size must be either positive or -1')

        train_dataset = torch.cat((x, y), dim=-1)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
        )
        return train_dataloader

    def _normalize_data(self, x, y=None):
        x = (x - self.x_mean) / self.x_std
        if y is None:
            return x
        else:
            y = (y - self.y_mean) / self.y_std
            return x, y

    def _unnormalize_preds(self, y):
        return y * self.y_std + self.y_mean

    def _unnormalize_predictive_dist(self, pred_dist):
        return self.affine_pred_dist_transform.apply(pred_dist)

    @staticmethod
    def _handle_input_data(x, y=None, convert_to_tensor=True, dtype=torch.float32):
        if x.ndim == 1:
            x = x.unsqueeze(dim=-1)

        assert x.ndim == 2

        if y is not None:
            if y.ndim == 1:
                y = y.unsqueeze(dim=-1)
            assert x.shape[0] == y.shape[0]
            assert y.ndim == 2

            if convert_to_tensor:
                x, y = torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)
            return x, y
        else:
            if convert_to_tensor:
                x = torch.tensor(x, dtype=dtype)
            return x

    def _split_into_nn_params_and_likelihood_std(self, params):
        assert params.ndim == 2
        assert params.shape[-1] == self.nn_param_size + self.likelihood_param_size
        n_particles = params.shape[0]
        nn_params = params[:, :self.nn_param_size]
        if self.likelihood_param_size > 0:
            likelihood_std = params[:, -self.likelihood_param_size:].exp()
        else:
            likelihood_std = torch.ones((n_particles, self.output_dim)) * self.likelihood_std

        assert likelihood_std.shape == (n_particles, self.output_dim)
        return nn_params, likelihood_std

    def predict(self, x):
        raise NotImplementedError

    def step(self, x_batch, y_batch):
        raise NotImplementedError
