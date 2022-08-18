import math
import torch


class GaussianLikelihood(torch.nn.Module):

    def __init__(self, output_dim, n_batched_models, std=0.2, trainable=True, name='gaussian_likelihood'):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.n_batched_models = n_batched_models
        self.log_std = torch.ones((n_batched_models, output_dim)) * math.log(std)
        if trainable:
            self.log_std.requires_grad = True

    @property
    def std(self):
        return self.log_std.exp()

    def log_prob(self, y_pred, y_true, std=None):
        if std is None:
            std = self.std
        assert len(y_pred.shape) == 3
        assert y_pred.shape[2] == self.output_dim
        batch_size = y_pred.shape[1]

        likelihood_std = torch.stack([std] * batch_size, dim=1)
        assert likelihood_std.shape == y_pred.shape
        likelihood = torch.distributions.Independent(
            torch.distributions.Normal(y_pred, likelihood_std),
            reinterpreted_batch_ndims=1
        )
        log_likelihood = likelihood.log_prob(y_true)
        avg_log_likelihood = log_likelihood.mean(dim=-1)  # average over batch
        return avg_log_likelihood

    def calculate_eval_metrics(self, pred_dist, y_true):
        eval_results = {
            'avg_ll': pred_dist.log_prob(y_true).mean().numpy(),
            'avg_rmse': self.rmse(pred_dist.mean, y_true).numpy(),
        }

        if self.output_dim == 1:
            eval_results['cal_err'] = self.calib_error(pred_dist, y_true).numpy()

        return eval_results

    @staticmethod
    def calib_error(pred_dist, y_true, use_circular_region=False):
        if y_true.ndim == 3:
            y_true = y_true[0]

        if use_circular_region or y_true.shape[-1] > 1:
            cdf_vals = pred_dist.cdf(y_true, circular=True)
        else:
            cdf_vals = pred_dist.cdf(y_true)
        cdf_vals = cdf_vals.reshape((-1, 1))

        num_points = torch.tensor(cdf_vals.numel()).type(torch.float32)
        conf_levels = torch.linspace(0.05, 0.95, 20)
        emp_freq_per_conf_level = (cdf_vals <= conf_levels).type(torch.float32).sum(dim=0) / num_points

        calib_err = (emp_freq_per_conf_level - conf_levels).abs().mean()
        return calib_err

    def get_pred_mixture_dist(self, y_pred, std=None):
        if std is None:
            std = self.std

        # check shapes
        assert len(y_pred.shape) == 3
        assert y_pred.shape[0] == std.shape[0]
        assert y_pred.shape[-1] == std.shape[-1]

        num_mixture_components = y_pred.shape[0]

        y_pred_transposed = torch.permute(y_pred, (1, 2, 0))
        std = std.t().unsqueeze(0).expand_as(y_pred_transposed)

        components = torch.distributions.Normal(loc=y_pred_transposed, scale=std)
        categorical = torch.distributions.Categorical(logits=torch.ones(y_pred_transposed.shape))
        return torch.distributions.MixtureSameFamily(categorical, components)

    def rmse(self, y_pred_mean, y_true):
        """
        Args:
            y_pred_mean (tf.Tensor): mean prediction
            y_true (tf.Tensor): true target variable

        Returns: (tf.Tensor) Root mean squared error (RMSE)

        """
        assert y_pred_mean.shape == y_true.shape
        return (y_pred_mean - y_true).square().mean().sqrt()


""" helper function """
def _split(array, n_splits):
    """
        splits array into n_splits of potentially unequal sizes
    """
    assert array.ndim == 1
    n_elements = array.shape[0]

    remainder = n_elements % n_splits
    split_sizes = []
    for i in range(n_splits):
        if i < remainder:
            split_sizes.append(n_elements // n_splits + 1)
        else:
            split_sizes.append(n_elements // n_splits)
    return torch.split(array, split_sizes)

