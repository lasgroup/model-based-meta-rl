"""Implementation of different Bayesian Neural Networks with pytorch."""
import copy
import math
from functools import reduce
from collections import OrderedDict

import torch

from rllib.util.utilities import safe_cholesky
from lib.meta_rl.algorithms.pacoh.modules.kernels import RBFKernel
from lib.meta_rl.algorithms.pacoh.modules.likelihood import GaussianLikelihood
from lib.meta_rl.algorithms.pacoh.modules.prior_posterior import GaussianPrior
from lib.meta_rl.algorithms.pacoh.modules.vectorized_nn import VectorizedModel, LinearVectorized


class FeedForwardBNN(VectorizedModel):
    """Trainable neural network that batches multiple sets of parameters. That is, each
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        layers=(),
        num_particles=5,
        prediction_strategy="moment_matching",
        include_aleatoric_uncertainty=False,
        deterministic=True,
        non_linearity="tanh",
        biased_head=True,
        min_scale=1e-6,
        max_scale=1,
        likelihood_std=0.1,
        learn_likelihood=True,
        prior_std=0.1,
        prior_weight=1e-4,
        likelihood_prior_mean=math.log(0.1),
        likelihood_prior_std=1.0,
        sqrt_mode=False,
        bandwidth=100.,
        meta_learned_prior=None,
        *args,
        **kwargs
    ):

        assert biased_head, "Unbiased head not implemented for Bayesian Neural Network"

        super().__init__(in_dim, out_dim)
        self.kwargs = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "layers": layers,
            "num_particles": num_particles,
            "prediction_strategy": prediction_strategy,
            "include_aleatoric_uncertainty": include_aleatoric_uncertainty,
            "non_linearity": non_linearity,
            "biased_head": biased_head,
            "min_scale": min_scale,
            "max_scale": max_scale,
        }

        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.sqrt_mode = sqrt_mode

        self.n_layers = len(layers)
        self.layers_sizes = layers
        self.num_particles = num_particles
        self.head_ptr = 0
        self.head_indexes = torch.zeros(1).long()
        self.deterministic = deterministic
        self.prediction_strategy = prediction_strategy
        self.include_aleatoric_uncertainty = include_aleatoric_uncertainty

        in_dim, self.non_linearity = self.parse_layers(layers, in_dim, non_linearity)
        self.embedding_dim = in_dim + 1 if biased_head else in_dim
        self.output_shape = out_dim[0]

        self.head = LinearVectorized(
            in_dim,
            reduce(lambda x, y: x * y, list(out_dim)),
            num_particles,
            self.non_linearity
        )

        self._min_scale = torch.log(torch.tensor(min_scale)).item()
        self._max_scale = torch.log(torch.tensor(max_scale)).item()

        # setup prior
        self.nn_param_size = self.num_parameters
        if learn_likelihood:
            self.likelihood_param_size = self.output_shape
        else:
            self.likelihood_param_size = 0

        if meta_learned_prior is None:
            self.prior = GaussianPrior(self.nn_param_size, nn_prior_std=prior_std,
                                       likelihood_param_size=self.likelihood_param_size,
                                       likelihood_prior_mean=likelihood_prior_mean,
                                       likelihood_prior_std=likelihood_prior_std)
            self.meta_learned_prior_mode = False
        else:
            self.prior = meta_learned_prior
            assert meta_learned_prior.get_parameters_stacked_per_prior().shape[-1] == \
                   2 * (self.nn_param_size + self.likelihood_param_size)
            assert num_particles % self.prior.n_batched_priors == 0, "num_particles must be multiple of n_batched_priors"
            self.meta_learned_prior_mode = True

        # Likelihood
        self.likelihood = GaussianLikelihood(self.output_shape, num_particles)

        # setup particles
        if self.meta_learned_prior_mode:
            # initialize posterior particles from meta-learned prior
            params = self.prior.sample(num_particles // self.prior.n_batched_priors).reshape((num_particles, -1))
            self.particles = torch.tensor(params, requires_grad=True)
        else:
            # initialize posterior particles from model initialization
            nn_params = self.parameters_as_vector()
            likelihood_params = torch.ones((self.num_particles, self.likelihood_param_size), requires_grad=True) * likelihood_prior_mean
            self.particles = torch.cat([nn_params.detach(), likelihood_params.detach()], dim=-1)
            self.particles.requires_grad = True

        # setup kernel and optimizer
        self.kernel = RBFKernel(bandwidth=bandwidth)

    def parse_layers(self, layers, in_dim, non_linearity):
        non_linearity = non_linearity.lower()
        if non_linearity == "relu":
            non_linearity = torch.relu
        elif non_linearity == "tanh":
            non_linearity = torch.tanh
        else:
            raise NotImplementedError

        prev_size = in_dim[0]
        for i, size in enumerate(layers):
            setattr(self, 'fc_%i' % (i + 1), LinearVectorized(prev_size, size, self.num_particles, non_linearity))
            prev_size = size

        return prev_size, non_linearity

    def forward_nn(self, x, nn_params=None):
        if nn_params is None:
            nn_params, _ = self._split_into_nn_params_and_likelihood_std(self.particles)

        self.set_parameters_as_vector(nn_params)

        out = x
        for i in range(1, self.n_layers + 1):
            out = getattr(self, 'fc_%i' % i)(out)
            out = self.non_linearity(out)
        out = self.head(out)

        return out

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Tensor of size [batch_size x out_dim].
        """

        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(self.particles)

        out = x
        if out.ndim > 2:
            out = out.reshape((-1, x.shape[-1]))
        out = self.forward_nn(out, nn_params)
        out = out.reshape([self.num_particles] + list(x.shape[:-1]) + [out.shape[-1]])
        out = torch.permute(out, [dim for dim in range(1, out.ndim)] + [0])

        if self.deterministic:
            scale = torch.zeros_like(out)
        else:
            scale = likelihood_std.clamp(self._min_scale, self._max_scale).transpose(0, 1)
            scale = torch.tile(scale.unsqueeze(0), (out.shape[0], 1, 1))

        if self.prediction_strategy == "moment_matching":
            mean = out.mean(-1)
            if self.include_aleatoric_uncertainty:
                variance = (scale.square() + out.square()).mean(-1) - mean.square()
                scale = safe_cholesky(torch.diag_embed(variance))
            else:
                variance = out.square().mean(-1) - mean.square()
                scale = safe_cholesky(torch.diag_embed(variance))
        elif self.prediction_strategy == "sample_head":  # TS-1
            head_ptr = torch.randint(self.num_particles, (1,))
            mean = out[..., head_ptr]
            scale = torch.diag_embed(scale[..., head_ptr])
        elif self.prediction_strategy in ["set_head", "posterior"]:  # Thompson sampling
            mean = out[..., self.head_ptr]
            scale = torch.diag_embed(scale[..., self.head_ptr])
        elif self.prediction_strategy == "sample_multiple_head":  # TS-1
            head_idx = torch.randint(self.num_particles, out.shape[:-1]).unsqueeze(-1)
            mean = out.gather(-1, head_idx).squeeze(-1)
            scale = torch.diag_embed(scale.gather(-1, head_idx).squeeze(-1))
        elif self.prediction_strategy == "set_head_idx":  # TS-INF
            mean = out.gather(-1, self.head_idx)
            scale = torch.diag_embed(scale.gather(-1, self.head_idx))
        elif self.prediction_strategy == "multi_head":
            mean = out.transpose(-1, -2)
            scale = torch.diag_embed(scale.transpose(-1, -2))
        else:
            raise NotImplementedError

        return mean, scale

    @torch.jit.export
    def last_layer_embeddings(self, x):
        """Get last layer embeddings of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Tensor of size [batch_size x embedding_dim].
        """
        output = x
        for i in range(1, self.n_layers + 1):
            output = getattr(self, 'fc_%i' % i)(output)
            output = self.non_linearity(output)

        if self.head.bias is not None:
            output = torch.cat((output, torch.ones(output.shape[:-1] + (1,))), dim=-1)

        return output

    @torch.jit.export
    def set_head(self, new_head: int):
        """Set the Ensemble head.

        Parameters
        ----------
        new_head: int
            If new_head == num_particles, then forward returns the average of all heads.
            If new_head < num_particles, then forward returns the output of `new_head' head.

        Raises
        ------
        ValueError: If new_head > num_particles.
        """
        self.head_ptr = new_head
        if not (0 <= self.head_ptr < self.num_particles):
            raise ValueError("head_ptr has to be between zero and num_particles - 1.")

    @torch.jit.export
    def get_head(self) -> int:
        """Get current head."""
        return self.head_ptr

    @torch.jit.export
    def set_head_idx(self, head_indexes):
        """Set ensemble head for particles.."""
        self.head_indexes = head_indexes

    @torch.jit.export
    def get_head_idx(self):
        """Get ensemble head index."""
        return self.head_indexes

    @torch.jit.export
    def set_prediction_strategy(self, prediction: str):
        """Set ensemble prediction strategy."""
        self.prediction_strategy = prediction

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get ensemble head."""
        return self.prediction_strategy

    def parameter_shapes(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).parameter_shapes().items():
                param_dict[layer_name + '.' + name] = param

        # head layer
        for name, param in self.head.parameter_shapes().items():
            param_dict['head' + '.' + name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).named_parameters().items():
                param_dict[layer_name + '.' + name] = param

        # head layer
        for name, param in self.head.named_parameters().items():
            param_dict['head' + '.' + name] = param

        return param_dict

    def get_forward_fn(self, params):
        reg_model = copy.deepcopy(self)
        reg_model.set_parameters_as_vector(params)
        return reg_model

    def get_avg_log_likelihood(self, y_pred, y_target):
        _, likelihood_std = self._split_into_nn_params_and_likelihood_std(self.particles)
        avg_log_likelihood = self.likelihood.log_prob(y_pred, y_target, likelihood_std)
        return avg_log_likelihood

    def get_prior_prob(self):
        if self.meta_learned_prior_mode:
            particles_reshaped = self.particles.reshape(
                (self.prior.n_batched_priors, self.num_particles // self.prior.n_batched_priors, -1)
            )
            prior_prob = self.prior.log_prob(
                particles_reshaped, model_params_prior_weight=self.prior_weight
            ).reshape((self.num_particles,))
        else:
            prior_prob = self.prior.log_prob(self.particles, model_params_prior_weight=self.prior_weight)
        return prior_prob

    def _split_into_nn_params_and_likelihood_std(self, params):
        assert params.ndim == 2
        assert params.shape[-1] == self.nn_param_size + self.likelihood_param_size
        num_particles = params.shape[0]
        nn_params = params[:, :self.nn_param_size]
        if self.likelihood_param_size > 0:
            likelihood_std = params[:, -self.likelihood_param_size:].exp()
        else:
            likelihood_std = torch.ones((num_particles, self.output_shape)) * self.likelihood_std

        assert likelihood_std.shape == (num_particles, self.output_shape)
        return nn_params, likelihood_std

    @property
    def num_parameters(self):
        return self.parameters_as_vector().shape[-1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
