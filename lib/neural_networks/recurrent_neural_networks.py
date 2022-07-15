"""Implementation of Recurrent Neural Networks"""

import torch

from torch import nn
from functools import reduce
from rllib.util.neural_networks.utilities import parse_layers, inverse_softplus, update_parameters


class RecurrentNN(nn.Module):
    """Recurrent Neural Network Implementation.

    Parameters
    ----------
    in_dim: Tuple[int]
        input dimension of neural network.
    out_dim: Tuple[int]
        output dimension of neural network.
    embedding_layers: list of int, optional
        list of width of embedding layers before the RNN, each separated with a non-linearity.
    layers: int, optional
        number of recurrent layers before the output
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(
        self,
        in_dim,
        out_dim,
        embedding_layers=(),
        layers=(),
        non_linearity="Tanh",
        biased_head=True,
        squashed_output=False,
        initial_scale=0.5,
        log_scale=False,
        min_scale=1e-6,
        max_scale=1,
    ):
        super().__init__()
        self.kwargs = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "embedding_layers": embedding_layers,
            "layers": layers,
            "non_linearity": non_linearity,
            "biased_head": biased_head,
            "squashed_output": squashed_output,
            "initial_scale": initial_scale,
            "log_scale": log_scale,
            "min_scale": min_scale,
            "max_scale": max_scale,
        }

        self.embedding_layers, in_dim = parse_layers(embedding_layers, in_dim, non_linearity)
        self.recurrent_layers, in_dim = self.parse_recurrent_layers(in_dim, len(layers), layers[0])
        self.embedding_dim = in_dim + 1 if biased_head else in_dim
        self.output_shape = out_dim
        self.head = nn.Linear(
            in_dim, reduce(lambda x, y: x * y, list(out_dim)), bias=biased_head
        )
        self.hidden_state = torch.zeros(len(layers), 1, layers[0])
        self.squashed_output = squashed_output
        self.log_scale = log_scale
        if self.log_scale:
            self._init_scale_transformed = torch.log(torch.tensor([initial_scale]))
            self._min_scale = torch.log(torch.tensor(min_scale)).item()
            self._max_scale = torch.log(torch.tensor(max_scale)).item()
        else:
            self._init_scale_transformed = inverse_softplus(
                torch.tensor([initial_scale])
            )
            self._min_scale = min_scale
            self._max_scale = max_scale

    def parse_recurrent_layers(self, input_size, num_layers, hidden_size):
        recurrent_layers = nn.GRU(
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            batch_first=True
        )
        return recurrent_layers, hidden_size

    @classmethod
    def from_other(cls, other, copy=True):
        """Initialize Feedforward NN from other NN Network."""
        out = cls(**other.kwargs)
        if copy:
            update_parameters(target_module=out, new_module=other)
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
        if x.ndim == 1:
            x, self.hidden_state = self.recurrent_layers(self.embedding_layers(x.view(1, 1, -1)), self.hidden_state)
            x = x.squeeze()
        else:
            x, _ = self.recurrent_layers(self.embedding_layers(x))
        out = self.head(x)
        if self.squashed_output:
            return torch.tanh(out)
        return out.reshape(*out.shape[:-1], *self.output_shape)

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
        out, self.hidden_state = self.recurrent_layers(self.embedding_layers(x), self.hidden_state)
        out = self.head(out)
        if self.head.bias is not None:
            out = torch.cat((out, torch.ones(out.shape[:-1] + (1,))), dim=-1)

        return out

    def reset(self):
        self.hidden_state = torch.zeros_like(self.hidden_state)


class HeteroGaussianRNN(RecurrentNN):
    """A Module that parametrizes a diagonal heteroscedastic Normal distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale = nn.Linear(
            in_features=self.head.in_features,
            out_features=self.kwargs["out_dim"][0],
            bias=self.kwargs["biased_head"],
        )

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        if x.ndim == 1:
            x, self.hidden_state = self.recurrent_layers(self.embedding_layers(x.view(1, 1, -1)), self.hidden_state)
            x = x.squeeze()
        else:
            x, _ = self.recurrent_layers(self.embedding_layers(x))
        mean = self.head(x)
        if self.squashed_output:
            mean = torch.tanh(mean)

        if self.log_scale:
            log_scale = self._scale(x).clamp(self._min_scale, self._max_scale)
            scale = torch.exp(log_scale + self._init_scale_transformed)
        else:
            scale = nn.functional.softplus(
                self._scale(x) + self._init_scale_transformed
            ).clamp(self._min_scale, self._max_scale)
        return mean, torch.diag_embed(scale)


class CategoricalRNN(RecurrentNN):
    """A Module that parametrizes a Categorical distribution."""
    pass


class DeterministicRNN(RecurrentNN):
    """Declaration of a Deterministic Recurrent Neural Network."""
    pass
