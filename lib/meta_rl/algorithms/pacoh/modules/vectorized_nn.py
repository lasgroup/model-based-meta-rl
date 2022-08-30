import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def parameter_shapes(self):
        raise NotImplementedError

    def named_parameters(self):
        raise NotImplementedError

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        if len(name.split('.')) == 1:
            setattr(self, name, value)
        else:
            remaining_name = ".".join(name.split('.')[1:])
            getattr(self, name.split('.')[0]).set_parameter(remaining_name, value)

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def set_parameters_as_vector(self, value):
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            if value.ndim == 1:
                self.set_parameter(name, value[idx:idx_next])
            elif value.ndim == 2:
                self.set_parameter(name, value[:, idx:idx_next])
            else:
                raise AssertionError
            idx = idx_next
        assert idx_next == value.shape[-1]


class LinearVectorized(VectorizedModel):
    def __init__(self, input_dim, output_dim, n_batched_models=1, nonlinearity='tanh'):
        super().__init__(input_dim, output_dim)

        self.weight = torch.normal(0, 1, size=(n_batched_models, input_dim * output_dim,), requires_grad=True)
        self.bias = torch.zeros((n_batched_models, output_dim), requires_grad=True)

        self.reset_parameters(nonlinearity)

    def reset_parameters(self, non_linearity, zero_bias=False):
        self.weight = _kaiming_uniform_batched(self.weight, fan_in=self.input_dim, fan_out=self.output_dim, a=math.sqrt(5), nonlinearity=non_linearity)
        if self.bias is not None and not zero_bias:
            fan_in = self.output_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.weight.ndim == 2 or self.weight.ndim == 3:
            model_batch_size = self.weight.shape[0]
            # batched computation
            if self.weight.ndim == 3:
                assert self.weight.shape[-2] == 1 and self.bias.shape[-2] == 1

            W = self.weight.view(model_batch_size, self.output_dim, self.input_dim)
            b = self.bias.view(model_batch_size, self.output_dim)

            if x.ndim == 2:
                # introduce new dimension 0
                x = x.reshape((1, x.shape[0], x.shape[1]))
                # tile dimension 0 to model_batch size
                x = x.repeat(model_batch_size, 1, 1)
            else:
                assert x.ndim == 3 and x.shape[0] == model_batch_size
            # out dimensions correspond to [nn_batch_size, data_batch_size, out_features)
            return torch.bmm(x, W.permute(0, 2, 1)) + b[:, None, :]
        elif self.weight.ndim == 1:
            return F.linear(x, self.weight.view(self.output_dim, self.input_dim), self.bias)
        else:
            raise NotImplementedError

    def parameter_shapes(self):
        return OrderedDict(bias=self.bias.shape, weight=self.weight.shape)

    def named_parameters(self):
        return OrderedDict(bias=self.bias, weight=self.weight)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NeuralNetworkVectorized(VectorizedModel):
    """Trainable neural network that batches multiple sets of parameters. That is, each
    """

    def __init__(self, input_dim, output_dim, layer_sizes=(64, 64), activation="tanh", n_batched_models=1):
        super().__init__(input_dim, output_dim)

        activation = activation.lower()
        if activation == "relu":
            nonlinearlity = torch.relu
        elif activation == "tanh":
            nonlinearlity = torch.tanh
        else:
            raise NotImplementedError

        self.nonlinearlity = nonlinearlity
        self.n_layers = len(layer_sizes)
        self.layers_sizes = layer_sizes
        self.n_batched_models = n_batched_models

        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, 'fc_%i' % (i + 1), LinearVectorized(prev_size, size, n_batched_models, activation))
            prev_size = size
        setattr(self, 'out', LinearVectorized(prev_size, output_dim, n_batched_models, activation))

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers + 1):
            output = getattr(self, 'fc_%i' % i)(output)
            output = self.nonlinearlity(output)
        output = getattr(self, 'out')(output)
        return output

    def parameter_shapes(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).parameter_shapes().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).parameter_shapes().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).named_parameters().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).named_parameters().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def get_forward_fn(self, params):
        reg_model = copy.deepcopy(self)
        reg_model.set_parameters_as_vector(params)
        return reg_model

    @property
    def num_parameters(self):
        return self.parameters_as_vector().shape[-1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _kaiming_uniform_batched(tensor, fan_in, fan_out, a=0.0, nonlinearity='tanh', initalization='glorot'):
    """ Initialization Helper """
    if initalization == 'glorot':
        std = 1.0 / math.sqrt(fan_in + fan_out)
        bound = math.sqrt(6.0) * std  # Calculate uniform bounds from standard deviation
    elif initalization == 'kaimin':
        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    else:
        raise NotImplementedError
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
