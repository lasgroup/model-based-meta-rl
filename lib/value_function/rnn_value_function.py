"""Value Functions parametrized with Recurrent Neural Networks."""

import torch

from rllib.util.neural_networks.utilities import one_hot_encode
from lib.neural_networks.recurrent_neural_networks import DeterministicRNN
from rllib.value_function.abstract_value_function import AbstractValueFunction


class RNNValueFunction(AbstractValueFunction):
    """Implementation of a Value Function implemented with a Recurrent Neural Network.

    Parameters
    ----------
    embedding_layers: list of int, optional
        list of width of embedding layers before the RNN, each separated with a non-linearity.
    layers: int, optional
        number of recurrent layers before the output
    biased_head: bool, optional (default=True).
        flag that indicates if head of NN has a bias term or not.
    non_linearity: string, optional (default=Tanh).
        Neural Network non-linearity.
    input_transform: nn.Module, optional (default=None).
        Module with which to transform inputs.

    """

    def __init__(
        self,
        layers=(200,),
        embedding_layers=(200, 200),
        biased_head=True,
        non_linearity="Tanh",
        input_transform=None,
        jit_compile=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.discrete_state:
            num_inputs = (self.num_states,)
        else:
            num_inputs = self.dim_state

        self.input_transform = input_transform
        if self.input_transform is not None:
            assert len(num_inputs) == 1, "Only implemented N x 1 inputs."
            num_inputs = (num_inputs[0] + self.input_transform.extra_dim,)

        self.nn = DeterministicRNN(
            num_inputs,
            self.dim_reward,
            layers=layers,
            embedding_layers=embedding_layers,
            squashed_output=False,
            non_linearity=non_linearity,
            biased_head=biased_head,
        )
        if jit_compile:
            self.nn = torch.jit.script(self.nn)
        self.dimension = self.nn.embedding_dim

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractValueFunction.default."""
        return super().default(environment, *args, **kwargs)

    @classmethod
    def from_other(cls, other, copy=True):
        """Create new Value Function from another Value Function."""
        new = cls(
            dim_state=other.dim_state,
            num_states=other.num_states,
            tau=other.tau,
            input_transform=other.input_transform,
            dim_reward=other.dim_reward,
        )
        new.nn = other.nn.__class__.from_other(other.nn, copy=copy)
        return new

    @classmethod
    def from_nn(
        cls,
        module,
        dim_state,
        num_states=-1,
        dim_reward=(1,),
        tau=0.0,
        input_transform=None,
    ):
        """Create new Value Function from a Neural Network Implementation."""
        new = cls(
            dim_state=dim_state,
            num_states=num_states,
            tau=tau,
            input_transform=input_transform,
            dim_reward=dim_reward,
        )
        new.nn = module
        return new

    def forward(self, state, action=torch.tensor(float("nan"))):
        """Get value of the value-function at a given state."""
        if self.input_transform is not None:
            state = self.input_transform(state)

        if self.discrete_state:
            state = one_hot_encode(state, self.num_states)
        return self.nn(state)

    @torch.jit.export
    def embeddings(self, state):
        """Get embeddings of the value-function at a given state."""
        if self.discrete_state:
            state = one_hot_encode(state, self.num_states)
        return self.nn.last_layer_embeddings(state).squeeze(-1)

    def reset(self):
        self.nn.reset()
