"""
Implementation of Model Agnostic Meta-Learning (MAML)
Adapted from https://github.com/learnables/learn2learn/
"""
import traceback

from torch import nn
from torch.autograd import grad

from lib.meta_rl.algorithms.utilities import clone_module, update_module


class MAML(nn.Module):
    """
    Implementation of the Model Agnostic Meta Learning Algorithm.

    References
    ----------
    Finn, C., Abbeel, P., Levine, S., (2017)
    Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
    """

    def __init__(
            self,
            model: nn.Module,
            lr: float = 0.01,
            first_order: bool = False,
            allow_unused: bool = None,
            allow_nograd: bool = False
    ):
        super().__init__()

        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss):
        """
        Takes a gradient step on the loss and updates the cloned parameters in place.

        Parameters
        ----------
        loss: Loss to minimize upon update
        """
        second_order = not self.first_order

        if self.allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=self.allow_unused)
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=self.allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # Update the module
        self.maml_update(gradients)

    def maml_update(self, gradients=None):
        """
        Performs a MAML update on model using gradients
        The model itself is updated in-place (no deepcopy), but the parameters' tensors are not.

        Parameters
        ----------
        gradients: A list of gradients for each model parameter.
        """
        if gradients is not None:
            params = list(self.module.parameters())
            if not len(gradients) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(gradients)) + ')'
                print(msg)
            for p, g in zip(params, gradients):
                if g is not None:
                    p.update = - self.lr * g
        self.module = update_module(self.module)

    def clone(self):
        """
        Returns a copy of the class with the module parameters cloned from the module of the original module.
        This implies that back-propagating losses on the cloned module will populate the buffers of the original module.
        """
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=self.first_order,
                    allow_unused=self.allow_unused,
                    allow_nograd=self.allow_nograd)
