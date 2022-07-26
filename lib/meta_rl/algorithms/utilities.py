"""
Utilities for meta-learning algorithms.
Adapted from https://github.com/learnables/learn2learn/
"""

import torch


def clone_module(module, memo=None):
    """
    Creates a copy of a module, whose parameters/buffers/submodules are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute the derivatives of the new
    modules' parameters w.r.t the original parameters.

    Parameters
    ----------
    module: nn.Module
        The module to be cloned

    Returns
    -------
    cloned_module: nn.Module
        The cloned module
    """
    if memo is None:
        memo = {}

    # First, create a copy of the module.
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def detach_module(module, keep_requires_grad=False):
    """
    Detaches all parameters/buffers of a previously cloned module from its computational graph.
    Note: detach works in-place, so it does not return a copy.

    Parameters
    ----------
    module: nn.Module
        Module to be detached.
    keep_requires_grad: bool
        If this flag is set to True, then the required_grad field will be the same as the pre-detached module.
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            requires_grad = module._parameters[param_key].requires_grad
            detached = module._parameters[param_key].detach_()
            if keep_requires_grad and requires_grad:
                module._parameters[param_key].requires_grad_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()
            if keep_requires_grad:  # requires_grad checked above
                module._buffers[buffer_key].requires_grad_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key], keep_requires_grad=keep_requires_grad)


def update_module(module, updates=None, memo=None):
    """
    Updates the parameters of a module in-place, in a way that preserves differentiability.
    The parameters of the module are swapped with their update values, according to:
    p -> p + u
    where p is the parameter, and u is its corresponding update.

    Parameters
    ----------
    module: nn.Module
        The module to update.
    updates (optional): list
        A list of gradients for each parameter of the model. If None, will use the tensors in .update attributes.
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module
