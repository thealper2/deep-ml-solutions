import torch
import torch.nn as nn


def capture_activations(model, layer_names, x):
    captured = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            captured[name] = output.detach()
        return hook

    for name in layer_names:
        module = model.get_submodule(name)
        handle = module.register_forward_hook(make_hook(name))
        hooks.append(handle)

    with torch.no_grad():
        _ = model(x)

    for handle in hooks:
        handle.remove()

    return captured