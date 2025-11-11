import torch

def register(dict_name, *keys):
    def decorate(f_constructor):
        key = keys[0] if len(keys) == 1 else tuple(keys)
        dict_name[key] = f_constructor
        return f_constructor
    return decorate

activations = {}

@register(activations, "cos")
def initialise_cos(**kwargs):
    return torch.cos