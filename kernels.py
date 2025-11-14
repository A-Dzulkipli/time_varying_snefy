import torch

def register(dict_name, *keys):
    def decorate(f):
        key = keys[0] if len(keys) == 1 else tuple(keys)
        dict_name[key] = f
        return f
    return decorate

kernels = {}

@register(kernels, "cos", "gauss")

def cosine_gauss_kernel(W, b, mu, std):
    squeeze = False
    if W.dim() == 2:
        squeeze = True
        W = W.unsqueeze(0)
        b = b.unsqueeze(0)
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)

    Wp = W * std.unsqueeze(1)
    bp = b + torch.einsum('bnd,bd->bn', W, mu)

    W2 = (Wp * Wp).sum(-1, keepdim=True)
    G = torch.einsum('bnd,bmd->bnm', Wp, Wp) 
    
    Emin = torch.exp(-0.5 * (W2 + W2.transpose(-1,-2) - 2*G))
    Eplu = torch.exp(-0.5 * (W2 + W2.transpose(-1,-2) + 2*G))

    Bc = bp.unsqueeze(-1)
    bm = Bc - Bc.transpose(-1,-2)
    bpp = Bc + Bc.transpose(-1,-2)

    K = 0.5 * (torch.cos(bm) * Emin + torch.cos(bpp) * Eplu)

    return K.squeeze(0) if squeeze else K