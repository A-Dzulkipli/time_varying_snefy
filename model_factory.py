import torch
import math

from model_utils import *

class LinearTime(torch.nn.Module):
    def __init__(self, output_size=16, bias=True):
        super().__init__()
        self.output_size = output_size
        self.param = torch.nn.Linear(1, output_size, bias)
    def forward(self, t):
        return self.param(t)

class MLP_1_Time(torch.nn.Module):
    def __init__(self, output_size=16, bias=True):
        super().__init__()
        self.output_size=output_size
        self.param = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )
    def forward(self, t):
        return self.param(t)

class MLP_2_Time(torch.nn.Module):
    def __init__(self, output_size=16, bias=True):
        super().__init__()
        self.output_size = output_size
        self.param = torch.nn.Sequential(
            torch.nn.Linear(1, 32, bias), torch.nn.ReLU(),
            torch.nn.Linear(32, 32, bias), torch.nn.ReLU(),
            torch.nn.Linear(32, output_size, bias)
        )
    def forward(self, t):
        return self.param(t)

class MLP_3_Time(torch.nn.Module):
    def __init__(self, output_size=16, bias=True):
        super().__init__()
        self.output_size = output_size
        self.param = torch.nn.Sequential(
            torch.nn.Linear(1, 32, bias), torch.nn.ReLU(),
            torch.nn.Linear(32, 32, bias), torch.nn.ReLU(),
            torch.nn.Linear(32, 32, bias), torch.nn.ReLU(),
            torch.nn.Linear(32, output_size, bias)
        )
    def forward(self, t):
        return self.param(t)

class SinusoidTime(torch.nn.Module):
    def __init__(self, output_size=16, w_min=1.0, w_max=10.0):
        super().__init__()
        freqs = torch.logspace(math.log10(w_min), math.log10(w_max), output_size // 2)
        self.register_buffer("freqs", freqs) 
        self.output_size = output_size
    
    def forward(self, t):
        w = t*self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(w), torch.cos(w)], dim=-1)
    
class RandomFourierTime(torch.nn.Module):
    def __init__(self, output_size=16, sigma=1.0):
        super().__init__()
        self.output_size = output_size
        W = torch.randn(1, output_size // 2) * sigma
        self.register_buffer("W", W)

    def forward(self, t):
        val = t @ self.W
        return torch.cat([torch.sin(val), torch.cos(val)], dim=-1)
    
def time_net_factory(output_size):
    return {
        "Linear": LinearTime(output_size=output_size),
        "MLP_1": MLP_1_Time(output_size=output_size),
        "MLP_2": MLP_2_Time(output_size=output_size),
        "MLP_3": MLP_3_Time(output_size=output_size),
        "Sinusoid": SinusoidTime(output_size=output_size),
        "Fourier": RandomFourierTime(output_size=output_size)
    }


class IdentityParamHead(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
    
    def forward(self, adapter_output):
        return adapter_output

class LinearParamHead(torch.nn.Module):
    def __init__(self, input_size, target_shape):
        super().__init__()
        self.input_size = input_size
        in_t = torch.as_tensor(input_size)
        self._in_shape = tuple(in_t.tolist()) if in_t.ndim > 0 else (int(in_t.item()),)
        flat_in = int(in_t.prod().item())

        tgt_t = torch.as_tensor(target_shape)
        self.target_shape = tuple(tgt_t.tolist()) if tgt_t.ndim > 0 else (int(tgt_t.item()),)
        flat_out = int(tgt_t.prod().item())

        self.proj = torch.nn.Linear(flat_in, flat_out)

    def forward(self, a):
        B = a.shape[0]
        y = self.proj(a.view(B, -1))
        return y.view(B, *self.target_shape)

class MLPParamHead(torch.nn.Module):
    def __init__(self, input_size, target_shape, hidden=(128, 128), act=torch.nn.ReLU):
        super().__init__()
        self.input_size = input_size
        in_t = torch.as_tensor(input_size)
        self._in_shape = tuple(in_t.tolist()) if in_t.ndim > 0 else (int(in_t.item()),)
        flat_in = int(in_t.prod().item())

        tgt_t = torch.as_tensor(target_shape)
        self.target_shape = tuple(tgt_t.tolist()) if tgt_t.ndim > 0 else (int(tgt_t.item()),)
        flat_out = int(tgt_t.prod().item())

        layers = []
        in_f = flat_in
        for h in hidden:
            layers += [torch.nn.Linear(in_f, h), act()]
            in_f = h
        layers += [torch.nn.Linear(in_f, flat_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, a):
        B = a.shape[0]
        y = self.net(a.view(B, -1))
        return y.view(B, *self.target_shape)

