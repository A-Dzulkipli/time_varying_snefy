import numpy as np
import torch
from kernels import *
# from activations import activations
from activation_functions import *
from model_utils import *

class Snefy(torch.nn.Module):
    def __init__(self, measure, activation, d, n, m, **kwargs):
        super().__init__()
        self.d = d
        self.n = n
        self.m = m
        self.activation = activations[activation](**kwargs) 
        self.kernel = kernels[(activation, measure)](**kwargs)
        self.initialise_time_net(**kwargs)
        self.initialise_params(n, m, d, **kwargs)
        self.initialise_mixture(n, m, d, **kwargs)
        self.eps = 1e-6

    def initialise_time_net(self, **kwargs):
        if "time_net" in kwargs:
            self.time_net = kwargs["time_net"]
        else:
            self.time_net = ZeroNet()

    def initialise_params(self, n, m, d, **kwargs):
        self.b = Param(n, m, d, "b", **kwargs)
        self.W = Param(n, m, d, "W", **kwargs)
        self.V = Param(n, m, d, "V", **kwargs)
        self.v0 = Param(n, m, d, "v0", **kwargs)

    def initialise_mixture(self, n, m, d, **kwargs):
        self.mixture = BaseMixture(n, m, d, **kwargs)

    def neural_component(self, x, W, b, V, v0):
        v0 = v0.squeeze(-1) if v0.dim() > 1 else v0
        scores = torch.einsum('bd,bnd->bn', x, W)
        neuron = self.activation(scores + b)
        outs = torch.einsum('bn,bmn->bm', neuron, V)
        return (outs * outs).sum(dim=1) + v0.pow(2)
    
    def log_Z(self, time_net_output, W, b, V, v0):
        v0 = v0.squeeze(-1) if v0.dim() > 1 else v0
        G = torch.matmul(V.transpose(1, 2), V)
        pi = self.mixture.weights(time_net_output)

        K = torch.zeros_like(G) 
        for k, component in enumerate(self.mixture.components):
            mu_k = component.loc_t(time_net_output)
            std_k = component.std_t(time_net_output)
            K_k = self.kernel(W, b, mu_k, std_k)
            K = K + pi[:, k].view(-1, 1, 1) * K_k
        Z = (G*K).sum(dim=(1,2)) + v0.pow(2)

        return torch.log(Z.clamp_min(self.eps))
    
    def log_prob(self, z):
        t = z[:, :1]
        x = z[:, 1:]
        time_net_output = self.time_net(t)
        W = self.W(time_net_output)
        b = self.b(time_net_output)
        V = self.V(time_net_output)
        v0 = self.v0(time_net_output)
        neural = self.neural_component(x, W, b, V, v0)
        logZ = self.log_Z(time_net_output, W, b, V, v0)
        log_phi = self.mixture.log_prob(time_net_output, x)
        return log_phi + torch.log(neural.clamp_min(self.eps)) - logZ
    
    def forward(self, z):
        return self.log_prob(z)
