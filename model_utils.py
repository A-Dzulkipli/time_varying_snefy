import torch
from torch import nn
import numpy as np
from normflows.distributions import GaussianMixture

class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 1
        self.register_buffer("_zeros", torch.zeros(1)) 
    
    def forward(self, t):
        if t.dim() > 0:
            B = t.shape[0]
        else:
            B = 1
        return self._zeros.unsqueeze(0).expand(B, self.output_size)

class DimensionAdapter(nn.Module):
    def __init__(self, time_net, param):
        super().__init__()
        size_tensor = torch.as_tensor(param.input_size)
        self._target_shape = tuple(size_tensor.tolist()) if size_tensor.ndim > 0 else (int(size_tensor.item()),)

        self.param_size = int(size_tensor.prod().item())
        self.time_size = time_net.output_size
        self.adapter = torch.nn.Linear(self.time_size, self.param_size)

    def forward(self, time_net_output):
        if time_net_output.dim() == 0:
            time_net_output = time_net_output.unsqueeze(0)

        B = time_net_output.shape[0]
        flat = self.adapter(time_net_output.view(B, self.time_size))
        return flat.view(B, *self._target_shape)

    
class StaticParam(nn.Module):
    def __init__(self, param_tensor):
        super().__init__()
        self.param = torch.nn.Parameter(param_tensor)
    
    def forward(self, time_net_output):
        if time_net_output.dim() > 0:
            B = time_net_output.shape[0]
            return self.param.unsqueeze(0).expand(B, *self.param.shape)
        return self.param
        

base_params = {
    "b": lambda n, m, d, **kwargs: StaticParam(torch.zeros(n)),
    "W": lambda n, m, d, **kwargs: StaticParam(2*torch.randn(n, d)),
    "V": lambda n, m, d, **kwargs: StaticParam(torch.randn(m, n)/np.sqrt(n)),
    "logits": lambda n, m, d, **kwargs: StaticParam(torch.zeros(kwargs["num_mix_components"])),
    "v0": lambda n, m, d, **kwargs: StaticParam(torch.tensor(0.0))
}

    
class Param(nn.Module):
    def __init__(self, n, m, d, param_type, **kwargs):
        super().__init__()
        if "time_dep" in kwargs and param_type in kwargs["time_dep"]:
            self.time_varying = True
            if "time_net" in kwargs:
                t_net = kwargs["time_net"]
            else:
                t_net = ZeroNet()
            self.adapter = DimensionAdapter(t_net, kwargs["time_dep"][param_type])
            self.param = kwargs["time_dep"][param_type]
        else:
            self.time_varying = False
            self.param = base_params[param_type](n, m, d, **kwargs)

    def forward(self, time_net_output):
        if self.time_varying:
            return self.param(self.adapter(time_net_output))
        else:
            return self.param(time_net_output)

class BaseMeasure(nn.Module):
    def __init__(self, n, m, d, **kwargs):
        super().__init__()
        self.initialise_parameters(n, m, d, **kwargs)

    def initialise_parameters(self, n, m, d, **kwargs):
        if "time_dep" not in kwargs or ("mixture" not in kwargs["time_dep"] and "mixture" in kwargs and "trainable" in kwargs["mixture"] and kwargs["mixture"]["trainable"] is False):
            self.trainable = False
            loc = kwargs.get("frozen_loc", torch.zeros(d))
            scale = kwargs.get("frozen_scale", torch.ones(d))
            log_scale = torch.log(torch.exp(scale) - 1.0)
            self.register_buffer("loc", loc.detach().clone())
            self.register_buffer("_log_scale", log_scale.detach().clone())
        elif "time_dep" not in kwargs or "mixture" not in kwargs["time_dep"]:
            self.trainable = True
            init_loc = kwargs.get("init_loc", None)
            init_std = kwargs.get("init_scale", None)

            if init_loc is None:
                loc_t = torch.zeros(d)
            else:
                loc_t = torch.as_tensor(init_loc, dtype=torch.get_default_dtype())

            if init_std is None:
                log_scale_t = torch.zeros(d)
            else:
                std_t = torch.as_tensor(init_std, dtype=torch.get_default_dtype())
                log_scale_t = torch.log(torch.clamp(std_t, min=1e-8).exp() - 1.0)

            self.loc = StaticParam(loc_t)
            self._log_scale = StaticParam(log_scale_t)
        else:
            self.trainable = True
            self.loc = kwargs["time_dep"]["mixture"].loc
            self._log_scale = kwargs["time_dep"]["mixture"].scale

    def loc_t(self, time_net_output):
        if not self.trainable:
            dev, dt = time_net_output.device, time_net_output.dtype
            loc = self.loc.to(device=dev, dtype=dt)
            if time_net_output.dim() > 0:
                B = time_net_output.shape[0]
                loc = loc.unsqueeze(0).expand(B, -1)
            return loc
        return self.loc(time_net_output)

    def _std(self, time_net_output):
        if not self.trainable:
            dev, dt = time_net_output.device, time_net_output.dtype
            log_scale = self._log_scale.to(device=dev, dtype=dt)
            std = torch.nn.functional.softplus(log_scale) + 1e-6
            if time_net_output.dim() > 0:
                B = time_net_output.shape[0]
                std = std.unsqueeze(0).expand(B, -1)
            return std
        log_scale = self._log_scale(time_net_output)
        return torch.nn.functional.softplus(log_scale) + 1e-6

    
    def std_t(self, time_net_output):
        return self._std(time_net_output)


    def log_prob(self, time_net_output, x):
        loc = self.loc_t(time_net_output)
        scale = self._std(time_net_output)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=loc, scale=scale), 1
        ).log_prob(x)
    
    def forward(self, time_net_out, x):
        return self.log_prob(time_net_out, x)
    
class _PerKMeasureParam(nn.Module):
    def __init__(self, all_param: nn.Module, k: int):
        super().__init__()
        self.all = all_param
        self.k = k

    def forward(self, time_net_output):
        a = self.all(time_net_output)
        return a[:, self.k, :]


class BaseMixture(nn.Module):
    def __init__(self, n, m, d, **kwargs):
        super().__init__()
        K = kwargs["num_mix_components"]

        f_loc = kwargs.get("frozen_loc", None)
        f_scale = kwargs.get("frozen_scale", None)
        i_loc = kwargs.get("init_loc", None)
        i_scale = kwargs.get("init_scale", None)

        td_in = kwargs.get("time_dep", None)
        has_tv_mix = (td_in is not None) and ("mixture" in td_in)
        if has_tv_mix:
            loc_all = td_in["mixture"].loc
            scale_all = td_in["mixture"].scale

        self.components = nn.ModuleList()
        for k in range(K):
            kw_k = dict(kwargs)

            if isinstance(f_loc, torch.Tensor):
                kw_k["frozen_loc"] = f_loc[k]
            if isinstance(f_scale, torch.Tensor):
                kw_k["frozen_scale"] = f_scale[k]
            if isinstance(i_loc, torch.Tensor):
                kw_k["init_loc"] = i_loc[k]
            if isinstance(i_scale, torch.Tensor):
                kw_k["init_scale"] = i_scale[k]

            if has_tv_mix:
                td_k = dict(td_in)

                class _TD(nn.Module):
                    def __init__(self, loc_all, scale_all, kk):
                        super().__init__()
                        self.loc   = _PerKMeasureParam(loc_all,   kk)
                        self.scale = _PerKMeasureParam(scale_all, kk)

                td_k["mixture"] = _TD(loc_all, scale_all, k)
                kw_k["time_dep"] = td_k

            self.components.append(BaseMeasure(n, m, d, **kw_k))

        self.logits = Param(n, m, d, "logits", **kwargs)

    def _logits(self, time_net_output):
        return self.logits(time_net_output)

    def weights(self, time_net_output):
        return torch.softmax(self._logits(time_net_output), dim=-1)

    def log_prob(self, time_net_output, x):
        log_w = torch.log_softmax(self._logits(time_net_output), dim=-1)
        lps = torch.stack(
            [component.log_prob(time_net_output, x) for component in self.components],
            dim=-1
        )
        return torch.logsumexp(lps + log_w, dim=-1)

    def forward(self, time_net_out, x):
        return self.log_prob(time_net_out, x)