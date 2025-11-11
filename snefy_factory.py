import torch
from torch import nn
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Sequence, Set, Dict, Any

from model_utils import *
from model_factory import *
from Snefy import *


TIME_NETS = {
    "Linear",
    "MLP_1",
    "MLP_2",
    "MLP_3",
    "Sinusoid",
    "Fourier"
}

PARAMETER_HEADS = {
    "Identity",
    "Linear",
    "MLP"
}

PARAMS = {
    "W", 
    "V", 
    "b", 
    "v0", 
    "mixture"
}

@dataclass
class TimeNetSpec:
    kind: str = "Linear"
    output_size: int = 16

@dataclass
class ParamHeadSpec:
    kind: str = "Identity"
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MixtureSpec:
    num_components: int = 8
    time_varying: bool = False
    trainable_if_static: bool = True
    frozen_loc: Optional[Sequence[Sequence[float]]] = None
    frozen_scale: Optional[Sequence[Sequence[float]]] = None


@dataclass
class SnefyConfig:
    n: int = 16
    m: int = 16
    d: int = 1

    time_varying: Set[str] = field(default_factory=set)

    time_net: TimeNetSpec = field(default_factory=TimeNetSpec)
    param_head: ParamHeadSpec = field(default_factory=ParamHeadSpec)
    activation: str = "cos"

    mixture: MixtureSpec = field(default_factory=MixtureSpec)

    seed: Optional[int] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
    
def make_snefy_config(
        *,
        n: int,
        m: int,
        d: int,
        time_varying: Sequence[str] = (),
        time_net_kind: str = "Linear",
        time_net_out: int = 16,
        head_kind: str = "Identity",
        head_kwargs: Optional[Dict[str, Any]] = None,
        activation: str = "cos",
        num_mixture_components: int = 8,
        mixture_trainable_if_static: bool = True,
        frozen_loc: Optional[Sequence[Sequence[float]]] = None,
        frozen_scale: Optional[Sequence[Sequence[float]]] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None
) -> SnefyConfig:
    time_varying_set = set(time_varying)
    head_kw = dict(head_kwargs or {})

    return SnefyConfig(
        n=n,
        m=m,
        d=d,
        time_varying=time_varying_set,
        time_net=TimeNetSpec(
            kind=time_net_kind,
            output_size=time_net_out
        ),
        param_head=ParamHeadSpec(
            kind=head_kind,
            kwargs=head_kw
        ),
        activation=activation,
        mixture=MixtureSpec(
            num_components=num_mixture_components,
            time_varying=("mixture" in time_varying_set),
            trainable_if_static=mixture_trainable_if_static,
            frozen_loc=frozen_loc,
            frozen_scale=frozen_scale
        ),
        seed=seed,
        name=name
    )
    
class MixtureParamBlock(nn.Module):
    def __init__(self, loc_head, scale_head):
        super().__init__()
        self.loc = loc_head
        self.scale = scale_head

def _target_shape(param: str, n: int, m: int, d: int) -> Tuple[int, ...]:
    if param == "W":
        return (n, d)
    if param == "V":
        return (m, n)
    if param == "b":
        return (n,)
    if param == "v0":
        return (1,)
    if param == "mixture":
        return (d,)
    raise ValueError(f"Unknown param: {param}")

def _input_size_for(param: str, head_kind: str, target_shape, k_default: int, time_net_out: int):
    if head_kind == "Identity":
        return target_shape
    return k_default

def _make_head(head_kind: str, input_size, target_shape, head_kwargs):
    if head_kind == "Identity":
        return IdentityParamHead(input_size=input_size)
    if head_kind == "Linear":
        return LinearParamHead(input_size=input_size, target_shape=target_shape)
    if head_kind == "MLP":
        return MLPParamHead(
            input_size=input_size,
            target_shape=target_shape,
            hidden=head_kwargs["hidden"],
            act=head_kwargs["act"]
        )
    raise ValueError(f"Unsupported head kind: {head_kind}")

def build_time_dep_non_mixture(cfg):
    n, m, d = cfg.n, cfg.m, cfg.d
    tv = set(cfg.time_varying)

    time_net = time_net_factory(output_size=cfg.time_net.output_size)[cfg.time_net.kind]

    head_kind = cfg.param_head.kind
    head_kwargs = dict(cfg.param_head.kwargs or {})
    k_default = head_kwargs["k"] if head_kind in {"Linear", "MLP"} else None

    time_dep = {}
    for pname in ["W", "V", "b", "v0"]:
        if pname in tv:
            tgt = _target_shape(pname, n, m, d)
            in_size = _input_size_for(pname, head_kind, tgt, k_default, cfg.time_net.output_size)
            time_dep[pname] = _make_head(head_kind, in_size, tgt, head_kwargs)

    model_kwargs = {"num_mix_components": cfg.mixture.num_components, "mixture": {}}

    K, d = cfg.mixture.num_components, cfg.d

    if not cfg.mixture.time_varying and not cfg.mixture.trainable_if_static:
        model_kwargs["mixture"]["trainable"] = False

        if cfg.mixture.frozen_loc is not None:
            loc = torch.as_tensor(cfg.mixture.frozen_loc, dtype=torch.get_default_dtype())
            if loc.ndim != 2 or loc.shape != (K, d):
                raise ValueError(f"frozen_loc must have shape (K,d)=({K},{d}); got {tuple(loc.shape)}")
            model_kwargs["frozen_loc"] = loc

        if cfg.mixture.frozen_scale is not None:
            scale = torch.as_tensor(cfg.mixture.frozen_scale, dtype=torch.get_default_dtype())
            if scale.ndim != 2 or scale.shape != (K, d):
                raise ValueError(f"frozen_scale must have shape (K,d)=({K},{d}); got {tuple(scale.shape)}")
            model_kwargs["frozen_scale"] = scale
    else:
        if cfg.mixture.frozen_loc is not None:
            loc = torch.as_tensor(cfg.mixture.frozen_loc, dtype=torch.get_default_dtype())
            if loc.ndim != 2 or loc.shape != (K, d):
                raise ValueError(f"frozen_loc must have shape (K,d)=({K},{d}); got {tuple(loc.shape)}")
            model_kwargs["init_loc"] = loc
        if cfg.mixture.frozen_scale is not None:
            scale = torch.as_tensor(cfg.mixture.frozen_scale, dtype=torch.get_default_dtype())
            if scale.ndim != 2 or scale.shape != (K, d):
                raise ValueError(f"frozen_scale must have shape (K,d)=({K},{d}); got {tuple(scale.shape)}")
            model_kwargs["init_scale"] = scale

    return time_net, time_dep, model_kwargs


def wire_mixture_time_dep(cfg, time_net, time_dep, model_kwargs):
    n, m, d = cfg.n, cfg.m, cfg.d
    tv = set(cfg.time_varying)

    head_kind   = cfg.param_head.kind
    head_kwargs = dict(cfg.param_head.kwargs or {})
    k_default   = head_kwargs["k"] if head_kind in {"Linear", "MLP"} else None

    K = cfg.mixture.num_components

    if "mixture" not in tv:
        return time_dep, model_kwargs

    mix_tgt = (K, d)
    in_size_mix = _input_size_for(
        "mixture", head_kind, mix_tgt, k_default, cfg.time_net.output_size
    )

    loc_head   = _make_head(head_kind, input_size=in_size_mix, target_shape=mix_tgt, head_kwargs=head_kwargs)
    scale_head = _make_head(head_kind, input_size=in_size_mix, target_shape=mix_tgt, head_kwargs=head_kwargs)

    loc_param   = Param(n, m, d, param_type="mixture_loc",   time_net=time_net, time_dep={"mixture_loc": loc_head})
    scale_param = Param(n, m, d, param_type="mixture_scale", time_net=time_net, time_dep={"mixture_scale": scale_head})

    class _MixtureParamBlock(nn.Module):
        def __init__(self, loc, scale):
            super().__init__()
            self.loc = loc
            self.scale = scale

    time_dep["mixture"] = _MixtureParamBlock(loc_param, scale_param)

    logits_tgt = (K,)
    in_size_logits = _input_size_for(
        "logits", head_kind, logits_tgt, k_default, cfg.time_net.output_size
    )
    logits_head = _make_head(head_kind, input_size=in_size_logits, target_shape=logits_tgt, head_kwargs=head_kwargs)
    time_dep["logits"] = logits_head

    model_kwargs["mixture"]["trainable"] = True
    return time_dep, model_kwargs




def build_snefy_from_config(cfg):
    time_net, time_dep, model_kwargs = build_time_dep_non_mixture(cfg)
    time_dep, model_kwargs = wire_mixture_time_dep(cfg, time_net, time_dep, model_kwargs)

    # for (key, val) in model_kwargs.items():
    #     print(f"key: {key}, val: {val}")

    model = Snefy(
        measure="gauss",
        activation=cfg.activation,
        d=cfg.d, 
        n=cfg.n, 
        m=cfg.m,
        time_net=time_net,
        time_dep=time_dep,
        **model_kwargs
    )

    model.meta = cfg.to_dict()
    if cfg.name:
        model.name = cfg.name
    return model
