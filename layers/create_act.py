""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .activations import *
from .activations_jit import *
from .activations_me import *


_has_silu = 'silu' in dir(torch.nn.functional)


_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
    hard_mish=HardMish,
)

_ACT_LAYER_ME = dict(
    silu=nn.SiLU if _has_silu else SwishMe,
    swish=nn.SiLU if _has_silu else SwishMe,
    mish=MishMe,
    hard_sigmoid=HardSigmoidMe,
    hard_swish=HardSwishMe,
    hard_mish=HardMishMe,
)


def get_act_layer(name='relu'):
    if not name:
        return None
    if name in _ACT_LAYER_ME:
        return _ACT_LAYER_ME[name]
    return _ACT_LAYER_DEFAULT[name]



