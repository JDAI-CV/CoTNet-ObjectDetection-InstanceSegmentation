import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

from detectron2.cupy_layers.aggregation_zeropad import LocalConvolution
from detectron2.layers.create_act import get_act_layer

__all__ = [
    "SECoTNetDBlockBase",
    "SECoTNetD",
    "make_secotnetd_stage",
    "build_secotnetd_backbone",
]

class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class SplitAttnConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, radix=2, reduction_factor=4,
                 act_layer=nn.ReLU, norm=None, drop_block=None, **kwargs):
        super(SplitAttnConv2d, self).__init__()
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        attn_chs = max(in_channels * radix // reduction_factor, 32)

        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        #self.bn0 = norm_layer(mid_chs) if norm_layer is not None else None
        self.bn0 = get_norm(norm, mid_chs)
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        #self.bn1 = norm_layer(attn_chs) if norm_layer is not None else None
        self.bn1 = get_norm(norm, attn_chs)
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.fc1.out_channels

    def forward(self, x):
        x = self.conv(x)
        if self.bn0 is not None:
            x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = F.adaptive_avg_pool2d(x_gap, 1)
        x_gap = self.fc1(x_gap)
        if self.bn1 is not None:
            x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()

class CoTLayer(nn.Module):
    def __init__(self, dim, kernel_size, norm=None):
        super(CoTLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False, norm=get_norm(norm, dim)),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            Conv2d(2*dim, dim//factor, 1, bias=False, norm=get_norm(norm, dim//factor)),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            get_norm(norm, dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = get_norm(norm, dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            Conv2d(dim, attn_chs, 1, norm=get_norm(norm, attn_chs)),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        
        return out.contiguous()

class CoTBlock(CNNBlockBase):
    def __init__(
        self,
        block_idx, 
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            #self.shortcut = Conv2d(
            #    in_channels,
            #    out_channels,
            #    kernel_size=1,
            #    stride=stride,
            #    bias=False,
            #    norm=get_norm(norm, out_channels),
            #)
            self.shortcut = nn.Sequential(*[
                nn.AvgPool2d(2, 2, ceil_mode=True, count_include_pad=False),
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                )
            ])
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        if block_idx == 0 and stride_3x3 == 1:
            stride_3x3 = 2

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        #self.conv2 = Conv2d(
        #    bottleneck_channels,
        #    bottleneck_channels,
        #    kernel_size=3,
        #    stride=stride_3x3,
        #    padding=1 * dilation,
        #    bias=False,
        #    groups=num_groups,
        #    dilation=dilation,
        #    norm=get_norm(norm, bottleneck_channels),
        #)

        res2_3_dim = {64, 128}
        self.avd = None

        if (bottleneck_channels in res2_3_dim) or (bottleneck_channels == 256 and block_idx % 2 != 0):
          self.conv2 = SplitAttnConv2d(
            bottleneck_channels, 
            bottleneck_channels, 
            kernel_size=3, 
            stride=stride_3x3, 
            padding=1, 
            reduction_factor=4, 
            dilation=1, 
            groups=num_groups, 
            radix=1, 
            norm=norm, 
            drop_block=None, 
            act_layer=get_act_layer('swish')
          )
        else:
          self.conv2 = CoTLayer(bottleneck_channels, kernel_size=3, norm=norm)
          if stride_3x3 > 1:
            self.avd = nn.AvgPool2d(3, 2, padding=1)
        
        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        #for layer in [self.conv1, self.conv3, self.shortcut]:
        #    if layer is not None:  # shortcut can be None
        #        weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.avd is not None:
            out = self.avd(out)

        out = self.conv2(out)
        #out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        stem_width = out_channels // 2
        self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
                get_norm(norm, stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                get_norm(norm, stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, out_channels, 3, stride=1, padding=1, bias=False),
                get_norm(norm, out_channels)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class SECoTNetD(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        #self._out_features.append('linear')
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(
        block_class, num_blocks, first_stride=None, *, in_channels, out_channels, **kwargs
    ):
        if first_stride is not None:
            assert "stride" not in kwargs and "stride_per_block" not in kwargs
            kwargs["stride_per_block"] = [first_stride] + [1] * (num_blocks - 1)
            #logger = logging.getLogger(__name__)
            #logger.warning(
            #    "ResNet.make_stage(first_stride=) is deprecated!  "
            #    "Use 'stride_per_block' or 'stride' instead."
            #)

        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(i, in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

SECoTNetDBlockBase = CNNBlockBase

def make_secotnetd_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return SECoTNetD.make_stage(*args, **kwargs)


def build_secotnetd_backbone(cfg, input_shape):
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features if f != "stem"
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        #first_stride = 1 if (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        ## Use BasicBlock for R18 and R34.
        #if depth in [18, 34]:
        #    stage_kargs["block_class"] = BasicBlock
        #else:
        #    stage_kargs["bottleneck_channels"] = bottleneck_channels
        #    stage_kargs["stride_in_1x1"] = stride_in_1x1
        #    stage_kargs["dilation"] = dilation
        #    stage_kargs["num_groups"] = num_groups
        #    if deform_on_per_stage[idx]:
        #        stage_kargs["block_class"] = DeformBottleneckBlock
        #        stage_kargs["deform_modulated"] = deform_modulated
        #        stage_kargs["deform_num_groups"] = deform_num_groups
        #    else:
        #        stage_kargs["block_class"] = CoTBlock
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        stage_kargs["block_class"] = CoTBlock

        blocks = SECoTNetD.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return SECoTNetD(stem, stages, out_features=out_features).freeze(freeze_at)
