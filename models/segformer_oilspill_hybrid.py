from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


def _resolve_group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _round_up_to_multiple(value: int, divisor: int) -> int:
    return max(divisor, int(math.ceil(value / max(divisor, 1))) * max(divisor, 1))


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(
                _resolve_group_norm_groups(out_channels),
                out_channels,
            ),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        super().__init__(*layers)


class ResidualPostBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_resolve_group_norm_groups(channels), channels),
            nn.GELU(),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GroupedCBAMEnhancer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        reduction_ratio: int,
        spatial_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_groups={num_groups}"
            )
        reduced = _round_up_to_multiple(
            max(channels // max(reduction_ratio, 1), num_groups),
            num_groups,
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                reduced,
                kernel_size=1,
                groups=num_groups,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                reduced,
                channels,
                kernel_size=1,
                groups=num_groups,
                bias=True,
            ),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.post = ResidualPostBlock(channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        channel_gate = self.channel_mlp(avg_pool + max_pool)
        channel_refined = x * channel_gate

        spatial_input = torch.cat(
            [
                channel_refined.mean(dim=1, keepdim=True),
                channel_refined.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_gate = self.spatial_gate(spatial_input)
        refined = channel_refined * spatial_gate
        return self.post(refined)


class ASPPContextFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        branches: list[nn.Module] = []
        for dilation in dilations:
            kernel_size = 1 if int(dilation) == 1 else 3
            branches.append(
                ConvGNAct(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNAct(in_channels, out_channels, kernel_size=1),
        )
        merged_channels = out_channels * (len(dilations) + 1)
        self.project = ConvGNAct(
            merged_channels,
            out_channels,
            kernel_size=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(
            pooled,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class FuseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvGNAct(in_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.block(torch.cat([x, skip], dim=1))


class EdgeSupervisionEnhancementModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge_predictor = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.refine = ConvGNAct(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_gate = torch.sigmoid(self.edge_predictor(x))
        return self.refine(x * (1.0 + edge_gate))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_channels, in_channels, kernel_size=3),
            nn.Dropout2d(p=float(dropout)),
            nn.Conv2d(in_channels, num_labels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OilSpillHybridDecoder(nn.Module):
    def __init__(self, config: SegformerConfig, decoder_cfg: dict) -> None:
        super().__init__()
        hidden_sizes = list(config.hidden_sizes)
        decoder_channels = int(decoder_cfg.get("channels", 192))
        gce_groups = int(decoder_cfg.get("gce_groups", 8))
        gce_reduction = int(decoder_cfg.get("gce_reduction", 4))
        spatial_kernel = int(decoder_cfg.get("spatial_kernel", 7))
        aspp_dilations = [int(v) for v in decoder_cfg.get("aspp_dilations", [1, 6, 12, 18])]
        dropout = float(decoder_cfg.get("dropout", 0.1))
        aux_heads = {
            str(level).lower() for level in decoder_cfg.get("esem_aux_heads", ["s16", "s8"])
        }

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False)
                for in_channels in hidden_sizes
            ]
        )
        self.enhancers = nn.ModuleList(
            [
                GroupedCBAMEnhancer(
                    decoder_channels,
                    num_groups=gce_groups,
                    reduction_ratio=gce_reduction,
                    spatial_kernel=spatial_kernel,
                    dropout=dropout,
                )
                for _ in hidden_sizes
            ]
        )
        self.aspp = ASPPContextFusion(
            decoder_channels,
            decoder_channels,
            dilations=aspp_dilations,
            dropout=dropout,
        )
        self.fuse_s16 = FuseBlock(decoder_channels * 2, decoder_channels)
        self.fuse_s8 = FuseBlock(decoder_channels * 2, decoder_channels)
        self.fuse_s4 = FuseBlock(decoder_channels * 2, decoder_channels)
        self.esem_s16 = EdgeSupervisionEnhancementModule(decoder_channels)
        self.esem_s8 = EdgeSupervisionEnhancementModule(decoder_channels)
        self.esem_s4 = EdgeSupervisionEnhancementModule(decoder_channels)
        self.aux_head_s16 = (
            SegmentationHead(
                decoder_channels,
                int(config.num_labels),
                dropout=dropout,
            )
            if "s16" in aux_heads
            else None
        )
        self.aux_head_s8 = (
            SegmentationHead(
                decoder_channels,
                int(config.num_labels),
                dropout=dropout,
            )
            if "s8" in aux_heads
            else None
        )
        self.classifier = SegmentationHead(
            decoder_channels,
            int(config.num_labels),
            dropout=dropout,
        )

    def forward(
        self,
        features: list[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        projected: list[torch.Tensor] = []
        for feature_map, projection, enhancer in zip(
            features,
            self.projections,
            self.enhancers,
        ):
            projected.append(enhancer(projection(feature_map)))

        s4, s8, s16, s32 = projected
        aspp_s32 = self.aspp(s32)
        d16 = self.esem_s16(self.fuse_s16(aspp_s32, s16))
        d8 = self.esem_s8(self.fuse_s8(d16, s8))
        d4 = self.esem_s4(self.fuse_s4(d8, s4))

        seg_logits = self.classifier(d4)
        aux_s16 = self.aux_head_s16(d16) if self.aux_head_s16 is not None else None
        aux_s8 = self.aux_head_s8(d8) if self.aux_head_s8 is not None else None

        outputs = {
            "seg_logits": seg_logits,
        }
        if aux_s16 is not None:
            outputs["aux_s16"] = aux_s16
        if aux_s8 is not None:
            outputs["aux_s8"] = aux_s8
        for key, tensor in list(outputs.items()):
            if tensor.shape[-2:] != output_size:
                outputs[key] = F.interpolate(
                    tensor,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
        return outputs


class SegformerOilSpillHybridModel(nn.Module):
    def __init__(self, config: SegformerConfig, decoder_cfg: dict) -> None:
        super().__init__()
        self.config = config
        self.encoder = SegformerModel(config)
        self.decoder = OilSpillHybridDecoder(config, decoder_cfg)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError("SegFormer encoder returned no hidden states.")
        return self.decoder(
            list(hidden_states),
            output_size=tuple(pixel_values.shape[-2:]),
        )
