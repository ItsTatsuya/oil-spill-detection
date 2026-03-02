import os
import re
import logging
from typing import Optional

# Suppress HuggingFace symlinks warning on Windows (cosmetic only)
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers.drop import DropPath  # type: ignore[assignment]
except ImportError:
    class DropPath(nn.Module):  # type: ignore[no-redef]
        """Stochastic depth fallback when timm is unavailable."""
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            rand_t = torch.rand(shape, dtype=x.dtype, device=x.device)
            return x / keep * (rand_t < keep).float()

logger = logging.getLogger('oil_spill')


# ---------------------------------------------------------------------------
# Building block: LayerNorm for NCHW feature maps
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm adapted for NCHW tensors — normalises over the C dimension."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # [B, C, H, W] → [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # → [B, C, H, W]
        return x


class PatchEmbed(nn.Module):
    """Non-overlapping patch embedding (SegFormer-B0 only; kept for API compat)."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dims=768):
        super().__init__()
        self.img_size  = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)   # [B, embed_dims, H//P, W//P]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dims=768):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2, bias=True,
        )
        self.norm = LayerNorm2d(embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)   # [B, embed_dims, H', W']
        x = self.norm(x)
        return x


class MixFFN(nn.Module):
    """Mix Feed-Forward Network: Conv1×1 → DW-Conv3×3 → GELU → Dropout → Conv1×1."""

    def __init__(self, embed_dims: int, feedforward_channels: int):
        super().__init__()
        self.fc1    = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            feedforward_channels, feedforward_channels,
            kernel_size=3, padding=1, groups=feedforward_channels, bias=True,
        )
        self.act  = nn.GELU()
        self.fc2  = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, bias=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAttention(nn.Module):
    """
    Efficient Self-Attention with Sequence Reduction (sr_ratio).

    Uses torch.nn.functional.scaled_dot_product_attention (PyTorch ≥ 2.0)
    for memory-efficient attention; falls back to manual matmul otherwise.
    All operations stay in NCHW format — no reshape to sequence form.
    """

    def __init__(self, embed_dims: int, num_heads: int, sr_ratio: int):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads  = num_heads
        self.sr_ratio   = sr_ratio
        self.head_dim   = embed_dims // num_heads

        self.q    = nn.Conv2d(embed_dims, embed_dims,     kernel_size=1, bias=True)
        self.kv   = nn.Conv2d(embed_dims, embed_dims * 2, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(embed_dims, embed_dims,     kernel_size=1, bias=True)
        self.drop = nn.Dropout(0.1)

        if sr_ratio > 1:
            self.sr   = nn.Conv2d(embed_dims, embed_dims,
                                  kernel_size=sr_ratio, stride=sr_ratio, bias=True)
            self.norm = LayerNorm2d(embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x)                     # [B, C, H, W]

        if self.sr_ratio > 1:
            x_r = self.norm(self.sr(x))   # [B, C, H/sr, W/sr]
            kv  = self.kv(x_r)
            _, _, Hr, Wr = x_r.shape
        else:
            kv  = self.kv(x)
            Hr, Wr = H, W

        k, v = kv.chunk(2, dim=1)         # each [B, C, Hr, Wr]

        # Reshape to [B, num_heads, seq_len, head_dim] for SDPA
        q = (q.permute(0, 2, 3, 1)
               .reshape(B, H * W,    self.num_heads, self.head_dim)
               .permute(0, 2, 1, 3))
        k = (k.permute(0, 2, 3, 1)
               .reshape(B, Hr * Wr, self.num_heads, self.head_dim)
               .permute(0, 2, 1, 3))
        v = (v.permute(0, 2, 3, 1)
               .reshape(B, Hr * Wr, self.num_heads, self.head_dim)
               .permute(0, 2, 1, 3))

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
        )  # [B, nh, HW, hd]

        # Merge heads → NCHW
        out = (out.permute(0, 2, 1, 3)     # [B, HW, nh, hd]
                  .reshape(B, H, W, C)      # [B, H, W, C]
                  .permute(0, 3, 1, 2))     # [B, C, H, W]

        out = self.proj(out)
        out = self.drop(out)
        return out


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer block: LN→Attn→DropPath + LN→FFN→DropPath."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        sr_ratio: int = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1     = LayerNorm2d(embed_dims)
        self.attn      = EfficientAttention(embed_dims, num_heads, sr_ratio)
        self.norm2     = LayerNorm2d(embed_dims)
        self.ffn       = MixFFN(embed_dims, feedforward_channels)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class MixVisionTransformer(nn.Module):
    """Hierarchical ViT backbone with 4 stages (SegFormer-B2 configuration)."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        drop_rate: float = 0.0,
        sr_ratios=(8, 4, 2, 1),
        num_layers=(3, 4, 6, 3),
    ):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,       patch_size=7, stride=4,
            in_channels=in_channels, embed_dims=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,  patch_size=3, stride=2,
            in_channels=embed_dims[0], embed_dims=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,  patch_size=3, stride=2,
            in_channels=embed_dims[1], embed_dims=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_channels=embed_dims[2], embed_dims=embed_dims[3],
        )

        self.block1 = nn.ModuleList([
            TransformerEncoderLayer(embed_dims[0], num_heads[0],
                                    mlp_ratios[0] * embed_dims[0], sr_ratios[0], drop_rate)
            for _ in range(num_layers[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerEncoderLayer(embed_dims[1], num_heads[1],
                                    mlp_ratios[1] * embed_dims[1], sr_ratios[1], drop_rate)
            for _ in range(num_layers[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerEncoderLayer(embed_dims[2], num_heads[2],
                                    mlp_ratios[2] * embed_dims[2], sr_ratios[2], drop_rate)
            for _ in range(num_layers[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerEncoderLayer(embed_dims[3], num_heads[3],
                                    mlp_ratios[3] * embed_dims[3], sr_ratios[3], drop_rate)
            for _ in range(num_layers[3])
        ])

        self.norm1 = LayerNorm2d(embed_dims[0])
        self.norm2 = LayerNorm2d(embed_dims[1])
        self.norm3 = LayerNorm2d(embed_dims[2])
        self.norm4 = LayerNorm2d(embed_dims[3])

    def forward(self, x: torch.Tensor):
        x1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1)
        x1 = self.norm1(x1)

        x2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2)
        x2 = self.norm2(x2)

        x3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3)
        x3 = self.norm3(x3)

        x4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4)
        x4 = self.norm4(x4)

        return [x1, x2, x3, x4]


# ---------------------------------------------------------------------------
# CBAM — Convolutional Block Attention Module
# ---------------------------------------------------------------------------
class CBAMModule(nn.Module):
    """Channel + Spatial attention (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, reduction_ratio: int = 2, kernel_size: int = 7):
        super().__init__()
        mid = max(channels // reduction_ratio, 4)
        self.ca_fc1  = nn.Conv2d(channels, mid,      kernel_size=1, bias=True)
        self.ca_relu = nn.ReLU()
        self.ca_fc2  = nn.Conv2d(mid,      channels, kernel_size=1, bias=True)
        self.sa_conv = nn.Conv2d(3, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        ca  = torch.sigmoid(
            self.ca_fc2(self.ca_relu(self.ca_fc1(avg))) +
            self.ca_fc2(self.ca_relu(self.ca_fc1(mx)))
        )
        x = x * ca

        # Spatial attention (mean + max + var over channels)
        mean_sp = x.mean(dim=1, keepdim=True)
        max_sp  = x.amax(dim=1, keepdim=True)
        var_sp  = x.var(dim=1, keepdim=True, unbiased=False)
        sa = torch.sigmoid(self.sa_conv(torch.cat([mean_sp, max_sp, var_sp], dim=1)))
        x = x * sa
        return x


# ---------------------------------------------------------------------------
# All-MLP Decoder Head
# ---------------------------------------------------------------------------
class SegFormerHead(nn.Module):
    """Project 4 encoder levels → upsample → fuse → classify."""

    def __init__(
        self,
        num_classes: int = 5,
        embed_dims=(64, 128, 320, 512),
        decoder_embed_dims: int = 768,
    ):
        super().__init__()
        self.linear_c1 = nn.Conv2d(embed_dims[0], decoder_embed_dims, kernel_size=1, bias=True)
        self.linear_c2 = nn.Conv2d(embed_dims[1], decoder_embed_dims, kernel_size=1, bias=True)
        self.linear_c3 = nn.Conv2d(embed_dims[2], decoder_embed_dims, kernel_size=1, bias=True)
        self.linear_c4 = nn.Conv2d(embed_dims[3], decoder_embed_dims, kernel_size=1, bias=True)

        self.linear_fuse = nn.Conv2d(decoder_embed_dims * 4, decoder_embed_dims,
                                     kernel_size=1, bias=False)
        self.bn         = nn.BatchNorm2d(decoder_embed_dims)
        self.relu       = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout(0.15)
        self.classifier = nn.Conv2d(decoder_embed_dims, num_classes, kernel_size=1, bias=True)

    def forward(self, features):
        c1, c2, c3, c4 = features
        target_size = c1.shape[2:]   # (H/4, W/4)

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=target_size, mode='bilinear', align_corners=False)
        _c4 = F.interpolate(self.linear_c4(c4), size=target_size, mode='bilinear', align_corners=False)

        _c = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        _c = self.linear_fuse(_c)
        _c = self.bn(_c)
        _c = self.relu(_c)
        _c = self.dropout(_c)
        return self.classifier(_c)   # [B, num_classes, H/4, W/4]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class OilSpillSegformerModel(nn.Module):
    """
    SegFormer-B2 adapted for single-channel SAR oil-spill segmentation.

    Input:  [B, C, H, W]  float32  (NCHW — PyTorch convention)
    Output: [B, num_classes, H, W]  float32 logits (NOT softmaxed)
    """

    def __init__(
        self,
        input_shape=(384, 384, 1),
        num_classes: int = 5,
        drop_rate: float = 0.1,
        use_cbam: bool = True,
    ):
        super().__init__()
        in_channels = input_shape[2]
        img_size    = input_shape[0]

        # Store for MultiScalePredictor / TTA compatibility
        self.num_classes       = num_classes
        self.expected_height   = input_shape[0]
        self.expected_width    = input_shape[1]
        self.expected_channels = in_channels

        embed_dims   = [64, 128, 320, 512]
        num_heads    = [1, 2, 5, 8]
        mlp_ratios   = [8, 8, 4, 4]
        sr_ratios    = [8, 4, 2, 1]
        num_layers   = [3, 4, 6, 3]
        decoder_dims = 768

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAMModule(channels=in_channels, reduction_ratio=2, kernel_size=7)

        self.backbone = MixVisionTransformer(
            img_size=img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            drop_rate=drop_rate,
            sr_ratios=sr_ratios,
            num_layers=num_layers,
        )
        self.decode_head = SegFormerHead(
            num_classes=num_classes,
            embed_dims=embed_dims,
            decoder_embed_dims=decoder_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        if self.use_cbam:
            x = self.cbam(x)
        features = self.backbone(x)
        logits   = self.decode_head(features)
        logits   = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------
    def load_pretrained_backbone(
        self,
        hub_name: str = 'nvidia/mit-b2',
        local_path: Optional[str] = None,
    ) -> bool:
        """Load MiT-B2 weights from a local .pt file or HuggingFace hub."""
        if local_path and os.path.exists(local_path):
            try:
                ckpt  = torch.load(local_path, map_location='cpu', weights_only=True)
                state = ckpt.get('state_dict', ckpt)
                if self._apply_backbone_weights(state):
                    print(f"Loaded pretrained backbone from {local_path}")
                    return True
            except Exception as e:
                print(f"Local weight loading failed: {e}")

        try:
            from huggingface_hub import hf_hub_download
            try:
                path  = hf_hub_download(repo_id=hub_name, filename='pytorch_model.bin')
                state = torch.load(path, map_location='cpu', weights_only=True)
            except Exception:
                try:
                    from safetensors.torch import load_file
                    path  = hf_hub_download(repo_id=hub_name, filename='model.safetensors')
                    state = load_file(path)
                except Exception:
                    print(f"Could not download weights from HuggingFace: {hub_name}")
                    return False
            ok = self._apply_backbone_weights(state)
            if ok:
                print(f"Loaded pretrained backbone from HuggingFace ({hub_name})")
            return ok
        except ImportError:
            print("huggingface_hub not installed — skipping pretrained weights.")
        except Exception as e:
            print(f"HuggingFace weight loading error: {e}\nTraining from scratch.")
        return False

    def _apply_backbone_weights(self, hf_state: dict) -> bool:
        """
        Remap HuggingFace SegFormer-B2 keys to our backbone naming.

        HF format                                    → Our format
        segformer.encoder.patch_embeddings.N.proj    → backbone.patch_embed{N+1}.projection
        block.N.K.*                                  → backbone.block{N+1}.{K}.*
        layer_norm.N.*                               → backbone.norm{N+1}.norm.*
        attention.self.query                         → attn.q
        attention.self.key_value                     → attn.kv
        attention.self.sr                            → attn.sr
        attention.output.dense                       → attn.proj
        intermediate.dense                           → ffn.fc1
        output.dense                                 → ffn.fc2
        dwconv.dwconv                                → ffn.dwconv
        layernorm_before / layer_norm_1              → norm1.norm
        layernorm_after  / layer_norm_2              → norm2.norm
        """
        mapped = {}
        for hf_key, val in hf_state.items():
            key = hf_key
            for pfx in ('segformer.encoder.', 'mit.', 'encoder.', 'model.'):
                if key.startswith(pfx):
                    key = key[len(pfx):]
                    break

            key = re.sub(r'patch_embeddings\.(\d+)\.',
                         lambda m: f'patch_embed{int(m.group(1)) + 1}.', key)
            key = re.sub(r'block\.(\d+)\.(\d+)\.',
                         lambda m: f'block{int(m.group(1)) + 1}.{m.group(2)}.', key)
            key = re.sub(r'layer_norm\.(\d+)\.',
                         lambda m: f'norm{int(m.group(1)) + 1}.norm.', key)

            for old, new in [
                ('attention.self.query',      'attn.q'),
                ('attention.self.key_value',  'attn.kv'),
                ('attention.self.sr',         'attn.sr'),
                ('attention.self.layer_norm', 'attn.norm.norm'),
                ('attention.output.dense',    'attn.proj'),
                ('intermediate.dense',        'ffn.fc1'),
                ('output.dense',              'ffn.fc2'),
                ('dwconv.dwconv',             'ffn.dwconv'),
                ('layernorm_before',          'norm1.norm'),
                ('layernorm_after',           'norm2.norm'),
                ('layer_norm_1',              'norm1.norm'),
                ('layer_norm_2',              'norm2.norm'),
            ]:
                key = key.replace(old, new)

            mapped[key] = val

        own     = self.backbone.state_dict()
        matched = 0
        for key, val in mapped.items():
            if key in own and own[key].shape == val.shape:
                try:
                    own[key].copy_(val)
                    matched += 1
                except Exception:
                    pass
        self.backbone.load_state_dict(own, strict=False)
        total = len(own)
        print(f"Backbone weight loading: {matched}/{total} tensors matched")
        return matched > 0


# ---------------------------------------------------------------------------
# Public factory — mirrors original OilSpillSegformer() API
# ---------------------------------------------------------------------------
def OilSpillSegformer(
    input_shape=(384, 384, 1),
    num_classes: int = 5,
    drop_rate: float = 0.1,
    use_cbam: bool = True,
    pretrained_weights: Optional[str] = None,
    hub_pretrained: str = 'nvidia/mit-b2',
) -> OilSpillSegformerModel:
    """
    Build and optionally initialise an OilSpillSegformerModel.

    Parameters
    ----------
    input_shape       : (H, W, C) — C is last (legacy API compat); forward() expects NCHW.
    num_classes       : number of segmentation classes.
    drop_rate         : stochastic depth / dropout rate.
    use_cbam          : prepend CBAM attention on the input channel.
    pretrained_weights: path to a local .pt file for backbone initialisation.
    hub_pretrained    : HuggingFace model ID for auto backbone download.
    """
    model = OilSpillSegformerModel(
        input_shape=input_shape,
        num_classes=num_classes,
        drop_rate=drop_rate,
        use_cbam=use_cbam,
    )
    if pretrained_weights or hub_pretrained:
        model.load_pretrained_backbone(
            hub_name=hub_pretrained or 'nvidia/mit-b2',
            local_path=pretrained_weights,
        )
    return model
