"""
Loss functions for oil spill segmentation — PyTorch implementation.

Key design points:
- All losses operate on NCHW logits [B, C, H, W] and long targets [B, H, W].
- HybridSegmentationLoss is an nn.Module; standalone functions kept for compat.
- Boundary Sobel kernels registered as buffers (auto device placement).
- Focal modulator applied to UNWEIGHTED CE, class weights applied after (corrected).
- Lovász-Softmax available as standalone function (directly optimises IoU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standalone losses (kept for backward compatibility)
# ---------------------------------------------------------------------------
def weighted_cross_entropy(class_weights=None, from_logits=True, epsilon=1e-5):
    """Return a callable weighted cross-entropy loss (standalone, not nn.Module)."""
    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    def loss(logits, targets):
        # logits:  [B, C, H, W]
        # targets: [B, H, W] long  —or—  [B, 1, H, W] uint8
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()

        B, C, H, W = logits.shape
        device = logits.device

        if cw_tensor is not None:
            weight = cw_tensor.to(device)
        else:
            weight = None

        probs = F.softmax(logits, dim=1).clamp(epsilon, 1 - epsilon)
        y_oh  = F.one_hot(targets, C).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        smooth = 0.1
        y_sm  = y_oh * (1 - smooth) + smooth / C
        ce    = -(y_sm * probs.log()).sum(dim=1)  # [B,H,W]

        if weight is not None:
            pw = (y_oh * weight.view(1, C, 1, 1)).sum(dim=1)
            ce = ce * pw

        assert torch.isfinite(ce).all(), "NaN/Inf in weighted_cross_entropy"
        return ce.mean()

    return loss


def focal_loss(class_weights=None, gamma=2.0, from_logits=True, epsilon=1e-5):
    """Return a callable focal loss (standalone)."""
    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    def loss(logits, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()

        B, C, H, W = logits.shape
        device = logits.device

        probs = F.softmax(logits, dim=1).clamp(epsilon, 1 - epsilon)
        y_oh  = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        ce    = -(y_oh * probs.log()).sum(dim=1)  # unweighted CE
        pt    = (y_oh * probs).sum(dim=1)
        focal = ce * (1 - pt).pow(gamma)

        if cw_tensor is not None:
            pw = (y_oh * cw_tensor.to(device).view(1, C, 1, 1)).sum(dim=1)
            focal = focal * pw

        assert torch.isfinite(focal).all(), "NaN/Inf in focal_loss"
        return focal.mean()

    return loss


def dice_loss(class_weights=None, from_logits=True, epsilon=1e-5):
    """Return a callable Dice loss (standalone)."""
    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    def loss(logits, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()

        B, C, H, W = logits.shape
        device = logits.device

        probs = F.softmax(logits, dim=1).clamp(epsilon, 1 - epsilon)
        y_oh  = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        y_flat = y_oh.view(B, C, -1)
        p_flat = probs.view(B, C, -1)
        inter  = (y_flat * p_flat).sum(2)
        sum_y  = y_flat.sum(2)
        sum_p  = p_flat.sum(2)
        dice   = (2 * inter + epsilon) / (sum_y + sum_p + epsilon)  # [B, C]

        if cw_tensor is not None:
            cw = cw_tensor.to(device)
            d  = 1 - (dice * cw.unsqueeze(0)).sum() / (cw.sum() * B)
        else:
            d = 1 - dice.mean()

        assert torch.isfinite(d), "NaN/Inf in dice_loss"
        return d

    return loss


def boundary_loss(class_weights=None, from_logits=False, epsilon=1e-5, boundary_weight=2.0):
    """Return a callable boundary loss using Sobel-based edge detection (standalone)."""
    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32).reshape(1, 1, 3, 3)

    def loss(logits, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()

        B, C, H, W = logits.shape
        device = logits.device
        sx = sobel_x.to(device)
        sy = sobel_y.to(device)

        probs = F.softmax(logits, dim=1) if from_logits else logits
        probs = probs.clamp(epsilon, 1 - epsilon)
        y_oh  = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        y_flat = y_oh.view(B * C, 1, H, W)
        p_flat = probs.view(B * C, 1, H, W)

        ey_y = F.conv2d(y_flat, sx, padding=1).abs() + F.conv2d(y_flat, sy, padding=1).abs()
        ey_p = F.conv2d(p_flat, sx, padding=1).abs() + F.conv2d(p_flat, sy, padding=1).abs()

        edges_y = ey_y.view(B, C, H, W)
        edges_p = ey_p.view(B, C, H, W)

        edge_mag   = edges_y.sum(1, keepdim=True).clamp(0, 1)
        rare_mask  = ((targets == 1) | (targets == 3)).float().unsqueeze(1)
        bound_mask = edge_mag * (1 + rare_mask * (boundary_weight - 1))

        bnd = ((edges_y - edges_p).pow(2) * bound_mask).mean()
        if cw_tensor is not None:
            pw  = (y_oh * cw_tensor.to(device).view(1, C, 1, 1)).sum(1).mean()
            bnd = bnd * pw

        assert torch.isfinite(bnd), "NaN/Inf in boundary_loss"
        return bnd

    return loss


# ---------------------------------------------------------------------------
# Lovász-Softmax  (directly optimises mean IoU)
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovász extension w.r.t. sorted errors."""
    p          = gt_sorted.shape[0]
    gts        = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union        = gts + torch.arange(1, p + 1, dtype=torch.float32, device=gt_sorted.device) - gt_sorted.cumsum(0)
    jaccard      = 1.0 - intersection / union
    jaccard      = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Multi-class Lovász-Softmax loss.

    Parameters
    ----------
    probas : (P, C) float  — per-pixel predicted probabilities
    labels : (P,)   long   — ground-truth class indices
    """
    losses = []
    for c in range(num_classes):
        fg     = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors        = (fg - probas[:, c]).abs()
        errors_sorted, perm = errors.sort(descending=True)
        fg_sorted     = fg[perm]
        grad          = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad.detach()))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(from_logits: bool = True, num_classes: int = 5):
    """Return a callable Lovász-Softmax loss."""

    def loss(logits, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()

        if from_logits:
            probas = F.softmax(logits, dim=1)
        else:
            probas = logits

        B, C, H, W = probas.shape
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = targets.reshape(-1)
        return lovasz_softmax_flat(probas_flat, labels_flat, num_classes)

    return loss


# ---------------------------------------------------------------------------
# Hybrid Segmentation Loss  (main loss used in training)
# ---------------------------------------------------------------------------
class HybridSegmentationLoss(nn.Module):
    """
    Combined segmentation loss: CE + Focal + Dice + Boundary (+ optional Lovász).

    Changes from original TF version
    ---------------------------------
    1. Label smoothing applied to targets before all components.
    2. Focal modulator on UNWEIGHTED CE; class weights applied after (bug-fix).
    3. Boundary Sobel kernels registered as buffers for correct device placement.
    4. Pure PyTorch — no TF dependency.
    5. Input format: logits [B, C, H, W], targets [B, H, W] long.
    """

    def __init__(
        self,
        class_weights=None,
        ce_weight: float = 0.35,
        focal_weight: float = 0.25,
        dice_weight: float = 0.3,
        boundary_weight: float = 0.1,
        lovasz_weight: float = 0.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        boundary_boost: float = 2.5,
        epsilon: float = 1e-7,
        from_logits: bool = True,
    ):
        super().__init__()
        self.ce_weight       = ce_weight
        self.focal_weight    = focal_weight
        self.dice_weight     = dice_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight   = lovasz_weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.boundary_boost  = boundary_boost
        self.epsilon         = epsilon
        self.from_logits     = from_logits

        if class_weights is not None:
            self.register_buffer(
                'class_weights',
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        # Sobel kernels for boundary loss — registered as buffers so they
        # automatically move to the correct device with .to(device).
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Pre-build Lovász closure once — avoids recreating it every forward pass
        self._lov_fn = lovasz_softmax_loss(from_logits=False)

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, C, H, W]  float  — raw logits (before softmax)
        targets : [B, H, W]     long   — class indices 0 … C-1
        """
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.long()

        B, C, H, W = logits.shape

        if self.from_logits:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits
        probs = probs.clamp(self.epsilon, 1 - self.epsilon)

        # One-hot  [B, C, H, W]
        y_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()

        # Label smoothing
        if self.label_smoothing > 0:
            y_sm = y_oh * (1 - self.label_smoothing) + self.label_smoothing / C
        else:
            y_sm = y_oh

        # ---- Unweighted CE (for focal) -----------------------------------
        unweighted_ce = -(y_sm * probs.log()).sum(dim=1)  # [B, H, W]

        # ---- Class-weighted CE ------------------------------------------
        if self.class_weights is not None:
            pw         = (y_oh * self.class_weights.view(1, C, 1, 1)).sum(dim=1)
            weighted_ce = unweighted_ce * pw
        else:
            weighted_ce = unweighted_ce
            pw          = None

        # ---- Focal -------------------------------------------------------
        if self.focal_weight > 0:
            pt       = (y_oh * probs).sum(dim=1)         # true-class prob
            focal_mod = (1 - pt).pow(self.gamma)
            focal    = unweighted_ce * focal_mod
            if pw is not None:
                focal = focal * pw
        else:
            focal = torch.zeros(1, device=logits.device)

        # ---- Dice --------------------------------------------------------
        if self.dice_weight > 0:
            y_flat    = y_oh.view(B, C, -1)
            p_flat    = probs.view(B, C, -1)
            inter     = (y_flat * p_flat).sum(2)
            sum_y     = y_flat.sum(2)
            sum_p     = p_flat.sum(2)
            dice      = (2 * inter + self.epsilon) / (sum_y + sum_p + self.epsilon)  # [B, C]
            if self.class_weights is not None:
                cw        = self.class_weights
                dice_loss = 1 - (dice * cw.unsqueeze(0)).sum() / (cw.sum() * B)
            else:
                dice_loss = 1 - dice.mean()
        else:
            dice_loss = torch.zeros(1, device=logits.device)

        # ---- Boundary ----------------------------------------------------
        if self.boundary_weight > 0:
            bnd_loss = self._boundary(y_oh, probs, targets)
        else:
            bnd_loss = torch.zeros(1, device=logits.device)

        # ---- Lovász-Softmax (optional) -----------------------------------
        if self.lovasz_weight > 0:
            lov_loss = self._lov_fn(probs, targets)
        else:
            lov_loss = torch.zeros(1, device=logits.device)

        # ---- Combine -----------------------------------------------------
        total = (
            self.ce_weight       * weighted_ce.mean()
            + self.focal_weight  * focal.mean()
            + self.dice_weight   * dice_loss
            + self.boundary_weight * bnd_loss
            + self.lovasz_weight * lov_loss
        )
        return total

    def _boundary(
        self,
        y_oh: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = y_oh.shape
        y_flat = y_oh.reshape(B * C, 1, H, W)
        p_flat = probs.reshape(B * C, 1, H, W)

        sx = self.sobel_x  # type: ignore[assignment]
        sy = self.sobel_y  # type: ignore[assignment]
        assert isinstance(sx, torch.Tensor) and isinstance(sy, torch.Tensor)
        ey_y = (F.conv2d(y_flat, sx, padding=1).abs() +
                F.conv2d(y_flat, sy, padding=1).abs())
        ey_p = (F.conv2d(p_flat, sx, padding=1).abs() +
                F.conv2d(p_flat, sy, padding=1).abs())

        edges_y = ey_y.view(B, C, H, W)
        edges_p = ey_p.view(B, C, H, W)

        edge_mag   = edges_y.sum(1, keepdim=True).clamp(0, 1)
        rare_mask  = ((targets == 1) | (targets == 3)).float().unsqueeze(1)
        bound_mask = edge_mag * (1 + rare_mask * (self.boundary_boost - 1))

        # Weight boundary loss per-pixel by class weight BEFORE taking the mean
        if self.class_weights is not None:
            per_pixel_weight = (y_oh * self.class_weights.view(1, C, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]
            bnd = ((edges_y - edges_p).pow(2) * bound_mask * per_pixel_weight).mean()
        else:
            bnd = ((edges_y - edges_p).pow(2) * bound_mask).mean()

        return bnd
