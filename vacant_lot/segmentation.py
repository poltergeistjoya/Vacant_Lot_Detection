"""Segmentation model definitions, loss functions, and metrics.

Provides U-Net and DeepLabV3+ via segmentation-models-pytorch, a combined
BCE + Dice loss with ignore-mask support, and pixel-level evaluation metrics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .logger import get_logger

log = get_logger()

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

_ARCH_MAP = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(
    arch: str = "unet",
    in_channels: int = 10,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    classes: int = 1,
    decoder_channels: list[int] | None = None,
) -> nn.Module:
    """Build a segmentation model.

    Args:
        arch: Architecture name (``"unet"`` or ``"deeplabv3plus"``).
        in_channels: Number of input channels (4 NAIP + 6 spectral indices).
        encoder_name: Encoder backbone name (any timm-compatible name).
        encoder_weights: Pretrained weights (``"imagenet"`` or ``None``).
        classes: Number of output classes (1 for binary segmentation).
        decoder_channels: UNet decoder channel widths per stage.  Only used
            when ``arch == "unet"``.  Controls model capacity (e.g.
            ``[256,128,64,32,16]`` for base_channels=32,
            ``[512,256,128,64,32]`` for base_channels=64).

    Returns:
        An ``nn.Module`` that maps ``(B, in_channels, H, W)`` →
        ``(B, classes, H, W)`` logits.
    """
    if arch not in _ARCH_MAP:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(_ARCH_MAP)}")

    kwargs: dict = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    if decoder_channels is not None and arch == "unet":
        kwargs["decoder_channels"] = decoder_channels

    model = _ARCH_MAP[arch](**kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Built {arch} ({encoder_name}): {n_params:,} params "
        f"({n_trainable:,} trainable), in_channels={in_channels}"
    )
    return model



# ---------------------------------------------------------------------------
# Lovász-hinge loss (binary)
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the Lovász gradient from sorted ground-truth labels."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász-hinge loss on flattened predictions.

    Args:
        logits: Flat (N,) raw logits (pre-sigmoid).
        labels: Flat (N,) binary ground-truth {0, 1}.
    """
    if len(logits) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0  # +1 for positive, -1 for negative
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """Combined BCE + Dice + Lovász loss with ignore mask.

    Label encoding:
        0 = non-vacant, 1 = vacant, 255 = ignore (buildings/water/roads/boundary).
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        lovasz_weight: float = 0.0,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: ``(B, 1, H, W)`` raw model output (pre-sigmoid).
            targets: ``(B, H, W)`` int64 with values 0, 1, 255.

        Returns:
            Scalar loss tensor.
        """
        logits_2d = logits.squeeze(1)  # (B, H, W)
        valid = targets != 255  # (B, H, W)
        targets_f = targets.float()

        # --- BCE on valid pixels ---
        logits_valid = logits_2d[valid]
        targets_valid = targets_f[valid]
        pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits_valid, targets_valid, pos_weight=pw,
        )

        # --- Dice on valid pixels ---
        probs = torch.sigmoid(logits_2d)
        probs_valid = probs * valid
        targets_valid_f = targets_f * valid
        intersection = (probs_valid * targets_valid_f).sum()
        dice = (2.0 * intersection + 1.0) / (probs_valid.sum() + targets_valid_f.sum() + 1.0)
        dice_loss = 1.0 - dice

        total = self.bce_weight * bce + self.dice_weight * dice_loss

        # --- Lovász-hinge on valid pixels ---
        if self.lovasz_weight > 0:
            lovasz = lovasz_hinge_flat(logits_valid, targets_valid)
            total = total + self.lovasz_weight * lovasz

        return total


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute segmentation metrics over valid (non-255) pixels.

    Args:
        preds: Sigmoid probabilities, ``(B, 1, H, W)`` or ``(B, H, W)``.
        targets: ``(B, H, W)`` int64 with values 0, 1, 255.
        threshold: Binarization threshold for predictions.

    Returns:
        Dict with keys: iou, dice, precision, recall, kappa.
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    valid = targets != 255
    pred_bin = (preds > threshold) & valid
    target_bin = (targets == 1) & valid

    tp = (pred_bin & target_bin).sum().float()
    fp = (pred_bin & ~target_bin).sum().float()
    fn = (~pred_bin & target_bin).sum().float()
    tn = (~pred_bin & ~target_bin & valid).sum().float()

    eps = 1e-7
    iou = (tp / (tp + fp + fn + eps)).item()
    dice = (2.0 * tp / (2.0 * tp + fp + fn + eps)).item()
    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()

    # Cohen's Kappa
    total = tp + fp + fn + tn + eps
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total)
    kappa = ((po - pe) / (1.0 - pe + eps)).item()

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        mode: ``"max"`` (e.g. IoU) or ``"min"`` (e.g. loss).
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def __call__(self, score: float) -> bool:
        """Update with new score. Returns ``True`` if training should stop."""
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
