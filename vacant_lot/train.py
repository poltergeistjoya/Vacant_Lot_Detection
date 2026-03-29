"""Segmentation training loop with checkpointing and TensorBoard logging.

Provides :class:`SegmentationTrainer` which orchestrates model training,
validation, gradient accumulation, learning-rate scheduling, early stopping,
and checkpoint management.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .segmentation import (
    EarlyStopping,
    SegmentationLoss,
    build_model,
    compute_metrics,
)
from .logger import get_logger

log = get_logger()


def _auto_device() -> torch.device:
    """Detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SegmentationTrainer:
    """Train and validate a segmentation model.

    The trainer builds the model, optimizer, scheduler, and loss from explicit
    config objects.  It does **not** create :class:`DataLoader`\\s — those are
    passed in so that data-loading concerns stay in the training script.

    Args:
        model_cfg: :class:`DLModelConfig` with architecture and encoder params.
        training_cfg: :class:`DLTrainingConfig` with epochs, lr, etc.
        loss_cfg: :class:`DLLossConfig` with BCE/Dice weights.
        train_loader: Training :class:`DataLoader`.
        val_loader: Validation :class:`DataLoader`.
        checkpoint_dir: Directory for best.pt / latest.pt checkpoints.
        log_dir: Directory for TensorBoard event files.
        device: Torch device.  Auto-detected (MPS > CUDA > CPU) if ``None``.
        run_id: Identifier for this run (used only in log messages).
    """

    def __init__(
        self,
        model_cfg,
        training_cfg,
        loss_cfg,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        log_dir: Path,
        device: torch.device | None = None,
        run_id: str | None = None,
    ):
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.loss_cfg = loss_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.run_id = run_id or f"{model_cfg.type}_{int(time.time())}"

        # Device
        self.device = device or _auto_device()

        # Model
        self.model = build_model(
            arch=model_cfg.type,
            in_channels=model_cfg.in_channels,
            encoder_name=model_cfg.encoder_name,
            encoder_weights=model_cfg.encoder_weights,
            classes=model_cfg.classes,
            decoder_channels=model_cfg.decoder_channels if hasattr(model_cfg, "decoder_channels") else None,
        ).to(self.device)

        # Loss
        self.criterion = SegmentationLoss(
            pos_weight=loss_cfg.pos_weight,
            bce_weight=loss_cfg.bce_weight,
            dice_weight=loss_cfg.dice_weight,
            soft_positive_weight=loss_cfg.soft_positive_weight,
            soft_positive_target=loss_cfg.soft_positive_target,
        )

        # Optimizer & scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=training_cfg.max_epochs,
            eta_min=1e-6,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=training_cfg.patience, mode="max")

        # Gradient accumulation
        self.accumulation_steps = training_cfg.accumulation_steps

        # State
        self.start_epoch = 0
        self.best_iou = 0.0
        self.writer: SummaryWriter | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict with ``train_loss``.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Train", leave=True)
        for i, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks) / self.accumulation_steps
            loss.backward()

            # Step optimizer every accumulation_steps or on last batch
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.accumulation_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        self.scheduler.step()

        return {"train_loss": running_loss / max(n_batches, 1)}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, epoch: int) -> dict[str, float]:
        """Evaluate on the validation set (micro-averaged metrics).

        Returns:
            Dict with ``val_loss``, ``val_iou``, ``val_dice``,
            ``val_precision``, ``val_recall``, ``val_kappa``.
        """
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        # Micro-averaging: accumulate TP/FP/FN/TN as plain floats (not tensors)
        # to avoid building an implicit computation graph chain across batches.
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_tn = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Val  ", leave=True)
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, masks)
                running_loss += loss.item()
                n_batches += 1

                probs = torch.sigmoid(logits).squeeze(1)
                valid = masks != 255
                pred_bin = (probs > 0.5) & valid
                target_bin = (masks == 1) & valid

                total_tp += (pred_bin & target_bin).sum().item()
                total_fp += (pred_bin & ~target_bin).sum().item()
                total_fn += (~pred_bin & target_bin).sum().item()
                total_tn += (~pred_bin & ~target_bin & valid).sum().item()

                eps = 1e-7
                _iou = total_tp / (total_tp + total_fp + total_fn + eps)
                pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}", iou=f"{_iou:.4f}")

        eps = 1e-7
        tp, fp, fn, tn = total_tp, total_fp, total_fn, total_tn
        iou = tp / (tp + fp + fn + eps)
        dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        total = tp + fp + fn + tn + eps
        po = (tp + tn) / total
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total)
        kappa = (po - pe) / (1.0 - pe + eps)

        return {
            "val_loss": running_loss / max(n_batches, 1),
            "val_iou": iou,
            "val_dice": dice,
            "val_precision": precision,
            "val_recall": recall,
            "val_kappa": kappa,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        """Save model checkpoint.

        Always saves ``latest.pt``.  When *is_best* also saves ``best.pt``.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_iou": self.best_iou,
            "metrics": metrics,
            "model_cfg": self.model_cfg.model_dump(),
            "run_id": self.run_id,
        }

        torch.save(state, self.checkpoint_dir / "latest.pt")

        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")

        del state

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore training state from a checkpoint.

        Args:
            path: Path to a ``.pt`` checkpoint file.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_iou = ckpt.get("best_iou", 0.0)

        log.info(f"Resumed from {path.name} (epoch {ckpt['epoch']}, best IoU {self.best_iou:.4f})")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> tuple[dict[str, float], list[dict]]:
        """Run the full training loop.

        Returns:
            Tuple of (best_metrics, history) where history is a list of
            per-epoch dicts with train_loss, val_loss, val_iou, etc.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        cfg = self.training_cfg
        weights_desc = self.model_cfg.encoder_weights or "random init (no pretrained weights)"
        log.info(
            f"Training {self.model_cfg.type}/{self.model_cfg.encoder_name} "
            f"[encoder_weights={weights_desc}] on {self.device} | "
            f"epochs={cfg.max_epochs}, batch={cfg.batch_size}, "
            f"accum={cfg.accumulation_steps}, lr={cfg.learning_rate}"
        )

        best_metrics: dict[str, float] = {}
        history: list[dict] = []

        for epoch in range(self.start_epoch, cfg.max_epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            all_metrics = {**train_metrics, **val_metrics}
            lr = self.optimizer.param_groups[0]["lr"]

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Metrics/val_iou", val_metrics["val_iou"], epoch)
            self.writer.add_scalar("Metrics/val_dice", val_metrics["val_dice"], epoch)
            self.writer.add_scalar("Metrics/val_precision", val_metrics["val_precision"], epoch)
            self.writer.add_scalar("Metrics/val_recall", val_metrics["val_recall"], epoch)
            self.writer.add_scalar("Metrics/val_kappa", val_metrics["val_kappa"], epoch)
            self.writer.add_scalar("LR", lr, epoch)

            # Best model tracking
            is_best = val_metrics["val_iou"] > self.best_iou
            if is_best:
                self.best_iou = val_metrics["val_iou"]
                best_metrics = val_metrics.copy()
                best_metrics["best_epoch"] = epoch

            self.save_checkpoint(epoch, all_metrics, is_best)

            # Accumulate history
            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "val_iou": val_metrics["val_iou"],
                "val_dice": val_metrics["val_dice"],
                "val_precision": val_metrics["val_precision"],
                "val_recall": val_metrics["val_recall"],
                "val_kappa": val_metrics["val_kappa"],
                "lr": lr,
            })

            elapsed = time.time() - t0
            marker = " *" if is_best else ""
            log.info(
                f"Epoch {epoch:3d}/{cfg.max_epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_iou={val_metrics['val_iou']:.4f}{marker} | "
                f"{elapsed:.1f}s"
            )

            # Clear memory caches to prevent accumulation over epochs.
            # On MPS, synchronize() MUST be called before empty_cache() —
            # without it, pending operations prevent the cache from being freed.
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Log process memory (RSS) to track leaks
            rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports in bytes, Linux in KB
            rss_gb = rss_bytes / (1024 ** 3) if os.uname().sysname == "Darwin" else rss_bytes / (1024 ** 2)
            log.info(f"  Memory: {rss_gb:.2f} GB (RSS)")

            if self.early_stopping(val_metrics["val_iou"]):
                log.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {cfg.patience} epochs)"
                )
                break

        self.writer.close()
        log.info(f"Training complete. Best val IoU: {self.best_iou:.4f}")
        log.info(f"Checkpoints: {self.checkpoint_dir}")
        log.info(f"TensorBoard:  {self.log_dir}")

        return best_metrics, history
