"""
PyTorch training loop with dual-head loss, early stopping, and LR scheduling.

The :class:`Trainer` orchestrates the training of any :class:`BaseTorchModel`
whose ``forward`` method returns ``(reg_output, cls_output)``.

Usage::

    from src.training.trainer import Trainer

    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.models.base_model import BaseTorchModel
from src.utils.config import AppConfig, load_config
from src.utils.constants import CHECKPOINTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Manages the dual-head PyTorch training loop.

    Parameters
    ----------
    model:
        A :class:`BaseTorchModel` instance whose inner ``nn.Module`` has
        been initialised and moved to the target device.
    config:
        Full application configuration (``AppConfig``).  When *None* the
        default configuration is loaded from disk.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        model: BaseTorchModel,
        config: Optional[AppConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config or load_config()
        self.device = torch.device(device)
        self.best_val_loss: float = float("inf")
        self._checkpoint_path: Optional[Path] = None

        logger.info(
            "Trainer initialised — model=%s, device=%s, params=%d",
            self.model.name,
            self.device,
            self.model.num_parameters,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: float = 1e-5,
        scheduler_type: str = "plateau",
        checkpoint_dir: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader:
            Training ``DataLoader`` yielding ``(x, y_reg, y_cls)`` tuples.
        val_loader:
            Validation ``DataLoader``.
        max_epochs:
            Maximum number of epochs.  Falls back to ``config.model.common.max_epochs``.
        patience:
            Early-stopping patience.  Falls back to ``config.model.common.patience``.
        lr:
            Initial learning rate.  Falls back to ``config.model.common.learning_rate``.
        weight_decay:
            L2 regularisation coefficient for Adam.
        scheduler_type:
            ``"plateau"`` for :class:`ReduceLROnPlateau` or ``"cosine"``
            for :class:`CosineAnnealingLR`.
        checkpoint_dir:
            Directory for saving the best model checkpoint.  Defaults to
            ``results/checkpoints/``.

        Returns
        -------
        dict[str, Any]
            Training history with keys ``train_loss``, ``val_loss``,
            ``learning_rate``, ``best_epoch``, ``best_val_loss``.
        """
        cfg = self.config.model
        max_epochs = max_epochs or cfg.common.max_epochs
        patience = patience or cfg.common.patience
        lr = lr or cfg.common.learning_rate
        alpha: float = cfg.dual_head.alpha

        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINTS_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = checkpoint_dir / f"{self.model.name}_best.pt"

        # Loss functions
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.BCELoss()

        # Optimiser
        assert self.model.model is not None, "Inner nn.Module must be initialised before training."
        optimizer = Adam(
            self.model.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler
        scheduler: ReduceLROnPlateau | CosineAnnealingLR
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=max(1, patience // 3),
            )

        # History tracking
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
        best_epoch = 0
        epochs_without_improvement = 0

        logger.info(
            "Starting training — max_epochs=%d, patience=%d, lr=%.6f, "
            "alpha=%.2f, scheduler=%s",
            max_epochs,
            patience,
            lr,
            alpha,
            scheduler_type,
        )

        for epoch in range(1, max_epochs + 1):
            # --- Train ---
            train_loss = self._train_one_epoch(
                train_loader, optimizer, criterion_reg, criterion_cls, alpha,
            )

            # --- Validate ---
            val_loss = self._validate(val_loader, criterion_reg, criterion_cls, alpha)

            # --- LR scheduler step ---
            current_lr = optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # --- Record history ---
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)

            logger.info(
                "Epoch %03d/%03d — train_loss: %.6f | val_loss: %.6f | lr: %.2e",
                epoch,
                max_epochs,
                train_loss,
                val_loss,
                current_lr,
            )

            # --- Checkpoint & early stopping ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self.model.save(self._checkpoint_path)
                logger.info(
                    "  -> New best model saved (val_loss=%.6f)", val_loss,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(no improvement for %d epochs).",
                        epoch,
                        patience,
                    )
                    break

        # Restore best weights
        if self._checkpoint_path and self._checkpoint_path.exists():
            self.model.load(self._checkpoint_path)
            logger.info("Restored best model weights from epoch %d.", best_epoch)

        result: dict[str, Any] = {
            **history,
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(history["train_loss"]),
        }
        logger.info(
            "Training complete — best_epoch=%d, best_val_loss=%.6f",
            best_epoch,
            self.best_val_loss,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Adam,
        criterion_reg: nn.MSELoss,
        criterion_cls: nn.BCELoss,
        alpha: float,
    ) -> float:
        """Run a single training epoch.

        Parameters
        ----------
        train_loader : Training data loader.
        optimizer : Adam optimiser.
        criterion_reg : MSE loss for the regression head.
        criterion_cls : BCE loss for the classification head.
        alpha : Weight for the regression loss component.

        Returns
        -------
        float
            Average combined loss over all batches.
        """
        assert self.model.model is not None
        self.model.model.train()

        total_loss = 0.0
        n_batches = 0

        for x_batch, y_reg_batch, y_cls_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_reg_batch = y_reg_batch.to(self.device)
            y_cls_batch = y_cls_batch.to(self.device)

            optimizer.zero_grad()

            reg_out, cls_out = self.model.model(x_batch)

            loss_reg = criterion_reg(reg_out, y_reg_batch)
            loss_cls = criterion_cls(cls_out, y_cls_batch)
            loss = alpha * loss_reg + (1.0 - alpha) * loss_cls

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def _validate(
        self,
        val_loader: DataLoader,
        criterion_reg: nn.MSELoss,
        criterion_cls: nn.BCELoss,
        alpha: float,
    ) -> float:
        """Run a validation pass (no gradient computation).

        Parameters
        ----------
        val_loader : Validation data loader.
        criterion_reg : MSE loss for the regression head.
        criterion_cls : BCE loss for the classification head.
        alpha : Weight for the regression loss component.

        Returns
        -------
        float
            Average combined loss over all validation batches.
        """
        assert self.model.model is not None
        self.model.model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_reg_batch, y_cls_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_reg_batch = y_reg_batch.to(self.device)
                y_cls_batch = y_cls_batch.to(self.device)

                reg_out, cls_out = self.model.model(x_batch)

                loss_reg = criterion_reg(reg_out, y_reg_batch)
                loss_cls = criterion_cls(cls_out, y_cls_batch)
                loss = alpha * loss_reg + (1.0 - alpha) * loss_cls

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
