"""
PyTorch Dataset and DataLoader factories for time-series crypto data.

Usage::

    from src.data.dataset import CryptoDataset, create_dataloaders

    train_loader, val_loader = create_dataloaders(
        X_train, y_train_reg, y_train_cls,
        X_val, y_val_reg, y_val_cls,
        batch_size=64,
    )
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CryptoDataset(Dataset):
    """A PyTorch Dataset that wraps pre-built time-series sequences.

    Each sample is a tuple of:

    * ``x``     -- input feature sequence, shape ``(lookback, n_features)``
    * ``y_reg`` -- regression target (future close prices), shape ``(horizon,)``
    * ``y_cls`` -- classification target (direction labels), shape ``(horizon,)``

    Parameters
    ----------
    X:
        Input sequences, shape ``(n_samples, lookback, n_features)``.
    y_reg:
        Regression targets, shape ``(n_samples, horizon)``.
    y_cls:
        Classification targets, shape ``(n_samples, horizon)``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
    ) -> None:
        super().__init__()

        if len(X) != len(y_reg) or len(X) != len(y_cls):
            raise ValueError(
                f"Length mismatch: X={len(X)}, y_reg={len(y_reg)}, y_cls={len(y_cls)}."
            )

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.float32)

        logger.debug(
            "CryptoDataset created: %d samples, lookback=%d, features=%d.",
            len(self.X),
            self.X.shape[1],
            self.X.shape[2],
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


# ---------------------------------------------------------------------------
# DataLoader factory helpers
# ---------------------------------------------------------------------------

def create_dataloaders(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_cls: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders.

    Parameters
    ----------
    X_train, y_train_reg, y_train_cls:
        Training-set arrays produced by
        :meth:`DataPreprocessor.create_sequences`.
    X_val, y_val_reg, y_val_cls:
        Validation-set arrays.
    batch_size:
        Mini-batch size.
    num_workers:
        Number of sub-processes for data loading.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``.  The training loader shuffles
        samples (safe because sequences are already constructed);
        the validation loader does not.
    """
    train_ds = CryptoDataset(X_train, y_train_reg, y_train_cls)
    val_ds = CryptoDataset(X_val, y_val_reg, y_val_cls)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    logger.info(
        "DataLoaders created – train: %d batches (%d samples), "
        "val: %d batches (%d samples), batch_size=%d.",
        len(train_loader),
        len(train_ds),
        len(val_loader),
        len(val_ds),
        batch_size,
    )
    return train_loader, val_loader


def create_test_loader(
    X_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """Build a test DataLoader (no shuffling).

    Parameters
    ----------
    X_test, y_test_reg, y_test_cls:
        Test-set arrays produced by
        :meth:`DataPreprocessor.create_sequences`.
    batch_size:
        Mini-batch size.
    num_workers:
        Number of sub-processes for data loading.

    Returns
    -------
    DataLoader
        A DataLoader that iterates the test set in order.
    """
    test_ds = CryptoDataset(X_test, y_test_reg, y_test_cls)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    logger.info(
        "Test DataLoader created – %d batches (%d samples), batch_size=%d.",
        len(test_loader),
        len(test_ds),
        batch_size,
    )
    return test_loader
