"""
1D-CNN for fall detection on raw windowed IMU signals.

Architecture follows Santos et al. (2019) and Ordóñez & Roggen (2016):
  Input: (batch, channels, window_len)   [PyTorch channels-first convention]
  3 × Conv1D blocks (Conv → BN → ReLU → MaxPool)
  2 × FC layers → sigmoid output

Can be extended to CNN-LSTM by replacing the FC head with an LSTM stack.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    nn    = None  # type: ignore


class CNN1D(nn.Module if _HAS_TORCH else object):
    """
    Parameters
    ----------
    n_channels  : number of IMU channels (3 for acc-only, 6 for acc+gyro)
    window_len  : number of time steps per window
    n_classes   : 2 for binary fall detection
    hidden_dim  : size of fully-connected hidden layer
    """

    def __init__(
        self,
        n_channels:  int = 3,
        window_len:  int = 128,
        n_classes:   int = 2,
        hidden_dim:  int = 128,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for CNN1D. Install with: pip install torch")
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Compute flattened size after conv blocks
        conv_out_len = window_len // 8  # 3 MaxPool(2) layers
        flat_dim = 128 * conv_out_len

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        """x: (B, C, T)"""
        x = self.conv_blocks(x)
        return self.classifier(x)


def train_cnn(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
) -> list[float]:
    """
    Train CNN1D and return per-epoch training losses.

    X_train : (N, window_len, n_channels)  — will be transposed to (N, C, T)
    y_train : (N,) int
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required.")

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transpose to (N, C, T) for Conv1D
    X_t = torch.tensor(X_train.transpose(0, 2, 1), dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    # Class imbalance — inverse-frequency weights
    n_pos = int(y_t.sum().item())
    n_neg = len(y_t) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)

    dataset   = TensorDataset(X_t, y_t)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device).train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(dataset))
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={losses[-1]:.4f}")

    return losses


def predict_cnn(
    model,
    X: np.ndarray,
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Return predicted class labels (0/1) for X."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch required.")

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t  = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32).to(device)
    ds   = TensorDataset(X_t)
    dl   = DataLoader(ds, batch_size=batch_size)

    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds)
