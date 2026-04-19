"""
utils.py
--------
Data loading, splitting, loss, and activation utilities.
All tensor operations are explicit — no autograd.
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset(path: str, device: torch.device = torch.device('cpu')):
    """
    Load a .pth dataset file and return feature matrix and label vector.

    The file must contain a dict with keys 'features' and 'labels'.

    Args:
        path:   Path to the .pth data file.
        device: Device to place the returned tensors on.

    Returns:
        x: Feature matrix  (N, D)  — float32
        y: Label column    (N, 1)  — float32
    """
    data = torch.load(path, map_location='cpu')
    x = data['features'].float()
    y = data['labels'].float().view(-1, 1)
    print(f"[data]  loaded  x={tuple(x.shape)}  y={tuple(y.shape)}")
    return x.to(device), y.to(device)


def make_synthetic_dataset(n: int = 2000, d: int = 256,
                           device: torch.device = torch.device('cpu'),
                           seed: int = 0):
    """
    Generate a synthetic regression dataset for smoke-testing without real data.

    The label is a sparse linear combination of the first 10 features plus
    Gaussian noise, giving a learnable but non-trivial signal.

    Args:
        n:      Number of samples.
        d:      Input feature dimension.
        device: Target device.
        seed:   Random seed for reproducibility.

    Returns:
        x: (n, d)  float32
        y: (n, 1)  float32
    """
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    w_true = torch.zeros(d)
    w_true[:10] = torch.randn(10)
    y = (x @ w_true).unsqueeze(1) + 0.1 * torch.randn(n, 1)
    print(f"[data]  synthetic  x={tuple(x.shape)}  y={tuple(y.shape)}")
    return x.to(device), y.to(device)


def split_dataset(x: torch.Tensor, y: torch.Tensor,
                  train_ratio: float = 0.7, seed: int = 42):
    """
    Randomly split (x, y) into train and test sets.

    Args:
        x:           Feature matrix (N, D).
        y:           Label matrix   (N, 1).
        train_ratio: Fraction of samples for training.
        seed:        Manual seed for the permutation.

    Returns:
        x_train, y_train, x_test, y_test
    """
    torch.manual_seed(seed)
    n = x.shape[0]
    idx = torch.randperm(n)
    split = int(n * train_ratio)

    x_train, y_train = x[idx[:split]], y[idx[:split]]
    x_test,  y_test  = x[idx[split:]], y[idx[split:]]

    print(f"[split] train={x_train.shape[0]}  test={x_test.shape[0]}")
    return x_train, y_train, x_test, y_test


# ── Loss & activation ─────────────────────────────────────────────────────────

def mse_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error: (1/N) Σ (ŷᵢ − yᵢ)²"""
    return torch.mean((y_hat - y) ** 2)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Rectified Linear Unit: max(0, x)"""
    return torch.clamp(x, min=0.0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_loss_curves(train_history: list, test_history: list,
                     save_path: str = 'figures/loss_curves.png'):
    """
    Save a publication-quality loss curve plot to disk.

    Args:
        train_history: Per-epoch training MSE (all epochs).
        test_history:  Per-epoch test MSE; None for epochs where it was skipped.
        save_path:     Output file path.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    test_epochs  = [i for i, v in enumerate(test_history) if v is not None]
    test_values  = [v for v in test_history if v is not None]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_history, color='#2563EB', linewidth=1.4,
            alpha=0.85, label='Training loss')
    ax.plot(test_epochs, test_values, 'o--', color='#DC2626',
            linewidth=1.6, markersize=4, alpha=0.95, label='Test loss')

    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE loss (log scale)', fontsize=12)
    ax.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot]  saved → {save_path}")
