"""
train.py
--------
Training loop and early-stopping logic.

The training loop uses mini-batch SGD with periodic validation.
EarlyStopper monitors the test loss and restores the best-seen
parameters when no improvement is observed for `patience` checks.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import FCNet
from src.utils  import mse_loss


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopper:
    """
    Monitors validation loss and signals when training should stop.

    The best parameters seen so far are preserved in memory and can be
    restored to the model after training terminates.

    Args:
        patience  (int):   Validation checks to wait after last improvement.
        min_delta (float): Minimum absolute improvement to count as progress.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.wait       = 0
        self.best_params: list | None = None

    def step(self, loss: float, params: list) -> bool:
        """
        Record the current validation loss and decide whether to stop.

        Args:
            loss:   Current validation MSE.
            params: List of parameter tensors [W1, b1, W2, b2].

        Returns:
            True if training should stop, False otherwise.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss   = loss
            self.best_params = [p.clone() for p in params]
            self.wait        = 0
            return False

        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model: FCNet) -> None:
        """Copy the best saved parameters back into the model."""
        if self.best_params is not None:
            model.W1.copy_(self.best_params[0])
            model.b1.copy_(self.best_params[1])
            model.W2.copy_(self.best_params[2])
            model.b2.copy_(self.best_params[3])


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model:      FCNet,
          x_train:    torch.Tensor,
          y_train:    torch.Tensor,
          x_test:     torch.Tensor,
          y_test:     torch.Tensor,
          lr:         float = 1e-4,
          epochs:     int   = 2000,
          batch_size: int   = 128,
          patience:   int   = 100,
          eval_every: int   = 10,
          device:     torch.device = torch.device('cpu'),
          verbose:    bool  = True) -> tuple[list, list]:
    """
    Run mini-batch SGD training with early stopping.

    Validation is performed every `eval_every` epochs to reduce overhead.
    The best model parameters are restored if early stopping triggers.

    Args:
        model:      FCNet instance to train.
        x_train:    Training features  (N_train, D).
        y_train:    Training labels    (N_train, 1).
        x_test:     Test features      (N_test,  D).
        y_test:     Test labels        (N_test,  1).
        lr:         SGD learning rate η.
        epochs:     Maximum training epochs.
        batch_size: Mini-batch size.
        patience:   Early-stopping patience (in validation checks).
        eval_every: Evaluate on test set every this many epochs.
        device:     Compute device.
        verbose:    Print progress to stdout.

    Returns:
        train_history: Per-epoch mean training MSE.
        test_history:  Per-epoch test MSE; None for non-evaluated epochs.
    """
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    stopper       = EarlyStopper(patience=patience)
    train_history = []
    test_history  = []

    for epoch in range(epochs):

        # ── Mini-batch SGD ──────────────────────────────────────────
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat  = model.forward(xb, training=True)
            loss   = mse_loss(y_hat, yb)
            grads  = model.backward(y_hat, yb)
            model.step(*grads, lr=lr)
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        train_history.append(train_loss)

        # ── Periodic validation ─────────────────────────────────────
        if epoch % eval_every == 0:
            with torch.no_grad():
                y_hat_test = model.forward(x_test.to(device), training=False)
                test_loss  = mse_loss(y_hat_test, y_test.to(device)).item()
            test_history.append(test_loss)

            if verbose:
                print(f"  epoch {epoch:>5d}  "
                      f"train={train_loss:.6f}  test={test_loss:.6f}")

            params = [model.W1, model.b1, model.W2, model.b2]
            if stopper.step(test_loss, params):
                if verbose:
                    print(f"  [early stop]  "
                          f"best test loss={stopper.best_loss:.6f}  "
                          f"restoring params")
                stopper.restore(model)
                # Pad history so indices align with epoch numbers
                remaining = epochs - epoch - 1
                train_history.extend([None] * remaining)
                test_history.extend( [None] * remaining)
                break
        else:
            test_history.append(None)

    return train_history, test_history
