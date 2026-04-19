"""
model.py
--------
Two-layer fully connected network with manual forward / backward passes.

No autograd is used. All gradients are derived analytically and computed
explicitly. The implementation follows the mathematics in docs/report.pdf.

Architecture
------------
    Input (D=256)  →  Linear  →  ReLU  →  [Dropout]  →  Linear  →  Output (1)

    ŷ = W₂ · ReLU(W₁x + b₁) + b₂
"""

import torch
from src.utils import relu


class FCNet:
    """
    Two-layer fully connected network with manual gradient computation.

    Parameters are stored as plain tensors (requires_grad=False).
    Weight initialisation follows Kaiming uniform scaling, which is
    empirically better suited to ReLU activations than naive Gaussian init.

    Args:
        input_dim    (int):   Input feature dimension D.
        hidden_dim   (int):   Hidden layer width H.
        output_dim   (int):   Output dimension (1 for regression).
        dropout_rate (float): Dropout probability p ∈ [0, 1).
        use_dropout  (bool):  Toggle dropout on/off without changing p.
        device       (str):   'cuda' or 'cpu'.
    """

    def __init__(self,
                 input_dim:    int   = 256,
                 hidden_dim:   int   = 512,
                 output_dim:   int   = 1,
                 dropout_rate: float = 0.0,
                 use_dropout:  bool  = False,
                 device:       str   = 'cpu'):

        self.device       = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = dropout_rate
        self.use_dropout  = use_dropout

        # ── Kaiming uniform initialisation ────────────────────────────────
        # gain = √2 for ReLU; std = gain / √fan_in
        std1 = (2.0 / input_dim)  ** 0.5
        std2 = (2.0 / hidden_dim) ** 0.5

        self.W1 = torch.randn(hidden_dim, input_dim,  device=self.device) * std1
        self.b1 = torch.zeros(hidden_dim,             device=self.device)
        self.W2 = torch.randn(output_dim, hidden_dim, device=self.device) * std2
        self.b2 = torch.zeros(output_dim,             device=self.device)

        for p in [self.W1, self.b1, self.W2, self.b2]:
            p.requires_grad_(False)

        # ── Forward-pass cache (populated during forward, read in backward) ─
        self._X:            torch.Tensor | None = None
        self._Z1:           torch.Tensor | None = None
        self._A1:           torch.Tensor | None = None
        self._dropout_mask: torch.Tensor | None = None

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Compute the network output for a batch X.

        Forward equations:
            Z₁ = X W₁ᵀ + b₁          (pre-activation, layer 1)
            A₁ = ReLU(Z₁)             (hidden activations)
            A₁ = A₁ ⊙ M / (1 − p)   (inverted dropout, optional)
            ŷ  = A₁ W₂ᵀ + b₂         (output layer)

        Args:
            X:        Input batch (B, D).
            training: If True and dropout is enabled, applies dropout mask.

        Returns:
            ŷ: Predictions (B, output_dim).
        """
        Z1 = X @ self.W1.T + self.b1
        A1 = relu(Z1)

        self._X  = X
        self._Z1 = Z1
        self._A1 = A1

        if training and self.use_dropout and self.dropout_rate > 0:
            mask = (torch.rand_like(A1) > self.dropout_rate).float()
            A1   = A1 * mask / (1.0 - self.dropout_rate)
            self._dropout_mask = mask
        else:
            self._dropout_mask = None
            # A1 written above already; keep self._A1 pointing to pre-dropout
            # value so backward can read it correctly even when dropout is off.

        self._A1_post = A1          # post-dropout activations used in backward

        y_hat = A1 @ self.W2.T + self.b2
        return y_hat

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, y_hat: torch.Tensor,
                 y: torch.Tensor) -> tuple:
        """
        Compute gradients analytically (no autograd).

        Backpropagation equations (MSE loss):
            δ_out = (2/B)(ŷ − y)

            ∂L/∂W₂ = δ_out^T A₁
            ∂L/∂b₂ = Σ δ_out

            ∂L/∂A₁ = δ_out W₂               (propagate through output layer)
            ∂L/∂Z₁ = ∂L/∂A₁ ⊙ 𝟙[Z₁ > 0]   (backprop through ReLU)

            ∂L/∂W₁ = (∂L/∂Z₁)^T X
            ∂L/∂b₁ = Σ ∂L/∂Z₁

        Args:
            y_hat: Model output (B, 1).
            y:     Ground truth (B, 1).

        Returns:
            dW1, db1, dW2, db2 — gradients matching parameter shapes.
        """
        B = y.shape[0]

        delta = (2.0 / B) * (y_hat - y)          # (B, 1)

        dW2 = delta.T @ self._A1_post             # (1, H)
        db2 = delta.sum(dim=0)                    # (1,)

        dA1 = delta @ self.W2                     # (B, H)  back through W₂

        # Backprop through optional dropout
        if self.use_dropout and self.dropout_rate > 0 and self._dropout_mask is not None:
            dA1 = dA1 * self._dropout_mask / (1.0 - self.dropout_rate)

        # Backprop through ReLU
        dZ1 = dA1.clone()
        dZ1[self._Z1 <= 0] = 0.0

        dW1 = dZ1.T @ self._X                     # (H, D)
        db1 = dZ1.sum(dim=0)                      # (H,)

        return dW1, db1, dW2, db2

    # ── SGD update ────────────────────────────────────────────────────────────

    def step(self, dW1: torch.Tensor, db1: torch.Tensor,
             dW2: torch.Tensor, db2: torch.Tensor,
             lr: float) -> None:
        """
        Apply a vanilla SGD parameter update:  W ← W − η · ∇L

        Args:
            dW1, db1, dW2, db2: Gradients from backward().
            lr: Learning rate η.
        """
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    # ── Persistence ───────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return a dict of parameter tensors (for checkpointing)."""
        return {'W1': self.W1, 'b1': self.b1,
                'W2': self.W2, 'b2': self.b2}

    def load_state_dict(self, sd: dict) -> None:
        """Restore parameters from a state dict produced by state_dict()."""
        self.W1.copy_(sd['W1'])
        self.b1.copy_(sd['b1'])
        self.W2.copy_(sd['W2'])
        self.b2.copy_(sd['b2'])

    def __repr__(self) -> str:
        d = self.W1.shape[1]
        h = self.W1.shape[0]
        o = self.W2.shape[0]
        p = f"  dropout p={self.dropout_rate}" if self.use_dropout else ""
        return (f"FCNet(\n"
                f"  ({d}) → Linear → ReLU{p} → ({h}) → Linear → ({o})\n"
                f"  params: {d*h + h + h*o + o:,}\n"
                f")")
