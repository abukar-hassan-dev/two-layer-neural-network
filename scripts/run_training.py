"""
run_training.py
---------------
Entry point. Edit CONFIG below to change any training setting.

Two modes
---------
  synthetic  — runs entirely on generated data; no files needed.
  real       — loads a .pth dataset from DATA_PATH.

Usage
-----
    python scripts/run_training.py
"""

import os
import sys
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_dataset, make_synthetic_dataset, split_dataset, plot_loss_curves
from src.model import FCNet
from src.train import train

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    'mode':        'synthetic',   # 'synthetic' | 'real'
    'data_path':   'data/homework_features_256_50000.pth',
    'train_ratio': 0.70,

    # Architecture
    'input_dim':    256,
    'hidden_dim':   512,
    'output_dim':   1,
    'dropout_rate': 0.5,
    'use_dropout':  False,        # disabled in final config — see report

    # Optimisation
    'lr':          1e-4,
    'epochs':      10_000,
    'batch_size':  128,
    'patience':    30,
    'eval_every':  10,

    # Output
    'save_model':  True,
    'model_path':  'checkpoints/fcnet.pth',
    'figure_path': 'figures/loss_curves.png',
}
# ──────────────────────────────────────────────────────────────────────────────


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}\n")

    # ── Load data ────────────────────────────────────────────────────────
    if CONFIG['mode'] == 'synthetic':
        x, y = make_synthetic_dataset(n=4000, d=CONFIG['input_dim'], device=device)
    else:
        x, y = load_dataset(CONFIG['data_path'], device=device)

    x_train, y_train, x_test, y_test = split_dataset(
        x, y, train_ratio=CONFIG['train_ratio'])

    # ── Build model ───────────────────────────────────────────────────────
    model = FCNet(
        input_dim    = CONFIG['input_dim'],
        hidden_dim   = CONFIG['hidden_dim'],
        output_dim   = CONFIG['output_dim'],
        dropout_rate = CONFIG['dropout_rate'],
        use_dropout  = CONFIG['use_dropout'],
        device       = str(device),
    )
    print(model, '\n')

    # ── Train ─────────────────────────────────────────────────────────────
    print("Training …")
    train_hist, test_hist = train(
        model,
        x_train, y_train,
        x_test,  y_test,
        lr          = CONFIG['lr'],
        epochs      = CONFIG['epochs'],
        batch_size  = CONFIG['batch_size'],
        patience    = CONFIG['patience'],
        eval_every  = CONFIG['eval_every'],
        device      = device,
        verbose     = True,
    )

    # ── Save model ────────────────────────────────────────────────────────
    if CONFIG['save_model']:
        os.makedirs(os.path.dirname(CONFIG['model_path']), exist_ok=True)
        torch.save(model.state_dict(), CONFIG['model_path'])
        print(f"\n[checkpoint]  saved → {CONFIG['model_path']}")

    # ── Plot ──────────────────────────────────────────────────────────────
    # Strip trailing Nones before plotting
    t_hist = [v for v in train_hist if v is not None]
    plot_loss_curves(t_hist, test_hist[:len(t_hist)], CONFIG['figure_path'])
    print(f"[done]  figure → {CONFIG['figure_path']}")


if __name__ == '__main__':
    main()