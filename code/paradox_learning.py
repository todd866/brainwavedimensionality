#!/usr/bin/env python3
"""
Paradox Learning: Computational Proof of the k=3 Requirement

This script uses an AUTOENCODER to compress and reconstruct a 3D helix
(the Liar's Paradox trajectory) through a k-dimensional bottleneck.

The helix is the natural representation of a cyclic process (True→False→True)
that progresses through time: x = sin(t), y = cos(t), z = t

Key insight:
- k=2 bottleneck: Cannot preserve the helix structure; must collapse time,
  causing overlapping cycles = high reconstruction error on time dimension
- k=3 bottleneck: Can preserve the full helix = low reconstruction error

This proves the topological collision is a FUNCTIONAL barrier to learning.

Author: Ian Todd
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ============================================================================
# 1. Data Generation (The Helix = Unfolded Paradox)
# ============================================================================

def generate_helix_data(n_cycles=4, points_per_cycle=100):
    """
    Generate the 3D helix: the natural embedding of the paradox trajectory.

    x = sin(t) = Truth value oscillation
    y = cos(t) = Confidence oscillation
    z = t      = Elapsed time / meta-level
    """
    n_samples = n_cycles * points_per_cycle
    t = np.linspace(0, n_cycles * 2 * np.pi, n_samples)

    x = np.sin(t)
    y = np.cos(t)
    z = t / (n_cycles * 2 * np.pi)  # Normalized to [0, 1]

    # The helix lives in 3D
    data = np.stack([x, y, z], axis=1)

    return torch.FloatTensor(data)


# ============================================================================
# 2. The Autoencoder Model
# ============================================================================

class LinearAutoencoder(nn.Module):
    """
    LINEAR autoencoder: compress 3D helix through k-dimensional bottleneck.

    With linear encoder/decoder, this is equivalent to PCA truncation.
    The topological constraint becomes evident: you cannot linearly embed
    a 3D helix into 2D without losing the time dimension.
    """
    def __init__(self, bottleneck_dim):
        super().__init__()

        # Linear encoder: 3D -> k
        self.encoder = nn.Linear(3, bottleneck_dim, bias=False)

        # Linear decoder: k -> 3D
        self.decoder = nn.Linear(bottleneck_dim, 3, bias=False)

    def forward(self, x):
        code = self.encoder(x)
        reconstruction = self.decoder(code)
        return reconstruction, code


# ============================================================================
# 3. Experiment
# ============================================================================

def train_and_evaluate(k, data, epochs=2000):
    model = LinearAutoencoder(bottleneck_dim=k)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, codes = model(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, model


def run_comparison():
    print("="*60)
    print("PARADOX LEARNING: Linear Compression of Helix Trajectory")
    print("="*60)
    print("\nTask: Compress 3D helix (sin(t), cos(t), t) through LINEAR bottleneck.")
    print("      This is equivalent to PCA truncation.")
    print("\nChallenge: The helix has 3 independent dimensions.")
    print("   k=2: Must discard one dimension → reconstruction error")
    print("   k=3: Can preserve all 3 dimensions → perfect reconstruction")

    data = generate_helix_data()

    # Train k=2 (Gamma Regime)
    print("\n" + "-"*40)
    print("Training k=2 autoencoder...")
    losses_k2, model_k2 = train_and_evaluate(2, data)
    final_loss_k2 = losses_k2[-1]
    print(f"  > Final reconstruction MSE: {final_loss_k2:.6f}")

    # Train k=3 (Beta Regime)
    print("\nTraining k=3 autoencoder...")
    losses_k3, model_k3 = train_and_evaluate(3, data)
    final_loss_k3 = losses_k3[-1]
    print(f"  > Final reconstruction MSE: {final_loss_k3:.6f}")

    # Calculate improvement
    improvement = (final_loss_k2 - final_loss_k3) / final_loss_k2 * 100
    print(f"\n{'='*60}")
    print(f"RESULT: k=3 reduced reconstruction error by {improvement:.1f}%")
    if improvement > 50:
        print("SUCCESS: k=2 fails due to topological collision, k=3 succeeds!")
    print(f"{'='*60}")

    # Analyze per-dimension error
    with torch.no_grad():
        recon_k2, codes_k2 = model_k2(data)
        recon_k3, codes_k3 = model_k3(data)

    data_np = data.numpy()
    recon_k2_np = recon_k2.numpy()
    recon_k3_np = recon_k3.numpy()

    # Per-dimension MSE
    mse_xy_k2 = np.mean((data_np[:, :2] - recon_k2_np[:, :2])**2)
    mse_t_k2 = np.mean((data_np[:, 2] - recon_k2_np[:, 2])**2)
    mse_xy_k3 = np.mean((data_np[:, :2] - recon_k3_np[:, :2])**2)
    mse_t_k3 = np.mean((data_np[:, 2] - recon_k3_np[:, 2])**2)

    print(f"\nPer-dimension analysis:")
    print(f"  k=2: Circle (x,y) error = {mse_xy_k2:.6f}, Time (z) error = {mse_t_k2:.6f}")
    print(f"  k=3: Circle (x,y) error = {mse_xy_k3:.6f}, Time (z) error = {mse_t_k3:.6f}")

    # Plotting
    fig = plt.figure(figsize=(12, 4))

    # Panel A: Loss curves
    ax1 = fig.add_subplot(131)
    ax1.semilogy(losses_k2, label='k=2 (Gamma)', color='#d62728', linewidth=2)
    ax1.semilogy(losses_k3, label='k=3 (Beta)', color='#2ca02c', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Reconstruction MSE (log)', fontsize=11)
    ax1.set_title('A. Learning Dynamics', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Reconstruction quality (time dimension)
    ax2 = fig.add_subplot(132)
    ax2.plot(data_np[:, 2], recon_k2_np[:, 2], alpha=0.7, linewidth=1.5,
             c='#d62728', label=f'k=2 (MSE={mse_t_k2:.4f})')
    ax2.plot(data_np[:, 2], recon_k3_np[:, 2], alpha=0.7, linewidth=1.5,
             c='#2ca02c', label=f'k=3 (MSE={mse_t_k3:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
    ax2.set_xlabel('True Time (z)', fontsize=11)
    ax2.set_ylabel('Reconstructed Time', fontsize=11)
    ax2.set_title('B. Time Reconstruction', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Panel C: 3D latent space of k=3
    ax3 = fig.add_subplot(133, projection='3d')
    codes_np = codes_k3.numpy()

    # Color by time to show the helix
    colors = plt.cm.viridis(np.linspace(0, 1, len(codes_np)))

    for i in range(len(codes_np) - 1):
        ax3.plot([codes_np[i, 0], codes_np[i+1, 0]],
                 [codes_np[i, 1], codes_np[i+1, 1]],
                 [codes_np[i, 2], codes_np[i+1, 2]],
                 color=colors[i], linewidth=1.5, alpha=0.8)

    ax3.set_title('C. k=3 Latent Space\n(Helix Preserved)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('z₁', fontsize=10)
    ax3.set_ylabel('z₂', fontsize=10)
    ax3.set_zlabel('z₃', fontsize=10)

    plt.suptitle("Linear Compression: k=2 Cannot Preserve Time Dimension", fontsize=13)
    plt.tight_layout()

    outfile = FIGURES / "fig4b_paradox_learning.pdf"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "fig4b_paradox_learning.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {outfile}")

    plt.close()

    return final_loss_k2, final_loss_k3


if __name__ == "__main__":
    run_comparison()
