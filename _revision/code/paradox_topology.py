#!/usr/bin/env python3
"""
Paradox Topology: Why k=2 Collides and k=3 Resolves

Demonstrates that self-referential logic (Liar's Paradox) requires
3D to avoid trajectory collisions.

The Liar's Paradox: "This statement is false"
- If TRUE → must be FALSE
- If FALSE → must be TRUE
- Creates an infinite cycle: T→F→T→F→...

In k=2 (flat plane): The cycle must cross itself (collision)
In k=3 (volume): The cycle can spiral upward (helix, no collision)

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def generate_paradox_cycle(n_cycles=3, points_per_cycle=100):
    """
    Generate the trajectory of a system processing the Liar's Paradox.

    State oscillates: TRUE (1) → FALSE (-1) → TRUE (1) → ...
    Each transition takes time, creating a continuous trajectory.
    """
    n_points = n_cycles * points_per_cycle
    t = np.linspace(0, n_cycles * 2 * np.pi, n_points)

    # The "truth value" oscillates
    truth_value = np.sin(t)  # +1 = TRUE, -1 = FALSE

    # The "assertion strength" also varies (how confident)
    confidence = np.cos(t)

    # Time progresses linearly
    time = t / (2 * np.pi)  # Normalized time

    return truth_value, confidence, time


def embed_2d(truth, confidence):
    """Embed the paradox in 2D - must collapse time dimension."""
    return truth, confidence


def embed_3d(truth, confidence, time):
    """Embed the paradox in 3D - can use time as third dimension."""
    return truth, confidence, time


def count_self_intersections_2d(x, y, threshold=0.1):
    """
    Estimate self-intersections in a 2D trajectory.
    Counts pairs of points that are close but far apart in sequence.
    """
    n = len(x)
    intersections = 0

    for i in range(n):
        for j in range(i + 10, n):  # Skip nearby points in sequence
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < threshold:
                intersections += 1

    return intersections


def run_learning_experiment():
    """
    Run the linear autoencoder experiment for the combined figure.
    Returns losses and reconstructions for k=2 and k=3.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(42)

    # Generate helix data
    n_cycles, points_per_cycle = 4, 100
    n_samples = n_cycles * points_per_cycle
    t = np.linspace(0, n_cycles * 2 * np.pi, n_samples)
    x = np.sin(t)
    y = np.cos(t)
    z = t / (n_cycles * 2 * np.pi)
    data = torch.FloatTensor(np.stack([x, y, z], axis=1))

    class LinearAE(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.encoder = nn.Linear(3, k, bias=False)
            self.decoder = nn.Linear(k, 3, bias=False)
        def forward(self, x):
            return self.decoder(self.encoder(x))

    results = {}
    for k in [2, 3]:
        model = LinearAE(k)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        losses = []
        for _ in range(2000):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(data), data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            recon = model(data).numpy()
        results[k] = {'losses': losses, 'recon': recon}

    return results, data.numpy()


def visualize_paradox_topology():
    """
    Combined 4-panel visualization: topology + learning proof
    """
    # ========== Generate topology data ==========
    truth, confidence, time = generate_paradox_cycle(n_cycles=3)
    x2d, y2d = embed_2d(truth, confidence)
    x3d, y3d, z3d = embed_3d(truth, confidence, time)
    n_collisions = count_self_intersections_2d(x2d, y2d, threshold=0.15)

    # ========== Run learning experiment ==========
    print("Running linear autoencoder experiment...")
    learning_results, helix_data = run_learning_experiment()

    # ========== Create 4-panel figure ==========
    fig = plt.figure(figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(truth)))

    # ========== Panel A: 2D Collision ==========
    ax1 = fig.add_subplot(221)
    for i in range(len(x2d) - 1):
        ax1.plot([x2d[i], x2d[i+1]], [y2d[i], y2d[i+1]],
                color=colors[i], linewidth=2, alpha=0.8)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.text(1.1, 1.1, 'TRUE', fontsize=10, alpha=0.5, ha='center')
    ax1.text(-1.1, -1.1, 'FALSE', fontsize=10, alpha=0.5, ha='center')
    circle = plt.Circle((0, 0), 0.2, color='red', alpha=0.2)
    ax1.add_patch(circle)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Truth Value', fontsize=11)
    ax1.set_ylabel('Confidence', fontsize=11)
    ax1.set_title(f'A. Gamma Regime (k=2): {n_collisions} Collisions',
                  fontweight='bold', fontsize=12)

    # ========== Panel B: 3D Helix ==========
    ax2 = fig.add_subplot(222, projection='3d')
    for i in range(len(x3d) - 1):
        ax2.plot([x3d[i], x3d[i+1]], [y3d[i], y3d[i+1]], [z3d[i], z3d[i+1]],
                color=colors[i], linewidth=2, alpha=0.8)
    ax2.set_xlabel('Truth Value', fontsize=10)
    ax2.set_ylabel('Confidence', fontsize=10)
    ax2.set_zlabel('Time', fontsize=10)
    ax2.set_title('B. Beta Regime (k=3): 0 Collisions',
                  fontweight='bold', fontsize=12)

    # ========== Panel C: Loss Curves ==========
    ax3 = fig.add_subplot(223)
    ax3.semilogy(learning_results[2]['losses'], color='#d62728', linewidth=2,
                 label='k=2 (Gamma)')
    ax3.semilogy(learning_results[3]['losses'], color='#2ca02c', linewidth=2,
                 label='k=3 (Beta)')
    ax3.set_xlabel('Training Epochs', fontsize=11)
    ax3.set_ylabel('Reconstruction MSE (log)', fontsize=11)
    ax3.set_title('C. Learning Dynamics', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    # Add final MSE annotations
    final_k2 = learning_results[2]['losses'][-1]
    final_k3 = learning_results[3]['losses'][-1]
    ax3.axhline(final_k2, color='#d62728', linestyle='--', alpha=0.5)
    ax3.axhline(final_k3, color='#2ca02c', linestyle='--', alpha=0.5)
    ax3.text(1800, final_k2*1.5, f'MSE={final_k2:.3f}', color='#d62728', fontsize=9)
    ax3.text(1800, final_k3*0.5, f'MSE={final_k3:.6f}', color='#2ca02c', fontsize=9)

    # ========== Panel D: Time Reconstruction ==========
    ax4 = fig.add_subplot(224)
    true_time = helix_data[:, 2]
    recon_k2_time = learning_results[2]['recon'][:, 2]
    recon_k3_time = learning_results[3]['recon'][:, 2]

    ax4.plot([0, 1], [0, 1], 'k-', linewidth=3, alpha=0.4, label='Perfect')  # Thicker, visible
    ax4.plot(true_time, recon_k2_time, color='#d62728', linewidth=2, alpha=0.8,
             label='k=2 (fails)')
    ax4.plot(true_time, recon_k3_time, color='#2ca02c', linewidth=2, alpha=0.7,
             label='k=3 (succeeds)', linestyle='--')  # Dashed so Perfect line shows through
    ax4.set_xlabel('True Time (z)', fontsize=11)
    ax4.set_ylabel('Reconstructed Time', fontsize=11)
    ax4.set_title('D. Time Dimension Reconstruction', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper left')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Self-Reference Benefits from k≥3: Geometric and Computational Illustration",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(FIGURES / "fig5_paradox_topology.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "fig5_paradox_topology.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGURES / 'fig5_paradox_topology.pdf'}")

    plt.close()

    return n_collisions, learning_results


def print_analysis():
    """Print the theoretical analysis."""
    print("="*70)
    print("PARADOX TOPOLOGY: Why Self-Reference Needs k≥3")
    print("="*70)
    print("""
The Liar's Paradox: "This statement is false"

If we represent this as a dynamical system:
  - State cycles: TRUE → FALSE → TRUE → FALSE → ...
  - Each transition takes time
  - The trajectory must be continuous

In k=2 (Gamma / symbolic logic):
  - Trajectory lives in a plane
  - After one cycle, path returns to start
  - COLLISION: Return path crosses outgoing path
  - System cannot distinguish "going TRUE→FALSE" from "going FALSE→TRUE"
  - Result: Confusion, oscillation, paradox

In k=3 (Beta / meta-cognition):
  - Third dimension encodes TIME or META-LEVEL
  - Trajectory spirals upward (helix)
  - NO COLLISION: Return path is "above" outgoing path
  - System can represent the PROCESS without confusion
  - Result: Understanding without resolution

This explains why:
  - You can THINK about paradoxes (Beta, k=3, meta-cognition)
  - You cannot DECIDE paradoxes (Gamma, k=2, assertion)
  - The brain toggles between these modes via frequency shifts
""")


if __name__ == "__main__":
    print_analysis()
    n_collisions, learning_results = visualize_paradox_topology()
    print(f"\n2D embedding: {n_collisions} self-intersections detected")
    print("3D embedding: 0 self-intersections (helix)")
    print(f"\nLinear autoencoder proof:")
    print(f"  k=2 final MSE: {learning_results[2]['losses'][-1]:.6f}")
    print(f"  k=3 final MSE: {learning_results[3]['losses'][-1]:.6f}")
    print("\nConclusion: Self-referential logic requires k≥3 to avoid collision.")
