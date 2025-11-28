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


def visualize_paradox_topology():
    """
    Main visualization: k=2 collision vs k=3 helix
    """
    # Generate the paradox cycle
    truth, confidence, time = generate_paradox_cycle(n_cycles=3)

    # Get embeddings
    x2d, y2d = embed_2d(truth, confidence)
    x3d, y3d, z3d = embed_3d(truth, confidence, time)

    # Count collisions in 2D
    n_collisions = count_self_intersections_2d(x2d, y2d, threshold=0.15)

    # Create figure
    fig = plt.figure(figsize=(12, 5))

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(truth)))

    # ========== Panel A: k=2 (Collision) ==========
    ax1 = fig.add_subplot(121)

    # Plot trajectory
    for i in range(len(x2d) - 1):
        ax1.plot([x2d[i], x2d[i+1]], [y2d[i], y2d[i+1]],
                color=colors[i], linewidth=2, alpha=0.8)

    # Mark TRUE and FALSE regions
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.text(0.8, 0.8, 'TRUE', fontsize=10, alpha=0.5)
    ax1.text(-0.8, -0.8, 'FALSE', fontsize=10, alpha=0.5)

    # Mark collision zone
    circle = plt.Circle((0, 0), 0.2, color='red', alpha=0.2, label='Collision zone')
    ax1.add_patch(circle)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Truth Value', fontsize=11)
    ax1.set_ylabel('Confidence', fontsize=11)
    ax1.set_title(f'A. Gamma Regime (k=2)\nPlanar: {n_collisions} self-intersections',
                  fontweight='bold', fontsize=12)

    # ========== Panel B: k=3 (Helix) ==========
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot trajectory as helix
    for i in range(len(x3d) - 1):
        ax2.plot([x3d[i], x3d[i+1]], [y3d[i], y3d[i+1]], [z3d[i], z3d[i+1]],
                color=colors[i], linewidth=2, alpha=0.8)

    ax2.set_xlabel('Truth Value', fontsize=10)
    ax2.set_ylabel('Confidence', fontsize=10)
    ax2.set_zlabel('Time / Meta-level', fontsize=10)
    ax2.set_title('B. Beta Regime (k=3)\nHelix: 0 self-intersections',
                  fontweight='bold', fontsize=12)

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 3))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.6, label='Cycle')

    plt.suptitle("The Liar's Paradox: Dimensional Requirements for Self-Reference",
                 fontsize=13)
    plt.tight_layout()

    # Save
    plt.savefig(FIGURES / "fig4_paradox_topology.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "fig4_paradox_topology.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGURES / 'fig4_paradox_topology.png'}")

    plt.close()

    return n_collisions


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
    n_collisions = visualize_paradox_topology()
    print(f"\n2D embedding: {n_collisions} self-intersections detected")
    print("3D embedding: 0 self-intersections (helix)")
    print("\nConclusion: Self-referential logic requires k≥3 to avoid collision.")
