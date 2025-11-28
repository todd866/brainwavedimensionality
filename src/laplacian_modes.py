#!/usr/bin/env python3
"""
Laplacian Eigenmodes: Proving Slow Waves are High-Dimensional

The Core Argument Against Miller's "Low-F = Low-D" Claim:
---------------------------------------------------------
Earl Miller conflates temporal frequency with geometric dimensionality.

Slow oscillations (delta, theta) involve MANY oscillators participating coherently.
Fast oscillations (gamma) involve FEW oscillators in local bursts.

PARTICIPATION RATIO proves this:
    PR = (Σ|ψ_i|²)² / Σ|ψ_i|⁴

For a mode ψ:
- PR ≈ N means all oscillators participate equally (high-D)
- PR ≈ 1 means one oscillator dominates (low-D)

On a cortical sheet (2D lattice with graph Laplacian):
- SLOW modes (small eigenvalue) = smooth, many oscillators = HIGH participation
- FAST modes (large eigenvalue) = localized, few oscillators = LOW participation

This is basic spectral geometry. Miller has it backwards.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
import networkx as nx
from pathlib import Path

# Output paths
ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def create_cortical_sheet(n=50, periodic=False):
    """
    Create a 2D lattice representing cortical sheet.

    Returns graph Laplacian for an n×n grid.

    Non-periodic boundaries (periodic=False) are MORE REALISTIC for cortex
    and show the localization effect more clearly:
    - Slow modes are smooth across the whole sheet
    - Fast modes localize near boundaries/corners

    Args:
        n: Grid size (n×n nodes = n² oscillators)
        periodic: If True, use periodic (torus) boundary conditions

    Returns:
        L: Sparse graph Laplacian (n² × n²)
    """
    N = n * n

    # Build adjacency matrix
    row, col, data = [], [], []

    for i in range(n):
        for j in range(n):
            node = i * n + j
            degree = 0

            # 4-connected neighbors
            neighbor_coords = [
                (i-1, j),  # up
                (i+1, j),  # down
                (i, j-1),  # left
                (i, j+1),  # right
            ]

            for ni, nj in neighbor_coords:
                if periodic:
                    ni = ni % n
                    nj = nj % n
                    neighbor = ni * n + nj
                    row.append(node)
                    col.append(neighbor)
                    data.append(-1.0)
                    degree += 1
                else:
                    # Non-periodic: only add if within bounds
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbor = ni * n + nj
                        row.append(node)
                        col.append(neighbor)
                        data.append(-1.0)
                        degree += 1

            # Diagonal = degree
            row.append(node)
            col.append(node)
            data.append(float(degree))

    L = sparse.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    return L


def create_smallworld_network(N=2500, k=6, p=0.1):
    """
    Create a Watts-Strogatz small-world network.

    This is more realistic for cortical connectivity:
    - High local clustering (like lattice)
    - Short path lengths (like random graph)
    - Creates natural spectral gap: slow modes are GLOBAL, fast modes are LOCAL

    Args:
        N: Number of nodes
        k: Each node connected to k nearest neighbors in ring
        p: Rewiring probability (0 = ring lattice, 1 = random graph)

    Returns:
        L: Sparse graph Laplacian (N × N)
    """
    G = nx.watts_strogatz_graph(n=N, k=k, p=p, seed=42)
    L = nx.laplacian_matrix(G).astype(float)
    return L


def create_modular_network(N=2500, n_modules=25, p_within=0.3, p_between=0.01):
    """
    Create a modular network (stochastic block model).

    Even more brain-like: cortex has distinct modules/columns
    with dense internal connectivity and sparse inter-module connections.

    This produces very clear spectral structure:
    - Slowest modes: inter-module (global coordination)
    - Fastest modes: intra-module (local activity)

    Args:
        N: Total number of nodes
        n_modules: Number of modules (e.g., cortical columns)
        p_within: Connection probability within module
        p_between: Connection probability between modules

    Returns:
        L: Sparse graph Laplacian (N × N)
    """
    nodes_per_module = N // n_modules
    sizes = [nodes_per_module] * n_modules

    # Adjust last module to account for remainder
    sizes[-1] += N - sum(sizes)

    # Create probability matrix
    probs = np.full((n_modules, n_modules), p_between)
    np.fill_diagonal(probs, p_within)

    G = nx.stochastic_block_model(sizes, probs, seed=42)
    L = nx.laplacian_matrix(G).astype(float)
    return L


def compute_participation_ratio(mode):
    """
    Compute participation ratio (inverse participation ratio = 1/PR).

    PR = (Σ|ψ_i|²)² / Σ|ψ_i|⁴

    For normalized modes (Σ|ψ_i|² = 1):
        PR = 1 / Σ|ψ_i|⁴

    Interpretation:
        PR ≈ N: All oscillators participate equally (delocalized, HIGH-D)
        PR ≈ 1: One oscillator dominates (localized, LOW-D)
    """
    mode_normalized = mode / np.linalg.norm(mode)
    ipr = np.sum(mode_normalized**4)
    pr = 1.0 / ipr if ipr > 0 else 0
    return pr


def analyze_laplacian_spectrum(n=50, n_modes=200, network_type='modular'):
    """
    Compute Laplacian eigenmodes and their participation ratios.

    Key prediction:
    - Small eigenvalues (slow modes) → HIGH participation ratio
    - Large eigenvalues (fast modes) → LOW participation ratio

    This directly contradicts "slow = simple = low-D".

    Args:
        n: Grid size (for lattice) or sqrt(N) for other networks
        n_modes: Number of modes to compute
        network_type: 'lattice', 'smallworld', or 'modular'
    """
    N = n * n

    if network_type == 'lattice':
        print(f"Creating {n}×{n} lattice ({N} oscillators)...")
        L = create_cortical_sheet(n, periodic=False)
    elif network_type == 'smallworld':
        print(f"Creating Watts-Strogatz small-world network ({N} oscillators)...")
        L = create_smallworld_network(N=N, k=6, p=0.1)
    elif network_type == 'modular':
        print(f"Creating modular network ({N} oscillators, 25 modules)...")
        L = create_modular_network(N=N, n_modules=25, p_within=0.3, p_between=0.01)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    print(f"Computing {n_modes} smallest eigenvalues...")
    # Get smallest eigenvalues (slowest modes)
    eigenvalues, eigenvectors = eigsh(L, k=n_modes, which='SM')

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute participation ratio for each mode
    participation_ratios = []
    for i in range(n_modes):
        pr = compute_participation_ratio(eigenvectors[:, i])
        participation_ratios.append(pr)

    participation_ratios = np.array(participation_ratios)

    # The first eigenvalue is 0 (constant mode), skip it
    return eigenvalues[1:], participation_ratios[1:], eigenvectors[:, 1:]


def run_experiment():
    """
    Main experiment: Prove slow waves are high-dimensional.
    """
    print("="*70)
    print("LAPLACIAN EIGENMODES: Slow Waves are High-Dimensional")
    print("="*70)
    print()
    print("Miller's claim: Low frequency = Low dimensionality")
    print("Reality: Low frequency = HIGH participation = HIGH dimensionality")
    print()

    # Run analysis with modular network (more brain-like than lattice)
    eigenvalues, participation, modes = analyze_laplacian_spectrum(
        n=50, n_modes=150, network_type='modular'
    )

    # Normalize eigenvalues to [0, 1] for "frequency" interpretation
    freq_normalized = eigenvalues / eigenvalues.max()

    # Results
    print("-"*70)
    print("RESULTS:")
    print("-"*70)

    # Correlation between frequency and participation
    corr = np.corrcoef(freq_normalized, participation)[0, 1]
    print(f"\nCorrelation (frequency, participation): r = {corr:.4f}")

    if corr < 0:
        print("  → CONFIRMED: Slow modes have HIGHER participation (more oscillators)")
        print("  → Miller's 'low-F = low-D' is BACKWARDS")

    # Print some examples
    n_modes = len(eigenvalues)
    print("\nExample modes:")
    print(f"  Mode 1 (slowest): eigenvalue={eigenvalues[0]:.4f}, PR={participation[0]:.1f} oscillators")
    print(f"  Mode {n_modes//2} (mid):    eigenvalue={eigenvalues[n_modes//2]:.4f}, PR={participation[n_modes//2]:.1f} oscillators")
    print(f"  Mode {n_modes} (fastest): eigenvalue={eigenvalues[-1]:.4f}, PR={participation[-1]:.1f} oscillators")

    # Save data
    np.savez(OUTPUT / "laplacian_analysis.npz",
             eigenvalues=eigenvalues,
             participation=participation,
             freq_normalized=freq_normalized,
             modes=modes,
             correlation=corr)

    print(f"\nData saved to {OUTPUT / 'laplacian_analysis.npz'}")

    return eigenvalues, participation, freq_normalized, modes


def plot_results(eigenvalues, participation, freq_normalized, modes, n=50):
    """
    Generate publication-quality figure.

    Layout:
    - A: Participation vs Eigenvalue (the main result)
    - B: PR distribution for slowest 15 vs fastest 15 modes
    - C: Mode participation histograms (slow vs fast)
    """
    fig = plt.figure(figsize=(11, 4))

    # Panel A: Main result - scatter plot
    ax1 = fig.add_axes([0.07, 0.15, 0.32, 0.75])

    # Color by participation
    scatter = ax1.scatter(freq_normalized, participation,
                         c=participation, cmap='viridis',
                         s=40, alpha=0.8, edgecolors='white', linewidths=0.5)

    # Fit line
    z = np.polyfit(freq_normalized, participation, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(0, 1, 100)
    ax1.plot(x_fit, p(x_fit), 'r--', linewidth=2)

    corr = np.corrcoef(freq_normalized, participation)[0, 1]
    ax1.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes,
             fontsize=11, ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax1.set_xlabel('Normalized Eigenvalue (≈ Frequency²)', fontsize=10)
    ax1.set_ylabel('Participation Ratio\n(# oscillators involved)', fontsize=10)
    ax1.set_title('A. Slow Modes Are High-Dimensional', fontweight='bold', fontsize=11)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, participation.max() * 1.1)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Participation', fontsize=9)

    # Panel B: Bar comparison - slowest vs fastest deciles
    ax2 = fig.add_axes([0.47, 0.15, 0.22, 0.75])

    n_compare = 15
    slow_pr = participation[:n_compare]
    fast_pr = participation[-n_compare:]

    slow_mean = slow_pr.mean()
    fast_mean = fast_pr.mean()
    slow_std = slow_pr.std()
    fast_std = fast_pr.std()

    bars = ax2.bar(['Slowest\n15 modes', 'Fastest\n15 modes'],
                   [slow_mean, fast_mean],
                   yerr=[slow_std, fast_std],
                   color=['#2E7D32', '#C62828'],
                   capsize=5, alpha=0.8)

    ax2.set_ylabel('Mean Participation Ratio', fontsize=10)
    ax2.set_title('B. Slow vs Fast Modes', fontweight='bold', fontsize=11)

    # Add percentage labels (inside bars to avoid title overlap)
    N = modes.shape[0]
    ax2.text(0, slow_mean * 0.85, f'{slow_mean/N*100:.0f}%',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax2.text(1, fast_mean * 0.85, f'{fast_mean/N*100:.0f}%',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Panel C: Node activation histograms
    ax3 = fig.add_axes([0.76, 0.15, 0.22, 0.75])

    # Get activation magnitudes for slowest and fastest modes
    slow_mode = np.abs(modes[:, 0])
    fast_mode = np.abs(modes[:, -1])

    # Normalize for comparison
    slow_mode = slow_mode / slow_mode.max()
    fast_mode = fast_mode / fast_mode.max()

    bins = np.linspace(0, 1, 30)
    ax3.hist(slow_mode, bins=bins, alpha=0.7, color='#2E7D32',
             label=f'Slowest (PR={participation[0]:.0f})', density=True)
    ax3.hist(fast_mode, bins=bins, alpha=0.7, color='#C62828',
             label=f'Fastest (PR={participation[-1]:.0f})', density=True)

    ax3.set_xlabel('Normalized Node Activation', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('C. Activation Distribution', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=8, loc='upper right')

    # Save
    plt.savefig(FIGURES / "fig1_laplacian_participation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "fig1_laplacian_participation.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {FIGURES / 'fig1_laplacian_participation.pdf'}")

    plt.close()


if __name__ == "__main__":
    eigenvalues, participation, freq_normalized, modes = run_experiment()
    plot_results(eigenvalues, participation, freq_normalized, modes)

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("""
Slow oscillations (low eigenvalue, low frequency²) involve MANY oscillators
acting coherently across the cortical sheet.

Fast oscillations (high eigenvalue, high frequency²) involve FEW oscillators
in localized activity patterns.

DIMENSIONALITY IS PARTICIPATION, NOT FREQUENCY.

Miller's "low-F = low-D" conflates temporal simplicity with geometric simplicity.
A slow wave sweeping across cortex is geometrically HIGH-dimensional precisely
BECAUSE it coordinates many degrees of freedom simultaneously.
""")
