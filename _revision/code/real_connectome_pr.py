#!/usr/bin/env python3
"""
Real Connectome Laplacian Analysis: PR vs Eigenvalue

Computes participation ratio vs eigenvalue on REAL human brain connectivity
derived from fMRI data to validate the modular network simulation predictions.

Uses nilearn's development fMRI dataset with Schaefer 100 parcellation.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def compute_participation_ratio(mode):
    """Compute participation ratio for a mode."""
    mode_normalized = mode / np.linalg.norm(mode)
    ipr = np.sum(mode_normalized**4)
    pr = 1.0 / ipr if ipr > 0 else 0
    return pr


def get_real_functional_connectivity():
    """
    Compute group-average functional connectivity from real fMRI data.

    Uses nilearn's development fMRI dataset with Schaefer 100 parcellation.
    Returns correlation-based connectivity matrix.
    """
    from nilearn import datasets
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiLabelsMasker

    print("Fetching Schaefer 100 atlas...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)

    print("Fetching development fMRI dataset (30 subjects)...")
    data = datasets.fetch_development_fmri(n_subjects=30, reduce_confounds=True, verbose=0)

    print(f"Extracting time series from {len(data.func)} subjects...")
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=True,
        memory='nilearn_cache',
        verbose=0
    )

    all_connectivity = []
    for i, (func_file, confounds) in enumerate(zip(data.func, data.confounds)):
        try:
            time_series = masker.fit_transform(func_file, confounds=confounds)
            conn_measure = ConnectivityMeasure(kind='correlation')
            connectivity = conn_measure.fit_transform([time_series])[0]
            all_connectivity.append(connectivity)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(data.func)} subjects")
        except Exception as e:
            print(f"  Subject {i} failed: {e}")
            continue

    print(f"Computing group average from {len(all_connectivity)} subjects...")
    mean_connectivity = np.mean(all_connectivity, axis=0)

    # Make positive for Laplacian (use absolute correlation)
    mean_connectivity = np.abs(mean_connectivity)
    np.fill_diagonal(mean_connectivity, 0)

    # Get network labels for interpretation
    network_labels = [label.decode() if isinstance(label, bytes) else label
                      for label in atlas.labels]

    return mean_connectivity, network_labels


def compute_laplacian(connectivity):
    """Compute unnormalized graph Laplacian."""
    W = (connectivity + connectivity.T) / 2
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L


def analyze_connectome_pr(connectivity, n_modes=None):
    """Analyze participation ratio vs eigenvalue."""
    N = connectivity.shape[0]
    if n_modes is None:
        n_modes = N - 2

    print(f"Computing Laplacian for {N} regions...")
    L = compute_laplacian(connectivity)

    print(f"Computing eigenmodes...")
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort and skip constant mode
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][1:n_modes+1]
    eigenvectors = eigenvectors[:, idx][:, 1:n_modes+1]

    print("Computing participation ratios...")
    participation = np.array([compute_participation_ratio(eigenvectors[:, i])
                               for i in range(len(eigenvalues))])

    freq_norm = eigenvalues / eigenvalues.max()
    r = np.corrcoef(freq_norm, participation)[0, 1]

    n_compare = min(10, len(eigenvalues) // 4)
    slow_pr = participation[:n_compare].mean()
    fast_pr = participation[-n_compare:].mean()
    pr_ratio = slow_pr / fast_pr

    return eigenvalues, participation, freq_norm, r, pr_ratio, slow_pr, fast_pr


def plot_results(freq_norm, participation, r, pr_ratio, slow_pr, fast_pr, title):
    """Generate publication figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Scatter
    ax1 = axes[0]
    scatter = ax1.scatter(freq_norm, participation, c=participation, cmap='viridis',
                          s=60, alpha=0.8, edgecolors='white', linewidths=0.5)

    z = np.polyfit(freq_norm, participation, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(0, 1, 100)
    ax1.plot(x_fit, p(x_fit), 'r--', linewidth=2)

    ax1.text(0.95, 0.95, f'r = {r:.3f}', transform=ax1.transAxes,
             fontsize=12, ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax1.set_xlabel('Normalized Laplacian Eigenvalue\n(spatial smoothness index)', fontsize=10)
    ax1.set_ylabel('Participation Ratio\n(# regions involved)', fontsize=10)
    ax1.set_title(f'A. {title}', fontweight='bold', fontsize=11)
    plt.colorbar(scatter, ax=ax1, shrink=0.8, label='PR')

    # Panel B: Bar comparison
    ax2 = axes[1]
    bars = ax2.bar(['Smoothest\nmodes', 'Most localized\nmodes'],
                   [slow_pr, fast_pr],
                   color=['#2E7D32', '#C62828'], alpha=0.8)

    ax2.set_ylabel('Mean Participation Ratio', fontsize=10)
    ax2.set_title('B. Slow vs Fast Modes', fontweight='bold', fontsize=11)

    # Annotate ratio - position above bar with black text
    if pr_ratio > 1:
        ax2.annotate(f'{pr_ratio:.1f}×',
                    xy=(0, slow_pr), xytext=(0, slow_pr + 0.5),
                    fontsize=11, ha='center', fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(FIGURES / "fig6_real_connectome_pr.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "fig6_real_connectome_pr.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {FIGURES / 'fig6_real_connectome_pr.pdf'}")
    plt.close()


def main():
    print("="*70)
    print("REAL HUMAN BRAIN CONNECTOME: Laplacian PR Analysis")
    print("="*70)

    # Get real connectivity
    connectivity, labels = get_real_functional_connectivity()
    print(f"\nConnectivity matrix: {connectivity.shape}")
    print(f"Density: {(connectivity > 0).sum() / (connectivity.shape[0]**2 - connectivity.shape[0]):.3f}")

    # Analyze
    eigenvalues, participation, freq_norm, r, pr_ratio, slow_pr, fast_pr = analyze_connectome_pr(connectivity)

    print("\n" + "="*70)
    print("RESULTS: Real Human Functional Connectome")
    print("="*70)
    print(f"  Correlation (eigenvalue, PR): r = {r:.3f}")
    print(f"  Slow modes mean PR: {slow_pr:.1f} regions")
    print(f"  Fast modes mean PR: {fast_pr:.1f} regions")
    print(f"  Slow/Fast ratio: {pr_ratio:.2f}×")

    # Plot
    plot_results(freq_norm, participation, r, pr_ratio, slow_pr, fast_pr,
                 title="Real fMRI Connectivity (Schaefer 100)")

    # Save
    np.savez(OUTPUT / "real_connectome_pr.npz",
             eigenvalues=eigenvalues, participation=participation,
             freq_norm=freq_norm, correlation=r, pr_ratio=pr_ratio,
             connectivity=connectivity)

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if r < -0.3:
        print(f"""
CONFIRMED: Real human brain connectivity shows the predicted pattern.

Slow/smooth structural modes (low eigenvalue) engage more regions
than fast/localized modes. Correlation r = {r:.3f} with {pr_ratio:.1f}×
more participation in slow modes.

This validates the modular network simulations and provides empirical
grounding for the substrate participation framework.
""")
    elif r < 0:
        print(f"""
PARTIAL SUPPORT: Real connectivity shows weak negative correlation (r = {r:.3f}).

The effect is weaker than idealized modular simulations, likely because:
- Real connectomes have rich-club hubs that distribute participation
- Functional connectivity includes long-range correlations
- The Schaefer parcellation may not optimally capture modularity

The direction of effect is consistent with predictions, supporting the
theoretical framework while indicating topology-dependent effect size.
""")
    else:
        print(f"""
NOT SUPPORTED: Real connectivity shows positive correlation (r = {r:.3f}).

This may indicate that functional connectivity (correlation-based) behaves
differently from structural connectivity (anatomical). The theoretical
predictions are specifically for modular anatomical networks.
""")

    return r, pr_ratio


if __name__ == "__main__":
    r, ratio = main()
