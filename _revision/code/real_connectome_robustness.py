#!/usr/bin/env python3
"""
Real Connectome Laplacian Analysis: Robustness Tests

Tests whether the smooth>localized PR pattern is robust to:
1. Positive-only weights (set negatives to 0, not abs)
2. Thresholded graphs (top 20%, 50% edges)
3. Normalized Laplacian (instead of unnormalized)

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


def compute_unnormalized_laplacian(W):
    """Compute unnormalized graph Laplacian."""
    W = (W + W.T) / 2
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L


def compute_normalized_laplacian(W):
    """Compute symmetric normalized Laplacian: I - D^{-1/2} W D^{-1/2}."""
    W = (W + W.T) / 2
    d = np.sum(W, axis=1)
    d_inv_sqrt = np.zeros_like(d)
    d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L_norm


def analyze_pr(L, n_modes=None):
    """Analyze participation ratio vs eigenvalue."""
    N = L.shape[0]
    if n_modes is None:
        n_modes = N - 2

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort and skip constant mode
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][1:n_modes+1]
    eigenvectors = eigenvectors[:, idx][:, 1:n_modes+1]

    participation = np.array([compute_participation_ratio(eigenvectors[:, i])
                               for i in range(len(eigenvalues))])

    freq_norm = eigenvalues / eigenvalues.max()
    r = np.corrcoef(freq_norm, participation)[0, 1]

    n_compare = min(10, len(eigenvalues) // 4)
    slow_pr = participation[:n_compare].mean()
    fast_pr = participation[-n_compare:].mean()
    pr_ratio = slow_pr / fast_pr if fast_pr > 0 else np.inf

    return r, pr_ratio, slow_pr, fast_pr


def get_connectivity():
    """Load connectivity from saved file or recompute."""
    npz_path = OUTPUT / "real_connectome_pr.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if 'connectivity' in data:
            return data['connectivity']

    # Recompute if not cached
    from nilearn import datasets
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.maskers import NiftiLabelsMasker

    print("Fetching data...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
    data = datasets.fetch_development_fmri(n_subjects=30, reduce_confounds=True, verbose=0)

    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True,
                                memory='nilearn_cache', verbose=0)

    all_connectivity = []
    for func_file, confounds in zip(data.func, data.confounds):
        try:
            time_series = masker.fit_transform(func_file, confounds=confounds)
            conn_measure = ConnectivityMeasure(kind='correlation')
            connectivity = conn_measure.fit_transform([time_series])[0]
            all_connectivity.append(connectivity)
        except:
            continue

    mean_connectivity = np.mean(all_connectivity, axis=0)
    return mean_connectivity


def run_robustness_tests(raw_connectivity):
    """Run robustness tests on connectivity matrix."""
    results = {}

    # 1. Original: abs(correlation)
    print("Testing: abs(correlation)...")
    W_abs = np.abs(raw_connectivity.copy())
    np.fill_diagonal(W_abs, 0)
    L_abs = compute_unnormalized_laplacian(W_abs)
    results['abs(corr)'] = analyze_pr(L_abs)

    # 2. Positive-only (set negatives to 0)
    print("Testing: positive-only...")
    W_pos = raw_connectivity.copy()
    W_pos[W_pos < 0] = 0
    np.fill_diagonal(W_pos, 0)
    L_pos = compute_unnormalized_laplacian(W_pos)
    results['positive-only'] = analyze_pr(L_pos)

    # 3. Thresholded: top 50% of edges
    print("Testing: top 50% edges...")
    W_50 = np.abs(raw_connectivity.copy())
    np.fill_diagonal(W_50, 0)
    threshold = np.percentile(W_50[W_50 > 0], 50)
    W_50[W_50 < threshold] = 0
    L_50 = compute_unnormalized_laplacian(W_50)
    results['top 50%'] = analyze_pr(L_50)

    # 4. Thresholded: top 20% of edges
    print("Testing: top 20% edges...")
    W_20 = np.abs(raw_connectivity.copy())
    np.fill_diagonal(W_20, 0)
    threshold = np.percentile(W_20[W_20 > 0], 80)
    W_20[W_20 < threshold] = 0
    L_20 = compute_unnormalized_laplacian(W_20)
    results['top 20%'] = analyze_pr(L_20)

    # 5. Normalized Laplacian
    print("Testing: normalized Laplacian...")
    W_norm = np.abs(raw_connectivity.copy())
    np.fill_diagonal(W_norm, 0)
    L_norm = compute_normalized_laplacian(W_norm)
    results['normalized L'] = analyze_pr(L_norm)

    return results


def plot_robustness(results):
    """Generate robustness figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = list(results.keys())
    correlations = [results[m][0] for m in methods]
    ratios = [results[m][1] for m in methods]

    # Panel A: Correlations
    ax1 = axes[0]
    colors = ['#2E7D32' if r < 0 else '#C62828' for r in correlations]
    bars1 = ax1.bar(range(len(methods)), correlations, color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Correlation (eigenvalue, PR)')
    ax1.set_title('A. Effect direction is robust', fontweight='bold')
    ax1.set_ylim(-0.7, 0.1)

    # Annotate values
    for i, (bar, r) in enumerate(zip(bars1, correlations)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                f'{r:.2f}', ha='center', va='top', fontsize=9, fontweight='bold')

    # Panel B: PR ratios
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(methods)), ratios, color='#1976D2', alpha=0.8)
    ax2.axhline(1, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Smooth/Localized PR ratio')
    ax2.set_title('B. Magnitude varies with preprocessing', fontweight='bold')

    # Annotate values
    for bar, ratio in zip(bars2, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{ratio:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES / "figS5_connectome_robustness.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "figS5_connectome_robustness.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {FIGURES / 'figS5_connectome_robustness.pdf'}")
    plt.close()


def main():
    print("=" * 70)
    print("REAL CONNECTOME: Robustness Analysis")
    print("=" * 70)

    # Get connectivity
    raw_connectivity = get_connectivity()
    print(f"Connectivity matrix: {raw_connectivity.shape}")

    # Run tests
    results = run_robustness_tests(raw_connectivity)

    # Print results
    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'r':<10} {'Ratio':<10} {'Smooth PR':<12} {'Local PR':<10}")
    print("-" * 70)
    for method, (r, ratio, slow_pr, fast_pr) in results.items():
        print(f"{method:<20} {r:<10.3f} {ratio:<10.2f} {slow_pr:<12.1f} {fast_pr:<10.1f}")

    # Check if all correlations are negative
    all_negative = all(r < 0 for r, _, _, _ in results.values())
    print("\n" + "=" * 70)
    if all_negative:
        print("CONFIRMED: Effect is ROBUST across all preprocessing choices.")
        print("All correlations are negative (smooth modes engage more regions).")
    else:
        print("WARNING: Some preprocessing choices show different patterns.")
    print("=" * 70)

    # Plot
    plot_robustness(results)

    # Save results
    np.savez(OUTPUT / "connectome_robustness.npz", **{
        k: np.array(v) for k, v in results.items()
    })

    return results


if __name__ == "__main__":
    results = main()
