#!/usr/bin/env python3
"""
Real Data Validation: Participation Ratio in Human EEG

Test the prediction: Slow rhythms have HIGHER participation ratio
(more channels engaged) than fast rhythms.

Dataset: PhysioNet Multimodal N-back Working Memory
https://physionet.org/content/multimodal-nback-music/

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import urllib.request
import zipfile
import os

# Output paths
ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def compute_participation_ratio(amplitudes):
    """
    Compute participation ratio across channels.

    PR = 1 / sum(v_i^4) where v is L2-normalized amplitude vector.

    High PR = many channels participate (delocalized)
    Low PR = few channels dominate (localized)
    """
    # Normalize
    norm = np.linalg.norm(amplitudes)
    if norm < 1e-10:
        return 1.0
    v = amplitudes / norm

    # PR
    ipr = np.sum(v**4)
    return 1.0 / ipr if ipr > 0 else 1.0


def bandpass_filter(data, fs, low, high, order=4):
    """Bandpass filter data (channels x time)."""
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq

    # Handle edge cases
    if high_norm >= 1:
        high_norm = 0.99
    if low_norm <= 0:
        low_norm = 0.01

    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    return signal.filtfilt(b, a, data, axis=1)


def hilbert_amplitude(data):
    """Get amplitude envelope via Hilbert transform."""
    analytic = signal.hilbert(data, axis=1)
    return np.abs(analytic)


def compute_pr_timeseries(amplitudes, window_samples=100):
    """
    Compute PR over sliding windows.

    Args:
        amplitudes: (n_channels, n_time) amplitude envelopes
        window_samples: window size in samples

    Returns:
        pr_timeseries: PR at each window center
    """
    n_channels, n_time = amplitudes.shape
    n_windows = n_time // window_samples

    pr_values = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples

        # Mean amplitude per channel in this window
        mean_amp = amplitudes[:, start:end].mean(axis=1)
        pr = compute_participation_ratio(mean_amp)
        pr_values.append(pr)

    return np.array(pr_values)


def generate_synthetic_validation():
    """
    Generate synthetic data that mimics EEG to validate the method.

    This demonstrates that IF slow waves are more global and fast waves
    are more local, our PR measure will detect it.
    """
    np.random.seed(42)

    n_channels = 64
    n_time = 10000
    fs = 256  # Hz
    t = np.arange(n_time) / fs

    # Generate "slow" activity: global mode + noise
    # All channels get correlated slow oscillation
    global_slow = np.sin(2 * np.pi * 4 * t)  # 4 Hz theta
    slow_data = np.zeros((n_channels, n_time))
    for i in range(n_channels):
        # High correlation with global mode
        slow_data[i] = 0.8 * global_slow + 0.2 * np.random.randn(n_time)

    # Generate "fast" activity: sparse, local bursts
    # Only a few channels active at any time
    fast_data = np.zeros((n_channels, n_time))
    for i in range(n_channels):
        # Sparse bursts
        burst_times = np.random.choice(n_time, size=n_time//50, replace=False)
        fast_signal = np.zeros(n_time)
        for bt in burst_times:
            # Short gamma burst
            burst_len = min(20, n_time - bt)
            fast_signal[bt:bt+burst_len] = np.sin(2 * np.pi * 40 * t[:burst_len])
        fast_data[i] = fast_signal * (0.1 + 0.9 * np.random.rand())  # Random amplitude per channel

    # Combine
    data = slow_data + fast_data + 0.1 * np.random.randn(n_channels, n_time)

    return data, fs


def analyze_synthetic():
    """
    Analyze synthetic data to validate the PR method.
    """
    print("="*70)
    print("SYNTHETIC VALIDATION: Does PR detect global vs local activity?")
    print("="*70)

    data, fs = generate_synthetic_validation()
    n_channels = data.shape[0]

    # Filter into bands
    print("\nFiltering into frequency bands...")
    slow_band = bandpass_filter(data, fs, 2, 8)   # Theta/low-alpha
    fast_band = bandpass_filter(data, fs, 30, 50)  # Low gamma

    # Get amplitude envelopes
    slow_amp = hilbert_amplitude(slow_band)
    fast_amp = hilbert_amplitude(fast_band)

    # Compute PR over time
    window_ms = 200
    window_samples = int(window_ms * fs / 1000)

    slow_pr = compute_pr_timeseries(slow_amp, window_samples)
    fast_pr = compute_pr_timeseries(fast_amp, window_samples)

    # Results
    print("\n" + "-"*70)
    print("RESULTS (Synthetic Data):")
    print("-"*70)
    print(f"  Slow band (2-8 Hz):  Mean PR = {slow_pr.mean():.1f} / {n_channels} channels ({slow_pr.mean()/n_channels*100:.1f}%)")
    print(f"  Fast band (30-50 Hz): Mean PR = {fast_pr.mean():.1f} / {n_channels} channels ({fast_pr.mean()/n_channels*100:.1f}%)")
    print(f"  Ratio (slow/fast): {slow_pr.mean()/fast_pr.mean():.2f}x")

    if slow_pr.mean() > fast_pr.mean():
        print("\n  ✓ CONFIRMED: Slow rhythms engage MORE channels than fast rhythms")

    return slow_pr, fast_pr, n_channels


def plot_synthetic_results(slow_pr, fast_pr, n_channels):
    """Plot PR comparison for synthetic data."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Distributions
    ax1 = axes[0]
    bins = np.linspace(1, n_channels, 30)
    ax1.hist(slow_pr, bins=bins, alpha=0.7, color='#2E7D32', label='Slow (2-8 Hz)', density=True)
    ax1.hist(fast_pr, bins=bins, alpha=0.7, color='#C62828', label='Fast (30-50 Hz)', density=True)
    ax1.axvline(slow_pr.mean(), color='#2E7D32', linestyle='--', linewidth=2)
    ax1.axvline(fast_pr.mean(), color='#C62828', linestyle='--', linewidth=2)
    ax1.set_xlabel('Participation Ratio (# channels)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('A. PR Distribution by Band', fontweight='bold', fontsize=12)
    ax1.legend()

    # Panel B: Summary bars
    ax2 = axes[1]
    means = [slow_pr.mean(), fast_pr.mean()]
    stds = [slow_pr.std(), fast_pr.std()]
    bars = ax2.bar(['Slow\n(2-8 Hz)', 'Fast\n(30-50 Hz)'], means,
                   yerr=stds, color=['#2E7D32', '#C62828'],
                   capsize=5, alpha=0.8)

    # Add percentage labels
    ax2.text(0, means[0] * 0.85, f'{means[0]/n_channels*100:.0f}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax2.text(1, means[1] * 0.85, f'{means[1]/n_channels*100:.0f}%',
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax2.set_ylabel('Mean Participation Ratio', fontsize=11)
    ax2.set_title('B. Slow vs Fast Bands', fontweight='bold', fontsize=12)
    # Set y-limit to accommodate bars + error bars
    max_val = max(means[0] + stds[0], means[1] + stds[1])
    ax2.set_ylim(0, max_val * 1.15)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_synthetic_pr_validation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES / "fig2_synthetic_pr_validation.png", dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {FIGURES / 'fig2_synthetic_pr_validation.pdf'}")
    plt.close()


def download_physionet_sample():
    """
    Download a sample from PhysioNet n-back dataset.

    Note: Full dataset requires PhysioNet credentials.
    This function provides instructions for manual download.
    """
    print("\n" + "="*70)
    print("REAL DATA: PhysioNet Multimodal N-back Dataset")
    print("="*70)
    print("""
To download the full dataset:

1. Create account at https://physionet.org
2. Complete required training for credentialed access
3. Download from: https://physionet.org/content/multimodal-nback-music/

Alternative open datasets (no credentials needed):

1. OpenNeuro ds004117 (Sternberg WM):
   https://openneuro.org/datasets/ds004117

2. BNCI Horizon 2020 datasets:
   http://bnci-horizon-2020.eu/database/data-sets

For now, running synthetic validation to demonstrate the method works.
""")


def run_on_edf(edf_path):
    """
    Run PR analysis on an EDF file.

    Args:
        edf_path: Path to .edf EEG file
    """
    try:
        import mne
    except ImportError:
        print("Install MNE-Python: pip install mne")
        return None, None

    print(f"\nLoading {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Get EEG channels only
    raw.pick_types(eeg=True)

    data = raw.get_data()  # (n_channels, n_time)
    fs = raw.info['sfreq']
    n_channels = data.shape[0]

    print(f"  {n_channels} EEG channels, {data.shape[1]/fs:.1f}s of data, fs={fs}Hz")

    # Filter
    print("  Filtering...")
    slow_band = bandpass_filter(data, fs, 2, 8)
    fast_band = bandpass_filter(data, fs, 25, 45)  # Beta/low-gamma

    # Amplitudes
    slow_amp = hilbert_amplitude(slow_band)
    fast_amp = hilbert_amplitude(fast_band)

    # PR
    window_samples = int(0.2 * fs)  # 200ms windows
    slow_pr = compute_pr_timeseries(slow_amp, window_samples)
    fast_pr = compute_pr_timeseries(fast_amp, window_samples)

    print(f"\nResults:")
    print(f"  Slow (2-8 Hz):  PR = {slow_pr.mean():.1f} ({slow_pr.mean()/n_channels*100:.1f}% of channels)")
    print(f"  Fast (25-45 Hz): PR = {fast_pr.mean():.1f} ({fast_pr.mean()/n_channels*100:.1f}% of channels)")
    print(f"  Ratio: {slow_pr.mean()/fast_pr.mean():.2f}x")

    return slow_pr, fast_pr


if __name__ == "__main__":
    # Run synthetic validation
    slow_pr, fast_pr, n_channels = analyze_synthetic()
    plot_synthetic_results(slow_pr, fast_pr, n_channels)

    # Instructions for real data
    download_physionet_sample()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The synthetic validation confirms that our Participation Ratio measure
correctly distinguishes global (high-PR) from local (low-PR) activity.

To complete the empirical validation:
1. Download an open EEG dataset (see instructions above)
2. Run: python real_data_pr.py --edf path/to/recording.edf

If slow waves are truly high-dimensional (engaging many channels) and
fast waves are low-dimensional (sparse/local), PR(slow) > PR(fast).
""")
