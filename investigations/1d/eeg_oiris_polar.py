#!/usr/bin/env python3
"""
O.IRIS Polar Map Comparison: Original vs Exotic Geometry Metrics
================================================================

Side-by-side circular pupil maps:
  Left:  O.IRIS original (spectral slope, ACF decay, flicker)
  Right: Exotic geometry top discriminators

Uses the same 10-second windowed approach as O.IRIS, applied to the
actual ds001787 (meditation) and ds003768 (sleep/rest) datasets.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import signal
from scipy.stats import linregress

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Re-use loaders from eeg_oiris
from eeg_oiris import (get_meditation_files, get_sleep_rest_files,
                        get_signal, to_uint8, _get_mne, BANDS)

from exotic_geometry_framework import GeometryAnalyzer

WINDOW_SEC = 10  # O.IRIS uses 10-30s windows
TARGET_SFREQ = 250
MAX_WINDOWS_PER_STATE = 200

# O.IRIS constants
PHI_M = 2.18

# Top metrics from D7 results
TOP_METRICS = [
    'Sol (Thurston):dz_persistence',
    'H² × ℝ (Thurston):radial_temporal_memory',
    'Cantor Set:jump_entropy',
    'SL(2,ℝ) (Thurston):trace_autocorrelation',
    'Lorentzian:crossing_density',
]


# =========================================================================
# O.IRIS ORIGINAL METRICS (from their extract_pupil_metrics.py)
# =========================================================================

def compute_spectral_slope(data, sfreq):
    """Spectral slope m: PSD ~ 1/f^m. Higher = steeper = more persistent."""
    nperseg = int(4 * sfreq)
    if len(data) < nperseg:
        nperseg = len(data)
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= 1) & (freqs <= 40) & (psd > 0)
    if np.sum(mask) < 5:
        return np.nan
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, _, _, _, _ = linregress(log_f, log_p)
    return -slope  # O.IRIS convention: m = -slope


def compute_acf_decay(data, sfreq, max_lag_sec=10.0):
    """ACF decay time: first lag where ACF of Hilbert envelope < 1/e."""
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)
    max_lag = int(min(max_lag_sec * sfreq, len(envelope) // 2))
    if max_lag < 10:
        return np.nan
    acf = np.correlate(envelope[:max_lag*2], envelope[:max_lag*2], mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / (acf[0] + 1e-15)
    threshold = 1.0 / np.e
    below = np.where(acf < threshold)[0]
    if len(below) == 0:
        return max_lag / sfreq
    return below[0] / sfreq


def compute_flicker(data, sfreq):
    """Flicker: variance of Hilbert envelope normalized by mean^2."""
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic)
    mean_env = np.mean(envelope)
    if mean_env < 1e-15:
        return np.nan
    return np.var(envelope) / (mean_env ** 2)


# =========================================================================
# WINDOWED METRIC EXTRACTION
# =========================================================================

def extract_windows(filepath, target_sfreq=TARGET_SFREQ, window_sec=WINDOW_SEC):
    """Load EEG file and extract non-overlapping windows.
    Returns list of (signal_window, sfreq) tuples."""
    sig, sfreq = get_signal(filepath, target_sfreq)
    window_samples = int(window_sec * sfreq)
    n_windows = len(sig) // window_samples
    windows = []
    for i in range(n_windows):
        start = i * window_samples
        w = sig[start:start + window_samples]
        windows.append((w, sfreq))
    return windows


def compute_oiris_metrics(windows):
    """Compute O.IRIS original 3 metrics per window."""
    results = []
    for w, sfreq in windows:
        m_slope = compute_spectral_slope(w, sfreq)
        acf_decay = compute_acf_decay(w, sfreq)
        flicker = compute_flicker(w, sfreq)
        if np.isfinite(m_slope) and np.isfinite(acf_decay) and np.isfinite(flicker):
            results.append({
                'm_slope': m_slope,
                'acf_decay': acf_decay,
                'flicker': flicker,
                'r': abs(m_slope - PHI_M),
                'theta': np.arctan2(np.log10(max(flicker, 1e-10)),
                                    np.log10(max(acf_decay, 1e-10))),
            })
    return results


def compute_egf_metrics(windows, analyzer):
    """Compute exotic geometry metrics per window. Extract top discriminators."""
    results = []
    for i, (w, sfreq) in enumerate(windows):
        data = to_uint8(w)
        if len(data) < 500:
            continue
        res = analyzer.analyze(data)
        metrics = {}
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if np.isfinite(mv):
                    metrics[key] = mv
        if metrics:
            results.append(metrics)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(windows)} windows...", flush=True)
    return results


# =========================================================================
# POLAR PLOT
# =========================================================================

def make_polar_figure(oiris_by_state, egf_by_state):
    """Generate side-by-side polar plots."""

    BG = '#181818'
    FG = '#e0e0e0'
    STATE_COLORS = {
        'Meditation': '#FF9800',
        'Rest':       '#2196F3',
        'Sleep':      '#9C27B0',
    }

    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'text.color': FG,
    })

    fig = plt.figure(figsize=(22, 10), facecolor=BG)

    # === LEFT: O.IRIS Original ===
    ax1 = fig.add_subplot(121, polar=True, facecolor=BG)
    ax1.set_title("O.IRIS Original\n(spectral slope → r, ACF/flicker → θ)",
                   fontsize=12, fontweight='bold', color=FG, pad=20)

    for state, points in oiris_by_state.items():
        if not points:
            continue
        thetas = [p['theta'] for p in points]
        rs = [p['r'] for p in points]
        ax1.scatter(thetas, rs, c=STATE_COLORS.get(state, '#666'),
                    s=12, alpha=0.5, label=state, edgecolors='none')

    # Phi marker at center
    ax1.plot(0, 0, 'o', color='gold', markersize=10, zorder=5)
    ax1.annotate('φ', xy=(0, 0), fontsize=14, color='gold',
                 fontweight='bold', ha='center', va='center')

    ax1.set_rmax(2.5)
    ax1.tick_params(colors='#888888', labelsize=7)
    ax1.grid(True, color='#333333', alpha=0.5)
    ax1.set_facecolor(BG)
    ax1.legend(loc='upper right', fontsize=8, facecolor='#222',
               edgecolor='#444', labelcolor=FG,
               bbox_to_anchor=(1.15, 1.1))

    # === RIGHT: EGF Top Metrics ===
    ax2 = fig.add_subplot(122, polar=True, facecolor=BG)
    ax2.set_title("Exotic Geometry Framework\n(PCA on rank-transformed top-5 metrics)",
                   fontsize=12, fontweight='bold', color=FG, pad=20)

    # Build feature matrix from top 5 metrics across all windows
    from scipy.stats import rankdata
    from sklearn.decomposition import PCA

    feature_keys = TOP_METRICS
    rows = []       # (state_idx, feature_vector)
    state_labels = []
    state_order = []
    for state, metrics_list in egf_by_state.items():
        for m in metrics_list:
            vec = [m.get(k, np.nan) for k in feature_keys]
            if all(np.isfinite(v) for v in vec):
                rows.append(vec)
                state_labels.append(state)
                state_order.append(state)

    if len(rows) < 10:
        print("  Too few valid EGF windows — skipping right panel")
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'figures', 'eeg_oiris_polar.png')
        fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
        return

    X = np.array(rows)
    n = len(X)

    # Rank-transform each feature to [0,1] — eliminates d=36 compression
    X_rank = np.zeros_like(X)
    for col in range(X.shape[1]):
        X_rank[:, col] = rankdata(X[:, col]) / n

    # PCA on rank-transformed features
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X_rank)
    print(f"  PCA on rank-transformed top-5 metrics: "
          f"PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    # Map PC1 → r (inverted: high PC1 for meditation = small radius)
    # Map PC2 → θ
    # Use rank again on PC1 for uniform radial spread
    pc1_rank = rankdata(pc[:, 0]) / n  # [0, 1]
    pc2_rank = rankdata(pc[:, 1]) / n

    # Check which direction = meditation (should be constricted = small r)
    med_mask = np.array([s == 'Meditation' for s in state_labels])
    if np.mean(pc1_rank[med_mask]) > 0.5:
        pc1_rank = 1.0 - pc1_rank  # flip so meditation is near center

    for state in egf_by_state:
        mask = np.array([s == state for s in state_labels])
        if not np.any(mask):
            continue
        # r: rank-based, range [0.15, 2.3] with uniform spread
        rs = 0.15 + 2.15 * pc1_rank[mask]
        # θ: spread PC2 rank across full circle
        thetas = 2 * np.pi * pc2_rank[mask] - np.pi

        ax2.scatter(thetas, rs, c=STATE_COLORS.get(state, '#666'),
                    s=12, alpha=0.5, label=state, edgecolors='none')

    # Center marker
    ax2.plot(0, 0.15, 'o', color='gold', markersize=10, zorder=5)

    ax2.set_rmax(2.5)
    ax2.tick_params(colors='#888888', labelsize=7)
    ax2.grid(True, color='#333333', alpha=0.5)
    ax2.set_facecolor(BG)
    ax2.legend(loc='upper right', fontsize=8, facecolor='#222',
               edgecolor='#444', labelcolor=FG,
               bbox_to_anchor=(1.15, 1.1))

    fig.suptitle("O.IRIS Brain State Maps: Original vs Exotic Geometry",
                 fontsize=16, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'eeg_oiris_polar.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()

    med_files = get_meditation_files()
    rest_files, sleep_files = get_sleep_rest_files()
    print(f"Data: {len(med_files)} meditation, {len(rest_files)} rest, "
          f"{len(sleep_files)} sleep files")

    # Setup analyzer for EGF metrics
    print("Initializing analyzer...")
    analyzer = GeometryAnalyzer()
    analyzer.add_tier_geometries(tier='complete')

    # Collect windows per state
    state_files = {}
    if med_files:
        state_files['Meditation'] = med_files
    if rest_files:
        state_files['Rest'] = rest_files
    if sleep_files:
        state_files['Sleep'] = sleep_files

    oiris_by_state = {}
    egf_by_state = {}

    for state, files in state_files.items():
        print(f"\n{'='*60}")
        print(f"Processing {state} ({len(files)} files)...")
        print(f"{'='*60}")

        # Collect windows from all files
        all_windows = []
        for fpath in files[:5]:  # limit to 5 files per state for speed
            print(f"  Loading {os.path.basename(fpath)}...", flush=True)
            try:
                windows = extract_windows(fpath)
                all_windows.extend(windows)
            except Exception as e:
                print(f"    Failed: {e}")
                continue

        # Cap windows
        if len(all_windows) > MAX_WINDOWS_PER_STATE:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_windows), MAX_WINDOWS_PER_STATE, replace=False)
            all_windows = [all_windows[i] for i in sorted(indices)]

        print(f"  {len(all_windows)} windows")

        # O.IRIS original metrics
        print(f"  Computing O.IRIS metrics...", flush=True)
        oiris_by_state[state] = compute_oiris_metrics(all_windows)
        print(f"    {len(oiris_by_state[state])} valid windows")

        # EGF metrics
        print(f"  Computing EGF metrics ({len(all_windows)} windows)...", flush=True)
        egf_by_state[state] = compute_egf_metrics(all_windows, analyzer)
        print(f"    {len(egf_by_state[state])} valid windows")

    # Print O.IRIS metric summaries (compare to paper)
    print(f"\n{'='*60}")
    print("O.IRIS METRIC COMPARISON (our extraction vs paper)")
    print(f"{'='*60}")
    print(f"  {'State':15s} {'median m':>10s} {'mean m':>10s} {'SD':>8s} {'|m-2.18|':>10s}")
    for state, points in oiris_by_state.items():
        if not points:
            continue
        ms = [p['m_slope'] for p in points]
        median_m = np.median(ms)
        mean_m = np.mean(ms)
        sd_m = np.std(ms)
        dist = abs(median_m - PHI_M)
        print(f"  {state:15s} {median_m:10.3f} {mean_m:10.3f} {sd_m:8.3f} {dist:10.3f}")

    # Generate figure
    print("\nGenerating polar comparison figure...")
    make_polar_figure(oiris_by_state, egf_by_state)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
