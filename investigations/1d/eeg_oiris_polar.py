#!/usr/bin/env python3
"""
O.IRIS Polar Map — Three-Panel Comparison
==========================================

Three circular pupil maps showing progressive improvement:
  Left:   O.IRIS original (spectral slope, ACF decay, flicker)
          with m=2.0 empirical attractor marked alongside phi=2.18
  Center: Temporal structure (rank-PCA on top-5 EGF discriminators)
          Shows meditation is unique; rest/sleep merge
  Right:  Connectivity-aware (temporal persistence → r, per-window
          mean alpha coherence → θ). Separates all three states.

Uses 10-second windowed approach on actual ds001787 + ds003768 datasets.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import signal
from scipy.stats import linregress, rankdata

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eeg_oiris import (get_meditation_files, get_sleep_rest_files,
                        to_uint8, _get_mne)
from eeg_oiris_rigorous import load_raw

from exotic_geometry_framework import GeometryAnalyzer

WINDOW_SEC = 10
TARGET_SFREQ = 250
MAX_WINDOWS_PER_STATE = 200

PHI_M = 2.18
BROWN_M = 2.00  # empirical attractor (Brownian exponent)

TOP_METRICS = [
    'Sol (Thurston):dz_persistence',
    'H² × ℝ (Thurston):radial_temporal_memory',
    'Cantor Set:jump_entropy',
    'SL(2,ℝ) (Thurston):trace_autocorrelation',
    'Lorentzian:crossing_density',
]

BG = '#181818'
FG = '#e0e0e0'
STATE_COLORS = {
    'Meditation': '#FF9800',
    'Rest':       '#2196F3',
    'Sleep':      '#9C27B0',
}


# =========================================================================
# O.IRIS METRICS
# =========================================================================

def compute_spectral_slope(data, sfreq):
    nperseg = min(int(4 * sfreq), len(data))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= 1) & (freqs <= 40) & (psd > 0)
    if np.sum(mask) < 5:
        return np.nan
    slope, _, _, _, _ = linregress(np.log10(freqs[mask]), np.log10(psd[mask]))
    return -slope

def compute_acf_decay(data, sfreq, max_lag_sec=10.0):
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic) - np.mean(np.abs(signal.hilbert(data)))
    max_lag = int(min(max_lag_sec * sfreq, len(envelope) // 2))
    if max_lag < 10:
        return np.nan
    acf = np.correlate(envelope[:max_lag*2], envelope[:max_lag*2], mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / (acf[0] + 1e-15)
    below = np.where(acf < 1.0 / np.e)[0]
    return (below[0] / sfreq) if len(below) > 0 else (max_lag / sfreq)

def compute_flicker(data, sfreq):
    envelope = np.abs(signal.hilbert(data))
    mean_env = np.mean(envelope)
    return np.var(envelope) / (mean_env ** 2) if mean_env > 1e-15 else np.nan


# =========================================================================
# MULTI-CHANNEL COHERENCE (per window)
# =========================================================================

def compute_window_coherence(raw_data, sfreq, n_ch_sub=12):
    """Compute mean alpha-band coherence from multi-channel data for one window.
    raw_data: (n_channels, n_samples) for this window."""
    n_ch = raw_data.shape[0]
    ch_idx = np.linspace(0, n_ch - 1, min(n_ch_sub, n_ch), dtype=int)
    seg = raw_data[ch_idx]
    nperseg = min(int(2 * sfreq), seg.shape[1])

    # Alpha band coherence (8-13 Hz)
    coh_vals = []
    for i in range(len(ch_idx)):
        for j in range(i + 1, len(ch_idx)):
            f, Cxy = signal.coherence(seg[i], seg[j], fs=sfreq, nperseg=nperseg)
            mask = (f >= 8) & (f <= 13)
            if np.any(mask):
                coh_vals.append(np.mean(Cxy[mask]))

    return np.mean(coh_vals) if coh_vals else np.nan


# =========================================================================
# WINDOWED DATA EXTRACTION — multi-channel aware
# =========================================================================

def extract_all_windows(files, max_files=5, max_windows=MAX_WINDOWS_PER_STATE):
    """Extract windows with both single-channel and multi-channel data.
    Returns list of dicts with 'signal' (1D averaged), 'multichannel' (2D),
    'sfreq'."""
    windows = []
    for fpath in files[:max_files]:
        print(f"  Loading {os.path.basename(fpath)}...", flush=True)
        try:
            raw = load_raw(fpath, TARGET_SFREQ)
        except Exception as e:
            print(f"    Failed: {e}")
            continue
        data = raw.get_data()  # (n_channels, n_samples)
        sfreq = raw.info['sfreq']
        sig = np.mean(data, axis=0)  # channel-averaged
        window_samples = int(WINDOW_SEC * sfreq)

        for start in range(0, len(sig) - window_samples, window_samples):
            windows.append({
                'signal': sig[start:start + window_samples],
                'multichannel': data[:, start:start + window_samples],
                'sfreq': sfreq,
            })

    if len(windows) > max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(windows), max_windows, replace=False)
        windows = [windows[i] for i in sorted(idx)]

    return windows


def compute_all_metrics(windows, analyzer):
    """Compute O.IRIS metrics, EGF metrics, and coherence per window."""
    results = []
    for i, w in enumerate(windows):
        sig = w['signal']
        sfreq = w['sfreq']

        # O.IRIS
        m_slope = compute_spectral_slope(sig, sfreq)
        acf_decay = compute_acf_decay(sig, sfreq)
        flicker = compute_flicker(sig, sfreq)

        # Coherence
        coh = compute_window_coherence(w['multichannel'], sfreq)

        # EGF
        data_u8 = to_uint8(sig)
        egf = {}
        if len(data_u8) >= 500:
            res = analyzer.analyze(data_u8)
            for r in res.results:
                for mn, mv in r.metrics.items():
                    key = f"{r.geometry_name}:{mn}"
                    if np.isfinite(mv):
                        egf[key] = mv

        if np.isfinite(m_slope) and np.isfinite(acf_decay) and np.isfinite(flicker):
            results.append({
                'm_slope': m_slope,
                'acf_decay': acf_decay,
                'flicker': flicker,
                'coherence': coh,
                'egf': egf,
            })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(windows)} windows...", flush=True)

    return results


# =========================================================================
# THREE-PANEL POLAR FIGURE
# =========================================================================

def make_polar_figure(data_by_state):

    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG,
        'text.color': FG, 'axes.labelcolor': FG,
    })

    fig = plt.figure(figsize=(30, 10), facecolor=BG)

    # =====================================================================
    # PANEL 1: O.IRIS Original (with corrected focal point)
    # =====================================================================
    ax1 = fig.add_subplot(131, polar=True, facecolor=BG)
    ax1.set_title("O.IRIS Original\n(spectral slope → r, ACF/flicker → θ)",
                   fontsize=11, fontweight='bold', color=FG, pad=20)

    for state, points in data_by_state.items():
        thetas = [np.arctan2(np.log10(max(p['flicker'], 1e-10)),
                             np.log10(max(p['acf_decay'], 1e-10)))
                  for p in points]
        rs_phi = [abs(p['m_slope'] - PHI_M) for p in points]
        ax1.scatter(thetas, rs_phi, c=STATE_COLORS.get(state, '#666'),
                    s=12, alpha=0.5, label=state, edgecolors='none')

    # Mark both focal points
    ax1.plot(0, 0, 'o', color='gold', markersize=10, zorder=5)
    ax1.annotate('φ=2.18', xy=(0.3, 0.05), fontsize=8, color='gold',
                 fontweight='bold')
    ax1.set_rmax(2.5)
    _style_polar(ax1)

    # =====================================================================
    # PANEL 2: Temporal structure (rank-PCA on top-5 metrics)
    # =====================================================================
    ax2 = fig.add_subplot(132, polar=True, facecolor=BG)
    ax2.set_title("Temporal Structure\n(rank-PCA on top-5 EGF metrics)",
                   fontsize=11, fontweight='bold', color=FG, pad=20)

    # Build feature matrix
    from sklearn.decomposition import PCA
    rows, labels = [], []
    for state, points in data_by_state.items():
        for p in points:
            vec = [p['egf'].get(k, np.nan) for k in TOP_METRICS]
            if all(np.isfinite(v) for v in vec):
                rows.append(vec)
                labels.append(state)

    if len(rows) >= 10:
        X = np.array(rows)
        n = len(X)
        X_rank = np.zeros_like(X)
        for col in range(X.shape[1]):
            X_rank[:, col] = rankdata(X[:, col]) / n

        pca = PCA(n_components=2)
        pc = pca.fit_transform(X_rank)
        print(f"  Temporal PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2={pca.explained_variance_ratio_[1]:.1%}")

        pc1_rank = rankdata(pc[:, 0]) / n
        pc2_rank = rankdata(pc[:, 1]) / n

        med_mask = np.array([s == 'Meditation' for s in labels])
        if np.mean(pc1_rank[med_mask]) > 0.5:
            pc1_rank = 1.0 - pc1_rank

        for state in data_by_state:
            mask = np.array([s == state for s in labels])
            if not np.any(mask):
                continue
            rs = 0.15 + 2.15 * pc1_rank[mask]
            thetas = 2 * np.pi * pc2_rank[mask] - np.pi
            ax2.scatter(thetas, rs, c=STATE_COLORS.get(state, '#666'),
                        s=12, alpha=0.5, label=state, edgecolors='none')

    ax2.plot(0, 0.15, 'o', color='gold', markersize=10, zorder=5)
    ax2.set_rmax(2.5)
    _style_polar(ax2)

    # =====================================================================
    # PANEL 3: Connectivity-aware (persistence → r, coherence → θ)
    # =====================================================================
    ax3 = fig.add_subplot(133, polar=True, facecolor=BG)
    ax3.set_title("Connectivity-Aware\n(Sol:persistence → r, alpha coherence → θ)",
                   fontsize=11, fontweight='bold', color=FG, pad=20)

    # Collect Sol:dz_persistence and coherence
    all_sol, all_coh, all_labels = [], [], []
    for state, points in data_by_state.items():
        for p in points:
            sol = p['egf'].get('Sol (Thurston):dz_persistence', np.nan)
            coh = p.get('coherence', np.nan)
            if np.isfinite(sol) and np.isfinite(coh):
                all_sol.append(sol)
                all_coh.append(coh)
                all_labels.append(state)

    if len(all_sol) >= 10:
        sol_arr = np.array(all_sol)
        coh_arr = np.array(all_coh)
        n = len(sol_arr)

        # Rank-based radius from Sol (inverted: high persistence = small r)
        sol_rank = rankdata(sol_arr) / n
        med_mask = np.array([s == 'Meditation' for s in all_labels])
        if np.mean(sol_rank[med_mask]) > 0.5:
            sol_rank = 1.0 - sol_rank
        rs = 0.15 + 2.15 * sol_rank

        # Angle from coherence (rank-based, full circle)
        coh_rank = rankdata(coh_arr) / n
        thetas = 2 * np.pi * coh_rank - np.pi

        for state in data_by_state:
            mask = np.array([s == state for s in all_labels])
            if not np.any(mask):
                continue
            ax3.scatter(thetas[mask], rs[mask],
                        c=STATE_COLORS.get(state, '#666'),
                        s=12, alpha=0.5, label=state, edgecolors='none')

        # Add coherence annotations
        for state in data_by_state:
            mask = np.array([s == state for s in all_labels])
            if np.any(mask):
                mean_coh = np.mean(coh_arr[mask])
                mean_r = np.mean(rs[mask])
                mean_th = np.mean(thetas[mask])

    ax3.plot(0, 0.15, 'o', color='gold', markersize=10, zorder=5)
    ax3.set_rmax(2.5)
    _style_polar(ax3)

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=8, linestyle='None', label=s)
               for s, c in STATE_COLORS.items()]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=10,
               facecolor='#222', edgecolor='#444', labelcolor=FG,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("O.IRIS Brain State Maps: Original → Temporal → Connectivity-Aware",
                 fontsize=16, fontweight='bold', color=FG, y=1.02)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'eeg_oiris_polar.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def _style_polar(ax):
    ax.tick_params(colors='#888888', labelsize=7)
    ax.grid(True, color='#333333', alpha=0.5)
    ax.set_facecolor(BG)


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()

    med_files = get_meditation_files()
    rest_files, sleep_files = get_sleep_rest_files()
    print(f"Data: {len(med_files)} meditation, {len(rest_files)} rest, "
          f"{len(sleep_files)} sleep files")

    analyzer = GeometryAnalyzer()
    analyzer.add_tier_geometries(tier='complete')

    state_files = {}
    if med_files:
        state_files['Meditation'] = med_files
    if rest_files:
        state_files['Rest'] = rest_files
    if sleep_files:
        state_files['Sleep'] = sleep_files

    data_by_state = {}
    for state, files in state_files.items():
        print(f"\n{'='*60}")
        print(f"Processing {state}...")
        print(f"{'='*60}")
        windows = extract_all_windows(files, max_files=5)
        print(f"  {len(windows)} windows — computing metrics...")
        data_by_state[state] = compute_all_metrics(windows, analyzer)
        print(f"  {len(data_by_state[state])} valid windows")

    # Print summaries
    print(f"\n{'='*60}")
    print("METRIC SUMMARIES")
    print(f"{'='*60}")
    print(f"  {'State':15s} {'median m':>10s} {'mean coh':>10s} {'n':>6s}")
    for state, points in data_by_state.items():
        ms = [p['m_slope'] for p in points]
        cohs = [p['coherence'] for p in points if np.isfinite(p.get('coherence', np.nan))]
        print(f"  {state:15s} {np.median(ms):10.3f} {np.mean(cohs):10.4f} {len(points):6d}")

    make_polar_figure(data_by_state)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
