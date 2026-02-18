#!/usr/bin/env python3
"""
Investigation: EEG Geometry
============================

What geometric signatures does EEG carry?  Run 8 EEG classes through the
full 233-metric exotic geometry framework with surrogate comparisons
and cross-class discrimination.

**Datasets:**
- Bonn/Andrzejak (5 classes): seizure, tumor EO, healthy EO, eyes closed, eyes open
- PhysioNet eegmmidb (3 regional aggregates): occipital, frontal, central

DIRECTIONS:
D1: Structure detection — each class vs shuffled surrogate
D2: Cross-class discrimination — pairwise 8×8 matrix
D3: IAAFT surrogate hierarchy — nonlinear structure beyond spectrum
D4: Per-geometry heatmap — which geometric lenses detect what
D5: EEG in the structure space — PCA with atlas reference sources
"""

import sys
import os
import csv
import json

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from exotic_geometry_framework import GeometryAnalyzer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =========================================================================
# CONFIG
# =========================================================================

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05
IAAFT_ITER = 100

EEG_PATH = '/tmp/eeg_geometry'
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'figures')
ATLAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'figures', 'structure_atlas_data.json')
BONN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'data', 'eeg',
                         'Epileptic Seizure Recognition.csv')

# Class definitions
BONN_CLASSES = [
    (1, 'Seizure'),
    (2, 'Tumor EO'),
    (3, 'Healthy EO'),
    (4, 'Eyes Closed'),
    (5, 'Eyes Open'),
]
PHYSIONET_REGIONS = [
    ('Occipital', ['O1..', 'Oz..', 'O2..']),
    ('Frontal',   ['Fz..', 'F3..', 'F4..']),
    ('Central',   ['Cz..', 'C3..', 'C4..']),
]
ALL_CLASSES = [b[1] for b in BONN_CLASSES] + [r[0] for r in PHYSIONET_REGIONS]
N_CLASSES = len(ALL_CLASSES)

CLASS_COLORS = [
    '#E91E63',   # Seizure — red/pink
    '#FF9800',   # Tumor EO — orange
    '#4CAF50',   # Healthy EO — green
    '#2196F3',   # Eyes Closed — blue
    '#00BCD4',   # Eyes Open — cyan
    '#9C27B0',   # Occipital — purple
    '#FFEB3B',   # Frontal — yellow
    '#795548',   # Central — brown
]

# =========================================================================
# FRAMEWORK SETUP
# =========================================================================

_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"Framework: {N_METRICS} metrics (Bonferroni α={BONF:.2e})")


# =========================================================================
# UTILITIES
# =========================================================================

def _to_uint8(signal):
    """Normalize float array to uint8 [0, 255]."""
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((signal - lo) / (hi - lo) * 255).astype(np.uint8)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def count_sig(data_a, data_b, alpha):
    """Count significant differences: p < alpha AND |d| > 0.8."""
    sig = 0
    findings = []
    for f in METRIC_NAMES:
        a = np.array(data_a.get(f, []))
        b = np.array(data_b.get(f, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < alpha and abs(d) > 0.8:
            sig += 1
            findings.append((f, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def collect_metrics(analyzer, data_arrays):
    """Run framework on multiple arrays, collect per-metric value lists."""
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def mean_profile(metrics_dict):
    """Collapse per-metric lists to means (for PCA projection)."""
    return [np.mean(metrics_dict.get(m, [0.0])) for m in METRIC_NAMES]


def iaaft_surrogate(data, rng, n_iter=IAAFT_ITER):
    """IAAFT surrogate preserving power spectrum + marginal distribution."""
    data_f = data.astype(np.float64)
    n = len(data_f)
    target_amp = np.abs(np.fft.rfft(data_f))
    sorted_data = np.sort(data_f)
    surr = data_f.copy()
    rng.shuffle(surr)
    for _ in range(n_iter):
        s_fft = np.fft.rfft(surr)
        s_fft = target_amp * np.exp(1j * np.angle(s_fft))
        surr = np.fft.irfft(s_fft, n=n)
        surr = sorted_data[np.argsort(np.argsort(surr))]
    return np.clip(surr, 0, 255).astype(np.uint8)


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# DATA LOADING
# =========================================================================

_BONN_DATA = None

def _get_bonn():
    """Load Bonn/Andrzejak EEG dataset. Returns (signals[11500,178], labels[11500])."""
    global _BONN_DATA
    if _BONN_DATA is None:
        with open(BONN_PATH) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = []
            for r in reader:
                rows.append([float(v) for v in r[1:]])  # skip unnamed col
        arr = np.array(rows)
        _BONN_DATA = (arr[:, :-1], arr[:, -1].astype(int))
    return _BONN_DATA


def gen_bonn_class(class_id, rng, size=DATA_SIZE):
    """Generate uint8 sample by concatenating random segments from a Bonn class."""
    signals, labels = _get_bonn()
    pool = signals[labels == class_id]
    n_segs = size // 178 + 1
    indices = rng.choice(len(pool), size=n_segs, replace=True)
    concat = pool[indices].flatten()[:size]
    return _to_uint8(concat)


def load_physionet():
    """Load PhysioNet eegmmidb eyes-closed resting EEG.
    Returns dict: channel_name -> list of (subject_id, signal_array).
    """
    import mne
    all_channels = {}
    needed = set()
    for _, chs in PHYSIONET_REGIONS:
        needed.update(chs)

    n_ok = 0
    for subj in range(1, 110):
        try:
            files = mne.datasets.eegbci.load_data(
                subjects=subj, runs=[2],
                path=EEG_PATH, update_path=False, verbose=False)
            raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
            raw.filter(1.0, 55.0, verbose=False)
            ch_names = raw.ch_names
            data = raw.get_data()
            for i, ch in enumerate(ch_names):
                if ch in needed:
                    if ch not in all_channels:
                        all_channels[ch] = []
                    all_channels[ch].append((subj, data[i]))
            n_ok += 1
        except Exception as e:
            if subj <= 5:
                print(f"  Subject {subj}: FAILED ({e})")
        if subj % 20 == 0:
            print(f"  PhysioNet: {subj}/109 subjects loaded")

    print(f"  PhysioNet: {n_ok} subjects OK, {len(all_channels)} channels")
    return all_channels


def gen_physionet_region(region_channels, physio_data, rng, size=DATA_SIZE):
    """Generate uint8 sample: random subject, random channel from region,
    contiguous block of `size` samples, quantized to uint8."""
    # Pool all (subject, signal) tuples from matching channels
    pool = []
    for ch in region_channels:
        if ch in physio_data:
            pool.extend(physio_data[ch])
    if not pool:
        raise ValueError(f"No data for channels {region_channels}")

    subj_id, signal = pool[rng.integers(len(pool))]
    # Pick random contiguous block
    max_start = len(signal) - size
    if max_start <= 0:
        # Signal too short, use what we have
        block = signal[:size] if len(signal) >= size else np.tile(signal, size // len(signal) + 1)[:size]
    else:
        start = rng.integers(max_start)
        block = signal[start:start + size]
    return _to_uint8(block)


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 78)
    print("INVESTIGATION: EEG Geometry")
    print("=" * 78)

    rng = np.random.default_rng(42)
    analyzer = GeometryAnalyzer().add_all_geometries()

    # ------------------------------------------------------------------
    # Generate data for all 8 classes
    # ------------------------------------------------------------------
    print("\nGenerating EEG data...")

    # Bonn classes (5)
    class_data = {}    # class_name -> list of uint8 arrays
    class_metrics = {} # class_name -> {metric: [values]}

    for class_id, name in BONN_CLASSES:
        print(f"  {name} (Bonn class {class_id})...", end=" ", flush=True)
        arrays = [gen_bonn_class(class_id, rng) for _ in range(N_TRIALS)]
        class_data[name] = arrays
        class_metrics[name] = collect_metrics(analyzer, arrays)
        print("done")

    # PhysioNet regions (3)
    print("\n  Loading PhysioNet data...")
    physio_data = load_physionet()

    for region_name, channels in PHYSIONET_REGIONS:
        print(f"  {region_name} (PhysioNet)...", end=" ", flush=True)
        arrays = [gen_physionet_region(channels, physio_data, rng) for _ in range(N_TRIALS)]
        class_data[region_name] = arrays
        class_metrics[region_name] = collect_metrics(analyzer, arrays)
        print("done")

    # ------------------------------------------------------------------
    # Generate shuffled reference (shared across D1)
    # ------------------------------------------------------------------
    print("\n  Shuffled reference...", end=" ", flush=True)
    shuffled_arrays = []
    for _ in range(N_TRIALS):
        arr = rng.integers(0, 256, DATA_SIZE, dtype=np.uint8)
        shuffled_arrays.append(arr)
    shuffled_metrics = collect_metrics(analyzer, shuffled_arrays)
    print("done")

    # Self-check: shuffled vs separate shuffled batch
    print("  Self-check (shuffled vs shuffled)...", end=" ", flush=True)
    shuffled2 = [rng.integers(0, 256, DATA_SIZE, dtype=np.uint8) for _ in range(N_TRIALS)]
    shuf2_metrics = collect_metrics(analyzer, shuffled2)
    sc_sig, _ = count_sig(shuffled_metrics, shuf2_metrics, BONF)
    print(f"{sc_sig}/{N_METRICS} sig (expect ~0)")

    # ==================================================================
    # D1: Structure Detection — each class vs shuffled
    # ==================================================================
    print("\n" + "=" * 78)
    print("D1: Structure Detection (vs shuffled)")
    print("=" * 78)

    d1_counts = {}
    d1_findings = {}
    for name in ALL_CLASSES:
        n_sig, findings = count_sig(class_metrics[name], shuffled_metrics, BONF)
        d1_counts[name] = n_sig
        d1_findings[name] = findings
        print(f"  {name:15s}: {n_sig:3d}/{N_METRICS} significant metrics")

    # ==================================================================
    # D2: Cross-Class Discrimination — pairwise 8×8
    # ==================================================================
    print("\n" + "=" * 78)
    print("D2: Cross-Class Discrimination")
    print("=" * 78)

    d2_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    d2_top_metrics = {}  # (i,j) -> top 3 findings
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            n_sig, findings = count_sig(
                class_metrics[ALL_CLASSES[i]],
                class_metrics[ALL_CLASSES[j]],
                BONF)
            d2_matrix[i, j] = n_sig
            d2_matrix[j, i] = n_sig
            d2_top_metrics[(i, j)] = findings[:3]
            if n_sig > 0:
                print(f"  {ALL_CLASSES[i]:15s} vs {ALL_CLASSES[j]:15s}: "
                      f"{n_sig:3d} sig metrics")

    # ==================================================================
    # D3: IAAFT Surrogate Hierarchy — nonlinear structure
    # ==================================================================
    print("\n" + "=" * 78)
    print("D3: IAAFT Surrogate Test (nonlinear structure)")
    print("=" * 78)

    d3_counts = {}
    for name in ALL_CLASSES:
        print(f"  {name:15s}...", end=" ", flush=True)
        # One IAAFT surrogate per trial (25 total — matched to real trials)
        iaaft_arrays = [iaaft_surrogate(arr, rng) for arr in class_data[name]]
        iaaft_metrics = collect_metrics(analyzer, iaaft_arrays)
        n_sig, _ = count_sig(class_metrics[name], iaaft_metrics, BONF)
        d3_counts[name] = n_sig
        print(f"{n_sig:3d}/{N_METRICS} nonlinear")

    # ==================================================================
    # D4: Per-Geometry Detection Heatmap
    # ==================================================================
    print("\n" + "=" * 78)
    print("D4: Per-Geometry Detection")
    print("=" * 78)

    geometry_names = sorted(set(m.split(':')[0] for m in METRIC_NAMES))
    geo_to_metrics = {g: [m for m in METRIC_NAMES if m.startswith(g + ':')]
                      for g in geometry_names}
    n_geos = len(geometry_names)

    d4_matrix = np.zeros((n_geos, N_CLASSES), dtype=int)
    for gi, geo in enumerate(geometry_names):
        geo_metrics = geo_to_metrics[geo]
        geo_alpha = ALPHA / len(geo_metrics) if geo_metrics else ALPHA
        for ci, name in enumerate(ALL_CLASSES):
            sig = 0
            for m in geo_metrics:
                a = np.array(class_metrics[name].get(m, []))
                b = np.array(shuffled_metrics.get(m, []))
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < geo_alpha and abs(d) > 0.8:
                    sig += 1
            d4_matrix[gi, ci] = sig

    # Print geometries that detect structure in any class
    active_geos = [(gi, geo) for gi, geo in enumerate(geometry_names)
                   if d4_matrix[gi].sum() > 0]
    print(f"  {len(active_geos)}/{n_geos} geometries detect structure in ≥1 class")
    for gi, geo in active_geos[:15]:
        counts = ' '.join(f'{d4_matrix[gi, ci]:2d}' for ci in range(N_CLASSES))
        print(f"    {geo:30s}: [{counts}]")

    # ==================================================================
    # D5: EEG in the Structure Space
    # ==================================================================
    print("\n" + "=" * 78)
    print("D5: EEG in Structure Space")
    print("=" * 78)

    # Load atlas data
    with open(ATLAS_PATH) as f:
        atlas = json.load(f)

    atlas_metric_names = atlas['metric_names']
    atlas_profiles = np.array(atlas['profiles'])  # (n_sources, n_metrics)
    atlas_sources = atlas['sources']

    # Align on shared metrics (atlas=233, current framework may differ)
    shared_metrics = [m for m in atlas_metric_names if m in METRIC_NAMES]
    atlas_idx = [atlas_metric_names.index(m) for m in shared_metrics]
    print(f"  Shared metrics: {len(shared_metrics)}/{len(atlas_metric_names)} atlas, "
          f"{len(shared_metrics)}/{N_METRICS} framework")

    # Build aligned profile matrices
    atlas_aligned = atlas_profiles[:, atlas_idx]

    eeg_profiles = []
    for name in ALL_CLASSES:
        profile = []
        for m in shared_metrics:
            vals = class_metrics[name].get(m, [])
            profile.append(np.mean(vals) if vals else 0.0)
        eeg_profiles.append(profile)
    eeg_profiles = np.array(eeg_profiles)

    # Combine and z-score before PCA (prevents high-variance metrics from dominating)
    all_profiles = np.vstack([atlas_aligned, eeg_profiles])
    all_profiles = np.nan_to_num(all_profiles, nan=0.0)

    mu = all_profiles.mean(axis=0)
    sd = all_profiles.std(axis=0)
    sd[sd < 1e-15] = 1.0
    all_z = (all_profiles - mu) / sd

    pca = PCA(n_components=10)
    all_pcs = pca.fit_transform(all_z)

    n_atlas = len(atlas_sources)
    atlas_pcs = all_pcs[:n_atlas]
    eeg_pcs = all_pcs[n_atlas:]

    var_expl = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={var_expl[0]:.1%}, PC2={var_expl[1]:.1%}")

    # Find nearest atlas neighbors for each EEG class (Euclidean in z-scored PCA)
    from scipy.spatial.distance import cdist
    dists = cdist(eeg_pcs[:, :10], atlas_pcs[:, :10])
    for ci, name in enumerate(ALL_CLASSES):
        nearest_idx = np.argsort(dists[ci])[:3]
        neighbors = [(atlas_sources[j]['name'], dists[ci, j]) for j in nearest_idx]
        print(f"  {name:15s} neighbors: " +
              ', '.join(f"{n} ({d:.2f})" for n, d in neighbors))

    # ==================================================================
    # FIGURE
    # ==================================================================
    print("\nGenerating figure...")

    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 24), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ------------------------------------------------------------------
    # Panel 1: D1 — Structure detection bar chart
    # ------------------------------------------------------------------
    ax1 = _dark_ax(fig.add_subplot(gs[0, 0]))
    bars = [d1_counts[name] for name in ALL_CLASSES]
    x = np.arange(N_CLASSES)
    ax1.bar(x, bars, color=CLASS_COLORS, edgecolor='none', width=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ALL_CLASSES, rotation=35, ha='right', fontsize=7)
    ax1.set_ylabel('Significant metrics (vs shuffled)', fontsize=9)
    ax1.set_title('D1: Structure Detection', fontsize=11, fontweight='bold')
    ax1.axhline(y=sc_sig, color='#666666', linestyle='--', linewidth=0.8,
                label=f'Self-check baseline ({sc_sig})')
    ax1.legend(fontsize=7, loc='upper right')
    for i, v in enumerate(bars):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=7, color='white')

    # ------------------------------------------------------------------
    # Panel 2: D2 — Cross-class discrimination heatmap
    # ------------------------------------------------------------------
    ax2 = _dark_ax(fig.add_subplot(gs[0, 1]))
    # Mask diagonal
    d2_display = d2_matrix.astype(float)
    np.fill_diagonal(d2_display, np.nan)
    im2 = ax2.imshow(d2_display, cmap='magma', aspect='auto',
                     interpolation='nearest')
    ax2.set_xticks(range(N_CLASSES))
    ax2.set_yticks(range(N_CLASSES))
    ax2.set_xticklabels(ALL_CLASSES, rotation=45, ha='right', fontsize=7)
    ax2.set_yticklabels(ALL_CLASSES, fontsize=7)
    ax2.set_title('D2: Cross-Class Discrimination', fontsize=11, fontweight='bold')
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cb2.set_label('Significant metrics', fontsize=8)
    cb2.ax.tick_params(labelsize=7, colors='#cccccc')
    # Annotate cells
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j:
                val = d2_matrix[i, j]
                color = 'white' if val < d2_matrix.max() * 0.6 else 'black'
                ax2.text(j, i, str(val), ha='center', va='center',
                         fontsize=6, color=color)

    # ------------------------------------------------------------------
    # Panel 3: D3 — IAAFT nonlinear structure
    # ------------------------------------------------------------------
    ax3 = _dark_ax(fig.add_subplot(gs[1, 0]))
    iaaft_bars = [d3_counts[name] for name in ALL_CLASSES]
    ax3.bar(x, iaaft_bars, color=CLASS_COLORS, edgecolor='none', width=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(ALL_CLASSES, rotation=35, ha='right', fontsize=7)
    ax3.set_ylabel('Significant metrics (vs IAAFT)', fontsize=9)
    ax3.set_title('D3: Nonlinear Structure (beyond spectrum)', fontsize=11,
                  fontweight='bold')
    for i, v in enumerate(iaaft_bars):
        ax3.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=7, color='white')

    # ------------------------------------------------------------------
    # Panel 4: D4 — Geometry × class heatmap
    # ------------------------------------------------------------------
    ax4 = _dark_ax(fig.add_subplot(gs[1, 1]))
    # Sort by total detections (descending), show top 30
    active_mask = d4_matrix.sum(axis=1) > 0
    active_indices = np.where(active_mask)[0]
    sort_order = np.argsort(-d4_matrix[active_indices].sum(axis=1))
    top_indices = active_indices[sort_order[:30]]
    top_data = d4_matrix[top_indices]
    top_names = [geometry_names[i] for i in top_indices]

    if len(top_data) > 0:
        im4 = ax4.imshow(top_data, cmap='YlOrRd', aspect='auto',
                         interpolation='nearest')
        ax4.set_xticks(range(N_CLASSES))
        ax4.set_xticklabels(ALL_CLASSES, rotation=45, ha='right', fontsize=7)
        ax4.set_yticks(range(len(top_names)))
        ax4.set_yticklabels(top_names, fontsize=5.5)
        ax4.set_title('D4: Geometry × Class Detections (top 30)', fontsize=11,
                      fontweight='bold')
        cb4 = fig.colorbar(im4, ax=ax4, shrink=0.8, pad=0.02)
        cb4.set_label('Sig. metrics', fontsize=8)
        cb4.ax.tick_params(labelsize=7, colors='#cccccc')

    # ------------------------------------------------------------------
    # Panel 5: D5 — PCA structure space
    # ------------------------------------------------------------------
    ax5 = _dark_ax(fig.add_subplot(gs[2, 0]))

    # Atlas sources (background)
    domain_colors = atlas.get('domain_colors', {})
    for i, src in enumerate(atlas_sources):
        domain = src.get('domain', 'other')
        c = domain_colors.get(domain, '#555555')
        ax5.scatter(atlas_pcs[i, 0], atlas_pcs[i, 1],
                    c=c, s=15, alpha=0.3, edgecolors='none')

    # EEG classes (foreground)
    for ci, name in enumerate(ALL_CLASSES):
        ax5.scatter(eeg_pcs[ci, 0], eeg_pcs[ci, 1],
                    c=CLASS_COLORS[ci], s=120, edgecolors='white',
                    linewidth=1.5, zorder=10, label=name)
        ax5.annotate(name, (eeg_pcs[ci, 0], eeg_pcs[ci, 1]),
                     fontsize=7, color='white',
                     xytext=(8, 4), textcoords='offset points')

    ax5.set_xlabel(f'PC1 ({var_expl[0]:.1%})', fontsize=9)
    ax5.set_ylabel(f'PC2 ({var_expl[1]:.1%})', fontsize=9)
    ax5.set_title('D5: EEG in Structure Space', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=6, loc='best', framealpha=0.3, ncol=2)

    # ------------------------------------------------------------------
    # Panel 6: Summary / top discriminating metrics
    # ------------------------------------------------------------------
    ax6 = _dark_ax(fig.add_subplot(gs[2, 1]))
    ax6.axis('off')

    # Collect top findings across all D1 comparisons
    all_findings = []
    for name in ALL_CLASSES:
        for f, d, p in d1_findings[name][:5]:
            all_findings.append((name, f, d, p))
    all_findings.sort(key=lambda x: -abs(x[2]))

    lines = ['TOP DISCRIMINATING METRICS (D1: vs shuffled)', '']
    lines.append(f'{"Class":15s}  {"Metric":40s}  {"Cohen d":>8s}  {"p":>10s}')
    lines.append('─' * 78)
    seen = set()
    count = 0
    for cls, metric, d, p in all_findings:
        if metric not in seen and count < 20:
            lines.append(f'{cls:15s}  {metric:40s}  {d:+8.2f}  {p:10.2e}')
            seen.add(metric)
            count += 1

    lines.append('')
    lines.append('─' * 78)
    lines.append('')

    # Add D2 highlights
    lines.append('KEY DISCRIMINATIONS (D2):')
    # Seizure vs healthy comparisons
    for i, j in [(0, 2), (0, 3), (0, 4), (3, 4)]:
        if i < N_CLASSES and j < N_CLASSES:
            val = d2_matrix[i, j]
            lines.append(f'  {ALL_CLASSES[i]} vs {ALL_CLASSES[j]}: {val} sig metrics')

    # PhysioNet inter-region
    for i in range(5, 8):
        for j in range(i + 1, 8):
            if i < N_CLASSES and j < N_CLASSES:
                val = d2_matrix[i, j]
                lines.append(f'  {ALL_CLASSES[i]} vs {ALL_CLASSES[j]}: {val} sig metrics')

    summary_text = '\n'.join(lines)
    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes,
             fontsize=6.5, fontfamily='monospace', color='#cccccc',
             verticalalignment='top')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, 'eeg_geometry.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='#181818', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ==================================================================
    # CONSOLE SUMMARY
    # ==================================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\n  Self-check (shuffled vs shuffled): {sc_sig}/{N_METRICS}")
    print(f"\n  D1 Structure detection (vs shuffled):")
    for name in ALL_CLASSES:
        print(f"    {name:15s}: {d1_counts[name]:3d}/{N_METRICS}")

    print(f"\n  D3 Nonlinear structure (vs IAAFT):")
    for name in ALL_CLASSES:
        print(f"    {name:15s}: {d3_counts[name]:3d}/{N_METRICS}")

    # D2 summary stats
    upper = d2_matrix[np.triu_indices(N_CLASSES, k=1)]
    print(f"\n  D2 Cross-class discrimination:")
    print(f"    Mean pairwise sig metrics: {upper.mean():.1f}")
    print(f"    Max:  {upper.max()} ({ALL_CLASSES[np.unravel_index(d2_matrix.argmax(), d2_matrix.shape)[0]]} "
          f"vs {ALL_CLASSES[np.unravel_index(d2_matrix.argmax(), d2_matrix.shape)[1]]})")
    print(f"    Min non-zero: {upper[upper > 0].min() if (upper > 0).any() else 0}")

    # D4 summary
    total_detections = d4_matrix.sum()
    print(f"\n  D4 Geometry detections: {total_detections} total across "
          f"{(d4_matrix.sum(axis=1) > 0).sum()}/{n_geos} geometries")

    # D5 summary
    print(f"\n  D5 Structure space:")
    print(f"    PCA: PC1={var_expl[0]:.1%}, PC2={var_expl[1]:.1%}, "
          f"PC1-10={sum(var_expl[:10]):.1%}")


if __name__ == '__main__':
    main()
