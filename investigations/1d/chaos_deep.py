#!/usr/bin/env python3
"""
Investigation: Deep Chaos Analysis — Dynamics & Geometry
=========================================================

Follow-up to chaos.py, probing the relationship between dynamical properties
(Lyapunov exponents, bifurcation points) and geometric signatures.

Directions:
  D1: Route to Chaos — Sweep the logistic map 'r' parameter (3.5 to 4.0) and
      track geometric metrics through the period-doubling cascade to chaos.
  D2: Lyapunov Correlation — Compare geometric metrics with the theoretical
      Lyapunov exponent. Which geometries track "chaos intensity" best?
  D3: Attractor Geometry — Compute correlation dimension (Grassberger-Procaccia)
      of the return map (x_n, x_{n+1}) and analyze 2D density images with
      spatial geometries. Compare periodic vs chaotic regimes.
  D4: Byte vs Unit Mode — Compare detection power against random baselines
      in uint8 (byte) vs float [0,1] (unit) mode at the edge of chaos.
  D5: Phase Space Volume Proxies — Correlate geometric metrics with the
      ground-truth occupied fraction of [0,1] to find volume proxy metrics.

Methodology: N_TRIALS=25, DATA_SIZE=2000, Cohen's d, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import pdist
from exotic_geometry_framework import GeometryAnalyzer, encode_float_to_unit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05

# --- Discover metric names ---
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn


# =========================================================================
# DYNAMICAL SYSTEMS
# =========================================================================

def logistic_map(r, x):
    return r * x * (1 - x)

def get_lyapunov_logistic(r, n_warmup=1000, n_iter=2000):
    x = 0.5
    for _ in range(n_warmup):
        x = logistic_map(r, x)
    
    lyap = 0
    for _ in range(n_iter):
        # derivative of r*x*(1-x) is r*(1-2x)
        deriv = abs(r * (1 - 2 * x))
        if deriv > 0:
            lyap += np.log(deriv)
        x = logistic_map(r, x)
    return lyap / n_iter

def gen_logistic_sweep(r, trial, size=DATA_SIZE):
    rng = np.random.default_rng(42 + trial)
    x = 0.1 + 0.8 * rng.uniform()
    warmup = 1000
    vals = []
    for _ in range(warmup + size):
        x = r * x * (1 - x)
        vals.append(x)
    return np.array(vals[warmup:], dtype=np.float64)


# =========================================================================
# UTILITIES
# =========================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps

def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: ROUTE TO CHAOS (BIFURCATION SWEEP)
# =========================================================================

def direction_1(analyzer):
    print("=" * 78)
    print("D1: ROUTE TO CHAOS — BIFURCATION SWEEP (r = 3.5 to 4.0)")
    print("=" * 78)
    
    r_vals = np.linspace(3.5, 4.0, 21)
    d1_results = {}
    
    for r in r_vals:
        print(f"  r = {r:.3f}...", end=" ", flush=True)
        metrics_acc = defaultdict(list)
        for t in range(N_TRIALS):
            data = gen_logistic_sweep(r, t)
            # Use 'auto' mode via uint8 encoding or direct if analyzer supported it
            # Framework's analyze handles floats if add_all_geometries(data_mode='auto') was used
            # But we'll manually encode to uint8 for consistency with chaos.py
            data_u8 = (data * 255).astype(np.uint8)
            res = analyzer.analyze(data_u8)
            for r_res in res.results:
                for mn, mv in r_res.metrics.items():
                    metrics_acc[f"{r_res.geometry_name}:{mn}"].append(mv)
        d1_results[r] = {m: np.mean(v) for m, v in metrics_acc.items()}
        print("done")
        
    return r_vals, d1_results


# =========================================================================
# D2: LYAPUNOV CORRELATION
# =========================================================================

def direction_2(r_vals, d1_results):
    print("\n" + "=" * 78)
    print("D2: LYAPUNOV CORRELATION")
    print("=" * 78)
    
    lyaps = [get_lyapunov_logistic(r) for r in r_vals]
    
    correlations = []
    for m in METRIC_NAMES:
        m_vals = [d1_results[r].get(m, 0) for r in r_vals]
        if np.std(m_vals) > 1e-10:
            corr, _ = stats.pearsonr(lyaps, m_vals)
            correlations.append((m, corr))
            
    correlations.sort(key=lambda x: -abs(x[1]))
    
    print(f"  Top 10 Lyapunov-tracking metrics:")
    for m, c in correlations[:10]:
        print(f"    {m:50s}  r = {c:+.4f}")
        
    return correlations, lyaps


# =========================================================================
# D3: ATTRACTOR GEOMETRY VIA RETURN MAP
# =========================================================================

def _correlation_dimension(points, n_samples=500):
    """Estimate correlation dimension via Grassberger-Procaccia algorithm.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        2D points (e.g. return map x_n vs x_{n+1}).
    n_samples : int
        Max points to subsample for pdist (O(N²) otherwise).

    Returns
    -------
    float : estimated correlation dimension (slope of log C(r) vs log r).
    """
    if len(points) > n_samples:
        idx = np.random.choice(len(points), n_samples, replace=False)
        points = points[idx]
    dists = pdist(points)
    dists = dists[dists > 0]
    if len(dists) < 10:
        return 0.0
    # Use a range of radii between 5th and 95th percentile of distances
    r_min, r_max = np.percentile(dists, 5), np.percentile(dists, 95)
    if r_min <= 0 or r_max <= r_min:
        return 0.0
    radii = np.geomspace(r_min, r_max, 20)
    n_pairs = len(dists)
    C_r = np.array([np.sum(dists < r) / n_pairs for r in radii])
    # Linear fit in log-log space
    mask = C_r > 0
    if np.sum(mask) < 5:
        return 0.0
    log_r = np.log(radii[mask])
    log_C = np.log(C_r[mask])
    slope, _, _, _, _ = stats.linregress(log_r, log_C)
    return slope


def _build_return_map_image(data, size=64):
    """Build a 2D density image from the return map (x_n, x_{n+1}).

    Returns a float64 array of shape (size, size).
    """
    x_n = data[:-1]
    x_n1 = data[1:]
    # Bin into a 2D histogram
    img, _, _ = np.histogram2d(x_n, x_n1, bins=size, range=[[0, 1], [0, 1]])
    return img.astype(np.float64)


def direction_3(analyzer_spatial):
    print("\n" + "=" * 78)
    print("D3: ATTRACTOR GEOMETRY VIA RETURN MAP")
    print("=" * 78)

    r_vals_d3 = [3.5, 3.57, 3.7, 3.83, 3.9, 4.0]
    corr_dims = {}
    spatial_results = {}

    for r in r_vals_d3:
        print(f"  r = {r:.2f}...", end=" ", flush=True)
        dims = []
        metrics_acc = defaultdict(list)
        for t in range(N_TRIALS):
            data = gen_logistic_sweep(r, t)
            # Correlation dimension of (x_n, x_{n+1}) attractor
            pts = np.column_stack([data[:-1], data[1:]])
            dims.append(_correlation_dimension(pts))
            # Build density image and analyze with spatial geometries
            img = _build_return_map_image(data)
            res = analyzer_spatial.analyze(img)
            for r_res in res.results:
                for mn, mv in r_res.metrics.items():
                    metrics_acc[f"{r_res.geometry_name}:{mn}"].append(mv)
        corr_dims[r] = (np.mean(dims), np.std(dims))
        spatial_results[r] = {m: np.array(v) for m, v in metrics_acc.items()}
        print(f"D_corr = {np.mean(dims):.3f} ± {np.std(dims):.3f}")

    # Compare periodic (r=3.5) vs chaotic (r=4.0) using spatial metrics
    spatial_metric_names = list(spatial_results[r_vals_d3[0]].keys())
    n_spatial = len(spatial_metric_names)
    bonf_spatial = ALPHA / n_spatial if n_spatial > 0 else ALPHA

    sig_count = 0
    findings = []
    a_data = spatial_results[3.5]
    b_data = spatial_results[4.0]
    for m in spatial_metric_names:
        a, b = a_data.get(m, []), b_data.get(m, [])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < bonf_spatial and abs(d) > 0.8:
            sig_count += 1
            findings.append((m, d, p))

    findings.sort(key=lambda x: -abs(x[1]))
    print(f"  {sig_count}/{n_spatial} spatial metrics distinguish r=3.5 from r=4.0")
    for m, d, p in findings[:5]:
        print(f"    {m:50s}  d={d:+.2f}")

    return r_vals_d3, corr_dims, findings


# =========================================================================
# D4: BYTE vs UNIT MODE DETECTION POWER
# =========================================================================

def direction_4(analyzer, analyzer_unit):
    print("\n" + "=" * 78)
    print("D4: BYTE vs UNIT MODE DETECTION POWER")
    print("=" * 78)
    print("  Comparing detection power vs random in byte (uint8) vs unit ([0,1]) mode")

    r_edge = 3.57

    # Collect signal metrics in both modes
    byte_sig = defaultdict(list)
    unit_sig = defaultdict(list)
    # Collect random baselines in both modes
    byte_rnd = defaultdict(list)
    unit_rnd = defaultdict(list)

    for t in range(N_TRIALS):
        rng = np.random.default_rng(9999 + t)
        raw = gen_logistic_sweep(r_edge, t)

        # Signal: byte mode
        res = analyzer.analyze((raw * 255).astype(np.uint8))
        for r_res in res.results:
            for mn, mv in r_res.metrics.items():
                byte_sig[f"{r_res.geometry_name}:{mn}"].append(mv)

        # Signal: unit mode
        res = analyzer_unit.analyze(raw)
        for r_res in res.results:
            for mn, mv in r_res.metrics.items():
                unit_sig[f"{r_res.geometry_name}:{mn}"].append(mv)

        # Random: byte mode
        res = analyzer.analyze(rng.integers(0, 256, DATA_SIZE, dtype=np.uint8))
        for r_res in res.results:
            for mn, mv in r_res.metrics.items():
                byte_rnd[f"{r_res.geometry_name}:{mn}"].append(mv)

        # Random: unit mode
        res = analyzer_unit.analyze(rng.uniform(0, 1, DATA_SIZE))
        for r_res in res.results:
            for mn, mv in r_res.metrics.items():
                unit_rnd[f"{r_res.geometry_name}:{mn}"].append(mv)

    # Count significant detections (d>0.8, Bonferroni) for each mode
    byte_only, unit_only, both_sig = [], [], []
    for m in METRIC_NAMES:
        b_s, b_r = byte_sig[m], byte_rnd[m]
        u_s, u_r = unit_sig[m], unit_rnd[m]
        if len(b_s) < 3 or len(b_r) < 3:
            continue
        d_byte = cohens_d(b_s, b_r)
        _, p_byte = stats.ttest_ind(b_s, b_r, equal_var=False)
        is_byte = p_byte < BONF and abs(d_byte) > 0.8

        d_unit = cohens_d(u_s, u_r)
        _, p_unit = stats.ttest_ind(u_s, u_r, equal_var=False)
        is_unit = p_unit < BONF and abs(d_unit) > 0.8

        if is_byte and is_unit:
            both_sig.append((m, d_byte, d_unit))
        elif is_byte:
            byte_only.append((m, d_byte))
        elif is_unit:
            unit_only.append((m, d_unit))

    n_byte = len(both_sig) + len(byte_only)
    n_unit = len(both_sig) + len(unit_only)
    print(f"  Significant detections vs random at r={r_edge}:")
    print(f"    Byte mode:  {n_byte} metrics")
    print(f"    Unit mode:  {n_unit} metrics")
    print(f"    Both:       {len(both_sig)}")
    print(f"    Byte-only:  {len(byte_only)}")
    print(f"    Unit-only:  {len(unit_only)}")
    if unit_only:
        print(f"  Unit-only metrics (rescued by float precision):")
        for m, d in sorted(unit_only, key=lambda x: -abs(x[1]))[:5]:
            print(f"    {m:50s}  d={d:+.2f}")

    return n_byte, n_unit, len(both_sig), byte_only, unit_only


# =========================================================================
# D5: PHASE SPACE VOLUME PROXIES
# =========================================================================

def direction_5(r_vals, d1_results):
    print("\n" + "=" * 78)
    print("D5: PHASE SPACE VOLUME PROXIES")
    print("=" * 78)

    # Ground truth: occupied fraction of [0,1] via histogram bin coverage
    n_bins = 100
    occupied = {}
    for r in r_vals:
        coverages = []
        for t in range(N_TRIALS):
            data = gen_logistic_sweep(r, t)
            counts, _ = np.histogram(data, bins=n_bins, range=(0, 1))
            coverages.append(np.sum(counts > 0) / n_bins)
        occupied[r] = np.mean(coverages)

    occ_vals = [occupied[r] for r in r_vals]
    print(f"  Occupied fraction range: {min(occ_vals):.3f} – {max(occ_vals):.3f}")

    # Spearman-correlate every metric from D1 with occupied fraction
    volume_corrs = []
    for m in METRIC_NAMES:
        m_vals = [d1_results[r].get(m, 0) for r in r_vals]
        if np.std(m_vals) < 1e-10:
            continue
        rho, p = stats.spearmanr(occ_vals, m_vals)
        if np.isfinite(rho):
            volume_corrs.append((m, rho, p))

    volume_corrs.sort(key=lambda x: -abs(x[1]))
    print(f"  Top 10 volume-proxy metrics (Spearman ρ with occupied fraction):")
    for m, rho, p in volume_corrs[:10]:
        print(f"    {m:50s}  ρ={rho:+.4f}  p={p:.2e}")

    return occupied, volume_corrs


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(r_vals, d1_results, correlations, lyaps,
                r_vals_d3, corr_dims, d3_findings,
                n_byte, n_unit, n_both, byte_only, unit_only,
                occupied, volume_corrs):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 22), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (0,0) D1: Bifurcation + Lyapunov
    ax1 = _dark_ax(fig.add_subplot(gs[0, 0]))
    r_diag = np.linspace(3.5, 4.0, 1000)
    x = np.full(len(r_diag), 0.5)
    for _ in range(200):
        x = r_diag * x * (1 - x)
    for _ in range(100):
        x = r_diag * x * (1 - x)
        ax1.plot(r_diag, x, ',w', alpha=0.1, markersize=1)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(r_vals, lyaps, 'r-', linewidth=2, label='Lyapunov λ')
    ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1_twin.set_ylabel('Lyapunov Exponent λ', color='red', fontsize=9)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1.set_title('D1: Bifurcation & Lyapunov Exponent', fontsize=11, fontweight='bold')
    ax1.set_xlabel('r', fontsize=9)

    # (0,1) D2: Top tracking metrics
    ax2 = _dark_ax(fig.add_subplot(gs[0, 1]))
    top_metrics = [m for m, c in correlations[:3]]
    colors_top = ['#E91E63', '#2196F3', '#4CAF50']
    for i, m in enumerate(top_metrics):
        vals = [d1_results[r][m] for r in r_vals]
        vals = (np.array(vals) - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-10)
        ax2.plot(r_vals, vals, 'o-', color=colors_top[i], label=m.split(':')[1], alpha=0.8)
    ax2.set_title('D2: Top Lyapunov-Tracking Metrics', fontsize=11, fontweight='bold')
    ax2.set_xlabel('r', fontsize=9)
    ax2.legend(fontsize=7, facecolor='#333', edgecolor='#666')

    # (1,0) D3: Correlation dimension vs r
    ax3 = _dark_ax(fig.add_subplot(gs[1, 0]))
    d3_r = sorted(corr_dims.keys())
    d3_means = [corr_dims[r][0] for r in d3_r]
    d3_stds = [corr_dims[r][1] for r in d3_r]
    ax3.errorbar(d3_r, d3_means, yerr=d3_stds, fmt='o-', color='#00BCD4',
                 capsize=4, linewidth=2, markersize=6)
    ax3.set_title('D3: Correlation Dimension of Return Map', fontsize=11, fontweight='bold')
    ax3.set_xlabel('r', fontsize=9)
    ax3.set_ylabel('D_corr (Grassberger-Procaccia)', fontsize=9)
    # Mark key regime transitions
    ax3.axvline(x=3.57, color='#FF5722', linestyle='--', alpha=0.5, label='Onset of chaos')
    ax3.legend(fontsize=7, facecolor='#333', edgecolor='#666')

    # (1,1) D4: Byte vs Unit bar chart
    ax4 = _dark_ax(fig.add_subplot(gs[1, 1]))
    categories = ['Both', 'Byte-only', 'Unit-only']
    counts = [n_both, len(byte_only), len(unit_only)]
    bar_colors = ['#9C27B0', '#FF9800', '#4CAF50']
    bars = ax4.bar(categories, counts, color=bar_colors, alpha=0.85, edgecolor='#666')
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10, color='white')
    ax4.set_title(f'D4: Detection Power at r=3.57 (byte={n_byte}, unit={n_unit})',
                  fontsize=11, fontweight='bold')
    ax4.set_ylabel('Significant metrics', fontsize=9)

    # (2,0) D5: Volume proxies vs r
    ax5 = _dark_ax(fig.add_subplot(gs[2, 0]))
    occ_r = sorted(occupied.keys())
    occ_vals = [occupied[r] for r in occ_r]
    ax5.plot(occ_r, occ_vals, 's-', color='#FF5722', linewidth=2, label='Occupied fraction')
    ax5.set_ylabel('Occupied fraction', color='#FF5722', fontsize=9)
    ax5.set_xlabel('r', fontsize=9)
    # Overlay top proxy metric
    if volume_corrs:
        best_m, best_rho, _ = volume_corrs[0]
        best_vals = [d1_results[r].get(best_m, 0) for r in occ_r]
        ax5_twin = ax5.twinx()
        ax5_twin.plot(occ_r, best_vals, 'o-', color='#2196F3', linewidth=2,
                      label=best_m.split(':')[1])
        ax5_twin.set_ylabel(best_m.split(':')[1], color='#2196F3', fontsize=9)
        ax5_twin.tick_params(axis='y', labelcolor='#2196F3')
        ax5.set_title(f'D5: Volume Proxy (best ρ={best_rho:+.3f})', fontsize=11, fontweight='bold')
    else:
        ax5.set_title('D5: Phase Space Occupied Fraction', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=7, facecolor='#333', edgecolor='#666', loc='upper left')

    # (2,1) Summary text panel
    ax6 = _dark_ax(fig.add_subplot(gs[2, 1]))
    ax6.axis('off')
    n_lyap_strong = len([c for m, c in correlations if abs(c) > 0.8])
    best_m, best_c = correlations[0]
    n_vol_strong = len([x for x in volume_corrs if abs(x[1]) > 0.8])
    summary_lines = [
        "DEEP CHAOS ANALYSIS — SUMMARY",
        "",
        f"D1: 21 r-values swept (3.50–4.00), {N_TRIALS} trials each",
        f"D2: {n_lyap_strong} metrics with |r| > 0.8 vs Lyapunov",
        f"    Best: {best_m} (r={best_c:+.4f})",
        f"D3: Corr dim: {corr_dims.get(3.5, (0,0))[0]:.2f} (r=3.5) → "
        f"{corr_dims.get(4.0, (0,0))[0]:.2f} (r=4.0)",
        f"    {len(d3_findings)} spatial metrics distinguish periodic vs chaotic",
        f"D4: Byte={n_byte}, Unit={n_unit} sig metrics at r=3.57",
        f"    Unit-only: {len(unit_only)}, Byte-only: {len(byte_only)}",
        f"D5: {n_vol_strong} metrics with |ρ| > 0.8 vs occupied fraction",
    ]
    if volume_corrs:
        summary_lines.append(f"    Best proxy: {volume_corrs[0][0]} (ρ={volume_corrs[0][1]:+.3f})")
    ax6.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             color='#cccccc', linespacing=1.5)

    fig.suptitle('Deep Chaos Analysis: Dynamics & Geometry',
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'chaos_deep.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#181818')
    print(f"  Saved {out_path}")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()
    analyzer_unit = GeometryAnalyzer().add_all_geometries(data_mode='unit')
    analyzer_spatial = GeometryAnalyzer().add_spatial_geometries()

    # D1: Bifurcation sweep (525 calls, ~3 min)
    r_vals, d1_results = direction_1(analyzer)

    # D2: Lyapunov correlation (0 calls, reuses D1)
    correlations, lyaps = direction_2(r_vals, d1_results)

    # D3: Attractor geometry via return map (150 calls, ~1 min)
    r_vals_d3, corr_dims, d3_findings = direction_3(analyzer_spatial)

    # D4: Byte vs unit detection power (100 calls, ~40s)
    n_byte, n_unit, n_both, byte_only, unit_only = direction_4(analyzer, analyzer_unit)

    # D5: Phase space volume proxies (0 calls, reuses D1)
    occupied, volume_corrs = direction_5(r_vals, d1_results)

    make_figure(r_vals, d1_results, correlations, lyaps,
                r_vals_d3, corr_dims, d3_findings,
                n_byte, n_unit, n_both, byte_only, unit_only,
                occupied, volume_corrs)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    n_lyap = len([c for m, c in correlations if abs(c) > 0.8])
    best_m, best_c = correlations[0]
    n_vol = len([x for x in volume_corrs if abs(x[1]) > 0.8])
    print(f"  D1: {len(r_vals)} r-values swept, {N_TRIALS} trials each")
    print(f"  D2: {n_lyap} metrics with |r| > 0.8 vs Lyapunov")
    print(f"      Best: {best_m} (r={best_c:+.4f})")
    print(f"  D3: Corr dim {corr_dims.get(3.5, (0,0))[0]:.2f} (r=3.5) → "
          f"{corr_dims.get(4.0, (0,0))[0]:.2f} (r=4.0)")
    print(f"      {len(d3_findings)} spatial metrics distinguish periodic vs chaotic")
    print(f"  D4: Byte={n_byte}, Unit={n_unit} sig at r=3.57")
    print(f"      Unit-only: {len(unit_only)}, Byte-only: {len(byte_only)}")
    print(f"  D5: {n_vol} metrics with |ρ| > 0.8 as volume proxies")
    if volume_corrs:
        print(f"      Best: {volume_corrs[0][0]} (ρ={volume_corrs[0][1]:+.3f})")
    print(f"\n[Investigation complete]")
