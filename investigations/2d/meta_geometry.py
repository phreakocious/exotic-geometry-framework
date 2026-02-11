#!/usr/bin/env python3
"""
Meta-Geometry Investigation: The Framework Analyzing Its Own Signature Space

The framework's 41 classifier signatures each encode a 524-element vector of
metric means (131 base metrics × 4 delay scales). This investigation turns the
framework on itself — treating the signature space as data to be analyzed.

Five directions:
  D1: Signature distance matrix as 2D field (spatial geometries)
  D2: Metric correlation matrix as 2D field (spatial geometries)
  D3: Signature profiles as 1D signals (1D geometries, category comparison)
  D4: Interference strength by geometry (coefficient of variation)
  D5: Spectral dimension of signature space (SVD vs Marchenko-Pastur)
"""

import sys, os, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats, spatial, cluster
from exotic_geometry_framework import GeometryAnalyzer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

N_TRIALS = 25
ALPHA = 0.05

# Category assignments
CATEGORIES = {
    'chaos': [
        'Henon Chaos', 'Logistic Chaos', 'Logistic Edge-of-Chaos',
        'Lorenz Attractor (x)', 'Rossler Attractor', 'Baker Map',
        'Standard Map (Chirikov)', 'Tent Map',
    ],
    'number_theory': [
        'Prime Gaps', 'Prime Gap Pairs', 'Prime Gap Diffs',
        'Divisor Count d(n)', 'Totient Ratio', 'Mertens Function',
        'Moebius Function',
    ],
    'collatz': [
        'Collatz (Hailstone)', 'Collatz (Parity)',
        'Collatz High Bits', 'Collatz Stopping Times',
    ],
    'noise': [
        'Gaussian White Noise', 'Pink Noise', 'Perlin Noise', 'Random',
    ],
    'waveform': ['Sine Wave', 'Sawtooth Wave', 'Square Wave'],
    'binary': [
        'Mach-O Binary', 'x86-64 Architecture', 'ARM64 Architecture',
        'Java Bytecode', 'WASM Bytecode',
    ],
    'bio': [
        'Bacterial DNA (E. coli)', 'Eukaryotic DNA (Human)', 'Viral DNA',
    ],
    'other': [
        'AES-ECB (Structured)', 'glibc LCG', 'RANDU',
        'LZ4 Compressed', 'NN Trained Dense', 'NN Pruned 90%', 'ASCII Text',
    ],
}

CATEGORY_COLORS = {
    'chaos': '#e74c3c',
    'number_theory': '#3498db',
    'collatz': '#e67e22',
    'noise': '#95a5a6',
    'waveform': '#2ecc71',
    'binary': '#9b59b6',
    'bio': '#1abc9c',
    'other': '#f1c40f',
}


# =============================================================================
# LOAD SIGNATURES
# =============================================================================

def load_signatures():
    """Load all signature JSONs into a matrix + metadata."""
    sig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'signatures')
    files = sorted(glob.glob(os.path.join(sig_dir, '*.json')))

    names = []
    means_list = []
    stds_list = []
    metric_names = None

    for f in files:
        with open(f) as fh:
            sig = json.load(fh)
        names.append(sig['name'])
        means_list.append(sig['means'])
        stds_list.append(sig['stds'])
        if metric_names is None:
            metric_names = sig['metrics']

    means_matrix = np.array(means_list, dtype=np.float64)  # (N_sigs, 524)
    stds_matrix = np.array(stds_list, dtype=np.float64)

    # Build name → category lookup
    name_to_cat = {}
    for cat, cat_names in CATEGORIES.items():
        for n in cat_names:
            name_to_cat[n] = cat

    categories = []
    for n in names:
        categories.append(name_to_cat.get(n, 'other'))

    print(f"Loaded {len(names)} signatures, {len(metric_names)} metrics each")
    return names, categories, metric_names, means_matrix, stds_matrix


# =============================================================================
# SPATIAL METRIC DISCOVERY (2D mode)
# =============================================================================

_analyzer_2d = GeometryAnalyzer().add_spatial_geometries()
_dummy_2d = _analyzer_2d.analyze(np.random.rand(16, 16))
SPATIAL_METRICS = []
for _r in _dummy_2d.results:
    for _mn in sorted(_r.metrics.keys()):
        SPATIAL_METRICS.append(f"{_r.geometry_name}:{_mn}")
N_SPATIAL = len(SPATIAL_METRICS)
del _analyzer_2d, _dummy_2d, _r, _mn

# 1D metric discovery
_analyzer_1d = GeometryAnalyzer().add_all_geometries()
_dummy_1d = _analyzer_1d.analyze(np.random.randint(0, 256, 524, dtype=np.uint8))
METRICS_1D = []
for _r in _dummy_1d.results:
    for _mn in sorted(_r.metrics.keys()):
        METRICS_1D.append(f"{_r.geometry_name}:{_mn}")
N_1D = len(METRICS_1D)
del _analyzer_1d, _dummy_1d, _r, _mn

BONF_2D = ALPHA / N_SPATIAL
BONF_1D = ALPHA / N_1D

print(f"2D spatial metrics: {N_SPATIAL}, 1D metrics: {N_1D}")


# =============================================================================
# HELPERS
# =============================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def collect_2d(analyzer, fields):
    """Collect spatial metrics from 2D fields."""
    out = {m: [] for m in SPATIAL_METRICS}
    for f in fields:
        res = analyzer.analyze(f)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def collect_1d(analyzer, signals):
    """Collect 1D metrics from uint8 signals."""
    out = {m: [] for m in METRICS_1D}
    for s in signals:
        res = analyzer.analyze(s)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def compare_metrics(data_a, data_b, metric_names, bonf_alpha):
    """Compare two metric dicts. Returns (n_sig, findings)."""
    sig = 0
    findings = []
    for m in metric_names:
        a = np.array(data_a.get(m, []))
        b = np.array(data_b.get(m, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        if not np.isfinite(d):
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < bonf_alpha and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =============================================================================
# D1: SIGNATURE DISTANCE MATRIX AS 2D FIELD
# =============================================================================

def direction_1(names, categories, metric_names, means_matrix, stds_matrix):
    """Compute and analyze the signature distance matrix."""
    print("\n" + "=" * 78)
    print("D1: Signature Distance Matrix as 2D Field")
    print("=" * 78)

    n_sigs, n_met = means_matrix.shape
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    # Z-score columns for cosine distance
    col_means = np.nanmean(means_matrix, axis=0)
    col_stds = np.nanstd(means_matrix, axis=0)
    col_stds[col_stds < 1e-15] = 1.0
    z_matrix = (means_matrix - col_means) / col_stds

    # 25 jackknife trials: random 80% of metrics each time
    rng = np.random.default_rng(42)
    n_keep = int(0.8 * n_met)

    real_fields = []
    null_fields = []

    t0 = time.time()
    for trial in range(N_TRIALS):
        idx = rng.choice(n_met, n_keep, replace=False)
        z_sub = z_matrix[:, idx]

        # Real distance matrix
        dist = spatial.distance.squareform(
            spatial.distance.pdist(z_sub, metric='cosine'))
        real_fields.append(dist.astype(np.float64))

        # Null: shuffle each column independently
        z_null = z_sub.copy()
        for c in range(z_null.shape[1]):
            rng.shuffle(z_null[:, c])
        dist_null = spatial.distance.squareform(
            spatial.distance.pdist(z_null, metric='cosine'))
        null_fields.append(dist_null.astype(np.float64))

    print(f"  Distance matrices computed: {time.time() - t0:.1f}s")

    # Analyze with spatial geometries
    t0 = time.time()
    real_metrics = collect_2d(analyzer, real_fields)
    null_metrics = collect_2d(analyzer, null_fields)
    print(f"  Spatial analysis: {time.time() - t0:.1f}s ({N_TRIALS*2} calls)")

    n_sig, findings = compare_metrics(real_metrics, null_metrics,
                                      SPATIAL_METRICS, BONF_2D)
    print(f"  Significant metrics: {n_sig} / {N_SPATIAL}")
    for m, d, p in findings[:5]:
        print(f"    {m}: d={d:+.2f}, p={p:.1e}")

    # Hierarchical clustering for plot order
    condensed = spatial.distance.pdist(z_matrix, metric='cosine')
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=0.0)
    full_dist = spatial.distance.squareform(condensed)
    linkage = cluster.hierarchy.linkage(condensed, method='ward')
    order = cluster.hierarchy.leaves_list(linkage)

    return {
        'n_sig': n_sig,
        'findings': findings,
        'dist_matrix': full_dist,
        'order': order,
        'names': names,
        'categories': categories,
    }


# =============================================================================
# D2: METRIC CORRELATION GEOMETRY AS 2D FIELD
# =============================================================================

def direction_2(names, categories, metric_names, means_matrix, stds_matrix):
    """Compute and analyze the metric correlation matrix."""
    print("\n" + "=" * 78)
    print("D2: Metric Correlation Geometry as 2D Field")
    print("=" * 78)

    n_sigs, n_met = means_matrix.shape
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    # Use tau1 metrics only for the correlation matrix
    tau1_idx = [i for i, m in enumerate(metric_names) if m.endswith(':tau1')]
    n_base = len(tau1_idx)
    print(f"  Using {n_base} tau1 base metrics")

    base_matrix = means_matrix[:, tau1_idx]  # (41, 131)

    rng = np.random.default_rng(123)

    real_fields = []
    null_fields = []

    t0 = time.time()
    for trial in range(N_TRIALS):
        # Bootstrap: resample 33 of 41 signatures with replacement
        boot_idx = rng.choice(n_sigs, size=33, replace=True)
        sub = base_matrix[boot_idx, :]

        # Real correlation matrix
        corr = np.corrcoef(sub.T)  # (131, 131)
        np.fill_diagonal(corr, 0)
        corr = np.nan_to_num(corr, nan=0.0)
        real_fields.append(corr.astype(np.float64))

        # Null: shuffle each metric column before correlating
        sub_null = sub.copy()
        for c in range(sub_null.shape[1]):
            rng.shuffle(sub_null[:, c])
        corr_null = np.corrcoef(sub_null.T)
        np.fill_diagonal(corr_null, 0)
        corr_null = np.nan_to_num(corr_null, nan=0.0)
        null_fields.append(corr_null.astype(np.float64))

    print(f"  Correlation matrices computed: {time.time() - t0:.1f}s")

    t0 = time.time()
    real_metrics = collect_2d(analyzer, real_fields)
    null_metrics = collect_2d(analyzer, null_fields)
    print(f"  Spatial analysis: {time.time() - t0:.1f}s ({N_TRIALS*2} calls)")

    n_sig, findings = compare_metrics(real_metrics, null_metrics,
                                      SPATIAL_METRICS, BONF_2D)
    print(f"  Significant metrics: {n_sig} / {N_SPATIAL}")
    for m, d, p in findings[:5]:
        print(f"    {m}: d={d:+.2f}, p={p:.1e}")

    # Full correlation matrix for display
    full_corr = np.corrcoef(base_matrix.T)
    np.fill_diagonal(full_corr, 0)
    full_corr = np.nan_to_num(full_corr, nan=0.0)

    # Geometry family boundaries for dividers
    tau1_names = [metric_names[i] for i in tau1_idx]
    families = []
    boundaries = []
    prev_fam = None
    for i, m in enumerate(tau1_names):
        fam = m.split(':')[0]
        if fam != prev_fam:
            boundaries.append(i)
            families.append(fam)
            prev_fam = fam

    return {
        'n_sig': n_sig,
        'findings': findings,
        'corr_matrix': full_corr,
        'families': families,
        'boundaries': boundaries,
        'n_base': n_base,
    }


# =============================================================================
# D3: SIGNATURE PROFILES AS 1D SIGNALS
# =============================================================================

def direction_3(names, categories, metric_names, means_matrix, stds_matrix):
    """Analyze signature profiles as 1D signals, compare categories."""
    print("\n" + "=" * 78)
    print("D3: Signature Profiles as 1D Signals")
    print("=" * 78)

    n_sigs, n_met = means_matrix.shape
    analyzer = GeometryAnalyzer().add_all_geometries()

    # Group signatures by category
    cat_sigs = {}
    for i, (name, cat) in enumerate(zip(names, categories)):
        cat_sigs.setdefault(cat, []).append(i)

    # For each signature: generate 25 perturbed profiles, normalize to uint8
    rng = np.random.default_rng(77)
    cat_metrics = {}

    t0 = time.time()
    for cat, sig_indices in sorted(cat_sigs.items()):
        all_signals = []
        for si in sig_indices:
            sig_means = means_matrix[si]
            sig_stds = stds_matrix[si]
            for trial in range(N_TRIALS):
                # Perturb by stored stds
                perturbed = sig_means + rng.normal(0, 1, n_met) * sig_stds
                # Percentile clip and normalize to uint8
                lo, hi = np.nanpercentile(perturbed, [2, 98])
                if hi - lo < 1e-15:
                    hi = lo + 1.0
                clipped = np.clip(perturbed, lo, hi)
                normalized = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
                all_signals.append(normalized)

        cat_metrics[cat] = collect_1d(analyzer, all_signals)
        print(f"  {cat}: {len(all_signals)} signals analyzed")

    print(f"  1D analysis: {time.time() - t0:.1f}s")

    # Pairwise comparison between categories
    cat_names = sorted(cat_metrics.keys())
    n_cats = len(cat_names)
    pw_matrix = np.zeros((n_cats, n_cats), dtype=int)

    print("\n  Pairwise comparisons:")
    for i in range(n_cats):
        for j in range(i + 1, n_cats):
            n_sig, findings = compare_metrics(
                cat_metrics[cat_names[i]], cat_metrics[cat_names[j]],
                METRICS_1D, BONF_1D)
            pw_matrix[i, j] = pw_matrix[j, i] = n_sig
            print(f"    {cat_names[i]:14s} vs {cat_names[j]:14s}: {n_sig:3d} sig")

    return {
        'matrix': pw_matrix,
        'cat_names': cat_names,
        'n_1d': N_1D,
    }


# =============================================================================
# D4: INTERFERENCE STRENGTH BY GEOMETRY
# =============================================================================

def direction_4(names, categories, metric_names, means_matrix, stds_matrix):
    """Compute discriminating power (SNR) per geometry family and tau scale."""
    print("\n" + "=" * 78)
    print("D4: Interference Strength by Geometry")
    print("=" * 78)

    n_met = len(metric_names)

    # Discriminating power = inter-signature std / mean intra-trial std
    # This is an SNR: how much does the metric vary across signatures
    # relative to its trial-to-trial noise within each signature.
    metric_snr = {}
    for j in range(n_met):
        inter_std = np.nanstd(means_matrix[:, j])
        intra_std = np.nanmean(stds_matrix[:, j])
        if intra_std > 1e-15:
            metric_snr[metric_names[j]] = inter_std / intra_std
        else:
            # Zero intra-trial variance: perfectly reproducible
            # Use inter_std directly (or inf if also zero)
            metric_snr[metric_names[j]] = inter_std if inter_std > 1e-15 else 0.0

    # Filter out non-finite SNR values before grouping
    finite_snr = {m: s for m, s in metric_snr.items() if np.isfinite(s)}
    print(f"  {len(finite_snr)} / {n_met} metrics have finite SNR")

    # Group by geometry family
    family_cvs = {}
    for m, snr in finite_snr.items():
        parts = m.rsplit(':', 2)  # "Family:metric:tauN"
        fam = parts[0]
        family_cvs.setdefault(fam, []).append(snr)

    family_mean_cv = {}
    for fam, cvs in family_cvs.items():
        family_mean_cv[fam] = np.mean(cvs)

    # Sort by mean CV descending
    sorted_families = sorted(family_mean_cv.items(), key=lambda x: -x[1])
    print(f"  Top 5 discriminating geometries (SNR = inter/intra std):")
    for fam, snr in sorted_families[:5]:
        print(f"    {fam}: SNR={snr:.2f}")
    print(f"  Bottom 5:")
    for fam, snr in sorted_families[-5:]:
        print(f"    {fam}: SNR={snr:.2f}")

    # Group by tau scale
    tau_cvs = {}
    for m, snr in finite_snr.items():
        tau = m.rsplit(':', 1)[-1]  # "tauN"
        tau_cvs.setdefault(tau, []).append(snr)

    tau_mean_cv = {}
    for tau, cvs in sorted(tau_cvs.items()):
        tau_mean_cv[tau] = np.mean(cvs)
        print(f"  {tau}: mean SNR={np.mean(cvs):.2f}")

    # Top 10 individual metrics
    sorted_metrics = sorted(finite_snr.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 discriminating metrics:")
    for m, snr in sorted_metrics[:10]:
        print(f"    {m}: SNR={snr:.2f}")

    return {
        'sorted_families': sorted_families,
        'tau_mean_cv': tau_mean_cv,
        'top_metrics': sorted_metrics[:10],
    }


# =============================================================================
# D5: SPECTRAL DIMENSION OF SIGNATURE SPACE
# =============================================================================

def direction_5(means_matrix):
    """SVD of signature matrix, compare to Marchenko-Pastur."""
    print("\n" + "=" * 78)
    print("D5: Spectral Dimension of Signature Space")
    print("=" * 78)

    n_sigs, n_met = means_matrix.shape

    # Z-score columns, dropping zero-variance columns
    col_means = np.nanmean(means_matrix, axis=0)
    col_stds = np.nanstd(means_matrix, axis=0)
    good_cols = col_stds > 1e-10
    means_good = means_matrix[:, good_cols]
    col_means_good = col_means[good_cols]
    col_stds_good = col_stds[good_cols]
    z_matrix = (means_good - col_means_good) / col_stds_good
    z_matrix = np.nan_to_num(z_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Using {z_matrix.shape[1]} / {n_met} non-degenerate metrics")

    # SVD of real matrix
    U, s_real, Vt = np.linalg.svd(z_matrix, full_matrices=False)
    variance_explained = s_real**2 / np.sum(s_real**2)
    cumvar = np.cumsum(variance_explained)

    # Participation ratio (effective dimensionality)
    pr = np.sum(s_real**2)**2 / np.sum(s_real**4)
    print(f"  Participation ratio (effective dim): {pr:.1f}")
    print(f"  Top singular value explains: {variance_explained[0]*100:.1f}%")
    print(f"  Top 5 cumulative: {cumvar[4]*100:.1f}%")
    print(f"  Top 10 cumulative: {cumvar[min(9, len(cumvar)-1)]*100:.1f}%")

    # 90% variance threshold
    dim_90 = np.searchsorted(cumvar, 0.90) + 1
    print(f"  Dimensions for 90% variance: {dim_90}")

    # Marchenko-Pastur envelope (25 random matrices)
    n_cols = z_matrix.shape[1]
    gamma = n_sigs / n_cols  # aspect ratio
    rng = np.random.default_rng(99)
    mp_spectra = []
    for _ in range(N_TRIALS):
        rand_mat = rng.standard_normal((n_sigs, n_cols))
        _, s_rand, _ = np.linalg.svd(rand_mat, full_matrices=False)
        mp_spectra.append(s_rand)

    mp_spectra = np.array(mp_spectra)
    mp_mean = np.mean(mp_spectra, axis=0)
    mp_upper = np.percentile(mp_spectra, 97.5, axis=0)
    mp_lower = np.percentile(mp_spectra, 2.5, axis=0)

    # Jackknife for real spectrum variance (leave-one-out)
    jack_spectra = []
    for i in range(n_sigs):
        z_loo = np.delete(z_matrix, i, axis=0)
        _, s_loo, _ = np.linalg.svd(z_loo, full_matrices=False)
        jack_spectra.append(s_loo)

    # Count how many real singular values exceed MP upper bound
    n_above_mp = np.sum(s_real > mp_upper)
    print(f"  Singular values above MP 97.5%: {n_above_mp} / {len(s_real)}")

    return {
        's_real': s_real,
        'variance_explained': variance_explained,
        'cumvar': cumvar,
        'participation_ratio': pr,
        'dim_90': dim_90,
        'mp_mean': mp_mean,
        'mp_upper': mp_upper,
        'mp_lower': mp_lower,
        'n_above_mp': n_above_mp,
        'jack_spectra': jack_spectra,
    }


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(d1, d2, d3, d4, d5):
    """Create the 6-panel figure."""
    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(18, 20), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.30,
                          left=0.07, right=0.96, top=0.94, bottom=0.04)
    fig.suptitle("Meta-Geometry: The Framework Analyzing Its Own Signatures",
                 fontsize=15, fontweight='bold', color='white')

    # ---- D1: Distance matrix heatmap (clustered, colored by category) -------
    ax1 = fig.add_subplot(gs[0, 0])
    dark_ax(ax1)

    order = d1['order']
    n = len(order)
    dist_ordered = d1['dist_matrix'][np.ix_(order, order)]

    im1 = ax1.imshow(dist_ordered, cmap='viridis', aspect='equal')
    ax1.set_title(f"D1: Signature Distance Matrix ({d1['n_sig']} sig / {N_SPATIAL})",
                  fontsize=10, fontweight='bold')

    # Category color sidebar
    ordered_cats = [d1['categories'][i] for i in order]
    ordered_names = [d1['names'][i] for i in order]
    for i, cat in enumerate(ordered_cats):
        ax1.add_patch(Rectangle((-1.5, i - 0.5), 1, 1,
                                color=CATEGORY_COLORS.get(cat, '#888888'),
                                clip_on=False))

    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    short_names = [nm[:12] for nm in ordered_names]
    ax1.set_xticklabels(short_names, rotation=90, fontsize=4, ha='center')
    ax1.set_yticklabels(short_names, fontsize=4)
    ax1.set_xlim(-0.5, n - 0.5)

    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.6, pad=0.02)
    cb1.ax.tick_params(labelsize=6, colors='#cccccc')
    cb1.set_label('Cosine distance', fontsize=7, color='#cccccc')

    # ---- D2: Correlation matrix heatmap (131×131) ---------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    dark_ax(ax2)

    im2 = ax2.imshow(d2['corr_matrix'], cmap='RdBu_r', vmin=-1, vmax=1,
                     aspect='equal')
    ax2.set_title(f"D2: Metric Correlation Field ({d2['n_sig']} sig / {N_SPATIAL})",
                  fontsize=10, fontweight='bold')

    # Geometry family dividers
    for b in d2['boundaries'][1:]:
        ax2.axhline(b - 0.5, color='#666666', linewidth=0.5, alpha=0.7)
        ax2.axvline(b - 0.5, color='#666666', linewidth=0.5, alpha=0.7)

    # Family labels at midpoints
    bounds = d2['boundaries'] + [d2['n_base']]
    mid_ticks = []
    mid_labels = []
    for k in range(len(bounds) - 1):
        mid = (bounds[k] + bounds[k + 1]) / 2
        mid_ticks.append(mid)
        # Abbreviate family names
        fam = d2['families'][k]
        abbr = fam[:8] if len(fam) > 8 else fam
        mid_labels.append(abbr)

    ax2.set_xticks(mid_ticks)
    ax2.set_xticklabels(mid_labels, rotation=90, fontsize=4)
    ax2.set_yticks(mid_ticks)
    ax2.set_yticklabels(mid_labels, fontsize=4)

    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.6, pad=0.02)
    cb2.ax.tick_params(labelsize=6, colors='#cccccc')
    cb2.set_label('Pearson r', fontsize=7, color='#cccccc')

    # ---- D3: Category pairwise sig-count matrix -----------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    dark_ax(ax3)

    n_cats = len(d3['cat_names'])
    vmax3 = d3['n_1d']
    im3 = ax3.imshow(d3['matrix'], cmap='YlOrRd', vmin=0,
                     vmax=min(vmax3, np.max(d3['matrix']) * 1.3 + 1),
                     aspect='equal')
    ax3.set_xticks(range(n_cats))
    ax3.set_yticks(range(n_cats))
    ax3.set_xticklabels(d3['cat_names'], rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(d3['cat_names'], fontsize=8)

    for i in range(n_cats):
        for j in range(n_cats):
            val = int(d3['matrix'][i, j])
            if i == j:
                continue
            color = 'white' if val > np.max(d3['matrix']) * 0.5 else '#cccccc'
            ax3.text(j, i, str(val), ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold')

    ax3.set_title(f"D3: Category Distinguishability (of {d3['n_1d']})",
                  fontsize=10, fontweight='bold')

    # ---- D4: Geometry discriminating power bars -----------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    dark_ax(ax4)

    # Filter out extreme outlier (Fisher det_fisher dominates at ~10^13)
    # Show it as annotation instead
    outlier_thresh = 1000
    outlier_fams = [(n, s) for n, s in d4['sorted_families'] if s > outlier_thresh]
    plot_fams = [(n, s) for n, s in d4['sorted_families'] if s <= outlier_thresh]

    fam_names = [f[0] for f in plot_fams]
    fam_snrs = [f[1] for f in plot_fams]
    n_fam = len(fam_names)

    norm = plt.Normalize(vmin=min(fam_snrs), vmax=max(fam_snrs))
    colors4 = plt.cm.plasma(norm(fam_snrs))

    y_pos = np.arange(n_fam)
    bars = ax4.barh(y_pos, fam_snrs, color=colors4, alpha=0.9, height=0.7)
    ax4.set_yticks(y_pos)
    short_fam = []
    for f in fam_names:
        if len(f) > 22:
            f = f[:20] + '..'
        short_fam.append(f)
    ax4.set_yticklabels(short_fam, fontsize=6)
    ax4.set_xlabel('Mean SNR (inter/intra std)', fontsize=8)
    ax4.set_title("D4: Discriminating Power by Geometry",
                  fontsize=10, fontweight='bold')
    ax4.invert_yaxis()

    for bar, snr in zip(bars, fam_snrs):
        ax4.text(bar.get_width() + 0.2,
                 bar.get_y() + bar.get_height() / 2,
                 f'{snr:.1f}', va='center', fontsize=5.5, color='#cccccc')

    # Annotate outliers
    if outlier_fams:
        outlier_text = '\n'.join(f'{n}: SNR={s:.0e}' for n, s in outlier_fams)
        ax4.text(0.98, 0.02, f'Off-scale:\n{outlier_text}',
                 transform=ax4.transAxes, ha='right', va='bottom',
                 fontsize=6, color='#f1c40f',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222',
                           edgecolor='#444444'))

    # ---- D5a: Singular value spectrum vs MP ---------------------------------
    ax5a = fig.add_subplot(gs[2, 0])
    dark_ax(ax5a)

    k = np.arange(1, len(d5['s_real']) + 1)
    ax5a.semilogy(k, d5['s_real'], 'o-', color='#e74c3c', markersize=4,
                  linewidth=1.5, label='Real spectrum', zorder=3)
    ax5a.fill_between(k, d5['mp_lower'], d5['mp_upper'],
                      color='#3498db', alpha=0.3, label='MP 95% envelope')
    ax5a.semilogy(k, d5['mp_mean'], '--', color='#3498db', linewidth=1,
                  alpha=0.8, label='MP mean')

    ax5a.set_xlabel('Component', fontsize=9)
    ax5a.set_ylabel('Singular value', fontsize=9)
    ax5a.set_title(f"D5: Singular Value Spectrum ({d5['n_above_mp']} above MP)",
                   fontsize=10, fontweight='bold')
    ax5a.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
                labelcolor='#cccccc')

    # Annotate participation ratio
    ax5a.text(0.98, 0.95,
              f"Eff. dim = {d5['participation_ratio']:.1f}",
              transform=ax5a.transAxes, ha='right', va='top',
              fontsize=8, color='#e74c3c',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222',
                        edgecolor='#444444'))

    # ---- D5b: Cumulative variance explained ---------------------------------
    ax5b = fig.add_subplot(gs[2, 1])
    dark_ax(ax5b)

    ax5b.plot(k, d5['cumvar'] * 100, 'o-', color='#2ecc71', markersize=4,
              linewidth=1.5)
    ax5b.fill_between(k, 0, d5['cumvar'] * 100, color='#2ecc71', alpha=0.15)
    ax5b.axhline(90, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    ax5b.axvline(d5['dim_90'], color='#e74c3c', linestyle='--', linewidth=1,
                 alpha=0.7)

    ax5b.text(d5['dim_90'] + 0.5, 85,
              f"90% at dim {d5['dim_90']}",
              fontsize=8, color='#e74c3c')

    ax5b.set_xlabel('Component', fontsize=9)
    ax5b.set_ylabel('Cumulative variance (%)', fontsize=9)
    ax5b.set_title("D5: Cumulative Variance Explained",
                   fontsize=10, fontweight='bold')
    ax5b.set_ylim(0, 102)

    # Individual variance as bars behind
    ax5b_twin = ax5b.twinx()
    ax5b_twin.bar(k, d5['variance_explained'] * 100, color='#2ecc71',
                  alpha=0.25, width=0.6)
    ax5b_twin.set_ylabel('Individual variance (%)', fontsize=7,
                         color='#888888')
    ax5b_twin.tick_params(labelsize=6, colors='#888888')
    ax5b_twin.set_ylim(0, d5['variance_explained'][0] * 100 * 2.5)

    # Category legend
    handles = []
    for cat in sorted(CATEGORY_COLORS.keys()):
        handles.append(plt.Line2D([0], [0], marker='s', color='none',
                                  markerfacecolor=CATEGORY_COLORS[cat],
                                  markersize=6, label=cat))
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=7,
               frameon=True, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc', bbox_to_anchor=(0.5, 0.005))

    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'figures', 'meta_geometry.png'),
                dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: figures/meta_geometry.png")


# =============================================================================
# MAIN
# =============================================================================

def run_investigation():
    print("=" * 78)
    print("META-GEOMETRY: The Framework Analyzing Its Own Signature Space")
    print("=" * 78)

    names, categories, metric_names, means_matrix, stds_matrix = load_signatures()

    d1 = direction_1(names, categories, metric_names, means_matrix, stds_matrix)
    d2 = direction_2(names, categories, metric_names, means_matrix, stds_matrix)
    d3 = direction_3(names, categories, metric_names, means_matrix, stds_matrix)
    d4 = direction_4(names, categories, metric_names, means_matrix, stds_matrix)
    d5 = direction_5(means_matrix)

    make_figure(d1, d2, d3, d4, d5)

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"D1: Distance matrix — {d1['n_sig']} sig / {N_SPATIAL} "
          f"spatial metrics detect structure")
    print(f"D2: Correlation field — {d2['n_sig']} sig / {N_SPATIAL} "
          f"spatial metrics detect structure")
    pw = d3['matrix']
    mean_pw = np.mean(pw[np.triu_indices(len(d3['cat_names']), k=1)])
    print(f"D3: Categories — mean {mean_pw:.0f} sig pairwise "
          f"(of {d3['n_1d']})")
    top_geom = d4['sorted_families'][0]
    print(f"D4: Top discriminator — {top_geom[0]} (SNR={top_geom[1]:.2f})")
    print(f"D5: Effective dimension — {d5['participation_ratio']:.1f} "
          f"({d5['n_above_mp']} above MP, 90% at dim {d5['dim_90']})")


if __name__ == "__main__":
    run_investigation()
