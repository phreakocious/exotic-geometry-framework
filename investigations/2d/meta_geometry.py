#!/usr/bin/env python3
"""
Meta-Geometry Investigation: The Framework Analyzing Its Own Signature Space

The framework's 179 source profiles each encode a 233-element vector of
metric means (44 geometries × varying metrics). This investigation turns the
framework on itself — treating the signature space as data to be analyzed.

Five directions:
  D1: Signature distance matrix as 2D field (spatial geometries)
  D2: Metric correlation matrix as 2D field (spatial geometries)
  D3: Domain distinguishability (Welch t-test + Cohen's d per metric pair)
  D4: Discriminating power by geometry (ANOVA F-statistic across domains)
  D5: Spectral dimension of signature space (SVD vs Marchenko-Pastur)

Data source: figures/structure_atlas_data.json (179 sources, 16 domains, 233 metrics)
"""

import sys, os, json, time
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


# =============================================================================
# LOAD ATLAS DATA
# =============================================================================

def load_atlas():
    """Load profiles, sources, and metadata from the atlas JSON."""
    atlas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', '..', 'figures', 'structure_atlas_data.json')
    with open(atlas_path) as f:
        atlas = json.load(f)

    names = [s['name'] for s in atlas['sources']]
    domains = [s['domain'] for s in atlas['sources']]
    metric_names = atlas['metric_names']
    domain_colors = atlas['domain_colors']
    profiles = np.array(atlas['profiles'], dtype=np.float64)  # (179, 233)

    # Replace non-finite values with NaN for consistent handling
    profiles[~np.isfinite(profiles)] = np.nan

    print(f"Loaded {len(names)} sources, {len(metric_names)} metrics, "
          f"{len(domain_colors)} domains")
    return names, domains, metric_names, domain_colors, profiles


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

BONF_2D = ALPHA / N_SPATIAL

print(f"2D spatial metrics: {N_SPATIAL}")


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
    ax.tick_params(colors='#cccccc', labelsize=9)
    return ax


# =============================================================================
# D1: SIGNATURE DISTANCE MATRIX AS 2D FIELD
# =============================================================================

def direction_1(names, domains, metric_names, profiles):
    """Compute and analyze the signature distance matrix."""
    print("\n" + "=" * 78)
    print("D1: Signature Distance Matrix as 2D Field")
    print("=" * 78)

    n_sigs, n_met = profiles.shape
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    # Z-score columns for cosine distance
    col_means = np.nanmean(profiles, axis=0)
    col_stds = np.nanstd(profiles, axis=0)
    col_stds[col_stds < 1e-15] = 1.0
    z_matrix = (profiles - col_means) / col_stds
    z_matrix = np.nan_to_num(z_matrix, nan=0.0)

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
        'domains': domains,
    }


# =============================================================================
# D2: METRIC CORRELATION GEOMETRY AS 2D FIELD
# =============================================================================

def direction_2(names, domains, metric_names, profiles):
    """Compute and analyze the metric correlation matrix."""
    print("\n" + "=" * 78)
    print("D2: Metric Correlation Geometry as 2D Field")
    print("=" * 78)

    n_sigs, n_met = profiles.shape
    analyzer = GeometryAnalyzer().add_spatial_geometries()

    # Use all 233 metrics (atlas metrics are already single-scale)
    print(f"  Using all {n_met} metrics")

    rng = np.random.default_rng(123)

    real_fields = []
    null_fields = []

    t0 = time.time()
    for trial in range(N_TRIALS):
        # Bootstrap: resample ~75% of sources with replacement
        boot_idx = rng.choice(n_sigs, size=int(0.75 * n_sigs), replace=True)
        sub = profiles[boot_idx, :]

        # Real correlation matrix
        corr = np.corrcoef(sub.T)  # (233, 233)
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
    full_corr = np.corrcoef(profiles.T)
    np.fill_diagonal(full_corr, 0)
    full_corr = np.nan_to_num(full_corr, nan=0.0)

    # Geometry family boundaries for dividers
    families = []
    boundaries = []
    prev_fam = None
    for i, m in enumerate(metric_names):
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
        'n_met': n_met,
    }


# =============================================================================
# D3: DOMAIN DISTINGUISHABILITY
# =============================================================================

def direction_3(names, domains, metric_names, domain_colors, profiles):
    """Pairwise domain distinguishability via Welch t-test + Cohen's d."""
    print("\n" + "=" * 78)
    print("D3: Domain Distinguishability")
    print("=" * 78)

    n_sigs, n_met = profiles.shape
    bonf_alpha = ALPHA / n_met

    # Group source indices by domain
    domain_indices = {}
    for i, dom in enumerate(domains):
        domain_indices.setdefault(dom, []).append(i)

    # Only keep domains with >= 3 sources
    domain_names = sorted([d for d, idx in domain_indices.items() if len(idx) >= 3])
    n_doms = len(domain_names)
    print(f"  {n_doms} domains with >= 3 sources")

    # Pairwise: count metrics where Welch t-test is significant AND |d| > 0.8
    pw_matrix = np.zeros((n_doms, n_doms), dtype=int)

    for i in range(n_doms):
        for j in range(i + 1, n_doms):
            idx_a = domain_indices[domain_names[i]]
            idx_b = domain_indices[domain_names[j]]
            n_sig = 0
            for k in range(n_met):
                a = profiles[idx_a, k]
                b = profiles[idx_b, k]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if len(a) < 3 or len(b) < 3:
                    continue
                d = cohens_d(a, b)
                if not np.isfinite(d) or abs(d) <= 0.8:
                    continue
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if p < bonf_alpha:
                    n_sig += 1
            pw_matrix[i, j] = pw_matrix[j, i] = n_sig
            if n_sig > 0:
                print(f"    {domain_names[i]:15s} vs {domain_names[j]:15s}: "
                      f"{n_sig:3d} sig metrics")

    return {
        'matrix': pw_matrix,
        'domain_names': domain_names,
        'n_met': n_met,
    }


# =============================================================================
# D4: DISCRIMINATING POWER BY GEOMETRY (ANOVA F)
# =============================================================================

def direction_4(names, domains, metric_names, profiles):
    """ANOVA F-statistic per metric across domains, grouped by geometry."""
    print("\n" + "=" * 78)
    print("D4: Discriminating Power by Geometry (ANOVA F)")
    print("=" * 78)

    n_sigs, n_met = profiles.shape

    # Group source indices by domain (>= 3 sources)
    domain_indices = {}
    for i, dom in enumerate(domains):
        domain_indices.setdefault(dom, []).append(i)
    valid_domains = {d: idx for d, idx in domain_indices.items() if len(idx) >= 3}

    # ANOVA F per metric
    metric_f = {}
    for j in range(n_met):
        groups = []
        for dom, idx in valid_domains.items():
            vals = profiles[idx, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            if np.isfinite(f_stat):
                metric_f[metric_names[j]] = f_stat

    print(f"  {len(metric_f)} / {n_met} metrics have finite ANOVA F")

    # Group by geometry family
    family_fs = {}
    for m, f in metric_f.items():
        fam = m.split(':')[0]
        family_fs.setdefault(fam, []).append(f)

    family_mean_f = {}
    for fam, fs in family_fs.items():
        family_mean_f[fam] = np.mean(fs)

    # Sort by mean F descending
    sorted_families = sorted(family_mean_f.items(), key=lambda x: -x[1])
    print(f"  Top 5 discriminating geometries (mean ANOVA F):")
    for fam, f in sorted_families[:5]:
        print(f"    {fam}: F={f:.2f}")
    print(f"  Bottom 5:")
    for fam, f in sorted_families[-5:]:
        print(f"    {fam}: F={f:.2f}")

    # Top 10 individual metrics
    sorted_metrics = sorted(metric_f.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 discriminating metrics:")
    for m, f in sorted_metrics[:10]:
        print(f"    {m}: F={f:.2f}")

    return {
        'sorted_families': sorted_families,
        'top_metrics': sorted_metrics[:10],
    }


# =============================================================================
# D5: SPECTRAL DIMENSION OF SIGNATURE SPACE
# =============================================================================

def direction_5(profiles):
    """SVD of profile matrix, compare to Marchenko-Pastur."""
    print("\n" + "=" * 78)
    print("D5: Spectral Dimension of Signature Space")
    print("=" * 78)

    n_sigs, n_met = profiles.shape

    # Z-score columns, dropping zero-variance columns
    col_means = np.nanmean(profiles, axis=0)
    col_stds = np.nanstd(profiles, axis=0)
    good_cols = col_stds > 1e-10
    means_good = profiles[:, good_cols]
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
    }


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(d1, d2, d3, d4, d5, domain_colors):
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

    fig = plt.figure(figsize=(18, 22), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.96, top=0.95, bottom=0.05)
    fig.suptitle("Meta-Geometry: The Framework Analyzing Its Own Signatures\n"
                 "179 sources · 233 metrics · 16 domains",
                 fontsize=15, fontweight='bold', color='white')

    # ---- D1: Distance matrix heatmap (clustered, domain color sidebar) ------
    ax1 = fig.add_subplot(gs[0, 0])
    dark_ax(ax1)

    order = d1['order']
    n = len(order)
    dist_ordered = d1['dist_matrix'][np.ix_(order, order)]

    # Percentile-based colormap limits to fix yellow saturation
    vmin1 = np.percentile(dist_ordered[dist_ordered > 0], 2)
    vmax1 = np.percentile(dist_ordered, 98)

    im1 = ax1.imshow(dist_ordered, cmap='viridis', aspect='equal',
                     vmin=vmin1, vmax=vmax1)
    ax1.set_title(f"D1: Signature Distance Matrix ({d1['n_sig']} sig / {N_SPATIAL})",
                  fontsize=12, fontweight='bold')

    # Domain color sidebar (no per-source labels at 179 sources)
    ordered_doms = [d1['domains'][i] for i in order]
    for i, dom in enumerate(ordered_doms):
        ax1.add_patch(Rectangle((-3.5, i - 0.5), 2.5, 1,
                                color=domain_colors.get(dom, '#888888'),
                                clip_on=False))
    # Also add top sidebar
    for i, dom in enumerate(ordered_doms):
        ax1.add_patch(Rectangle((i - 0.5, -3.5), 1, 2.5,
                                color=domain_colors.get(dom, '#888888'),
                                clip_on=False))

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(n - 0.5, -0.5)

    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.6, pad=0.02)
    cb1.ax.tick_params(labelsize=8, colors='#cccccc')
    cb1.set_label('Cosine distance', fontsize=9, color='#cccccc')

    # ---- D2: Correlation matrix heatmap (233×233) ---------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    dark_ax(ax2)

    im2 = ax2.imshow(d2['corr_matrix'], cmap='RdBu_r', vmin=-1, vmax=1,
                     aspect='equal')
    ax2.set_title(f"D2: Metric Correlation Field ({d2['n_sig']} sig / {N_SPATIAL})",
                  fontsize=12, fontweight='bold')

    # Geometry family dividers
    for b in d2['boundaries'][1:]:
        ax2.axhline(b - 0.5, color='#666666', linewidth=0.5, alpha=0.7)
        ax2.axvline(b - 0.5, color='#666666', linewidth=0.5, alpha=0.7)

    # Family labels at midpoints
    bounds = d2['boundaries'] + [d2['n_met']]
    mid_ticks = []
    mid_labels = []
    for k in range(len(bounds) - 1):
        mid = (bounds[k] + bounds[k + 1]) / 2
        mid_ticks.append(mid)
        fam = d2['families'][k]
        abbr = fam[:10] if len(fam) > 10 else fam
        mid_labels.append(abbr)

    ax2.set_xticks(mid_ticks)
    ax2.set_xticklabels(mid_labels, rotation=90, fontsize=5)
    ax2.set_yticks(mid_ticks)
    ax2.set_yticklabels(mid_labels, fontsize=5)

    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.6, pad=0.02)
    cb2.ax.tick_params(labelsize=8, colors='#cccccc')
    cb2.set_label('Pearson r', fontsize=9, color='#cccccc')

    # ---- D3: Domain pairwise distinguishability matrix ----------------------
    ax3 = fig.add_subplot(gs[1, 0])
    dark_ax(ax3)

    n_doms = len(d3['domain_names'])
    mat3 = d3['matrix']
    max3 = max(np.max(mat3), 1)

    im3 = ax3.imshow(mat3, cmap='YlOrRd', vmin=0,
                     vmax=min(d3['n_met'], max3 * 1.3 + 1),
                     aspect='equal')
    ax3.set_xticks(range(n_doms))
    ax3.set_yticks(range(n_doms))
    # Short domain labels
    short_doms = [dn[:8] for dn in d3['domain_names']]
    ax3.set_xticklabels(short_doms, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(short_doms, fontsize=8)

    for i in range(n_doms):
        for j in range(n_doms):
            if i == j:
                continue
            val = int(mat3[i, j])
            color = 'white' if val > max3 * 0.5 else '#cccccc'
            ax3.text(j, i, str(val), ha='center', va='center',
                     fontsize=6, color=color, fontweight='bold')

    ax3.set_title(f"D3: Domain Distinguishability (of {d3['n_met']} metrics)",
                  fontsize=12, fontweight='bold')

    cb3 = fig.colorbar(im3, ax=ax3, shrink=0.6, pad=0.02)
    cb3.ax.tick_params(labelsize=8, colors='#cccccc')
    cb3.set_label('# significant metrics', fontsize=9, color='#cccccc')

    # ---- D4: Geometry discriminating power bars (log scale) -----------------
    ax4 = fig.add_subplot(gs[1, 1])
    dark_ax(ax4)

    fam_names = [f[0] for f in d4['sorted_families']]
    fam_fs = [f[1] for f in d4['sorted_families']]
    n_fam = len(fam_names)

    # Log-scale x-axis; clip minimum to 0.1 for display
    fam_fs_plot = [max(f, 0.1) for f in fam_fs]

    norm = plt.Normalize(vmin=np.log10(min(fam_fs_plot)),
                         vmax=np.log10(max(fam_fs_plot)))
    colors4 = plt.cm.plasma(norm(np.log10(fam_fs_plot)))

    y_pos = np.arange(n_fam)
    bars = ax4.barh(y_pos, fam_fs_plot, color=colors4, alpha=0.9, height=0.7)
    ax4.set_xscale('log')
    ax4.set_yticks(y_pos)
    short_fam = [f[:22] + '..' if len(f) > 22 else f for f in fam_names]
    ax4.set_yticklabels(short_fam, fontsize=6)
    ax4.set_xlabel('Mean ANOVA F-statistic (log scale)', fontsize=9)
    ax4.set_title("D4: Discriminating Power by Geometry",
                  fontsize=12, fontweight='bold')
    ax4.invert_yaxis()

    # Value annotations
    for bar, f_val in zip(bars, fam_fs):
        ax4.text(bar.get_width() * 1.15,
                 bar.get_y() + bar.get_height() / 2,
                 f'{f_val:.1f}', va='center', fontsize=5.5, color='#cccccc')

    # ---- D5a: Singular value spectrum vs MP ---------------------------------
    ax5a = fig.add_subplot(gs[2, 0])
    dark_ax(ax5a)

    k = np.arange(1, len(d5['s_real']) + 1)
    ax5a.semilogy(k, d5['s_real'], 'o-', color='#e74c3c', markersize=3,
                  linewidth=1.5, label='Real spectrum', zorder=3)
    ax5a.fill_between(k, d5['mp_lower'], d5['mp_upper'],
                      color='#3498db', alpha=0.3, label='MP 95% envelope')
    ax5a.semilogy(k, d5['mp_mean'], '--', color='#3498db', linewidth=1,
                  alpha=0.8, label='MP mean')

    ax5a.set_xlabel('Component', fontsize=10)
    ax5a.set_ylabel('Singular value', fontsize=10)
    ax5a.set_title(f"D5: Singular Value Spectrum ({d5['n_above_mp']} above MP)",
                   fontsize=12, fontweight='bold')
    ax5a.legend(fontsize=8, facecolor='#222222', edgecolor='#444444',
                labelcolor='#cccccc')

    ax5a.text(0.98, 0.95,
              f"Eff. dim = {d5['participation_ratio']:.1f}",
              transform=ax5a.transAxes, ha='right', va='top',
              fontsize=9, color='#e74c3c',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222',
                        edgecolor='#444444'))

    # ---- D5b: Cumulative variance explained ---------------------------------
    ax5b = fig.add_subplot(gs[2, 1])
    dark_ax(ax5b)

    ax5b.plot(k, d5['cumvar'] * 100, 'o-', color='#2ecc71', markersize=3,
              linewidth=1.5, zorder=3)
    ax5b.axhline(90, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    ax5b.axvline(d5['dim_90'], color='#e74c3c', linestyle='--', linewidth=1,
                 alpha=0.7)

    ax5b.text(d5['dim_90'] + 0.5, 85,
              f"90% at dim {d5['dim_90']}",
              fontsize=9, color='#e74c3c')

    ax5b.set_xlabel('Component', fontsize=10)
    ax5b.set_ylabel('Cumulative variance (%)', fontsize=10)
    ax5b.set_title("D5: Cumulative Variance Explained",
                   fontsize=12, fontweight='bold')
    ax5b.set_ylim(0, 102)

    # Individual variance as bars behind — reduced fill opacity
    ax5b_twin = ax5b.twinx()
    ax5b_twin.bar(k, d5['variance_explained'] * 100, color='#2ecc71',
                  alpha=0.08, width=0.6)
    ax5b_twin.set_ylabel('Individual variance (%)', fontsize=8,
                         color='#888888')
    ax5b_twin.tick_params(labelsize=7, colors='#888888')
    ax5b_twin.set_ylim(0, d5['variance_explained'][0] * 100 * 2.5)

    # Domain color legend
    handles = []
    for dom in sorted(domain_colors.keys()):
        handles.append(plt.Line2D([0], [0], marker='s', color='none',
                                  markerfacecolor=domain_colors[dom],
                                  markersize=7, label=dom))
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=8,
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

    names, domains, metric_names, domain_colors, profiles = load_atlas()

    d1 = direction_1(names, domains, metric_names, profiles)
    d2 = direction_2(names, domains, metric_names, profiles)
    d3 = direction_3(names, domains, metric_names, domain_colors, profiles)
    d4 = direction_4(names, domains, metric_names, profiles)
    d5 = direction_5(profiles)

    make_figure(d1, d2, d3, d4, d5, domain_colors)

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"D1: Distance matrix — {d1['n_sig']} sig / {N_SPATIAL} "
          f"spatial metrics detect structure")
    print(f"D2: Correlation field — {d2['n_sig']} sig / {N_SPATIAL} "
          f"spatial metrics detect structure")
    pw = d3['matrix']
    dom_names = d3['domain_names']
    mean_pw = np.mean(pw[np.triu_indices(len(dom_names), k=1)])
    print(f"D3: Domains — mean {mean_pw:.0f} sig pairwise "
          f"(of {d3['n_met']} metrics)")
    top_geom = d4['sorted_families'][0]
    print(f"D4: Top discriminator — {top_geom[0]} (F={top_geom[1]:.2f})")
    print(f"D5: Effective dimension — {d5['participation_ratio']:.1f} "
          f"({d5['n_above_mp']} above MP, 90% at dim {d5['dim_90']})")


if __name__ == "__main__":
    run_investigation()
