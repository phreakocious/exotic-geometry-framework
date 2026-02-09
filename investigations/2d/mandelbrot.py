#!/usr/bin/env python3
"""
Mandelbrot & Julia Set Investigation: Geometry of Escape Times
==============================================================

Can 2D spatial geometries (80 metrics) distinguish different regions of the
Mandelbrot set? Can they detect the connected/disconnected Julia set transition?
Are mini-Mandelbrots geometrically self-similar to the full set?

DIRECTIONS:
D1: Region taxonomy — 6 Mandelbrot regions, pairwise distinguishability
D2: Julia parameter sweep — geometry as c crosses the M-set boundary
D3: Self-similarity — mini-Mandelbrots vs full set at different scales
D4: Boundary proximity — geometric complexity vs distance from boundary
D5: Escape times vs null models — shuffled, random, Gaussian random field
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from exotic_geometry_framework import GeometryAnalyzer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from scipy.ndimage import gaussian_filter

# ==============================================================
# CONFIG
# ==============================================================
FIELD_SIZE = 128
MAX_ITER = 256
N_TRIALS = 25
ALPHA = 0.05
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# Discover metric names from a dummy run
_analyzer = GeometryAnalyzer().add_spatial_geometries()
_dummy = _analyzer.analyze(np.random.rand(16, 16))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_ALPHA = ALPHA / N_METRICS
print(f"Spatial geometries: {len(_dummy.results)}, metrics: {N_METRICS}")


# ==============================================================
# ESCAPE TIME COMPUTATION
# ==============================================================
def mandelbrot_field(cx, cy, width, size=FIELD_SIZE, max_iter=MAX_ITER):
    """Compute smooth escape time field for the Mandelbrot set."""
    hw = width / 2
    re = np.linspace(cx - hw, cx + hw, size)
    im = np.linspace(cy - hw, cy + hw, size)
    C = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    Z = np.zeros_like(C)
    escape = np.full((size, size), float(max_iter))

    for i in range(max_iter):
        mask = escape == max_iter
        if not mask.any():
            break
        Z[mask] = Z[mask]**2 + C[mask]
        escaped = mask & (np.abs(Z) > 2)
        if escaped.any():
            absZ = np.abs(Z[escaped])
            # Smooth iteration count: i + 1 - log2(log2(|z|))
            log_zn = np.log(absZ)
            escape[escaped] = i + 1 - np.log(log_zn / np.log(2)) / np.log(2)

    return np.clip(escape, 0, max_iter)


def julia_field(c, cx=0, cy=0, width=4, size=FIELD_SIZE, max_iter=MAX_ITER):
    """Compute smooth escape time field for a Julia set with parameter c."""
    hw = width / 2
    re = np.linspace(cx - hw, cx + hw, size)
    im = np.linspace(cy - hw, cy + hw, size)
    Z = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    escape = np.full((size, size), float(max_iter))

    for i in range(max_iter):
        mask = escape == max_iter
        if not mask.any():
            break
        Z[mask] = Z[mask]**2 + c
        escaped = mask & (np.abs(Z) > 2)
        if escaped.any():
            absZ = np.abs(Z[escaped])
            log_zn = np.log(absZ)
            escape[escaped] = i + 1 - np.log(log_zn / np.log(2)) / np.log(2)

    return np.clip(escape, 0, max_iter)


def normalize_field(field):
    """Normalize to [0, 1]."""
    lo, hi = field.min(), field.max()
    if hi - lo > 1e-10:
        return (field - lo) / (hi - lo)
    return np.full_like(field, 0.5)


# ==============================================================
# STATISTICS
# ==============================================================
def collect_metrics(analyzer, fields):
    """Analyze fields, return dict of metric_name -> list of values."""
    out = {m: [] for m in METRIC_NAMES}
    for f in fields:
        res = analyzer.analyze(normalize_field(f))
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def compare(data_a, data_b, label=""):
    """Compare two metric dicts. Return (n_sig, findings_list)."""
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a, b = np.array(data_a[m]), np.array(data_b[m])
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        if not np.isfinite(d):
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if p < BONF_ALPHA and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


# ==============================================================
# DARK THEME
# ==============================================================
def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# ==============================================================
# MANDELBROT REGIONS
# ==============================================================
REGIONS = {
    'cardioid':  ((-0.25,  0.0),   1.0,   0.15),
    'period2':   ((-1.0,   0.0),   0.4,   0.05),
    'seahorse':  ((-0.745, 0.113), 0.02,  0.003),
    'elephant':  ((0.282,  0.01),  0.02,  0.003),
    'antenna':   ((-1.75,  0.0),   0.15,  0.02),
    'exterior':  ((0.5,    0.5),   0.5,   0.1),
}

# Julia parameter sweep along real axis
JULIA_C = [
    0.0, -0.4, -0.75, -1.0, -1.25, -1.368, -1.401,
    -1.5, -1.75, -1.95, -2.0, -2.25, -2.5,
]

# Mini-Mandelbrot locations
MINIS = {
    'full':   ((-0.5,    0.0),    3.0),
    'mini_1': ((-1.7686, 0.00175), 0.006),
    'mini_2': ((-0.1565, 1.0323),  0.004),
    'mini_3': ((-1.256,  0.3815),  0.008),
}

# Boundary probes along real axis (boundary at c = 0.25)
PROBES = [
    ('far_int',    -0.1),
    ('mid_int',     0.15),
    ('near_int',    0.24),
    ('boundary',    0.25),
    ('near_ext',    0.26),
    ('mid_ext',     0.35),
    ('far_ext',     0.50),
]


# ==============================================================
# D1: REGION TAXONOMY
# ==============================================================
def direction_1(analyzer):
    print("\n" + "=" * 60)
    print("D1: REGION TAXONOMY — 6 Mandelbrot regions")
    print("=" * 60)

    region_data = {}
    example_fields = {}
    rnames = list(REGIONS.keys())

    for rname in rnames:
        (cx0, cy0), width, jitter = REGIONS[rname]
        print(f"  {rname:12s}...", end=" ", flush=True)
        t0 = time.time()
        fields = []
        for t in range(N_TRIALS):
            cx = cx0 + np.random.uniform(-jitter, jitter)
            cy = cy0 + np.random.uniform(-jitter, jitter)
            field = mandelbrot_field(cx, cy, width)
            fields.append(field)
            if t == 0:
                example_fields[rname] = field
        region_data[rname] = collect_metrics(analyzer, fields)
        print(f"{time.time() - t0:.1f}s")

    # Pairwise
    n = len(rnames)
    sig_matrix = np.zeros((n, n), dtype=int)
    print(f"\n  Pairwise ({N_METRICS} metrics):")
    for i in range(n):
        for j in range(i + 1, n):
            ns, findings = compare(region_data[rnames[i]], region_data[rnames[j]])
            sig_matrix[i, j] = sig_matrix[j, i] = ns
            print(f"    {rnames[i]:12s} vs {rnames[j]:12s} = {ns:3d} sig")

    return dict(data=region_data, sig_matrix=sig_matrix, names=rnames,
                examples=example_fields)


# ==============================================================
# D2: JULIA PARAMETER SWEEP
# ==============================================================
def direction_2(analyzer):
    print("\n" + "=" * 60)
    print("D2: JULIA PARAMETER SWEEP — connected/disconnected transition")
    print("=" * 60)

    sweep_data = {}
    example_julias = {}

    for c_re in JULIA_C:
        c = complex(c_re, 0)
        print(f"  c = {c_re:+.3f}...", end=" ", flush=True)
        t0 = time.time()
        fields = []
        for t in range(N_TRIALS):
            jc = c + complex(np.random.uniform(-0.005, 0.005),
                             np.random.uniform(-0.005, 0.005))
            fields.append(julia_field(jc, width=4))
            if t == 0:
                example_julias[c_re] = fields[0]
        sweep_data[c_re] = collect_metrics(analyzer, fields)
        print(f"{time.time() - t0:.1f}s")

    # Connected (c inside M-set on real axis: -2 < c < 0.25) vs disconnected
    int_data = {m: [] for m in METRIC_NAMES}
    ext_data = {m: [] for m in METRIC_NAMES}
    for c_re, data in sweep_data.items():
        target = int_data if -2.0 <= c_re <= 0.25 else ext_data
        for m in METRIC_NAMES:
            target[m].extend(data[m])

    ns, findings = compare(int_data, ext_data)
    print(f"\n  Connected vs disconnected: {ns} sig")
    print("  Top 5 discriminators:")
    for name, d, p in findings[:5]:
        print(f"    {name:50s}  d={d:+.2f}")

    return dict(sweep_data=sweep_data, connected_vs_disc=(ns, findings),
                examples=example_julias)


# ==============================================================
# D3: SELF-SIMILARITY — mini-Mandelbrots
# ==============================================================
def direction_3(analyzer):
    print("\n" + "=" * 60)
    print("D3: SELF-SIMILARITY — mini-Mandelbrots vs full set")
    print("=" * 60)

    mini_data = {}
    examples = {}
    mnames = list(MINIS.keys())

    for mname in mnames:
        (cx0, cy0), width = MINIS[mname]
        jitter = width * 0.05
        print(f"  {mname:12s} width={width:.4f}...", end=" ", flush=True)
        t0 = time.time()
        fields = []
        for t in range(N_TRIALS):
            cx = cx0 + np.random.uniform(-jitter, jitter)
            cy = cy0 + np.random.uniform(-jitter, jitter)
            fields.append(mandelbrot_field(cx, cy, width))
            if t == 0:
                examples[mname] = fields[0]
        mini_data[mname] = collect_metrics(analyzer, fields)
        print(f"{time.time() - t0:.1f}s")

    # Compare each mini to full
    results = {}
    print(f"\n  vs full set ({N_METRICS} metrics):")
    for mname in mnames[1:]:
        ns, findings = compare(mini_data['full'], mini_data[mname])
        results[mname] = (ns, findings)
        print(f"    full vs {mname:12s} = {ns:3d} sig")

    # Pairwise minis
    print("  Mini pairwise:")
    for i in range(1, len(mnames)):
        for j in range(i + 1, len(mnames)):
            ns, _ = compare(mini_data[mnames[i]], mini_data[mnames[j]])
            print(f"    {mnames[i]:12s} vs {mnames[j]:12s} = {ns:3d} sig")

    return dict(data=mini_data, names=mnames, results=results, examples=examples)


# ==============================================================
# D4: BOUNDARY PROXIMITY
# ==============================================================
def direction_4(analyzer):
    print("\n" + "=" * 60)
    print("D4: BOUNDARY PROXIMITY — complexity vs distance from boundary")
    print("=" * 60)

    probe_width = 0.05
    probe_data = {}
    probe_fields = {}

    for pname, cx in PROBES:
        print(f"  {pname:12s} c_re={cx:.2f}...", end=" ", flush=True)
        t0 = time.time()
        fields = []
        for t in range(N_TRIALS):
            jcy = np.random.uniform(-0.005, 0.005)
            fields.append(mandelbrot_field(cx, jcy, probe_width))
        probe_data[pname] = collect_metrics(analyzer, fields)
        probe_fields[pname] = fields
        print(f"{time.time() - t0:.1f}s")

    # Compare each to its own shuffled version
    print(f"\n  Structure vs shuffled:")
    vs_shuffled = {}
    for pname, cx in PROBES:
        shuf_fields = []
        for f in probe_fields[pname]:
            flat = f.flatten().copy()
            np.random.shuffle(flat)
            shuf_fields.append(flat.reshape(f.shape))
        shuf_data = collect_metrics(analyzer, shuf_fields)
        ns, _ = compare(probe_data[pname], shuf_data)
        vs_shuffled[pname] = ns
        print(f"    {pname:12s} vs shuffled = {ns:3d} sig")

    return dict(data=probe_data, vs_shuffled=vs_shuffled)


# ==============================================================
# D5: ESCAPE TIMES VS NULL MODELS
# ==============================================================
def direction_5(analyzer):
    print("\n" + "=" * 60)
    print("D5: ESCAPE TIMES VS NULL MODELS (seahorse valley)")
    print("=" * 60)

    (cx0, cy0), width, jitter = REGIONS['seahorse']

    # Real fields
    print("  Generating seahorse fields...", end=" ", flush=True)
    t0 = time.time()
    real_fields = []
    for t in range(N_TRIALS):
        cx = cx0 + np.random.uniform(-jitter, jitter)
        cy = cy0 + np.random.uniform(-jitter, jitter)
        real_fields.append(mandelbrot_field(cx, cy, width))
    real_data = collect_metrics(analyzer, real_fields)
    print(f"{time.time() - t0:.1f}s")

    # Null 1: Shuffled
    print("  Shuffled...", end=" ", flush=True)
    t0 = time.time()
    shuf_fields = []
    for f in real_fields:
        flat = f.flatten().copy()
        np.random.shuffle(flat)
        shuf_fields.append(flat.reshape(f.shape))
    shuf_data = collect_metrics(analyzer, shuf_fields)
    print(f"{time.time() - t0:.1f}s")

    # Null 2: Uniform random
    print("  Uniform random...", end=" ", flush=True)
    t0 = time.time()
    rand_fields = [np.random.uniform(0, 1, (FIELD_SIZE, FIELD_SIZE))
                   for _ in range(N_TRIALS)]
    rand_data = collect_metrics(analyzer, rand_fields)
    print(f"{time.time() - t0:.1f}s")

    # Null 3: Gaussian random field (spatially correlated)
    print("  Gaussian random field...", end=" ", flush=True)
    t0 = time.time()
    grf_fields = []
    for _ in range(N_TRIALS):
        raw = np.random.randn(FIELD_SIZE, FIELD_SIZE)
        smooth = gaussian_filter(raw, sigma=3)
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-10)
        grf_fields.append(smooth)
    grf_data = collect_metrics(analyzer, grf_fields)
    print(f"{time.time() - t0:.1f}s")

    comparisons = [
        ('vs_shuffled', real_data, shuf_data),
        ('vs_random',   real_data, rand_data),
        ('vs_GRF',      real_data, grf_data),
    ]

    results = {}
    print(f"\n  Seahorse vs null models ({N_METRICS} metrics):")
    for cname, a, b in comparisons:
        ns, findings = compare(a, b)
        results[cname] = (ns, findings)
        print(f"    {cname:15s} = {ns:3d} sig")

    # Top discriminators vs shuffled
    print("  Top 5 (vs shuffled):")
    for name, d, p in results['vs_shuffled'][1][:5]:
        print(f"    {name:50s}  d={d:+.2f}")

    return results


# ==============================================================
# FIGURE
# ==============================================================
def make_figure(d1, d2, d3, d4, d5):
    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(22, 20), facecolor='#181818')
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.40, wspace=0.40,
                           left=0.05, right=0.97, top=0.94, bottom=0.05,
                           height_ratios=[0.6, 1.0, 1.0])

    fig.suptitle("Mandelbrot & Julia Set Investigation: Geometry of Escape Times",
                 fontsize=15, fontweight='bold', color='white')

    # ---- Row 0: Example escape time fields ----
    for idx, rname in enumerate(d1['names']):
        ax = fig.add_subplot(gs[0, idx])
        _dark_ax(ax)
        field = d1['examples'][rname]
        ax.imshow(np.log1p(field), cmap='inferno', aspect='equal',
                  interpolation='nearest')
        ax.set_title(rname, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- Row 1 left: D1 pairwise heatmap ----
    ax1 = fig.add_subplot(gs[1, :3])
    _dark_ax(ax1)
    rnames = d1['names']
    mat = d1['sig_matrix']
    im = ax1.imshow(mat, cmap='YlOrRd', vmin=0,
                    vmax=max(N_METRICS, mat.max()))
    ax1.set_xticks(range(len(rnames)))
    ax1.set_yticks(range(len(rnames)))
    ax1.set_xticklabels(rnames, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(rnames, fontsize=8)
    for i in range(len(rnames)):
        for j in range(len(rnames)):
            if i != j:
                c = 'white' if mat[i, j] > N_METRICS * 0.5 else '#cccccc'
                ax1.text(j, i, str(mat[i, j]), ha='center', va='center',
                         fontsize=9, color=c)
    ax1.set_title(f"D1: Region Pairwise (sig of {N_METRICS})", fontsize=11)
    cb = plt.colorbar(im, ax=ax1, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color='#cccccc')
    cb.outline.set_edgecolor('#444444')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#cccccc')

    # ---- Row 1 right: D2 Julia sweep ----
    ax2 = fig.add_subplot(gs[1, 3:])
    _dark_ax(ax2)
    c_vals = sorted(d2['sweep_data'].keys())

    # Pick metrics with highest variance across c values
    mean_by_c = {}
    for name in METRIC_NAMES:
        vals = []
        for c in c_vals:
            v = d2['sweep_data'][c][name]
            vals.append(np.mean(v) if len(v) > 0 else np.nan)
        arr = np.array(vals)
        if np.all(np.isfinite(arr)):
            rng = arr.max() - arr.min()
            if rng > 1e-10:
                mean_by_c[name] = (arr, rng)

    top = sorted(mean_by_c.keys(), key=lambda x: mean_by_c[x][1], reverse=True)[:6]
    colors = plt.cm.Set2(np.linspace(0, 1, len(top)))

    for name, color in zip(top, colors):
        arr = mean_by_c[name][0]
        lo, hi = arr.min(), arr.max()
        normed = (arr - lo) / (hi - lo)
        short = name.split(':')[-1][:20]
        ax2.plot(c_vals, normed, 'o-', color=color, markersize=3,
                 linewidth=1.5, label=short)

    ax2.axvline(x=-2.0, color='#e74c3c', ls='--', alpha=0.6, lw=1)
    ax2.text(-2.0, 1.07, 'M-set\nbdy', ha='center', fontsize=7, color='#e74c3c')
    ax2.axvline(x=-1.401, color='#f1c40f', ls=':', alpha=0.5, lw=1)
    ax2.text(-1.401, 1.07, 'Feigen-\nbaum', ha='center', fontsize=7,
             color='#f1c40f')
    ax2.set_xlabel("c (real axis)", fontsize=9)
    ax2.set_ylabel("Metric (normalized)", fontsize=9)
    ax2.set_title("D2: Julia Set Metrics vs Parameter c", fontsize=11)
    ax2.legend(fontsize=7, loc='upper left', facecolor='#222222',
               edgecolor='#444444', labelcolor='#cccccc')

    # ---- Row 2 left: D3 self-similarity ----
    ax3 = fig.add_subplot(gs[2, :2])
    _dark_ax(ax3)
    mnames = d3['names'][1:]  # skip 'full'
    sig_vals = [d3['results'][m][0] for m in mnames]
    bars = ax3.bar(range(len(mnames)), sig_vals, color='#e74c3c', alpha=0.85)
    ax3.set_xticks(range(len(mnames)))
    ax3.set_xticklabels(mnames, fontsize=8)
    ax3.set_ylabel(f"Sig metrics (of {N_METRICS})", fontsize=9)
    ax3.set_title("D3: Mini-Mandelbrot vs Full Set", fontsize=11)
    ax3.axhline(y=0, color='#444444', lw=0.5)
    for bar, val in zip(bars, sig_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 str(val), ha='center', fontsize=9, color='white')

    # ---- Row 2 middle: D4 boundary proximity ----
    ax4 = fig.add_subplot(gs[2, 2:4])
    _dark_ax(ax4)
    pnames = [p[0] for p in PROBES]
    c_positions = [p[1] for p in PROBES]
    shuf_vals = [d4['vs_shuffled'][p] for p in pnames]
    ax4.plot(c_positions, shuf_vals, 'o-', color='#3498db', markersize=7, lw=2)
    for cx, sv in zip(c_positions, shuf_vals):
        ax4.annotate(str(sv), (cx, sv), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8, color='white')
    ax4.axvline(x=0.25, color='#e74c3c', ls='--', alpha=0.6, lw=1)
    ax4.text(0.25, max(shuf_vals) * 1.08, 'boundary', ha='center',
             fontsize=8, color='#e74c3c')
    ax4.set_xlabel("c (real axis)", fontsize=9)
    ax4.set_ylabel("Sig metrics vs shuffled", fontsize=9)
    ax4.set_title("D4: Geometric Complexity vs Boundary Distance", fontsize=11)

    # ---- Row 2 right: D5 null models ----
    ax5 = fig.add_subplot(gs[2, 4:])
    _dark_ax(ax5)
    null_labels = ['Shuffled', 'Uniform\nRandom', 'Gaussian\nRF']
    null_keys = ['vs_shuffled', 'vs_random', 'vs_GRF']
    null_vals = [d5[k][0] for k in null_keys]
    null_colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax5.bar(range(len(null_labels)), null_vals, color=null_colors, alpha=0.85)
    ax5.set_xticks(range(len(null_labels)))
    ax5.set_xticklabels(null_labels, fontsize=9)
    ax5.set_ylabel(f"Sig metrics (of {N_METRICS})", fontsize=9)
    ax5.set_title("D5: Seahorse Valley vs Null Models", fontsize=11)
    for bar, val in zip(bars, null_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 str(val), ha='center', fontsize=9, color='white')

    out = FIG_DIR / "mandelbrot.png"
    fig.savefig(out, dpi=180, facecolor='#181818')
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ==============================================================
# MAIN
# ==============================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("MANDELBROT & JULIA SET INVESTIGATION")
    print(f"Field: {FIELD_SIZE}x{FIELD_SIZE}, max_iter={MAX_ITER}, "
          f"trials={N_TRIALS}, metrics={N_METRICS}")
    print("=" * 60)

    analyzer = GeometryAnalyzer().add_spatial_geometries()

    d1 = direction_1(analyzer)
    d2 = direction_2(analyzer)
    d3 = direction_3(analyzer)
    d4 = direction_4(analyzer)
    d5 = direction_5(analyzer)

    make_figure(d1, d2, d3, d4, d5)

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mat = d1['sig_matrix']
    pw = mat[np.triu_indices_from(mat, k=1)]
    print(f"D1: All 6 regions pairwise: {pw.min()}-{pw.max()} sig "
          f"(mean {pw.mean():.0f})")

    ns_cd = d2['connected_vs_disc'][0]
    print(f"D2: Connected vs disconnected Julia: {ns_cd} sig")

    for mname in d3['names'][1:]:
        ns = d3['results'][mname][0]
        print(f"D3: full vs {mname}: {ns} sig "
              f"({'similar' if ns < 10 else 'different'})")

    pnames = [p[0] for p in PROBES]
    peak_p = max(d4['vs_shuffled'], key=d4['vs_shuffled'].get)
    peak_v = d4['vs_shuffled'][peak_p]
    print(f"D4: Peak complexity at {peak_p} ({peak_v} sig vs shuffled)")

    for k in ['vs_shuffled', 'vs_random', 'vs_GRF']:
        print(f"D5: Seahorse {k} = {d5[k][0]} sig")


if __name__ == "__main__":
    main()
