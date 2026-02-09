#!/usr/bin/env python3
"""
Investigation: RNG Quality Testing via Exotic Geometry
=======================================================

Can exotic geometries detect structure in PRNG output that standard statistical
tests miss? We test 10 generators spanning a quality gradient from cryptographic
(os.urandom, SHA-256 CTR) through good (PCG64, MT19937, SFC64) to weak/bad
(XorShift128, MINSTD, RANDU, LFSR-16, Middle-Square).

Key self-check: os.urandom vs os.urandom should yield ~0 significant metrics.
CRYPTO/GOOD generators should also be ~0. BAD generators should be detectable.

Directions:
  D1: Baseline detection — each generator vs os.urandom reference
  D2: Standard features vs geometric metrics comparison
  D3: Delay embedding amplification — does DE2 reveal hidden structure?
  D4: Minimum sequence length for detection
  D5: Geometric taxonomy heatmap — which geometries catch which weaknesses?

Methodology: N_TRIALS=25, DATA_SIZE=2000, Cohen's d > 0.8, Bonferroni correction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import hashlib
import struct
import numpy as np
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, delay_embed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000
ALPHA = 0.05


# =========================================================================
# RNG IMPLEMENTATIONS
# =========================================================================

def gen_urandom(trial, size, seed=None):
    """os.urandom — true random reference."""
    # Use numpy for reproducibility in testing (seeded from os.urandom conceptually)
    # We need reproducible trials, so seed from trial index
    rng = np.random.default_rng(seed if seed is not None else (1000 + trial))
    return rng.integers(0, 256, size, dtype=np.uint8)


def gen_sha256_ctr(trial, size, seed=None):
    """SHA-256 in counter mode — hash(key || counter)."""
    key = struct.pack('<Q', seed if seed is not None else (2000 + trial))
    out = bytearray()
    ctr = 0
    while len(out) < size:
        block = hashlib.sha256(key + struct.pack('<Q', ctr)).digest()
        out.extend(block)
        ctr += 1
    return np.frombuffer(bytes(out[:size]), dtype=np.uint8).copy()


def gen_pcg64(trial, size, seed=None):
    """PCG64 — NumPy default, passes BigCrush."""
    rng = np.random.Generator(np.random.PCG64(seed if seed is not None else (3000 + trial)))
    return rng.integers(0, 256, size, dtype=np.uint8)


def gen_mt19937(trial, size, seed=None):
    """MT19937 — Python/old NumPy default, passes most tests."""
    rng = np.random.Generator(np.random.MT19937(seed if seed is not None else (4000 + trial)))
    return rng.integers(0, 256, size, dtype=np.uint8)


def gen_sfc64(trial, size, seed=None):
    """SFC64 — Small Fast Chaotic, passes BigCrush."""
    rng = np.random.Generator(np.random.SFC64(seed if seed is not None else (5000 + trial)))
    return rng.integers(0, 256, size, dtype=np.uint8)


class XorShift128:
    """Marsaglia 2003 XorShift128. Fails LinearComplexity test."""
    def __init__(self, seed):
        self.x = seed & 0xFFFFFFFF or 1
        self.y = ((seed >> 32) & 0xFFFFFFFF) or 362436069
        self.z = 521288629
        self.w = 88675123

    def next_u32(self):
        t = self.x ^ ((self.x << 11) & 0xFFFFFFFF)
        self.x = self.y
        self.y = self.z
        self.z = self.w
        self.w = (self.w ^ (self.w >> 19)) ^ (t ^ (t >> 8))
        self.w &= 0xFFFFFFFF
        return self.w


def gen_xorshift128(trial, size, seed=None):
    """XorShift128 — fails LinearComplexity."""
    rng = XorShift128(seed if seed is not None else (6000 + trial))
    return np.array([rng.next_u32() & 0xFF for _ in range(size)], dtype=np.uint8)


class LCG:
    """Linear Congruential Generator."""
    def __init__(self, seed, a, c, m):
        self.state = seed % m
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state


def gen_minstd(trial, size, seed=None):
    """MINSTD (Park-Miller) — a=16807, m=2^31-1. Lattice structure."""
    s = seed if seed is not None else (7000 + trial)
    lcg = LCG(s if s > 0 else 1, 16807, 0, 2**31 - 1)
    return np.array([lcg.next() & 0xFF for _ in range(size)], dtype=np.uint8)


def gen_randu(trial, size, seed=None):
    """RANDU — IBM 1968, a=65539, m=2^31. Falls on 15 hyperplanes in 3D."""
    s = seed if seed is not None else (8000 + trial)
    lcg = LCG(s if s > 0 else 1, 65539, 0, 2**31)
    return np.array([lcg.next() & 0xFF for _ in range(size)], dtype=np.uint8)


class LFSR16:
    """16-bit Linear Feedback Shift Register. Period = 65535."""
    def __init__(self, seed):
        self.state = seed & 0xFFFF or 1  # must be nonzero

    def next_byte(self):
        # Taps at bits 16, 14, 13, 11 (maximal-length for 16-bit)
        out = 0
        for bit in range(8):
            feedback = ((self.state >> 15) ^ (self.state >> 13) ^
                        (self.state >> 12) ^ (self.state >> 10)) & 1
            self.state = ((self.state << 1) | feedback) & 0xFFFF
            out = (out << 1) | (self.state & 1)
        return out


def gen_lfsr16(trial, size, seed=None):
    """LFSR-16 — linear, period 65535."""
    lfsr = LFSR16(seed if seed is not None else (9000 + trial))
    return np.array([lfsr.next_byte() for _ in range(size)], dtype=np.uint8)


class MiddleSquare:
    """von Neumann Middle-Square method (1949). Catastrophic cycle tendency."""
    def __init__(self, seed):
        self.state = seed % 10000
        if self.state < 1000:
            self.state += 1000  # need 4+ digit seed

    def next_byte(self):
        sq = self.state * self.state
        # Extract middle 4 digits from 8-digit square
        sq_str = f"{sq:08d}"
        self.state = int(sq_str[2:6])
        if self.state == 0:
            self.state = 1234  # escape absorbing state
        return self.state & 0xFF


def gen_middle_square(trial, size, seed=None):
    """Middle-Square — von Neumann 1949, catastrophic cycles."""
    ms = MiddleSquare(seed if seed is not None else (1234 + trial * 37))
    return np.array([ms.next_byte() for _ in range(size)], dtype=np.uint8)


# Generator registry: (name, function, category)
GENERATORS = [
    ('os.urandom',    gen_urandom,       'CRYPTO'),
    ('SHA-256 CTR',   gen_sha256_ctr,    'CRYPTO'),
    ('PCG64',         gen_pcg64,         'GOOD'),
    ('MT19937',       gen_mt19937,       'GOOD'),
    ('SFC64',         gen_sfc64,         'GOOD'),
    ('XorShift128',   gen_xorshift128,   'WEAK'),
    ('MINSTD',        gen_minstd,        'WEAK'),
    ('RANDU',         gen_randu,         'BAD'),
    ('LFSR-16',       gen_lfsr16,        'BAD'),
    ('Middle-Square', gen_middle_square,  'BAD'),
]

GEN_NAMES = [g[0] for g in GENERATORS]
GEN_CATEGORIES = {g[0]: g[2] for g in GENERATORS}
GEN_FUNCS = {g[0]: g[1] for g in GENERATORS}

CATEGORY_COLORS = {
    'CRYPTO': '#4CAF50',
    'GOOD':   '#2196F3',
    'WEAK':   '#FF9800',
    'BAD':    '#E91E63',
}


# =========================================================================
# STANDARD STATISTICAL FEATURES (for D2 comparison)
# =========================================================================

def compute_standard_features(data):
    """11 standard statistical features used in PRNG testing."""
    data_f = data.astype(np.float64)
    features = {}

    features['mean'] = float(np.mean(data_f))
    features['std'] = float(np.std(data_f))
    features['skewness'] = float(stats.skew(data_f))
    features['kurtosis'] = float(stats.kurtosis(data_f))

    # Chi-squared uniformity test statistic
    counts = np.bincount(data, minlength=256)
    expected = len(data) / 256.0
    features['chi2_uniformity'] = float(np.sum((counts - expected)**2 / expected))

    # Byte entropy
    p = counts[counts > 0] / len(data)
    features['entropy'] = float(-np.sum(p * np.log2(p)))

    # Autocorrelation at lags 1, 2, 5
    dm = data_f - np.mean(data_f)
    var = np.var(data_f)
    for lag in [1, 2, 5]:
        if var > 1e-15 and lag < len(data):
            features[f'autocorr_lag{lag}'] = float(np.mean(dm[:-lag] * dm[lag:]) / var)
        else:
            features[f'autocorr_lag{lag}'] = 0.0

    # Number of runs (above/below median)
    median = np.median(data_f)
    above = data_f >= median
    features['n_runs'] = float(np.sum(np.diff(above.astype(int)) != 0) + 1)

    # Unique byte pairs
    if len(data) >= 2:
        pairs = set(zip(data[:-1].tolist(), data[1:].tolist()))
        features['unique_byte_pairs'] = float(len(pairs))
    else:
        features['unique_byte_pairs'] = 0.0

    return features


STANDARD_NAMES = list(compute_standard_features(np.zeros(100, dtype=np.uint8)).keys())
N_STANDARD = len(STANDARD_NAMES)


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
BONF_FRAMEWORK = ALPHA / N_METRICS
BONF_STANDARD = ALPHA / N_STANDARD
del _analyzer, _dummy, _r, _mn

print(f"Standard: {N_STANDARD} features (Bonferroni α={BONF_STANDARD:.2e})")
print(f"Framework: {N_METRICS} metrics (Bonferroni α={BONF_FRAMEWORK:.2e})")


# =========================================================================
# UTILITIES
# =========================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def collect_framework(analyzer, data_arrays):
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def collect_standard(data_arrays):
    out = {f: [] for f in STANDARD_NAMES}
    for arr in data_arrays:
        feats = compute_standard_features(arr)
        for f, v in feats.items():
            if np.isfinite(v):
                out[f].append(v)
    return out


def count_sig(data_a, data_b, feature_names, alpha):
    sig = 0
    findings = []
    for f in feature_names:
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


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# DATA GENERATION
# =========================================================================

def generate_all(analyzer):
    """Generate N_TRIALS samples from each generator and collect metrics."""
    print("\nGenerating data for all generators...")

    # Reference: os.urandom (separate batch for self-check)
    print(f"  {'reference':15s} (os.urandom for comparison)...", end=" ", flush=True)
    ref_data = [gen_urandom(t, DATA_SIZE, seed=100 + t) for t in range(N_TRIALS)]
    ref_fw = collect_framework(analyzer, ref_data)
    ref_std = collect_standard(ref_data)
    print("done")

    # Self-check: second independent os.urandom batch
    print(f"  {'self-check':15s} (os.urandom vs os.urandom)...", end=" ", flush=True)
    selfcheck_data = [gen_urandom(t, DATA_SIZE, seed=200 + t) for t in range(N_TRIALS)]
    selfcheck_fw = collect_framework(analyzer, selfcheck_data)
    selfcheck_std = collect_standard(selfcheck_data)
    print("done")

    all_data = {
        '_reference_fw': ref_fw,
        '_reference_std': ref_std,
        '_selfcheck_fw': selfcheck_fw,
        '_selfcheck_std': selfcheck_std,
        '_raw': {},  # store raw arrays for D3
    }

    for name, gen_fn, cat in GENERATORS:
        print(f"  {name:15s} [{cat}]...", end=" ", flush=True)
        samples = [gen_fn(t, DATA_SIZE) for t in range(N_TRIALS)]
        all_data[name] = {
            'fw': collect_framework(analyzer, samples),
            'std': collect_standard(samples),
        }
        all_data['_raw'][name] = samples
        print("done")

    # Store reference raw for D3
    all_data['_raw']['_reference'] = ref_data

    return all_data


# =========================================================================
# D1: BASELINE DETECTION
# =========================================================================

def direction_1(all_data):
    print("\n" + "=" * 78)
    print("D1: BASELINE DETECTION — Each generator vs os.urandom reference")
    print("=" * 78)

    ref_fw = all_data['_reference_fw']

    # Self-check first
    sc_sig, sc_findings = count_sig(all_data['_selfcheck_fw'], ref_fw,
                                     METRIC_NAMES, BONF_FRAMEWORK)
    print(f"\n  SELF-CHECK (urandom vs urandom): {sc_sig}/{N_METRICS} sig", end="")
    print(f" {'PASS' if sc_sig <= 5 else 'WARN'}")

    d1 = {'_selfcheck': sc_sig}

    for name in GEN_NAMES:
        cat = GEN_CATEGORIES[name]
        n_sig, findings = count_sig(all_data[name]['fw'], ref_fw,
                                     METRIC_NAMES, BONF_FRAMEWORK)
        d1[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"  {name:15s} [{cat:6s}]: {n_sig:3d}/{N_METRICS} sig", end="")
        if n_sig == 0:
            print("  (indistinguishable)")
        else:
            print()
            for m, dval, p in findings[:3]:
                print(f"    {m:50s}  d={dval:+.2f}")

    return d1


# =========================================================================
# D2: STANDARD FEATURES VS GEOMETRIC METRICS
# =========================================================================

def direction_2(all_data):
    print("\n" + "=" * 78)
    print("D2: STANDARD FEATURES VS GEOMETRIC METRICS")
    print("=" * 78)
    print(f"  Standard: {N_STANDARD} features (α={BONF_STANDARD:.2e})")
    print(f"  Geometric: {N_METRICS} metrics (α={BONF_FRAMEWORK:.2e})")

    ref_fw = all_data['_reference_fw']
    ref_std = all_data['_reference_std']

    d2 = {}
    for name in GEN_NAMES:
        std_sig, std_findings = count_sig(all_data[name]['std'], ref_std,
                                           STANDARD_NAMES, BONF_STANDARD)
        geo_sig = all_data.get('_d1', {}).get(name, {}).get('n_sig', None)
        if geo_sig is None:
            geo_sig, _ = count_sig(all_data[name]['fw'], ref_fw,
                                    METRIC_NAMES, BONF_FRAMEWORK)

        d2[name] = {'std_sig': std_sig, 'geo_sig': geo_sig}

        cat = GEN_CATEGORIES[name]
        advantage = "geometric+" if geo_sig > std_sig else "equal" if geo_sig == std_sig else "standard+"
        print(f"  {name:15s} [{cat:6s}]: standard={std_sig:2d}/{N_STANDARD}  "
              f"geometric={geo_sig:3d}/{N_METRICS}  ({advantage})")
        if std_findings:
            for m, dval, p in std_findings[:2]:
                print(f"    std: {m:25s}  d={dval:+.2f}")

    return d2


# =========================================================================
# D3: DELAY EMBEDDING AMPLIFICATION
# =========================================================================

def direction_3(all_data, analyzer):
    print("\n" + "=" * 78)
    print("D3: DELAY EMBEDDING AMPLIFICATION — Does DE2 reveal hidden structure?")
    print("=" * 78)

    ref_raw = all_data['_raw']['_reference']
    ref_de2 = [delay_embed(arr, tau=2) for arr in ref_raw]
    ref_de2_fw = collect_framework(analyzer, ref_de2)

    d3 = {}
    for name in GEN_NAMES:
        raw_samples = all_data['_raw'][name]
        de2_samples = [delay_embed(arr, tau=2) for arr in raw_samples]
        de2_fw = collect_framework(analyzer, de2_samples)

        de2_sig, de2_findings = count_sig(de2_fw, ref_de2_fw,
                                           METRIC_NAMES, BONF_FRAMEWORK)
        raw_sig = all_data.get('_d1', {}).get(name, {}).get('n_sig', None)
        if raw_sig is None:
            raw_sig, _ = count_sig(all_data[name]['fw'],
                                    all_data['_reference_fw'],
                                    METRIC_NAMES, BONF_FRAMEWORK)

        d3[name] = {'raw_sig': raw_sig, 'de2_sig': de2_sig, 'de2_findings': de2_findings}

        cat = GEN_CATEGORIES[name]
        diff = de2_sig - raw_sig
        arrow = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "="
        print(f"  {name:15s} [{cat:6s}]: raw={raw_sig:3d}  DE2={de2_sig:3d}  ({arrow})")
        if de2_findings and de2_sig > raw_sig:
            for m, dval, p in de2_findings[:3]:
                print(f"    NEW: {m:50s}  d={dval:+.2f}")

    return d3


# =========================================================================
# D4: MINIMUM SEQUENCE LENGTH
# =========================================================================

def direction_4(all_data, analyzer, d1):
    print("\n" + "=" * 78)
    print("D4: MINIMUM SEQUENCE LENGTH FOR DETECTION")
    print("=" * 78)

    # Only test generators that were detectable in D1
    detectable = [name for name in GEN_NAMES if d1[name]['n_sig'] > 0]
    if not detectable:
        print("  No generators detected in D1 — skipping D4.")
        return {}

    print(f"  Testing: {', '.join(detectable)}")
    sizes = [500, 1000, 2000, 4000]

    d4 = {}
    for name in detectable:
        gen_fn = GEN_FUNCS[name]
        cat = GEN_CATEGORIES[name]
        print(f"\n  {name:15s} [{cat}]:")
        d4[name] = {}

        for sz in sizes:
            print(f"    N={sz:5d}...", end=" ", flush=True)

            # Fresh reference at this size
            ref_samples = [gen_urandom(t, sz, seed=300 + t) for t in range(N_TRIALS)]
            ref_fw = collect_framework(analyzer, ref_samples)

            gen_samples = [gen_fn(t, sz) for t in range(N_TRIALS)]
            gen_fw = collect_framework(analyzer, gen_samples)

            n_sig, _ = count_sig(gen_fw, ref_fw, METRIC_NAMES, BONF_FRAMEWORK)
            d4[name][sz] = n_sig
            print(f"{n_sig:3d} sig")

    return d4


# =========================================================================
# D5: GEOMETRIC TAXONOMY HEATMAP
# =========================================================================

def direction_5(all_data, d1):
    print("\n" + "=" * 78)
    print("D5: GEOMETRIC TAXONOMY — Which geometries catch which weaknesses?")
    print("=" * 78)

    geometry_groups = sorted(set(m.split(':')[0] for m in METRIC_NAMES))
    geo_to_metrics = {g: [m for m in METRIC_NAMES if m.startswith(g + ':')]
                      for g in geometry_groups}

    ref_fw = all_data['_reference_fw']

    # Build heatmap: geometry_group × generator
    d5 = {}
    for geo in geometry_groups:
        d5[geo] = {}
        metrics = geo_to_metrics[geo]
        bonf_geo = ALPHA / max(len(metrics), 1)
        for name in GEN_NAMES:
            n_sig, _ = count_sig(all_data[name]['fw'], ref_fw, metrics, bonf_geo)
            d5[geo][name] = n_sig

    # Print summary
    # Find geometries with any detection
    active_geos = [geo for geo in geometry_groups
                   if sum(d5[geo][n] for n in GEN_NAMES) > 0]

    print(f"\n  {'Geometry':<35s}", end="")
    for name in GEN_NAMES:
        print(f" {name[:8]:>8s}", end="")
    print()
    print(f"  {'─' * 35}", end="")
    for _ in GEN_NAMES:
        print(f" {'─' * 8}", end="")
    print()

    for geo in active_geos:
        print(f"  {geo:<35s}", end="")
        for name in GEN_NAMES:
            v = d5[geo][name]
            print(f" {v:>8d}" if v > 0 else f" {'·':>8s}", end="")
        print()

    return d5


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(d1, d2, d3, d4, d5):
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

    fig = plt.figure(figsize=(20, 24), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30,
                           height_ratios=[1.0, 1.0, 1.4])

    x = np.arange(len(GEN_NAMES))
    bar_colors = [CATEGORY_COLORS[GEN_CATEGORIES[n]] for n in GEN_NAMES]

    # ── D1: Detection bar chart (full width) ──
    ax = _dark_ax(fig.add_subplot(gs[0, :]))
    sig_vals = [d1[n]['n_sig'] if isinstance(d1[n], dict) else d1[n] for n in GEN_NAMES]
    bars = ax.bar(x, sig_vals, color=bar_colors, alpha=0.85, edgecolor='#333', linewidth=0.5)
    # Annotate self-check
    sc_val = d1['_selfcheck']
    ax.axhline(y=sc_val + 2, color='#666', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.annotate(f'self-check: {sc_val}', (0.02, 0.95), xycoords='axes fraction',
                fontsize=8, color='#888')
    ax.set_xticks(x)
    ax.set_xticklabels(GEN_NAMES, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Significant metrics vs os.urandom', fontsize=9)
    ax.set_title('D1: Baseline detection — each generator vs cryptographic reference',
                 fontsize=11, fontweight='bold')
    # Category legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_COLORS[c], label=c)
                       for c in ['CRYPTO', 'GOOD', 'WEAK', 'BAD']]
    ax.legend(handles=legend_elements, fontsize=8, facecolor='#333', edgecolor='#666',
              loc='upper left')

    # ── D2: Standard vs Geometric (left) ──
    ax = _dark_ax(fig.add_subplot(gs[1, 0]))
    std_vals = [d2[n]['std_sig'] for n in GEN_NAMES]
    geo_vals = [d2[n]['geo_sig'] for n in GEN_NAMES]
    ax.bar(x - 0.2, std_vals, 0.35, color='#FF9800', alpha=0.85,
           label=f'Standard ({N_STANDARD})')
    ax.bar(x + 0.2, geo_vals, 0.35, color='#2196F3', alpha=0.85,
           label=f'Geometric ({N_METRICS})')
    ax.set_xticks(x)
    ax.set_xticklabels(GEN_NAMES, fontsize=6, rotation=40, ha='right')
    ax.set_ylabel('Significant detections', fontsize=9)
    ax.set_title('D2: Standard features vs geometric metrics', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # ── D3: Raw vs DE2 (right) ──
    ax = _dark_ax(fig.add_subplot(gs[1, 1]))
    raw_vals = [d3[n]['raw_sig'] for n in GEN_NAMES]
    de2_vals = [d3[n]['de2_sig'] for n in GEN_NAMES]
    ax.bar(x - 0.2, raw_vals, 0.35, color='#4CAF50', alpha=0.85, label='Raw')
    ax.bar(x + 0.2, de2_vals, 0.35, color='#9C27B0', alpha=0.85, label='Delay embed τ=2')
    ax.set_xticks(x)
    ax.set_xticklabels(GEN_NAMES, fontsize=6, rotation=40, ha='right')
    ax.set_ylabel('Significant detections', fontsize=9)
    ax.set_title('D3: Raw vs delay-embedded detection', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # ── D4: Length sensitivity (bottom left) ──
    ax = _dark_ax(fig.add_subplot(gs[2, 0]))
    if d4:
        sizes = sorted(next(iter(d4.values())).keys())
        for name in d4:
            vals = [d4[name][sz] for sz in sizes]
            cat = GEN_CATEGORIES[name]
            ax.plot(sizes, vals, 'o-', color=CATEGORY_COLORS[cat],
                    label=name, linewidth=2, markersize=5)
        ax.set_xlabel('Sequence length N', fontsize=9)
        ax.set_ylabel('Significant metrics', fontsize=9)
        ax.legend(fontsize=7, facecolor='#333', edgecolor='#666')
    else:
        ax.text(0.5, 0.5, 'No generators detected in D1\n(all indistinguishable from random)',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='#666')
    ax.set_title('D4: Detection vs sequence length', fontsize=11, fontweight='bold')

    # ── D5: Geometry × Generator heatmap (bottom right) ──
    ax = _dark_ax(fig.add_subplot(gs[2, 1]))
    geometry_groups = sorted(d5.keys())
    # Filter to geometries with any detection
    active_geos = [g for g in geometry_groups if sum(d5[g][n] for n in GEN_NAMES) > 0]
    if active_geos:
        heatmap = np.array([[d5[g][n] for n in GEN_NAMES] for g in active_geos])
        im = ax.imshow(heatmap, aspect='auto', cmap='magma', interpolation='nearest')
        ax.set_xticks(range(len(GEN_NAMES)))
        ax.set_xticklabels(GEN_NAMES, fontsize=6, rotation=45, ha='right')
        ax.set_yticks(range(len(active_geos)))
        ax.set_yticklabels([g[:30] for g in active_geos], fontsize=6)
        # Annotate cells
        for i in range(len(active_geos)):
            for j in range(len(GEN_NAMES)):
                v = heatmap[i, j]
                if v > 0:
                    color = 'black' if v > heatmap.max() * 0.6 else 'white'
                    ax.text(j, i, str(int(v)), ha='center', va='center',
                            fontsize=6, color=color, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7, colors='#cccccc')
        cbar.set_label('Sig metrics', fontsize=8, color='white')
    else:
        ax.text(0.5, 0.5, 'No geometric detections',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='#666')
    ax.set_title('D5: Which geometries catch which weaknesses?',
                 fontsize=11, fontweight='bold')

    fig.suptitle('RNG Quality Testing via Exotic Geometry',
                 fontsize=14, fontweight='bold', color='white', y=0.995)

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'rng.png')
    fig.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='#181818')
    print(f"  Saved {outpath}")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    all_data = generate_all(analyzer)

    d1 = direction_1(all_data)

    # Stash D1 results for D2/D3 to reuse
    all_data['_d1'] = d1

    d2 = direction_2(all_data)
    d3 = direction_3(all_data, analyzer)
    d4 = direction_4(all_data, analyzer, d1)
    d5 = direction_5(all_data, d1)

    make_figure(d1, d2, d3, d4, d5)

    # ── VERDICT ──
    print(f"\n{'=' * 78}")
    print("VERDICT")
    print(f"{'=' * 78}")
    print(f"  Self-check (urandom vs urandom): {d1['_selfcheck']} sig")
    print()
    for name in GEN_NAMES:
        cat = GEN_CATEGORIES[name]
        d1_sig = d1[name]['n_sig']
        d2_std = d2[name]['std_sig']
        d2_geo = d2[name]['geo_sig']
        d3_raw = d3[name]['raw_sig']
        d3_de2 = d3[name]['de2_sig']
        print(f"  {name:15s} [{cat:6s}]: D1={d1_sig:3d}  "
              f"std={d2_std:2d} geo={d2_geo:3d}  "
              f"raw={d3_raw:3d} DE2={d3_de2:3d}")

    # Summary statistics
    n_crypto_good = sum(1 for n in GEN_NAMES
                        if GEN_CATEGORIES[n] in ('CRYPTO', 'GOOD') and d1[n]['n_sig'] == 0)
    n_detected = sum(1 for n in GEN_NAMES if d1[n]['n_sig'] > 0)
    n_weak_bad = sum(1 for n in GEN_NAMES if GEN_CATEGORIES[n] in ('WEAK', 'BAD'))
    n_weak_bad_detected = sum(1 for n in GEN_NAMES
                              if GEN_CATEGORIES[n] in ('WEAK', 'BAD') and d1[n]['n_sig'] > 0)
    geo_advantage = sum(1 for n in GEN_NAMES if d2[n]['geo_sig'] > d2[n]['std_sig'])
    de2_amplified = sum(1 for n in GEN_NAMES if d3[n]['de2_sig'] > d3[n]['raw_sig'])

    print(f"\n  CRYPTO/GOOD undetected (correct): {n_crypto_good}/5")
    print(f"  WEAK/BAD detected: {n_weak_bad_detected}/{n_weak_bad}")
    print(f"  Geometric > standard: {geo_advantage}/{len(GEN_NAMES)} generators")
    print(f"  DE2 amplified detection: {de2_amplified}/{len(GEN_NAMES)} generators")
