#!/usr/bin/env python3
"""
Investigation: Continued Fraction Geometry — Does Exotic Geometry Detect
Sequential Structure Beyond the Gauss-Kuzmin Distribution?

number_theory_deep.py found cf_pi=101, cf_sqrt2=95 sig vs random, and
√2 vs π = 105 sig pairwise. But those were basic detection tests. The key
question: does geometry detect *sequential* structure in CF coefficients
beyond what the Gauss-Kuzmin marginal distribution P(a_n=k) = log₂((k+1)²/(k(k+2)))
predicts?

5 constants:
  √2  — Period-1: [1; 2, 2, 2, ...]  (algebraic, trivially non-GK)
  √3  — Period-2: [1; 1, 2, 1, 2, ...] (simplest non-trivial periodic)
  e   — Patterned: [2; 1, 2k, 1, ...] (deterministic but not periodic)
  π   — Chaotic-looking (the "does it look GK?" test)
  ln2 — Chaotic-looking (second transcendental for robustness)

Five directions:
  D1: Detection baseline — each constant's CF vs uniform random uint8
  D2: Gauss-Kuzmin surrogates — the KEY test. If π or ln2 differ from iid GK,
      geometry detects sequential correlation beyond the marginal distribution.
  D3: Pairwise discrimination — all 10 pairs among 5 constants (free from D1)
  D4: Ordering dependence — original vs shuffled for each constant
  D5: Delay embedding — delay_embed at τ=1,2,3,5,8, compare vs embedded random

Budget: ~1075 analyzer calls at DATA_SIZE=200 (~0.2s/call) ≈ 4 min total.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, delay_embed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 200
TRIAL_STRIDE = 50
ALPHA = 0.05

# --- Discover metric names ---
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF_ALPHA = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn

print(f"1D metrics: {N_METRICS}, Bonferroni α={BONF_ALPHA:.2e}")


# =========================================================================
# PRECOMPUTATION — CF coefficients for all 5 constants
# =========================================================================

def compute_cf_coefficients(value_str, n_coeffs):
    """Compute continued fraction coefficients using mpmath for high precision."""
    try:
        from mpmath import mp, mpf, floor as mpfloor
        mp.dps = max(n_coeffs * 2, 1000)
        if value_str == 'sqrt2':
            x = mp.sqrt(2)
        elif value_str == 'sqrt3':
            x = mp.sqrt(3)
        elif value_str == 'e':
            x = mp.e
        elif value_str == 'pi':
            x = mp.pi
        elif value_str == 'ln2':
            x = mp.ln(2)
        else:
            raise ValueError(f"Unknown constant: {value_str}")

        coeffs = []
        for _ in range(n_coeffs):
            a = int(mpfloor(x))
            coeffs.append(a)
            frac = x - a
            if frac < mpf('1e-50'):
                break
            x = 1 / frac
        return coeffs
    except ImportError:
        # Fallback: known patterns for algebraic constants
        if value_str == 'sqrt2':
            return [1] + [2] * (n_coeffs - 1)
        elif value_str == 'sqrt3':
            return [1] + [1, 2] * ((n_coeffs - 1) // 2 + 1)
        elif value_str == 'e':
            coeffs = [2]
            k = 1
            while len(coeffs) < n_coeffs:
                coeffs.extend([1, 2 * k, 1])
                k += 1
            return coeffs[:n_coeffs]
        elif value_str == 'pi':
            pi_cf = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2,
                     1, 84, 2, 1, 1, 15, 3, 13, 1, 4, 2, 6, 6, 99, 1, 2, 2, 6, 3, 5,
                     1, 1, 6, 8, 1, 7, 1, 2, 3, 7]
            if n_coeffs > len(pi_cf):
                print(f"  Warning: Only {len(pi_cf)} π CF coeffs without mpmath")
            return pi_cf[:n_coeffs]
        elif value_str == 'ln2':
            ln2_cf = [0, 1, 2, 3, 1, 6, 3, 1, 1, 2, 1, 1, 1, 1, 3, 10, 1, 1, 1, 2,
                      1, 1, 3, 2, 4, 2, 1, 3, 2, 1]
            if n_coeffs > len(ln2_cf):
                print(f"  Warning: Only {len(ln2_cf)} ln2 CF coeffs without mpmath")
            return ln2_cf[:n_coeffs]
        raise


CF_N = 3000  # enough for 25 trials with stride 50 at size 200
CONSTANTS = ['sqrt2', 'sqrt3', 'e', 'pi', 'ln2']
CONST_LABELS = {'sqrt2': '√2', 'sqrt3': '√3', 'e': 'e', 'pi': 'π', 'ln2': 'ln 2'}

print("Precomputing CF coefficients...")
CF_DATA = {}
for name in CONSTANTS:
    print(f"  {CONST_LABELS[name]}...", end=" ", flush=True)
    CF_DATA[name] = compute_cf_coefficients(name, CF_N)
    print(f"{len(CF_DATA[name])} coefficients")

# Spot checks
assert CF_DATA['sqrt2'][0] == 1 and CF_DATA['sqrt2'][1] == 2
assert CF_DATA['sqrt3'][0] == 1 and CF_DATA['sqrt3'][1] == 1 and CF_DATA['sqrt3'][2] == 2
assert CF_DATA['e'][0] == 2 and CF_DATA['e'][1] == 1 and CF_DATA['e'][2] == 2
assert CF_DATA['pi'][0] == 3 and CF_DATA['pi'][1] == 7
assert CF_DATA['ln2'][0] == 0 and CF_DATA['ln2'][1] == 1
print("  Spot checks passed")


# =========================================================================
# ENCODING GENERATORS
# =========================================================================

def gen_cf(name, trial, size=DATA_SIZE):
    """Generate CF coefficient array for a constant, clipped to uint8."""
    start = trial * TRIAL_STRIDE
    coeffs = CF_DATA[name]
    end = min(start + size, len(coeffs))
    chunk = coeffs[start:end]
    if len(chunk) < size:
        # Wrap around if not enough coefficients
        n_extra = size - len(chunk)
        chunk = chunk + coeffs[:n_extra]
    return np.clip(chunk, 0, 255).astype(np.uint8)


def gen_random(trial, size=DATA_SIZE):
    """Uniform random uint8 bytes."""
    return np.random.default_rng(9000 + trial).integers(0, 256, size, dtype=np.uint8)


def gen_gauss_kuzmin(trial, size=DATA_SIZE):
    """IID samples from the Gauss-Kuzmin distribution, clipped to uint8.
    P(a_n = k) = log_2((k+1)^2 / (k*(k+2))) for k = 1, 2, 3, ...
    """
    rng = np.random.default_rng(7000 + trial)
    # Precompute CDF up to k=255
    max_k = 255
    probs = np.zeros(max_k)
    for k in range(1, max_k + 1):
        probs[k - 1] = np.log2((k + 1) ** 2 / (k * (k + 2)))
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]  # normalize (tail probability negligible)

    u = rng.random(size)
    samples = np.searchsorted(cdf, u) + 1  # k starts at 1
    return np.clip(samples, 0, 255).astype(np.uint8)


# =========================================================================
# ANALYSIS UTILITIES
# =========================================================================

def collect_metrics(analyzer, data_arrays):
    """Run analyzer on list of arrays, collect metrics into dict of lists."""
    out = {m: [] for m in METRIC_NAMES}
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if key in out and np.isfinite(mv):
                    out[key].append(mv)
    return out


def cohens_d(a, b):
    """Compute Cohen's d with degeneracy guard."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def compare(data_a, data_b):
    """Count significant metrics between two metric dicts."""
    sig = 0
    findings = []
    for m in METRIC_NAMES:
        a = np.array(data_a.get(m, []))
        b = np.array(data_b.get(m, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        if not np.isfinite(d):
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if not np.isfinite(p):
            continue
        if p < BONF_ALPHA and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax


# =========================================================================
# D1: Detection Baseline — each constant vs uniform random
# =========================================================================

def direction_1(analyzer):
    print("\n" + "=" * 78)
    print("D1: DETECTION BASELINE — EACH CONSTANT VS RANDOM")
    print("=" * 78)
    print(f"  {len(CONSTANTS)} constants × {N_TRIALS} trials + {N_TRIALS} random = "
          f"{(len(CONSTANTS) + 1) * N_TRIALS} calls")

    # Collect random baseline
    print("  random...", end=" ", flush=True)
    random_arrays = [gen_random(t) for t in range(N_TRIALS)]
    random_data = collect_metrics(analyzer, random_arrays)
    print("done")

    all_data = {}
    d1_results = {}

    for name in CONSTANTS:
        label = CONST_LABELS[name]
        print(f"  {label:5s}...", end=" ", flush=True)
        arrays = [gen_cf(name, t) for t in range(N_TRIALS)]

        # Quick stats
        sample = arrays[0]
        n_distinct = len(np.unique(sample))
        ent_vals = np.bincount(sample, minlength=256)
        ent_p = ent_vals[ent_vals > 0] / len(sample)
        entropy = float(-np.sum(ent_p * np.log2(ent_p)))

        data = collect_metrics(analyzer, arrays)
        all_data[name] = data

        n_sig, findings = compare(data, random_data)
        d1_results[name] = n_sig
        print(f"{n_sig:3d} sig  (H={entropy:.2f} bits, {n_distinct} distinct)")
        for m, d, p in findings[:3]:
            print(f"       {m:45s}  d={d:+8.2f}")

    return all_data, random_data, d1_results


# =========================================================================
# D2: Gauss-Kuzmin Surrogates — THE KEY DIRECTION
# =========================================================================

def direction_2(analyzer, all_data):
    print("\n" + "=" * 78)
    print("D2: GAUSS-KUZMIN SURROGATES — SEQUENTIAL STRUCTURE TEST")
    print("=" * 78)
    print("  For 'almost all' reals, CF coefficients are iid Gauss-Kuzmin.")
    print("  If a constant differs from GK surrogates, geometry detects")
    print("  sequential correlation beyond the marginal distribution.")
    print(f"  {N_TRIALS} GK surrogate trials")

    # Collect GK surrogates
    print("  GK surrogates...", end=" ", flush=True)
    gk_arrays = [gen_gauss_kuzmin(t) for t in range(N_TRIALS)]
    gk_data = collect_metrics(analyzer, gk_arrays)
    print("done")

    d2_results = {}
    for name in CONSTANTS:
        label = CONST_LABELS[name]
        n_sig, findings = compare(all_data[name], gk_data)
        d2_results[name] = n_sig
        print(f"  {label:5s} vs GK: {n_sig:3d} sig", end="")
        if findings:
            print(f"   top: {findings[0][0]:35s} d={findings[0][1]:+.2f}", end="")
        print()

    # Khinchin statistics
    print(f"\n  Khinchin's constant K ≈ 2.6854...")
    print(f"  Geometric mean of CF coefficients (excluding a₀):")
    for name in CONSTANTS:
        coeffs = CF_DATA[name][1:min(len(CF_DATA[name]), 2000)]  # skip a₀
        arr = np.array(coeffs, dtype=np.float64)
        arr_pos = arr[arr > 0]
        if len(arr_pos) > 0:
            geo_mean = np.exp(np.mean(np.log(arr_pos)))
        else:
            geo_mean = 0.0
        print(f"    {CONST_LABELS[name]:5s}: geo_mean = {geo_mean:.4f}"
              f"  (mean = {np.mean(arr):.1f}, median = {np.median(arr):.0f})")

    return gk_data, d2_results


# =========================================================================
# D3: Pairwise Discrimination — all 10 pairs (free from D1)
# =========================================================================

def direction_3(all_data):
    print("\n" + "=" * 78)
    print("D3: PAIRWISE DISCRIMINATION (0 extra calls)")
    print("=" * 78)
    print("  All 10 pairwise comparisons among 5 constants.")

    d3_matrix = {}
    for i, n1 in enumerate(CONSTANTS):
        for j, n2 in enumerate(CONSTANTS):
            if j > i:
                n_sig, findings = compare(all_data[n1], all_data[n2])
                d3_matrix[(n1, n2)] = n_sig
                top_str = ""
                if findings:
                    top_str = f"  top: {findings[0][0]:30s} d={findings[0][1]:+.2f}"
                print(f"  {CONST_LABELS[n1]:5s} vs {CONST_LABELS[n2]:5s}: "
                      f"{n_sig:3d} sig{top_str}")

    # Key questions
    print(f"\n  Key discriminations:")
    for pair, label in [
        (('sqrt2', 'sqrt3'), "algebraic: √2 vs √3"),
        (('pi', 'ln2'), "transcendental: π vs ln 2"),
        (('sqrt2', 'pi'), "algebraic vs transcendental: √2 vs π"),
        (('e', 'pi'), "patterned vs chaotic: e vs π"),
    ]:
        n_sig = d3_matrix.get(pair, d3_matrix.get((pair[1], pair[0]), 0))
        print(f"    {label:45s} → {n_sig:3d} sig")

    return d3_matrix


# =========================================================================
# D4: Ordering Dependence — original vs shuffled
# =========================================================================

def direction_4(analyzer, all_data):
    print("\n" + "=" * 78)
    print("D4: ORDERING DEPENDENCE — ORIGINAL VS SHUFFLED")
    print("=" * 78)
    print(f"  {len(CONSTANTS)} constants × {N_TRIALS} shuffled + {N_TRIALS} random self-check")
    print("  √2 = [2,2,...]: shuffled ≡ original → expect sig ≈ 0")
    print("  √3 = [1,2,1,2,...]: shuffling destroys period → expect sig > 0")

    d4_results = {}
    for name in CONSTANTS:
        label = CONST_LABELS[name]
        print(f"  {label:5s}...", end=" ", flush=True)
        # Generate shuffled versions
        shuf_arrays = []
        for t in range(N_TRIALS):
            arr = gen_cf(name, t)
            s = arr.copy()
            np.random.default_rng(1000 + t).shuffle(s)
            shuf_arrays.append(s)
        shuf_data = collect_metrics(analyzer, shuf_arrays)

        n_sig, findings = compare(all_data[name], shuf_data)
        d4_results[name] = n_sig
        note = ""
        if name == 'sqrt2':
            note = " (control: constant sequence)"
        elif name == 'sqrt3':
            note = " (period-2 destroyed)"
        print(f"{n_sig:3d} sig{note}")
        for m, d, p in findings[:2]:
            print(f"       {m:45s}  d={d:+8.2f}")

    # Random self-check
    print(f"  random self-check...", end=" ", flush=True)
    shuf_random = []
    for t in range(N_TRIALS):
        arr = gen_random(t)
        s = arr.copy()
        np.random.default_rng(2000 + t).shuffle(s)
        shuf_random.append(s)
    rand_orig = [gen_random(t) for t in range(N_TRIALS)]
    rand_orig_data = collect_metrics(analyzer, rand_orig)
    shuf_rand_data = collect_metrics(analyzer, shuf_random)
    n_sig_rand, _ = compare(rand_orig_data, shuf_rand_data)
    d4_results['random'] = n_sig_rand
    print(f"{n_sig_rand:3d} sig (expect ≈ 0)")

    return d4_results


# =========================================================================
# D5: Delay Embedding — amplify sequential correlations
# =========================================================================

def direction_5(analyzer):
    print("\n" + "=" * 78)
    print("D5: DELAY EMBEDDING — CF × τ")
    print("=" * 78)
    taus = [1, 2, 3, 5, 8]
    print(f"  τ = {taus}, {len(CONSTANTS)} constants + random × {N_TRIALS} trials × {len(taus)} τ")

    d5_results = {}  # d5_results[name][tau] = n_sig
    extra_size = DATA_SIZE + max(taus) + 10

    for name in CONSTANTS:
        label = CONST_LABELS[name]
        d5_results[name] = {}
        print(f"  {label:5s}:", end="", flush=True)
        for tau in taus:
            # Embedded constant vs embedded random
            const_emb_arrays = []
            rand_emb_arrays = []
            for t in range(N_TRIALS):
                raw_const = gen_cf(name, t, size=extra_size)
                raw_rand = gen_random(t, size=extra_size)
                emb_const = delay_embed(raw_const, tau)[:DATA_SIZE]
                emb_rand = delay_embed(raw_rand, tau)[:DATA_SIZE]
                const_emb_arrays.append(emb_const)
                rand_emb_arrays.append(emb_rand)

            const_emb_data = collect_metrics(analyzer, const_emb_arrays)
            rand_emb_data = collect_metrics(analyzer, rand_emb_arrays)
            n_sig, _ = compare(const_emb_data, rand_emb_data)
            d5_results[name][tau] = n_sig
            print(f"  τ={tau}→{n_sig}", end="", flush=True)
        print()

    return d5_results, taus


# =========================================================================
# FIGURE: 3×2 grid
# =========================================================================

def make_figure(d1_results, d2_results, d3_matrix, d4_results, d5_results, taus):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'

    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'axes.edgecolor': '#444444',
        'axes.labelcolor': FG,
        'text.color': FG,
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

    colors = {'sqrt2': '#E91E63', 'sqrt3': '#FF9800', 'e': '#4CAF50',
              'pi': '#2196F3', 'ln2': '#9C27B0'}

    # ── (0,0) D1: Detection bar chart ──
    ax = _dark_ax(fig.add_subplot(gs[0, 0]))
    names = CONSTANTS
    labels = [CONST_LABELS[n] for n in names]
    sigs = [d1_results[n] for n in names]
    bars = ax.bar(range(len(names)), sigs,
                  color=[colors[n] for n in names], alpha=0.85, edgecolor='#333')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Sig metrics vs random', fontsize=9, color=FG)
    ax.set_title('D1: Detection Baseline — CF vs Random', fontsize=11,
                 fontweight='bold', color=FG)
    for i, v in enumerate(sigs):
        ax.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9, fontweight='bold')

    # ── (0,1) D2: Real vs GK surrogate ──
    ax = _dark_ax(fig.add_subplot(gs[0, 1]))
    gk_sigs = [d2_results[n] for n in names]
    bars = ax.bar(range(len(names)), gk_sigs,
                  color=[colors[n] for n in names], alpha=0.85, edgecolor='#333')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Sig metrics vs Gauss-Kuzmin', fontsize=9, color=FG)
    ax.set_title('D2: Beyond Gauss-Kuzmin — Sequential Structure',
                 fontsize=11, fontweight='bold', color=FG)
    for i, v in enumerate(gk_sigs):
        ax.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9, fontweight='bold')
    # Add annotation for key result
    pi_sig = d2_results['pi']
    ln2_sig = d2_results['ln2']
    ax.text(0.98, 0.95,
            f"π vs GK: {pi_sig}\nln2 vs GK: {ln2_sig}",
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            color='#aaa', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    # ── (1,0) D3: Pairwise heatmap ──
    ax = _dark_ax(fig.add_subplot(gs[1, 0]))
    n = len(names)
    mat = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j > i:
                key = (n1, n2)
                val = d3_matrix.get(key, d3_matrix.get((n2, n1), 0))
                mat[i, j] = val
                mat[j, i] = val
    im = ax.imshow(mat, cmap='magma', interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{int(mat[i, j])}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if mat[i, j] > mat.max() / 2 else '#aaa')
    ax.set_title('D3: Pairwise Discrimination', fontsize=11,
                 fontweight='bold', color=FG)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── (1,1) D4: Ordering dependence ──
    ax = _dark_ax(fig.add_subplot(gs[1, 1]))
    d4_names = CONSTANTS + ['random']
    d4_labels = [CONST_LABELS.get(n, n) for n in d4_names]
    d4_sigs = [d4_results[n] for n in d4_names]
    d4_colors = [colors.get(n, '#666666') for n in d4_names]
    ax.bar(range(len(d4_names)), d4_sigs,
           color=d4_colors, alpha=0.85, edgecolor='#333')
    ax.set_xticks(range(len(d4_names)))
    ax.set_xticklabels(d4_labels, fontsize=10)
    ax.set_ylabel('Sig metrics (orig vs shuffled)', fontsize=9, color=FG)
    ax.set_title('D4: Ordering Dependence', fontsize=11,
                 fontweight='bold', color=FG)
    for i, v in enumerate(d4_sigs):
        ax.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=9, fontweight='bold')

    # ── (2,0) D5: Sig vs τ line plot ──
    ax = _dark_ax(fig.add_subplot(gs[2, :]))
    for name in CONSTANTS:
        label = CONST_LABELS[name]
        sig_vals = [d5_results[name][tau] for tau in taus]
        ax.plot(range(len(taus)), sig_vals, 'o-', color=colors[name],
                linewidth=2, markersize=6, alpha=0.85, label=label)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([str(t) for t in taus], fontsize=9, color=FG)
    ax.set_xlabel('Delay τ', fontsize=9, color=FG)
    ax.set_ylabel('Sig metrics (embedded CF vs embedded random)', fontsize=8, color=FG)
    ax.set_title('D5: Delay Embedding', fontsize=11, fontweight='bold', color=FG)
    ax.legend(fontsize=8, facecolor='#222', edgecolor='#444', labelcolor=FG,
              loc='best')

    fig.suptitle('Continued Fraction Geometry: Sequential Structure Beyond Gauss-Kuzmin',
                 fontsize=14, fontweight='bold', color=FG, y=0.995)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'continued_fractions.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved continued_fractions.png")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()

    # D1: Detection baseline (collects all_data for reuse)
    all_data, random_data, d1_results = direction_1(analyzer)

    # D2: Gauss-Kuzmin surrogates — the key test
    gk_data, d2_results = direction_2(analyzer, all_data)

    # D3: Pairwise discrimination (free from D1)
    d3_matrix = direction_3(all_data)

    # D4: Ordering dependence
    d4_results = direction_4(analyzer, all_data)

    # D5: Delay embedding
    d5_results, taus = direction_5(analyzer)

    # Figure
    make_figure(d1_results, d2_results, d3_matrix, d4_results, d5_results, taus)

    # Final summary
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    for name in CONSTANTS:
        label = CONST_LABELS[name]
        d1 = d1_results[name]
        d2 = d2_results[name]
        d4 = d4_results[name]
        peak_tau = max(taus, key=lambda t: d5_results[name][t])
        d5_peak = d5_results[name][peak_tau]
        print(f"  {label:5s}: D1={d1:3d} vs rand, D2={d2:3d} vs GK, "
              f"D4={d4:3d} ordering, D5={d5_peak:3d} (τ={peak_tau})")
    print(f"  rand:  D4={d4_results['random']:3d} self-check")
