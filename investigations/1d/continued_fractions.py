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
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from exotic_geometry_framework import delay_embed
from tools.investigation_runner import Runner

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000  # CF coefficients are expensive; 2000 is 10× the original
N_WORKERS = 8


# =========================================================================
# PRECOMPUTATION — CF coefficients for all 5 constants
# =========================================================================

def compute_cf_coefficients(value_str, n_coeffs):
    """Compute continued fraction coefficients.

    Algebraic constants (√2, √3) and e have known CF patterns — generated
    directly without mpmath. Transcendentals (π, ln2) require mpmath with
    high precision. The dps requirement is ~1.5× n_coeffs (each CF step
    consumes roughly log10(a_n) digits; mean a_n under Gauss-Kuzmin is ~3.4,
    so ~0.5 digits/step plus safety margin).
    """
    # Algebraic / patterned constants — exact, no mpmath needed
    if value_str == 'sqrt2':
        return [1] + [2] * (n_coeffs - 1)
    elif value_str == 'sqrt3':
        return ([1] + [1, 2] * ((n_coeffs - 1) // 2 + 1))[:n_coeffs]
    elif value_str == 'e':
        coeffs = [2]
        k = 1
        while len(coeffs) < n_coeffs:
            coeffs.extend([1, 2 * k, 1])
            k += 1
        return coeffs[:n_coeffs]

    # Transcendentals — need mpmath
    from mpmath import mp, mpf, floor as mpfloor
    mp.dps = max(int(n_coeffs * 1.5) + 500, 1000)
    if value_str == 'pi':
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


# Need enough coefficients for N_TRIALS non-overlapping windows of DATA_SIZE
CF_N = N_TRIALS * DATA_SIZE + 1000
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
# GENERATORS
# =========================================================================

def make_cf_gen(name):
    """Return a generator function (rng, size) -> uint8 for a CF constant."""
    coeffs = CF_DATA[name]
    max_start = max(1, len(coeffs) - DATA_SIZE)
    def gen(rng, size):
        start = int(rng.integers(0, max_start))
        end = min(start + size, len(coeffs))
        chunk = coeffs[start:end]
        if len(chunk) < size:
            chunk = chunk + coeffs[:size - len(chunk)]
        return np.clip(chunk, 0, 255).astype(np.uint8)
    return gen


def gen_random(rng, size):
    return rng.integers(0, 256, size, dtype=np.uint8)


def gen_gauss_kuzmin(rng, size):
    """IID samples from the Gauss-Kuzmin distribution, clipped to uint8.
    P(a_n = k) = log_2((k+1)^2 / (k*(k+2))) for k = 1, 2, 3, ...
    """
    max_k = 255
    probs = np.zeros(max_k)
    for k in range(1, max_k + 1):
        probs[k - 1] = np.log2((k + 1) ** 2 / (k * (k + 2)))
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]
    u = rng.random(size)
    samples = np.searchsorted(cdf, u) + 1
    return np.clip(samples, 0, 255).astype(np.uint8)


# =========================================================================
# D1: Detection Baseline — each constant vs uniform random
# =========================================================================

def direction_1(runner):
    print("\n" + "=" * 78)
    print("D1: DETECTION BASELINE — EACH CONSTANT VS RANDOM")
    print("=" * 78)

    # Collect random baseline
    print("  random...", end=" ", flush=True)
    random_chunks = [gen_random(rng, DATA_SIZE) for rng in runner.trial_rngs()]
    random_data = runner.collect(random_chunks)
    print("done")

    all_data = {}
    d1_results = {}

    for name in CONSTANTS:
        label = CONST_LABELS[name]
        print(f"  {label:5s}...", end=" ", flush=True)
        gen = make_cf_gen(name)
        chunks = [gen(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=100)]

        # Quick stats on first chunk
        sample = chunks[0]
        n_distinct = len(np.unique(sample))
        ent_vals = np.bincount(sample, minlength=256)
        ent_p = ent_vals[ent_vals > 0] / len(sample)
        entropy = float(-np.sum(ent_p * np.log2(ent_p)))

        data = runner.collect(chunks)
        all_data[name] = data

        n_sig, findings = runner.compare(data, random_data)
        d1_results[name] = n_sig
        print(f"{n_sig:3d} sig  (H={entropy:.2f} bits, {n_distinct} distinct)")
        for m, d, p in findings[:3]:
            print(f"       {m:45s}  d={d:+8.2f}")

    return all_data, random_data, d1_results


# =========================================================================
# D2: Gauss-Kuzmin Surrogates — THE KEY DIRECTION
# =========================================================================

def direction_2(runner, all_data):
    print("\n" + "=" * 78)
    print("D2: GAUSS-KUZMIN SURROGATES — SEQUENTIAL STRUCTURE TEST")
    print("=" * 78)
    print("  For 'almost all' reals, CF coefficients are iid Gauss-Kuzmin.")
    print("  If a constant differs from GK surrogates, geometry detects")
    print("  sequential correlation beyond the marginal distribution.")
    print(f"  {N_TRIALS} GK surrogate trials")

    print("  GK surrogates...", end=" ", flush=True)
    gk_chunks = [gen_gauss_kuzmin(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=200)]
    gk_data = runner.collect(gk_chunks)
    print("done")

    d2_results = {}
    for name in CONSTANTS:
        label = CONST_LABELS[name]
        n_sig, findings = runner.compare(all_data[name], gk_data)
        d2_results[name] = n_sig
        print(f"  {label:5s} vs GK: {n_sig:3d} sig", end="")
        if findings:
            print(f"   top: {findings[0][0]:35s} d={findings[0][1]:+.2f}", end="")
        print()

    # Khinchin statistics
    print(f"\n  Khinchin's constant K ≈ 2.6854...")
    print(f"  Geometric mean of CF coefficients (excluding a₀):")
    for name in CONSTANTS:
        coeffs = CF_DATA[name][1:min(len(CF_DATA[name]), 2000)]
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

def direction_3(runner, all_data):
    print("\n" + "=" * 78)
    print("D3: PAIRWISE DISCRIMINATION (0 extra calls)")
    print("=" * 78)
    print("  All 10 pairwise comparisons among 5 constants.")

    d3_matrix = {}
    for i, n1 in enumerate(CONSTANTS):
        for j, n2 in enumerate(CONSTANTS):
            if j > i:
                n_sig, findings = runner.compare(all_data[n1], all_data[n2])
                d3_matrix[(n1, n2)] = n_sig
                top_str = ""
                if findings:
                    top_str = f"  top: {findings[0][0]:30s} d={findings[0][1]:+.2f}"
                print(f"  {CONST_LABELS[n1]:5s} vs {CONST_LABELS[n2]:5s}: "
                      f"{n_sig:3d} sig{top_str}")

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

def direction_4(runner, all_data):
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
        gen = make_cf_gen(name)
        rngs = runner.trial_rngs(offset=100)
        orig_chunks = [gen(rng, DATA_SIZE) for rng in rngs]
        shuf_chunks = []
        for i, chunk in enumerate(orig_chunks):
            s = chunk.copy()
            np.random.default_rng(1000 + i).shuffle(s)
            shuf_chunks.append(s)
        shuf_data = runner.collect(shuf_chunks)

        n_sig, findings = runner.compare(all_data[name], shuf_data)
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
    rand_rngs = runner.trial_rngs(offset=300)
    rand_orig = [gen_random(rng, DATA_SIZE) for rng in rand_rngs]
    rand_shuf = []
    for i, chunk in enumerate(rand_orig):
        s = chunk.copy()
        np.random.default_rng(2000 + i).shuffle(s)
        rand_shuf.append(s)
    rand_orig_data = runner.collect(rand_orig)
    rand_shuf_data = runner.collect(rand_shuf)
    n_sig_rand, _ = runner.compare(rand_orig_data, rand_shuf_data)
    d4_results['random'] = n_sig_rand
    print(f"{n_sig_rand:3d} sig (expect ≈ 0)")

    return d4_results


# =========================================================================
# D5: Delay Embedding — amplify sequential correlations
# =========================================================================

def direction_5(runner):
    print("\n" + "=" * 78)
    print("D5: DELAY EMBEDDING — CF × τ")
    print("=" * 78)
    taus = [1, 2, 3, 5, 8]
    print(f"  τ = {taus}, {len(CONSTANTS)} constants + random × {N_TRIALS} trials × {len(taus)} τ")

    d5_results = {}
    extra_size = DATA_SIZE + max(taus) + 10

    for name in CONSTANTS:
        label = CONST_LABELS[name]
        gen = make_cf_gen(name)
        d5_results[name] = {}
        print(f"  {label:5s}:", end="", flush=True)
        for tau in taus:
            const_rngs = runner.trial_rngs(offset=400)
            rand_rngs = runner.trial_rngs(offset=500)
            const_emb = [delay_embed(gen(rng, extra_size), tau)[:DATA_SIZE]
                         for rng in const_rngs]
            rand_emb = [delay_embed(gen_random(rng, extra_size), tau)[:DATA_SIZE]
                        for rng in rand_rngs]
            const_data = runner.collect(const_emb)
            rand_data = runner.collect(rand_emb)
            n_sig, _ = runner.compare(const_data, rand_data)
            d5_results[name][tau] = n_sig
            print(f"  τ={tau}→{n_sig}", end="", flush=True)
        print()

    return d5_results, taus


# =========================================================================
# FIGURE: 3×2 grid (D5 spans bottom row)
# =========================================================================

def make_figure(runner, d1_results, d2_results, d3_matrix, d4_results, d5_results, taus):
    print("\nGenerating figure...", flush=True)

    runner._apply_dark_theme()
    BG = '#181818'
    FG = '#e0e0e0'

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

    colors = {'sqrt2': '#E91E63', 'sqrt3': '#FF9800', 'e': '#4CAF50',
              'pi': '#2196F3', 'ln2': '#9C27B0'}

    names = CONSTANTS
    labels = [CONST_LABELS[n] for n in names]

    # (0,0) D1: Detection bar chart
    ax = runner.dark_ax(fig.add_subplot(gs[0, 0]))
    sigs = [d1_results[n] for n in names]
    ax.bar(range(len(names)), sigs,
           color=[colors[n] for n in names], alpha=0.85, edgecolor='#333')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Sig metrics vs random', fontsize=9)
    ax.set_title('D1: Detection Baseline — CF vs Random', fontsize=11, fontweight='bold')
    for i, v in enumerate(sigs):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=9, fontweight='bold', color=FG)

    # (0,1) D2: Real vs GK surrogate
    ax = runner.dark_ax(fig.add_subplot(gs[0, 1]))
    gk_sigs = [d2_results[n] for n in names]
    ax.bar(range(len(names)), gk_sigs,
           color=[colors[n] for n in names], alpha=0.85, edgecolor='#333')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Sig metrics vs Gauss-Kuzmin', fontsize=9)
    ax.set_title('D2: Beyond Gauss-Kuzmin — Sequential Structure', fontsize=11, fontweight='bold')
    for i, v in enumerate(gk_sigs):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=9, fontweight='bold', color=FG)
    ax.text(0.98, 0.95,
            f"π vs GK: {d2_results['pi']}\nln2 vs GK: {d2_results['ln2']}",
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            color='#aaa', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    # (1,0) D3: Pairwise heatmap
    ax = runner.dark_ax(fig.add_subplot(gs[1, 0]))
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
    ax.set_title('D3: Pairwise Discrimination', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (1,1) D4: Ordering dependence
    ax = runner.dark_ax(fig.add_subplot(gs[1, 1]))
    d4_names = CONSTANTS + ['random']
    d4_labels = [CONST_LABELS.get(n, n) for n in d4_names]
    d4_sigs = [d4_results[n] for n in d4_names]
    d4_colors = [colors.get(n, '#666666') for n in d4_names]
    ax.bar(range(len(d4_names)), d4_sigs,
           color=d4_colors, alpha=0.85, edgecolor='#333')
    ax.set_xticks(range(len(d4_names)))
    ax.set_xticklabels(d4_labels, fontsize=10)
    ax.set_ylabel('Sig metrics (orig vs shuffled)', fontsize=9)
    ax.set_title('D4: Ordering Dependence', fontsize=11, fontweight='bold')
    for i, v in enumerate(d4_sigs):
        ax.text(i, v + 0.3, str(v), ha='center', fontsize=9, fontweight='bold', color=FG)

    # (2,:) D5: Sig vs τ line plot (spans bottom row)
    ax = runner.dark_ax(fig.add_subplot(gs[2, :]))
    for name in CONSTANTS:
        sig_vals = [d5_results[name][tau] for tau in taus]
        ax.plot(range(len(taus)), sig_vals, 'o-', color=colors[name],
                linewidth=2, markersize=6, alpha=0.85, label=CONST_LABELS[name])
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([str(t) for t in taus], fontsize=9)
    ax.set_xlabel('Delay τ', fontsize=9)
    ax.set_ylabel('Sig metrics (embedded CF vs embedded random)', fontsize=8)
    ax.set_title('D5: Delay Embedding', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#222', edgecolor='#444', labelcolor=FG, loc='best')

    fig.suptitle('Continued Fraction Geometry: Sequential Structure Beyond Gauss-Kuzmin',
                 fontsize=14, fontweight='bold', color=FG, y=0.995)

    runner.save(fig, "continued_fractions")


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()
    runner = Runner("Continued Fractions", mode="1d",
                    n_workers=N_WORKERS, data_size=DATA_SIZE,
                    n_trials=N_TRIALS)

    print("=" * 78)
    print("CONTINUED FRACTION GEOMETRY")
    print("=" * 78)

    try:
        all_data, random_data, d1_results = direction_1(runner)
        gk_data, d2_results = direction_2(runner, all_data)
        d3_matrix = direction_3(runner, all_data)
        d4_results = direction_4(runner, all_data)
        d5_results, taus = direction_5(runner)

        make_figure(runner, d1_results, d2_results, d3_matrix, d4_results,
                    d5_results, taus)

        elapsed = time.time() - t0
        print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        runner.print_summary({
            **{f"D1 {CONST_LABELS[n]}": f"{d1_results[n]} sig vs random"
               for n in CONSTANTS},
            **{f"D2 {CONST_LABELS[n]}": f"{d2_results[n]} sig vs GK"
               for n in CONSTANTS},
            **{f"D4 {CONST_LABELS[n]}": f"{d4_results[n]} ordering"
               for n in CONSTANTS},
            'D4 random': f"{d4_results['random']} self-check",
        })
    finally:
        runner.close()


if __name__ == "__main__":
    main()
