#!/usr/bin/env python3
"""
Investigation: GF(2) Linear Structure Detection
=================================================

Can the exotic geometry framework detect algebraic structure in GF(2)-linear
PRNGs (XorShift, LFSRs) that byte-level statistics miss?

Two complementary approaches:
  D1: Full framework scan — all geometries on PRNG sources vs White Noise.
      Which of the ~278 metrics detect GF(2) structure?
  D2: Custom GF(2) metrics — binary rank, Berlekamp-Massey linear complexity,
      Walsh-Hadamard linearity, bit serial correlation. These are the
      specialized tools. How do they compare to the framework?
  D3: Geometry × source heatmap — which geometries catch which PRNGs?
  D4: GF(2) metric effect sizes — Cohen's d for each custom metric,
      XorShift32 vs White Noise and PRNG cluster vs true random.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats as sp_stats
from collections import defaultdict
from tools.sources import get_sources
from tools.investigation_runner import Runner

N_TRIALS = 25
DATA_SIZE = 16384
N_WORKERS = 8

# Sources to test
PRNG_NAMES = ['XorShift32', 'MINSTD (Park-Miller)', 'glibc LCG', 'Wichmann-Hill', 'RANDU']
RANDOM_NAMES = ['White Noise', 'AES Encrypted']
STRUCTURED_NAMES = ['Logistic Chaos', 'Lorenz Attractor', 'Thue-Morse']
ALL_NAMES = PRNG_NAMES + RANDOM_NAMES + STRUCTURED_NAMES


# =========================================================================
# CUSTOM GF(2) METRICS
# =========================================================================

def bytes_to_bits(data):
    return np.unpackbits(data)


def berlekamp_massey(bits, max_len=None):
    """Berlekamp-Massey: shortest LFSR length. Random ≈ n/2, XorShift ≈ 32."""
    n = len(bits)
    if max_len is not None:
        n = min(n, max_len)
    s = bits[:n].astype(np.int8)
    c = np.zeros(n + 1, dtype=np.int8)
    b = np.zeros(n + 1, dtype=np.int8)
    c[0] = 1
    b[0] = 1
    L = 0
    m = 1
    for i in range(n):
        d = s[i]
        for j in range(1, L + 1):
            d ^= c[j] & s[i - j]
        d &= 1
        if d == 0:
            m += 1
        else:
            t = c.copy()
            for j in range(n + 1 - m):
                c[j + m] ^= b[j]
            if 2 * L <= i:
                L = i + 1 - L
                b = t.copy()
                m = 1
            else:
                m += 1
    return L


def gf2_rank(matrix):
    """Rank of binary matrix over GF(2) via Gaussian elimination."""
    m = matrix.copy().astype(np.uint8)
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col] == 1:
                m[row] = m[row] ^ m[rank]
        rank += 1
    return rank


def binary_rank_gf2(data, matrix_size=32):
    """GF(2) binary matrix rank test. Returns (mean_deficiency, full_rank_fraction)."""
    bits = bytes_to_bits(data)
    bits_per_matrix = matrix_size * matrix_size
    n_matrices = len(bits) // bits_per_matrix
    if n_matrices < 3:
        return float('nan'), float('nan')
    ranks = []
    for i in range(n_matrices):
        start = i * bits_per_matrix
        mat = bits[start:start + bits_per_matrix].reshape(matrix_size, matrix_size)
        ranks.append(gf2_rank(mat))
    ranks = np.array(ranks, dtype=np.float64)
    return matrix_size - np.mean(ranks), np.mean(ranks == matrix_size)


def linear_complexity_profile(data, n_bits=2048):
    """Normalized linear complexity and deviation from n/2 growth."""
    bits = bytes_to_bits(data)[:n_bits]
    n = len(bits)
    checkpoints = [64, 128, 256, 512, 1024, min(n, 2048)]
    lc_values = []
    for cp in checkpoints:
        if cp > n:
            break
        lc = berlekamp_massey(bits, max_len=cp)
        lc_values.append((cp, lc))
    if not lc_values:
        return float('nan'), float('nan')
    final_n, final_lc = lc_values[-1]
    normalized_lc = final_lc / (final_n / 2)
    expected = np.array([cp / 2 for cp, _ in lc_values])
    actual = np.array([lc for _, lc in lc_values])
    deviation = np.mean((expected - actual) / expected)
    return normalized_lc, deviation


def bit_serial_correlation(data):
    """Cross-bit-position correlation between consecutive bytes."""
    n = len(data) - 1
    bit_channels = np.zeros((8, n), dtype=np.int8)
    next_channels = np.zeros((8, n), dtype=np.int8)
    for b in range(8):
        bit_channels[b] = (data[:n] >> b) & 1
        next_channels[b] = (data[1:n+1] >> b) & 1
    corr_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            r = np.corrcoef(bit_channels[i].astype(float),
                            next_channels[j].astype(float))[0, 1]
            if np.isfinite(r):
                corr_matrix[i, j] = r
    return np.max(np.abs(corr_matrix)), np.linalg.norm(corr_matrix, 'fro') / 8


def compute_gf2_metrics(data):
    """Compute all custom GF(2) metrics, return dict."""
    gf2_def, gf2_fr = binary_rank_gf2(data, matrix_size=32)
    norm_lc, lc_dev = linear_complexity_profile(data, n_bits=2048)
    max_bc, bc_frob = bit_serial_correlation(data)
    return {
        'gf2_rank_deficiency': gf2_def,
        'gf2_full_rank_frac': gf2_fr,
        'linear_complexity': norm_lc,
        'lc_deviation': lc_dev,
        'max_bit_corr': max_bc,
        'bit_corr_frob': bc_frob,
    }


# =========================================================================
# DIRECTIONS
# =========================================================================

def direction_1(runner, source_map):
    """D1: Full framework — each source vs White Noise."""
    print("\n" + "=" * 78)
    print("D1: FULL FRAMEWORK — EACH SOURCE VS WHITE NOISE")
    print("=" * 78)

    wn_src = source_map['White Noise']
    with runner.timed("White Noise reference"):
        ref_chunks = [wn_src.gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs()]
        ref_met = runner.collect(ref_chunks)

    d1 = {}
    for name in PRNG_NAMES + STRUCTURED_NAMES:
        if name not in source_map:
            continue
        src = source_map[name]
        with runner.timed(name):
            chunks = [src.gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=100)]
            met = runner.collect(chunks)
        n_sig, findings = runner.compare(met, ref_met)
        d1[name] = n_sig
        print(f"  {name:<25s}: {n_sig:3d} sig")
        for m, d, p in findings[:3]:
            print(f"    {m:50s} d={d:+8.2f}")

    # AES self-check
    aes_src = source_map.get('AES Encrypted')
    if aes_src:
        with runner.timed("AES self-check"):
            aes_chunks = [aes_src.gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=200)]
            aes_met = runner.collect(aes_chunks)
        n_sig, _ = runner.compare(aes_met, ref_met)
        d1['AES Encrypted'] = n_sig
        print(f"  {'AES self-check':<25s}: {n_sig:3d} sig (expect ≈ 0)")

    return d1


def direction_2(source_map):
    """D2: Custom GF(2) metrics — specialized diagnostics."""
    print("\n" + "=" * 78)
    print("D2: CUSTOM GF(2) METRICS")
    print("=" * 78)

    metric_keys = ['gf2_rank_deficiency', 'gf2_full_rank_frac', 'linear_complexity',
                   'lc_deviation', 'max_bit_corr', 'bit_corr_frob']

    print(f"  {'Source':<25s}  {'rank_def':>8s}  {'full_r%':>7s}  "
          f"{'norm_lc':>7s}  {'lc_dev':>7s}  {'max_bc':>7s}  {'bc_frob':>7s}")
    print("  " + "-" * 90)

    all_results = {}
    for name in ALL_NAMES:
        if name not in source_map:
            continue
        src = source_map[name]
        trials = {k: [] for k in metric_keys}
        for t in range(N_TRIALS):
            rng = np.random.default_rng(42 + t)
            data = src.gen_fn(rng, DATA_SIZE)
            m = compute_gf2_metrics(data)
            for k in metric_keys:
                trials[k].append(m[k])

        means = {k: np.nanmean(v) for k, v in trials.items()}
        stds = {k: np.nanstd(v) for k, v in trials.items()}
        all_results[name] = (means, stds, trials)

        print(f"  {name:<25s}  {means['gf2_rank_deficiency']:>8.3f}  "
              f"{means['gf2_full_rank_frac']:>7.3f}  "
              f"{means['linear_complexity']:>7.4f}  {means['lc_deviation']:>7.4f}  "
              f"{means['max_bit_corr']:>7.4f}  {means['bit_corr_frob']:>7.4f}")

    return all_results


def direction_3(runner, source_map):
    """D3: Geometry × source heatmap — which geometries catch which PRNGs."""
    print("\n" + "=" * 78)
    print("D3: GEOMETRY × SOURCE HEATMAP")
    print("=" * 78)

    wn_src = source_map['White Noise']
    ref_chunks = [wn_src.gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs()]
    ref_met = runner.collect(ref_chunks)

    # Get geometry names from metric names
    geom_names = sorted(set(m.split(':')[0] for m in runner.metric_names))

    heatmap = {}  # (geom, source) -> n_sig
    for name in PRNG_NAMES:
        if name not in source_map:
            continue
        src = source_map[name]
        chunks = [src.gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=100)]
        met = runner.collect(chunks)

        for geom in geom_names:
            geom_metrics = [m for m in runner.metric_names if m.startswith(geom + ':')]
            n_sig = 0
            for m in geom_metrics:
                a = np.array(met.get(m, []))
                b = np.array(ref_met.get(m, []))
                if len(a) < 3 or len(b) < 3:
                    continue
                _, p = sp_stats.ttest_ind(a, b, equal_var=False)
                if np.isfinite(p) and p < runner.bonf_alpha:
                    n_sig += 1
            heatmap[(geom, name)] = n_sig

    # Print top detectors per PRNG
    for name in PRNG_NAMES:
        if name not in source_map:
            continue
        detectors = [(g, heatmap.get((g, name), 0)) for g in geom_names]
        detectors.sort(key=lambda x: -x[1])
        top = [f"{g}={n}" for g, n in detectors[:5] if n > 0]
        print(f"  {name:<25s}: {', '.join(top) if top else 'none'}")

    return heatmap, geom_names


def direction_4(gf2_results):
    """D4: GF(2) metric effect sizes."""
    print("\n" + "=" * 78)
    print("D4: GF(2) EFFECT SIZES — XorShift32 vs White Noise")
    print("=" * 78)

    metric_keys = ['gf2_rank_deficiency', 'gf2_full_rank_frac', 'linear_complexity',
                   'lc_deviation', 'max_bit_corr', 'bit_corr_frob']

    if 'XorShift32' in gf2_results and 'White Noise' in gf2_results:
        for k in metric_keys:
            xm, xs = gf2_results['XorShift32'][0][k], gf2_results['XorShift32'][1][k]
            wm, ws = gf2_results['White Noise'][0][k], gf2_results['White Noise'][1][k]
            pooled = np.sqrt((xs**2 + ws**2) / 2 + 1e-15)
            d = (xm - wm) / pooled
            print(f"  {k:<25s}: d = {d:+.2f}  "
                  f"(xor={xm:.4f}±{xs:.4f}, wn={wm:.4f}±{ws:.4f})")

    print(f"\n  PRNG cluster vs true random:")
    for k in metric_keys:
        prng_vals = [gf2_results[n][0][k] for n in PRNG_NAMES if n in gf2_results]
        rand_vals = [gf2_results[n][0][k] for n in RANDOM_NAMES if n in gf2_results]
        if prng_vals and rand_vals:
            pm, rm = np.mean(prng_vals), np.mean(rand_vals)
            ratio_str = f"ratio={pm/rm:.3f}" if rm != 0 else ""
            print(f"  {k:<25s}: PRNG={pm:.4f}  Random={rm:.4f}  {ratio_str}")


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(runner, d1, d3_heatmap, d3_geoms):
    fig, axes = runner.create_figure(2, "GF(2) Linear Structure Detection")

    # D1: Detection bar chart
    ax = axes[0]
    names = [n for n in PRNG_NAMES + STRUCTURED_NAMES if n in d1]
    sigs = [d1[n] for n in names]
    short = [n.split('(')[0].strip()[:15] for n in names]
    ax.bar(range(len(names)), sigs, alpha=0.85,
           color=['#FF5722' if n in PRNG_NAMES else '#2196F3' for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})', fontsize=9)
    ax.set_title('D1: Framework Detection vs White Noise', fontsize=11, fontweight='bold')
    for i, v in enumerate(sigs):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=8, fontweight='bold', color='white')

    # D3: Top-5 geometry detectors per PRNG (horizontal bars)
    ax = axes[1]
    prng_present = [n for n in PRNG_NAMES if any((g, n) in d3_heatmap for g in d3_geoms)]
    if prng_present:
        # For each PRNG, show top geometry
        top_geoms = set()
        for name in prng_present:
            detectors = [(g, d3_heatmap.get((g, name), 0)) for g in d3_geoms]
            detectors.sort(key=lambda x: -x[1])
            for g, n in detectors[:3]:
                if n > 0:
                    top_geoms.add(g)
        top_geoms = sorted(top_geoms)[:10]
        if top_geoms:
            data = np.array([[d3_heatmap.get((g, n), 0) for n in prng_present]
                             for g in top_geoms])
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(prng_present)))
            ax.set_xticklabels([n.split('(')[0].strip()[:12] for n in prng_present],
                               fontsize=7, rotation=30, ha='right')
            ax.set_yticks(range(len(top_geoms)))
            ax.set_yticklabels([g[:20] for g in top_geoms], fontsize=7)
            for i in range(len(top_geoms)):
                for j in range(len(prng_present)):
                    v = data[i, j]
                    if v > 0:
                        ax.text(j, i, str(int(v)), ha='center', va='center',
                                fontsize=8, color='white' if v > data.max() / 2 else '#333')
            ax.set_title('D3: Top Geometry Detectors per PRNG', fontsize=11, fontweight='bold')

    runner.save(fig, "gf2_explore")


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()

    sources = get_sources()
    source_map = {s.name: s for s in sources if s.name in ALL_NAMES}
    missing = [n for n in ALL_NAMES if n not in source_map]
    if missing:
        print(f"Missing sources: {missing}")

    runner = Runner("GF(2) Explore", mode="1d",
                    n_workers=N_WORKERS, data_size=DATA_SIZE,
                    n_trials=N_TRIALS)

    print("=" * 78)
    print("GF(2) LINEAR STRUCTURE DETECTION")
    print("=" * 78)

    try:
        d1 = direction_1(runner, source_map)
        gf2_results = direction_2(source_map)
        d3_heatmap, d3_geoms = direction_3(runner, source_map)
        direction_4(gf2_results)

        make_figure(runner, d1, d3_heatmap, d3_geoms)

        elapsed = time.time() - t0
        print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        runner.print_summary({
            **{f"D1 {n}": f"{d1[n]} sig vs White Noise" for n in PRNG_NAMES if n in d1},
            'D1 AES': f"{d1.get('AES Encrypted', '?')} sig (self-check, expect ≈ 0)",
        })
    finally:
        runner.close()


if __name__ == '__main__':
    main()
