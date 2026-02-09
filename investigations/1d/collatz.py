#!/usr/bin/env python3
"""
Investigation: Geometric signatures of Collatz (3n+1) sequences.

The Collatz conjecture states that iterating n → n/2 (even) or n → 3n+1 (odd)
always reaches 1. Despite its simplicity, the dynamics are not well understood.

Questions:
1. Do Collatz hailstone sequences look random to exotic geometries?
2. Can we distinguish Collatz from actual random data?
3. Do different starting ranges produce different geometric signatures?
4. Does the PARITY sequence (odd/even choices) carry detectable structure?
5. How do Collatz sequences compare to other integer sequences?
6. What does the stopping-time distribution look like geometrically?

Encodings tested:
- hailstone: raw trajectory values mod 256
- parity: the sequence of odd/even decisions (0/1 bits, packed into bytes)
- steps: consecutive stopping times for sequential starting values
- residues: trajectory values mod various small primes

Comparisons: os.urandom, logistic chaos, linear congruential sequences
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000


# =============================================================================
# COLLATZ GENERATORS
# =============================================================================

def collatz_step(n):
    """Single Collatz step."""
    return n // 2 if n % 2 == 0 else 3 * n + 1


def collatz_trajectory(start, max_steps=100000):
    """Full trajectory from start down to 1."""
    n = start
    traj = [n]
    while n != 1 and len(traj) < max_steps:
        n = collatz_step(n)
        traj.append(n)
    return traj


def collatz_stopping_time(n):
    """Number of steps to reach 1."""
    steps = 0
    while n != 1 and steps < 1000000:
        n = collatz_step(n)
        steps += 1
    return steps


def generate_collatz_data(encoding, trial_seed, size=DATA_SIZE):
    """Generate Collatz-derived byte data in various encodings."""

    if encoding == 'hailstone_small':
        # Trajectory from a starting number in [1000, 10000]
        # Take values mod 256 to get bytes
        rng = np.random.RandomState(trial_seed)
        start = rng.randint(1000, 10000)
        traj = collatz_trajectory(start)
        # If trajectory is short, chain multiple starting points
        while len(traj) < size:
            start = rng.randint(1000, 10000)
            traj.extend(collatz_trajectory(start))
        arr = np.array(traj[:size], dtype=np.uint64)
        return (arr % 256).astype(np.uint8)

    elif encoding == 'hailstone_large':
        # Trajectory from large starting numbers [10^9, 10^10]
        rng = np.random.RandomState(trial_seed)
        start = rng.randint(10**9, 10**10)
        traj = collatz_trajectory(start)
        while len(traj) < size:
            start = rng.randint(10**9, 10**10)
            traj.extend(collatz_trajectory(start))
        arr = np.array(traj[:size], dtype=np.uint64)
        return (arr % 256).astype(np.uint8)

    elif encoding == 'hailstone_sequential':
        # Chain trajectories from consecutive integers starting at seed*1000
        base = trial_seed * 1000 + 2
        traj = []
        n = base
        while len(traj) < size:
            traj.extend(collatz_trajectory(n))
            n += 1
        arr = np.array(traj[:size], dtype=np.uint64)
        return (arr % 256).astype(np.uint8)

    elif encoding == 'parity':
        # The odd/even decision sequence, packed into bytes
        # This is the "compressed" version of Collatz — just the turns
        rng = np.random.RandomState(trial_seed)
        bits = []
        while len(bits) < size * 8:
            start = rng.randint(10**6, 10**8)
            n = start
            while n != 1 and len(bits) < size * 8:
                bits.append(n % 2)
                n = collatz_step(n)
        # Pack bits into bytes
        bits = bits[:size * 8]
        result = np.zeros(size, dtype=np.uint8)
        for i in range(size):
            byte_val = 0
            for bit_idx in range(8):
                byte_val |= (bits[i * 8 + bit_idx] << bit_idx)
            result[i] = byte_val
        return result

    elif encoding == 'stopping_times':
        # Stopping times for consecutive integers
        base = trial_seed * 10000 + 2
        times = []
        for n in range(base, base + size):
            times.append(collatz_stopping_time(n))
        arr = np.array(times, dtype=np.uint64)
        return (arr % 256).astype(np.uint8)

    elif encoding == 'residues_mod7':
        # Trajectory values mod 7, mapped to byte range
        rng = np.random.RandomState(trial_seed)
        vals = []
        while len(vals) < size:
            start = rng.randint(10**6, 10**8)
            traj = collatz_trajectory(start)
            vals.extend([t % 7 for t in traj])
        arr = np.array(vals[:size])
        return (arr * 36).astype(np.uint8)  # scale 0-6 to 0-216

    elif encoding == 'high_bits':
        # High-order bits of trajectory values (log-scale structure)
        rng = np.random.RandomState(trial_seed)
        vals = []
        while len(vals) < size:
            start = rng.randint(10**6, 10**8)
            traj = collatz_trajectory(start)
            for t in traj:
                if t > 0:
                    vals.append(int(np.log2(max(t, 1)) * 8) % 256)
                else:
                    vals.append(0)
        return np.array(vals[:size], dtype=np.uint8)

    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def generate_comparison_data(comp_type, trial_seed, size=DATA_SIZE):
    """Generate comparison data for baseline."""
    rng = np.random.RandomState(trial_seed)

    if comp_type == 'random':
        return rng.randint(0, 256, size, dtype=np.uint8)

    elif comp_type == 'logistic':
        # Chaotic logistic map
        r = 3.99
        x = 0.1 + 0.001 * (trial_seed % 100)
        vals = []
        for _ in range(size + 500):
            x = r * x * (1 - x)
            vals.append(x)
        arr = np.array(vals[500:500 + size])
        return (arr * 255).astype(np.uint8)

    elif comp_type == 'lcg':
        # Simple LCG (structured but not Collatz)
        state = trial_seed * 7 + 1
        vals = []
        for _ in range(size):
            state = (1103515245 * state + 12345) % (2**31)
            vals.append((state >> 16) & 0xFF)
        return np.array(vals, dtype=np.uint8)

    elif comp_type == 'random_walk':
        # Random walk mod 256 (has sequential structure like Collatz)
        x = 128
        vals = []
        for _ in range(size):
            x = (x + rng.choice([-3, -2, -1, 0, 1, 2, 3])) % 256
            vals.append(x)
        return np.array(vals, dtype=np.uint8)

    elif comp_type == 'syracuse_variant':
        # Modified Collatz: n → n/2 (even) or 5n+1 (odd)
        # Different dynamics, same flavor
        vals = []
        while len(vals) < size:
            n = rng.randint(10**6, 10**8)
            steps = 0
            while n != 1 and steps < 10000:
                if n % 2 == 0:
                    n = n // 2
                else:
                    n = 5 * n + 1
                vals.append(n % 256)
                steps += 1
                if n > 10**15:  # 5n+1 can diverge
                    break
        return np.array(vals[:size], dtype=np.uint8)

    else:
        raise ValueError(f"Unknown comparison: {comp_type}")


# =============================================================================
# STATISTICS
# =============================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("INVESTIGATION: Collatz (3n+1) Sequences — Geometric Signatures")
    print("=" * 78)

    encodings = [
        'hailstone_small', 'hailstone_large', 'hailstone_sequential',
        'parity', 'stopping_times', 'residues_mod7', 'high_bits',
    ]
    comparisons = ['random', 'logistic', 'lcg', 'random_walk', 'syracuse_variant']

    analyzer = GeometryAnalyzer().add_all_geometries()

    # =========================================================================
    # STEP 1: Collect metrics for all Collatz encodings
    # =========================================================================
    print(f"\nParameters: {N_TRIALS} trials x {DATA_SIZE} bytes each")
    print(f"Collatz encodings: {len(encodings)}")
    print(f"Comparison types: {len(comparisons)}")

    all_metrics = {}

    print(f"\n--- Generating Collatz data ---")
    for enc in encodings:
        print(f"  {enc}...", end=" ", flush=True)
        all_metrics[enc] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_collatz_data(enc, trial)
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    all_metrics[enc][f"{r.geometry_name}:{mn}"].append(mv)
        print("done")

    print(f"\n--- Generating comparison data ---")
    for comp in comparisons:
        print(f"  {comp}...", end=" ", flush=True)
        all_metrics[comp] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_comparison_data(comp, trial)
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    all_metrics[comp][f"{r.geometry_name}:{mn}"].append(mv)
        print("done")

    # Get all metric names
    all_metric_names = sorted(set().union(*(m.keys() for m in all_metrics.values())))

    key_metrics = [
        'E8 Lattice:unique_roots', 'E8 Lattice:alignment_mean',
        'Torus T^2:coverage', 'Torus T^2:chi2_uniformity',
        'Heisenberg (Nil):twist_rate', 'Heisenberg (Nil) (centered):twist_rate',
        'Tropical:linearity', 'Sol (Thurston):anisotropy',
        'Penrose (Quasicrystal):fivefold_balance',
        'Fisher Information:trace_fisher',
        'Hyperbolic (Poincaré):mean_radius',
        'Persistent Homology:n_significant_features',
    ]

    # =========================================================================
    # STEP 2: Fingerprint table
    # =========================================================================
    print(f"\n{'='*78}")
    print("STEP 1: GEOMETRIC FINGERPRINT TABLE (key metrics)")
    print(f"{'='*78}\n")

    display_metrics = key_metrics[:8]
    header = f"{'Source':<22}" + "".join(f"{km.split(':')[1][:12]:>14}" for km in display_metrics)
    print(header)
    print("-" * len(header))

    for source in encodings + comparisons:
        row = f"{source:<22}"
        for km in display_metrics:
            vals = all_metrics[source].get(km, [0])
            row += f"{np.mean(vals):>14.3f}"
        print(row)

    # =========================================================================
    # STEP 3: Each Collatz encoding vs random
    # =========================================================================
    print(f"\n{'='*78}")
    print("STEP 2: EACH COLLATZ ENCODING vs RANDOM")
    print(f"{'='*78}")

    n_tests = len(all_metric_names)
    alpha = 0.05 / max(n_tests, 1)
    print(f"\nBonferroni alpha: {alpha:.2e} ({n_tests} metrics)")

    for enc in encodings:
        sig_results = []
        for km in all_metric_names:
            if km not in all_metrics['random'] or km not in all_metrics[enc]:
                continue
            r_vals = all_metrics['random'][km]
            e_vals = all_metrics[enc][km]
            if len(r_vals) < 2 or len(e_vals) < 2:
                continue
            d = cohens_d(e_vals, r_vals)
            _, p = stats.ttest_ind(e_vals, r_vals, equal_var=False)
            if abs(d) > 0.8 and p < alpha:
                sig_results.append((km, d, p))

        sig_results.sort(key=lambda x: -abs(x[1]))
        status = "*** DETECTED" if sig_results else "(random-looking)"
        print(f"\n  {enc}: {len(sig_results)} significant metrics  {status}")
        for km, d, p in sig_results[:5]:
            print(f"    {km:<50} d={d:+8.2f}  p={p:.2e}")

    # =========================================================================
    # STEP 4: Collatz vs other structured sequences
    # =========================================================================
    print(f"\n{'='*78}")
    print("STEP 3: COLLATZ vs OTHER STRUCTURED SEQUENCES")
    print(f"{'='*78}")

    # Use hailstone_large as the canonical Collatz encoding
    canonical = 'hailstone_large'
    structured = ['logistic', 'lcg', 'random_walk', 'syracuse_variant']

    n_tests2 = len(all_metric_names)
    alpha2 = 0.05 / max(n_tests2, 1)
    print(f"\nComparing {canonical} against each structured sequence")
    print(f"Bonferroni alpha: {alpha2:.2e}\n")

    for comp in structured:
        sig_results = []
        for km in all_metric_names:
            if km not in all_metrics[canonical] or km not in all_metrics[comp]:
                continue
            c_vals = all_metrics[canonical][km]
            o_vals = all_metrics[comp][km]
            if len(c_vals) < 2 or len(o_vals) < 2:
                continue
            d = cohens_d(c_vals, o_vals)
            _, p = stats.ttest_ind(c_vals, o_vals, equal_var=False)
            if abs(d) > 0.8 and p < alpha2:
                sig_results.append((km, d, p))

        sig_results.sort(key=lambda x: -abs(x[1]))
        print(f"  {canonical} vs {comp}: {len(sig_results)} significant metrics")
        for km, d, p in sig_results[:5]:
            print(f"    {km:<50} d={d:+8.2f}  p={p:.2e}")

    # =========================================================================
    # STEP 5: Pairwise encoding comparison
    # =========================================================================
    print(f"\n{'='*78}")
    print("STEP 4: PAIRWISE COLLATZ ENCODING COMPARISON")
    print(f"{'='*78}")
    print(f"\nCan different Collatz encodings be distinguished from each other?\n")

    import itertools
    enc_pairs = list(itertools.combinations(encodings, 2))
    n_pair_tests = len(all_metric_names)
    alpha_pair = 0.05 / max(n_pair_tests, 1)

    pair_results = []
    for e1, e2 in enc_pairs:
        best_d, best_km, best_p = 0, None, 1
        n_sig = 0
        for km in all_metric_names:
            if km not in all_metrics[e1] or km not in all_metrics[e2]:
                continue
            v1, v2 = all_metrics[e1][km], all_metrics[e2][km]
            if len(v1) < 2 or len(v2) < 2:
                continue
            d = cohens_d(v1, v2)
            _, p = stats.ttest_ind(v1, v2, equal_var=False)
            if abs(d) > 0.8 and p < alpha_pair:
                n_sig += 1
            if abs(d) > abs(best_d):
                best_d, best_km, best_p = d, km, p
        pair_results.append((e1, e2, n_sig, best_km, best_d, best_p))

    pair_results.sort(key=lambda x: -abs(x[4]))

    print(f"{'Encoding 1':<24} {'Encoding 2':<24} {'# Sig':>6} {'Best Metric':<30} {'d':>8}")
    print("-" * 98)
    for e1, e2, n_sig, km, d, p in pair_results:
        km_short = km.split(':')[1] if km else '?'
        sig = "***" if n_sig > 0 else ""
        print(f"{e1:<24} {e2:<24} {n_sig:>6} {km_short:<30} {d:>+8.2f} {sig}")

    # =========================================================================
    # STEP 6: Shuffle validation
    # =========================================================================
    print(f"\n{'='*78}")
    print("STEP 5: SHUFFLE VALIDATION")
    print(f"{'='*78}")
    print(f"\nDo Collatz signatures survive shuffling?")
    print(f"If YES → signal is from sequential structure (ordering matters)")
    print(f"If NO  → signal is from marginal distribution only\n")

    test_encodings = ['hailstone_large', 'parity', 'stopping_times', 'high_bits']
    test_metrics = [
        'E8 Lattice:unique_roots', 'Torus T^2:coverage',
        'Heisenberg (Nil):twist_rate', 'Tropical:linearity',
        'Penrose (Quasicrystal):fivefold_balance',
        'Fisher Information:trace_fisher',
    ]

    for enc in test_encodings:
        print(f"  {enc}:")
        # Generate one sample and its shuffle
        data = generate_collatz_data(enc, 42)
        shuffled = data.copy()
        np.random.RandomState(99).shuffle(shuffled)

        results_real = analyzer.analyze(data)
        results_shuf = analyzer.analyze(shuffled)

        real_metrics = {}
        shuf_metrics = {}
        for r in results_real.results:
            for mn, mv in r.metrics.items():
                real_metrics[f"{r.geometry_name}:{mn}"] = mv
        for r in results_shuf.results:
            for mn, mv in r.metrics.items():
                shuf_metrics[f"{r.geometry_name}:{mn}"] = mv

        for km in test_metrics:
            rv = real_metrics.get(km, 0)
            sv = shuf_metrics.get(km, 0)
            if sv != 0:
                ratio = rv / sv
            elif rv != 0:
                ratio = float('inf')
            else:
                ratio = 1.0
            label = "ORDERING" if abs(ratio - 1) > 0.1 else "marginal"
            km_short = km.split(':')[1]
            print(f"    {km_short:<30} real={rv:>10.3f}  shuf={sv:>10.3f}  ratio={ratio:.2f}  → {label}")
        print()

    # =========================================================================
    # STEP 7: Starting number effect
    # =========================================================================
    print(f"{'='*78}")
    print("STEP 6: STARTING NUMBER MAGNITUDE EFFECT")
    print(f"{'='*78}")
    print(f"\nDo larger starting numbers produce different geometric signatures?\n")

    ranges = {
        'tiny (10-100)':     (10, 100),
        'small (1K-10K)':    (1000, 10000),
        'medium (1M-10M)':   (10**6, 10**7),
        'large (1B-10B)':    (10**9, 10**10),
    }

    range_metrics = {}
    for label, (lo, hi) in ranges.items():
        range_metrics[label] = defaultdict(list)
        for trial in range(N_TRIALS):
            rng = np.random.RandomState(trial)
            start = rng.randint(lo, hi)
            traj = collatz_trajectory(start)
            while len(traj) < DATA_SIZE:
                start = rng.randint(lo, hi)
                traj.extend(collatz_trajectory(start))
            data = (np.array(traj[:DATA_SIZE], dtype=np.uint64) % 256).astype(np.uint8)
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    range_metrics[label][f"{r.geometry_name}:{mn}"].append(mv)

    range_labels = list(ranges.keys())
    range_pair_tests = len(all_metric_names)
    alpha_range = 0.05 / max(range_pair_tests, 1)

    for i, r1 in enumerate(range_labels):
        for r2 in range_labels[i+1:]:
            sig_count = 0
            best_d, best_km = 0, None
            for km in all_metric_names:
                if km not in range_metrics[r1] or km not in range_metrics[r2]:
                    continue
                v1, v2 = range_metrics[r1][km], range_metrics[r2][km]
                if len(v1) < 2 or len(v2) < 2:
                    continue
                d = cohens_d(v1, v2)
                _, p = stats.ttest_ind(v1, v2, equal_var=False)
                if abs(d) > 0.8 and p < alpha_range:
                    sig_count += 1
                if abs(d) > abs(best_d):
                    best_d, best_km = d, km
            km_short = best_km.split(':')[1] if best_km else '?'
            sig = "***" if sig_count > 0 else ""
            print(f"  {r1:<20} vs {r2:<20}: {sig_count:>3} sig metrics  best: {km_short} d={best_d:+.2f} {sig}")

    # =========================================================================
    # STEP 8: 5n+1 variant comparison
    # =========================================================================
    print(f"\n{'='*78}")
    print("STEP 7: COLLATZ (3n+1) vs SYRACUSE VARIANT (5n+1)")
    print(f"{'='*78}")
    print(f"\nCan exotic geometries distinguish the canonical 3n+1 from the 5n+1 variant?\n")

    n_tests_var = len(all_metric_names)
    alpha_var = 0.05 / max(n_tests_var, 1)

    sig_var = []
    for km in all_metric_names:
        if km not in all_metrics['hailstone_large'] or km not in all_metrics['syracuse_variant']:
            continue
        v1 = all_metrics['hailstone_large'][km]
        v2 = all_metrics['syracuse_variant'][km]
        if len(v1) < 2 or len(v2) < 2:
            continue
        d = cohens_d(v1, v2)
        _, p = stats.ttest_ind(v1, v2, equal_var=False)
        if abs(d) > 0.8 and p < alpha_var:
            sig_var.append((km, d, p))

    sig_var.sort(key=lambda x: -abs(x[1]))

    if sig_var:
        print(f"  {len(sig_var)} significant differences found:")
        for km, d, p in sig_var[:10]:
            print(f"    {km:<50} d={d:+8.2f}  p={p:.2e}")
    else:
        print(f"  No significant differences found!")
        print(f"  3n+1 and 5n+1 trajectories are geometrically indistinguishable at byte level.")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")

    print(f"\nCollatz vs Random:")
    for enc in encodings:
        sig_count = 0
        for km in all_metric_names:
            if km not in all_metrics['random'] or km not in all_metrics[enc]:
                continue
            r_vals = all_metrics['random'][km]
            e_vals = all_metrics[enc][km]
            if len(r_vals) < 2 or len(e_vals) < 2:
                continue
            d = cohens_d(e_vals, r_vals)
            _, p = stats.ttest_ind(e_vals, r_vals, equal_var=False)
            if abs(d) > 0.8 and p < 0.05 / max(len(all_metric_names), 1):
                sig_count += 1
        status = "DETECTED" if sig_count > 0 else "random-looking"
        print(f"  {enc:<24}: {sig_count:>3} sig metrics  → {status}")

    print(f"\n[Investigation complete]")

    return all_metrics, all_metric_names, encodings, comparisons, pair_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(all_metrics, all_metric_names, encodings, comparisons, pair_results):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 1.2],
                           hspace=0.40, wspace=0.35)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    enc_colors = {
        'hailstone_small': '#E91E63', 'hailstone_large': '#FF5722',
        'hailstone_sequential': '#FF9800', 'parity': '#4CAF50',
        'stopping_times': '#2196F3', 'residues_mod7': '#9C27B0',
        'high_bits': '#00BCD4',
    }

    # ── Row 0, left: Example hailstone trajectories ──
    ax_traj = fig.add_subplot(gs[0, :2])
    _dark_ax(ax_traj)
    starts = [27, 871, 6171, 77031]
    traj_colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']
    for start, c in zip(starts, traj_colors):
        traj = collatz_trajectory(start, max_steps=300)
        ax_traj.plot(range(len(traj)), traj, color=c, alpha=0.85, linewidth=1.0,
                     label=f'n={start} ({len(traj)} steps)')
    ax_traj.set_yscale('log')
    ax_traj.set_xlabel('Step', color=FG, fontsize=9)
    ax_traj.set_ylabel('Value', color=FG, fontsize=9)
    ax_traj.set_title('Hailstone Trajectories', fontsize=11, fontweight='bold', color=FG)
    ax_traj.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # ── Row 0, right: Detection strength (# sig metrics vs random) ──
    ax_det = fig.add_subplot(gs[0, 2:])
    _dark_ax(ax_det)

    alpha = 0.05 / max(len(all_metric_names), 1)
    det_counts = []
    for enc in encodings:
        n_sig = 0
        for km in all_metric_names:
            if km not in all_metrics['random'] or km not in all_metrics[enc]:
                continue
            r_vals = all_metrics['random'][km]
            e_vals = all_metrics[enc][km]
            if len(r_vals) < 2 or len(e_vals) < 2:
                continue
            d = cohens_d(e_vals, r_vals)
            _, p = stats.ttest_ind(e_vals, r_vals, equal_var=False)
            if abs(d) > 0.8 and p < alpha:
                n_sig += 1
        det_counts.append(n_sig)

    bars = ax_det.barh(range(len(encodings)), det_counts,
                       color=[enc_colors[e] for e in encodings],
                       alpha=0.85, edgecolor='#333')
    ax_det.set_yticks(range(len(encodings)))
    ax_det.set_yticklabels([e.replace('_', ' ') for e in encodings],
                           fontsize=8, color=FG)
    ax_det.set_xlabel('Significant metrics vs random', color=FG, fontsize=9)
    ax_det.set_title('Collatz Encodings: Detection Strength', fontsize=11,
                     fontweight='bold', color=FG)
    ax_det.invert_yaxis()
    for i, v in enumerate(det_counts):
        ax_det.text(v + 1, i, str(v), va='center', color=FG, fontsize=8, fontweight='bold')

    # ── Row 1, left: Radar fingerprint ──
    ax_radar = fig.add_subplot(gs[1, :2], polar=True)
    ax_radar.set_facecolor(BG)

    radar_metrics = [
        'E8 Lattice:unique_roots', 'Torus T^2:coverage',
        'Tropical:linearity', 'Penrose (Quasicrystal):fivefold_balance',
        'Fisher Information:trace_fisher', 'Heisenberg (Nil):twist_rate',
        'Higher-Order Statistics:perm_entropy', 'Lorentzian:mean_velocity',
    ]
    radar_labels = [km.split(':')[1].replace('_', ' ') for km in radar_metrics]
    n_axes = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize each metric across all sources for comparability
    radar_sources = ['hailstone_large', 'parity', 'stopping_times', 'high_bits', 'random']
    radar_source_colors = ['#FF5722', '#4CAF50', '#2196F3', '#00BCD4', '#888888']

    all_vals_per_metric = {}
    for km in radar_metrics:
        combined = []
        for s in radar_sources:
            combined.extend(all_metrics[s].get(km, [0]))
        mn, mx = min(combined), max(combined)
        all_vals_per_metric[km] = (mn, mx)

    for src, c in zip(radar_sources, radar_source_colors):
        vals = []
        for km in radar_metrics:
            raw = np.mean(all_metrics[src].get(km, [0]))
            mn, mx = all_vals_per_metric[km]
            if mx - mn > 1e-12:
                vals.append((raw - mn) / (mx - mn))
            else:
                vals.append(0.5)
        vals += vals[:1]
        lw = 2.0 if src != 'random' else 1.5
        ls = '-' if src != 'random' else '--'
        ax_radar.plot(angles, vals, color=c, linewidth=lw, linestyle=ls,
                      label=src.replace('_', ' '), alpha=0.85)
        ax_radar.fill(angles, vals, color=c, alpha=0.08)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=7, color=FG)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['', '', '', ''], fontsize=6)
    ax_radar.set_title('Geometric Fingerprints (normalized)', fontsize=11,
                       fontweight='bold', color=FG, pad=20)
    ax_radar.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.35, 1.1),
                    facecolor='#222', edgecolor='#444', labelcolor=FG)
    ax_radar.tick_params(colors='#666')
    ax_radar.grid(color='#333', linewidth=0.5)
    ax_radar.spines['polar'].set_color('#444')

    # ── Row 1, right: Shuffle validation ──
    ax_shuf = fig.add_subplot(gs[1, 2:])
    _dark_ax(ax_shuf)

    shuf_encodings = ['hailstone_large', 'parity', 'stopping_times', 'high_bits']
    shuf_metrics = [
        'E8 Lattice:unique_roots', 'Torus T^2:coverage',
        'Tropical:linearity', 'Penrose (Quasicrystal):fivefold_balance',
    ]
    shuf_labels = [km.split(':')[1].replace('_', ' ') for km in shuf_metrics]

    analyzer = GeometryAnalyzer().add_all_geometries()
    bar_width = 0.18
    x = np.arange(len(shuf_metrics))

    for idx, enc in enumerate(shuf_encodings):
        ratios = []
        for km in shuf_metrics:
            data = generate_collatz_data(enc, 42)
            shuffled = data.copy()
            np.random.RandomState(99).shuffle(shuffled)
            r_real = analyzer.analyze(data)
            r_shuf = analyzer.analyze(shuffled)

            real_v, shuf_v = 0, 0
            for r in r_real.results:
                for mn, mv in r.metrics.items():
                    if f"{r.geometry_name}:{mn}" == km:
                        real_v = mv
            for r in r_shuf.results:
                for mn, mv in r.metrics.items():
                    if f"{r.geometry_name}:{mn}" == km:
                        shuf_v = mv

            if shuf_v != 0:
                ratios.append(real_v / shuf_v)
            elif real_v != 0:
                ratios.append(2.0)  # cap for display
            else:
                ratios.append(1.0)

        offset = (idx - len(shuf_encodings)/2 + 0.5) * bar_width
        bars = ax_shuf.bar(x + offset, ratios, bar_width,
                          color=enc_colors[enc], alpha=0.85, edgecolor='#333',
                          label=enc.replace('_', ' '))

    ax_shuf.axhline(y=1.0, color='#888', linestyle='--', linewidth=1, alpha=0.7)
    ax_shuf.set_ylim(0, 3.5)
    ax_shuf.set_xticks(x)
    ax_shuf.set_xticklabels(shuf_labels, fontsize=8, color=FG, rotation=20, ha='right')
    ax_shuf.set_ylabel('Real / Shuffled ratio', color=FG, fontsize=9)
    ax_shuf.set_title('Shuffle Validation (ratio ≠ 1 → ordering matters)',
                      fontsize=11, fontweight='bold', color=FG)
    ax_shuf.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG,
                   loc='upper right')

    # ── Row 2, left: Pairwise encoding matrix ──
    ax_mat = fig.add_subplot(gs[2, :2])
    _dark_ax(ax_mat)

    n_enc = len(encodings)
    mat = np.zeros((n_enc, n_enc))
    for e1, e2, n_sig, km, d, p in pair_results:
        i1, i2 = encodings.index(e1), encodings.index(e2)
        mat[i1, i2] = n_sig
        mat[i2, i1] = n_sig
    im = ax_mat.imshow(mat, cmap='YlOrRd', interpolation='nearest', vmin=0)
    ax_mat.set_xticks(range(n_enc))
    ax_mat.set_yticks(range(n_enc))
    short_names = [e.replace('hailstone_', 'hail_').replace('_', '\n') for e in encodings]
    ax_mat.set_xticklabels(short_names, fontsize=7, color=FG, rotation=0, ha='center')
    ax_mat.set_yticklabels(short_names, fontsize=7, color=FG)
    for i in range(n_enc):
        for j in range(n_enc):
            if i != j:
                ax_mat.text(j, i, f'{int(mat[i,j])}', ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color='white' if mat[i,j] > 50 else 'black')
    ax_mat.set_title('Pairwise Encoding Distinguishability', fontsize=11,
                     fontweight='bold', color=FG)
    cb = plt.colorbar(im, ax=ax_mat, shrink=0.8)
    cb.ax.tick_params(colors=FG)

    # ── Row 2, right: 3n+1 vs 5n+1 and starting magnitude ──
    ax_comp = fig.add_subplot(gs[2, 2:])
    _dark_ax(ax_comp)

    # 3n+1 vs 5n+1 top discriminators
    alpha_var = 0.05 / max(len(all_metric_names), 1)
    var_results = []
    for km in all_metric_names:
        if km not in all_metrics['hailstone_large'] or km not in all_metrics['syracuse_variant']:
            continue
        v1 = all_metrics['hailstone_large'][km]
        v2 = all_metrics['syracuse_variant'][km]
        if len(v1) < 2 or len(v2) < 2:
            continue
        d = cohens_d(v1, v2)
        _, p = stats.ttest_ind(v1, v2, equal_var=False)
        if abs(d) > 0.8 and p < alpha_var:
            var_results.append((km, d, p))
    var_results.sort(key=lambda x: -abs(x[1]))
    top_var = var_results[:10]

    y_pos = np.arange(len(top_var))
    d_vals = [d for _, d, _ in top_var]
    colors_bar = ['#E91E63' if d > 0 else '#2196F3' for d in d_vals]
    ax_comp.barh(y_pos, d_vals, color=colors_bar, alpha=0.85, edgecolor='#333')
    ax_comp.set_yticks(y_pos)
    metric_labels = [km.split(':')[1].replace('_', ' ')[:25] for km, _, _ in top_var]
    ax_comp.set_yticklabels(metric_labels, fontsize=7, color=FG)
    ax_comp.set_xlabel("Cohen's d", color=FG, fontsize=9)
    ax_comp.set_title('3n+1 vs 5n+1: Top Discriminators', fontsize=11,
                      fontweight='bold', color=FG)
    ax_comp.invert_yaxis()
    ax_comp.axvline(x=0, color='#666', linewidth=0.8)

    fig.suptitle('Collatz (3n+1) Sequences: Exotic Geometry Analysis',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'collatz.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved collatz.png")
    plt.close(fig)


if __name__ == '__main__':
    all_metrics, all_metric_names, encodings, comparisons, pair_results = main()
    make_figure(all_metrics, all_metric_names, encodings, comparisons, pair_results)
