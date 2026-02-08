#!/usr/bin/env python3
"""
Investigation: Deep Collatz Geometric Exploration.

Follows up on collatz.py with five deeper questions:

1. (2k+1)n+1 Family Phase Transition — do geometric signatures change sharply
   at the convergent/divergent boundary (k=1 vs k>=2)?
2. Parity Bit Resolution — at what modular resolution does sequential structure
   emerge in Collatz trajectories?
3. Delay Embedding — which lag τ captures the most Collatz structure?
4. Bitplane Decomposition — which bit planes carry geometric signal?
5. Tropical Slope Analysis — do odd-step slopes match theoretical log₂(2k+1)?

Total budget: ~875 analyzer calls, estimated 3-5 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, TropicalGeometry
from exotic_geometry_framework import delay_embed, bitplane_extract
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 2000


# =============================================================================
# TRAJECTORY GENERATORS
# =============================================================================

def variant_step(n, k):
    """Single step for the (2k+1)n+1 family.
    k=1 → 3n+1 (Collatz), k=2 → 5n+1, k=3 → 7n+1, etc.
    """
    if n % 2 == 0:
        return n // 2
    return (2 * k + 1) * n + 1


def variant_trajectory(start, k, max_steps=100000, cap=10**15):
    """Trajectory for (2k+1)n+1 variant. Stops at 1, max_steps, or cap."""
    n = start
    traj = [n]
    while n != 1 and len(traj) < max_steps:
        n = variant_step(n, k)
        traj.append(n)
        if n > cap:
            break
    return traj


def generate_variant_data(k, trial_seed, size=DATA_SIZE):
    """Generate byte data from (2k+1)n+1 trajectory, chaining starts."""
    rng = np.random.RandomState(trial_seed)
    traj = []
    while len(traj) < size:
        start = rng.randint(10**6, 10**8)
        seg = variant_trajectory(start, k)
        traj.extend(seg)
    arr = np.array(traj[:size], dtype=np.uint64)
    return (arr % 256).astype(np.uint8)


def generate_hailstone_large(trial_seed, size=DATA_SIZE):
    """Hailstone large (3n+1) from collatz.py — canonical encoding."""
    rng = np.random.RandomState(trial_seed)
    traj = []
    while len(traj) < size:
        start = rng.randint(10**9, 10**10)
        seg = variant_trajectory(start, k=1)
        traj.extend(seg)
    arr = np.array(traj[:size], dtype=np.uint64)
    return (arr % 256).astype(np.uint8)


def generate_random(trial_seed, size=DATA_SIZE):
    """Uniform random baseline."""
    return np.random.RandomState(trial_seed).randint(0, 256, size, dtype=np.uint8)


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


def count_significant(metrics_a, metrics_b, metric_names, n_total_tests):
    """Count metrics significantly different between two groups (|d|>0.8, Bonferroni)."""
    alpha = 0.05 / max(n_total_tests, 1)
    sig = 0
    for km in metric_names:
        va = metrics_a.get(km, [])
        vb = metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb)
        if abs(d) > 0.8 and p < alpha:
            sig += 1
    return sig


# =============================================================================
# DIRECTION 1: (2k+1)n+1 FAMILY PHASE TRANSITION
# =============================================================================

def direction1_family_transition(analyzer, metric_names):
    """Test k=1..8 for geometric phase transition at convergent/divergent boundary."""
    print("\n" + "=" * 78)
    print("DIRECTION 1: (2k+1)n+1 Family Phase Transition")
    print("=" * 78)
    print("k=1 → 3n+1 (converges), k≥2 → diverges (growth ≥ 5/2 > 2)")
    print(f"Testing k=1..8, {N_TRIALS} trials each\n")

    # 5 key metrics to track across k
    track_metrics = [
        'E8 Lattice:unique_roots',
        'Torus T^2:coverage',
        'Tropical:linearity',
        'Fisher Information:trace_fisher',
        'Heisenberg (Nil):twist_rate',
    ]

    k_values = list(range(1, 9))
    k_metrics = {}  # k -> {metric_name: [values]}
    k_sig_vs_random = {}  # k -> count of significant metrics vs random

    # Collect random baseline
    print("  Collecting random baseline...", flush=True)
    random_metrics = defaultdict(list)
    for trial in range(N_TRIALS):
        data = generate_random(trial + 9000)
        results = analyzer.analyze(data)
        for r in results.results:
            for mn, mv in r.metrics.items():
                random_metrics[f"{r.geometry_name}:{mn}"].append(mv)

    for k in k_values:
        label = f"({2*k+1})n+1"
        print(f"  k={k} {label}...", end=" ", flush=True)
        k_metrics[k] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = generate_variant_data(k, trial + 1000 * k)
            results = analyzer.analyze(data)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    k_metrics[k][f"{r.geometry_name}:{mn}"].append(mv)

        n_sig = count_significant(k_metrics[k], random_metrics, metric_names,
                                  len(k_values) * len(metric_names))
        k_sig_vs_random[k] = n_sig
        conv = "converges" if k == 1 else "DIVERGES"
        print(f"{n_sig} sig metrics vs random  ({conv})")

    # Track key metrics means and stds across k
    print(f"\n  Key metrics across k:")
    print(f"  {'Metric':<35} " + "".join(f"{'k='+str(k):>10}" for k in k_values))
    print(f"  {'-'*35} " + "-" * (10 * len(k_values)))

    tracked_data = {}  # metric -> (means, stds)
    for km in track_metrics:
        means = []
        stds = []
        for k in k_values:
            vals = k_metrics[k].get(km, [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        tracked_data[km] = (means, stds)
        short = km.split(':')[1]
        row = f"  {short:<35} " + "".join(f"{m:>10.3f}" for m in means)
        print(row)

    # Adjacent-pair jumps
    print(f"\n  Adjacent-pair jumps (k→k+1):")
    for km in track_metrics:
        means = tracked_data[km][0]
        short = km.split(':')[1]
        jumps = []
        for i in range(len(means) - 1):
            if means[i] != 0:
                jumps.append((means[i+1] - means[i]) / abs(means[i]) * 100)
            else:
                jumps.append(0)
        # Highlight the k=1→k=2 boundary
        boundary = jumps[0] if jumps else 0
        print(f"  {short:<35} k=1→2: {boundary:+.1f}%  "
              f"max jump: {max(abs(j) for j in jumps):.1f}%")

    return {
        'k_values': k_values,
        'k_sig_vs_random': k_sig_vs_random,
        'tracked_data': tracked_data,
        'track_metrics': track_metrics,
    }


# =============================================================================
# DIRECTION 2: PARITY BIT RESOLUTION
# =============================================================================

def direction2_parity_resolution(analyzer, metric_names):
    """Test parity at mod 2, 4, 8, 16, 32 — real vs shuffled."""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Parity Bit Resolution")
    print("=" * 78)
    print("At what modular resolution does sequential structure emerge?")
    print(f"Testing mod 2,4,8,16,32 — real vs shuffled, {N_TRIALS} trials each\n")

    mod_levels = [2, 4, 8, 16, 32]
    results_data = {}  # mod -> {'real': {metric: [vals]}, 'shuffled': {metric: [vals]}}

    for mod in mod_levels:
        print(f"  mod {mod}...", end=" ", flush=True)
        real_metrics = defaultdict(list)
        shuf_metrics = defaultdict(list)

        for trial in range(N_TRIALS):
            # Generate hailstone trajectory, take values mod N, scale to bytes
            rng = np.random.RandomState(trial + 2000)
            traj = []
            while len(traj) < DATA_SIZE:
                start = rng.randint(10**9, 10**10)
                seg = variant_trajectory(start, k=1)
                traj.extend(seg)
            traj = traj[:DATA_SIZE]
            # mod N, then scale to [0, 255]
            arr = np.array(traj, dtype=np.uint64)
            modded = (arr % mod).astype(np.float64)
            scaled = (modded / max(mod - 1, 1) * 255).astype(np.uint8)

            # Shuffled version
            shuffled = scaled.copy()
            np.random.RandomState(trial + 5000).shuffle(shuffled)

            r_real = analyzer.analyze(scaled)
            r_shuf = analyzer.analyze(shuffled)

            for r in r_real.results:
                for mn, mv in r.metrics.items():
                    real_metrics[f"{r.geometry_name}:{mn}"].append(mv)
            for r in r_shuf.results:
                for mn, mv in r.metrics.items():
                    shuf_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        n_total = len(mod_levels) * len(metric_names)
        n_sig = count_significant(real_metrics, shuf_metrics, metric_names, n_total)
        results_data[mod] = {'real': dict(real_metrics), 'shuffled': dict(shuf_metrics),
                             'n_sig': n_sig}
        print(f"{n_sig} metrics where ordering matters")

    # Summary ratios for key metrics
    ratio_metrics = [
        'E8 Lattice:unique_roots', 'Tropical:linearity',
        'Heisenberg (Nil):twist_rate', 'Fisher Information:trace_fisher',
    ]
    print(f"\n  Real/Shuffled ratio for key metrics:")
    print(f"  {'Metric':<35} " + "".join(f"{'mod '+str(m):>10}" for m in mod_levels))
    print(f"  {'-'*35} " + "-" * (10 * len(mod_levels)))

    ratio_data = {}  # metric -> [ratios per mod]
    for km in ratio_metrics:
        ratios = []
        for mod in mod_levels:
            rv = np.mean(results_data[mod]['real'].get(km, [0]))
            sv = np.mean(results_data[mod]['shuffled'].get(km, [0]))
            if sv != 0:
                ratios.append(rv / sv)
            elif rv != 0:
                ratios.append(2.0)
            else:
                ratios.append(1.0)
        ratio_data[km] = ratios
        short = km.split(':')[1]
        row = f"  {short:<35} " + "".join(f"{r:>10.3f}" for r in ratios)
        print(row)

    return {
        'mod_levels': mod_levels,
        'results_data': results_data,
        'ratio_data': ratio_data,
        'ratio_metrics': ratio_metrics,
    }


# =============================================================================
# DIRECTION 3: DELAY EMBEDDING
# =============================================================================

def direction3_delay_embedding(analyzer, metric_names):
    """Apply delay_embed at various tau, count significant metrics vs random."""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Delay Embedding")
    print("=" * 78)
    print("Which lag τ captures the most Collatz structure?")
    print(f"Testing τ = 1,2,3,5,8,13,21 — {N_TRIALS} trials each\n")

    tau_values = [1, 2, 3, 5, 8, 13, 21]
    tau_results = {}  # tau -> {metric: [vals]}
    random_tau_results = {}  # tau -> {metric: [vals]}

    # For each tau, generate data, delay-embed, analyze
    for tau in tau_values:
        print(f"  τ={tau}...", end=" ", flush=True)
        tau_metrics = defaultdict(list)
        rand_metrics = defaultdict(list)

        for trial in range(N_TRIALS):
            # Collatz data — generate extra to account for delay reduction
            raw_size = DATA_SIZE + tau + 100
            data = generate_hailstone_large(trial + 3000, size=raw_size)
            embedded = delay_embed(data, tau)
            # Trim to standard size
            embedded = embedded[:DATA_SIZE]

            results = analyzer.analyze(embedded)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    tau_metrics[f"{r.geometry_name}:{mn}"].append(mv)

            # Random baseline with same delay embedding
            rand_data = generate_random(trial + 4000, size=raw_size)
            rand_embedded = delay_embed(rand_data, tau)
            rand_embedded = rand_embedded[:DATA_SIZE]

            r_rand = analyzer.analyze(rand_embedded)
            for r in r_rand.results:
                for mn, mv in r.metrics.items():
                    rand_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        n_total = len(tau_values) * len(metric_names)
        n_sig = count_significant(tau_metrics, rand_metrics, metric_names, n_total)
        tau_results[tau] = {'metrics': dict(tau_metrics), 'n_sig': n_sig}
        random_tau_results[tau] = dict(rand_metrics)
        print(f"{n_sig} sig metrics vs random")

    # Find optimal tau
    best_tau = max(tau_values, key=lambda t: tau_results[t]['n_sig'])
    print(f"\n  Optimal τ = {best_tau} ({tau_results[best_tau]['n_sig']} significant metrics)")

    return {
        'tau_values': tau_values,
        'tau_results': tau_results,
        'best_tau': best_tau,
    }


# =============================================================================
# DIRECTION 4: BITPLANE DECOMPOSITION
# =============================================================================

def direction4_bitplane(analyzer, metric_names):
    """Extract each bit plane (0-7) and test vs random."""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Bitplane Decomposition")
    print("=" * 78)
    print("Which bit planes carry geometric signal?")
    print(f"Testing planes 0-7 (LSB to MSB), {N_TRIALS} trials each\n")

    planes = list(range(8))
    plane_results = {}  # plane -> {metric: [vals]}

    # Generate base data at 16000 bytes so bitplane_extract yields 2000 bytes
    base_size = DATA_SIZE * 8  # 16000

    # Shared random baseline
    print("  Collecting random baseline...", flush=True)
    random_metrics = defaultdict(list)
    for trial in range(N_TRIALS):
        rand_data = generate_random(trial + 6000, size=base_size)
        # Pick plane 0 for random — all planes of uniform random are equally random
        extracted = bitplane_extract(rand_data, 0)
        extracted = extracted[:DATA_SIZE]
        results = analyzer.analyze(extracted)
        for r in results.results:
            for mn, mv in r.metrics.items():
                random_metrics[f"{r.geometry_name}:{mn}"].append(mv)

    for plane in planes:
        print(f"  plane {plane} ({'LSB' if plane == 0 else 'MSB' if plane == 7 else ''})...",
              end=" ", flush=True)
        plane_metrics = defaultdict(list)

        for trial in range(N_TRIALS):
            data = generate_hailstone_large(trial + 7000, size=base_size)
            extracted = bitplane_extract(data, plane)
            extracted = extracted[:DATA_SIZE]

            results = analyzer.analyze(extracted)
            for r in results.results:
                for mn, mv in r.metrics.items():
                    plane_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        n_total = len(planes) * len(metric_names)
        n_sig = count_significant(plane_metrics, random_metrics, metric_names, n_total)
        plane_results[plane] = {'metrics': dict(plane_metrics), 'n_sig': n_sig}
        print(f"{n_sig} sig metrics vs random")

    return {
        'planes': planes,
        'plane_results': plane_results,
    }


# =============================================================================
# DIRECTION 5: TROPICAL SLOPE ANALYSIS
# =============================================================================

def direction5_tropical_slopes():
    """Analyze per-step slopes in log₂ space for 3n+1 vs 5n+1."""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Tropical Slope Analysis")
    print("=" * 78)
    print("Odd-step slopes should peak at log₂(2k+1): 1.585 for 3n+1, 2.322 for 5n+1")
    print("Even-step slopes should all be -1.0\n")

    trop = TropicalGeometry(input_scale='auto')

    variants = {
        '3n+1': 1,
        '5n+1': 2,
    }
    theoretical = {
        '3n+1': np.log2(3),  # 1.585
        '5n+1': np.log2(5),  # 2.322
    }

    slope_data = {}  # label -> {'odd_slopes': [...], 'even_slopes': [...]}

    for label, k in variants.items():
        print(f"  {label}...", end=" ", flush=True)
        odd_slopes = []
        even_slopes = []

        # Generate many trajectories, filter for n > 100
        rng = np.random.RandomState(8000)
        n_trajs = 50
        for _ in range(n_trajs):
            start = rng.randint(10**6, 10**8)
            traj = variant_trajectory(start, k, max_steps=50000)

            # Compute log2 values and per-step slopes
            for i in range(len(traj) - 1):
                curr, nxt = traj[i], traj[i + 1]
                if curr <= 100 or nxt <= 0 or curr <= 0:
                    continue
                slope = np.log2(nxt) - np.log2(curr)
                if curr % 2 == 1:  # odd step: applied (2k+1)n+1
                    odd_slopes.append(slope)
                else:  # even step: applied n/2
                    even_slopes.append(slope)

        odd_slopes = np.array(odd_slopes)
        even_slopes = np.array(even_slopes)
        slope_data[label] = {
            'odd_slopes': odd_slopes,
            'even_slopes': even_slopes,
        }

        # Also run TropicalGeometry on a sample trajectory for metrics
        sample_data = generate_variant_data(k, 42)
        trop_result = trop.compute_metrics(sample_data)
        trop_metrics = trop_result.metrics

        print(f"odd slopes: μ={np.mean(odd_slopes):.4f} (theory: {theoretical[label]:.4f}), "
              f"even slopes: μ={np.mean(even_slopes):.4f} (theory: -1.0)")
        print(f"    Tropical metrics: linearity={trop_metrics.get('linearity', 0):.4f}, "
              f"slope_changes={trop_metrics.get('slope_changes', 0):.0f}, "
              f"unique_slopes={trop_metrics.get('unique_slopes', 0):.0f}")

    return {
        'slope_data': slope_data,
        'theoretical': theoretical,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("DEEP COLLATZ GEOMETRIC EXPLORATION")
    print("=" * 78)
    print(f"\nParameters: {N_TRIALS} trials, {DATA_SIZE} bytes per sample")
    print("Five directions: family transition, parity resolution,")
    print("  delay embedding, bitplane decomposition, tropical slopes\n")

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names from a quick dummy run
    dummy = generate_random(0)
    dummy_results = analyzer.analyze(dummy)
    metric_names = []
    for r in dummy_results.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    metric_names = sorted(set(metric_names))

    d1 = direction1_family_transition(analyzer, metric_names)
    d2 = direction2_parity_resolution(analyzer, metric_names)
    d3 = direction3_delay_embedding(analyzer, metric_names)
    d4 = direction4_bitplane(analyzer, metric_names)
    d5 = direction5_tropical_slopes()

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\n  D1 — Family transition:")
    for k in d1['k_values']:
        n_sig = d1['k_sig_vs_random'][k]
        conv = "converges" if k == 1 else "diverges"
        print(f"    ({2*k+1})n+1: {n_sig:>3} sig metrics vs random  ({conv})")

    print(f"\n  D2 — Parity resolution (real vs shuffled sig metrics):")
    for mod in d2['mod_levels']:
        n_sig = d2['results_data'][mod]['n_sig']
        print(f"    mod {mod:>2}: {n_sig:>3} metrics where ordering matters")

    print(f"\n  D3 — Delay embedding (sig metrics vs random):")
    for tau in d3['tau_values']:
        n_sig = d3['tau_results'][tau]['n_sig']
        marker = " ← best" if tau == d3['best_tau'] else ""
        print(f"    τ={tau:>2}: {n_sig:>3} sig metrics{marker}")

    print(f"\n  D4 — Bitplane (sig metrics vs random):")
    for plane in d4['planes']:
        n_sig = d4['plane_results'][plane]['n_sig']
        tag = "LSB" if plane == 0 else ("MSB" if plane == 7 else "")
        print(f"    plane {plane}: {n_sig:>3} sig metrics  {tag}")

    print(f"\n  D5 — Tropical slopes:")
    for label in ['3n+1', '5n+1']:
        sd = d5['slope_data'][label]
        theo = d5['theoretical'][label]
        print(f"    {label}: odd slope μ={np.mean(sd['odd_slopes']):.4f} "
              f"(theory {theo:.4f}), even slope μ={np.mean(sd['even_slopes']):.4f} "
              f"(theory -1.0)")

    print(f"\n[Deep investigation complete]")

    return d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION — 6-panel layout (22x14, dark theme)
# =============================================================================

def make_figure(d1, d2, d3, d4, d5):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Top-left: (2k+1)n+1 metrics vs k ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    k_vals = d1['k_values']
    colors1 = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

    for idx, km in enumerate(d1['track_metrics']):
        means, stds = d1['tracked_data'][km]
        short = km.split(':')[1].replace('_', ' ')
        c = colors1[idx % len(colors1)]
        ax1.errorbar(k_vals, means, yerr=stds, marker='o', markersize=4,
                     color=c, linewidth=1.5, capsize=3, label=short, alpha=0.9)

    # Mark convergent/divergent boundary
    ax1.axvline(x=1.5, color='#FF5722', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(1.55, ax1.get_ylim()[1] * 0.95, 'convergent | divergent',
             color='#FF5722', fontsize=7, va='top')

    ax1.set_xlabel('k  [(2k+1)n+1]', color=FG, fontsize=9)
    ax1.set_ylabel('Metric value (mean ± std)', color=FG, fontsize=9)
    ax1.set_title('D1: Family Phase Transition', fontsize=11, fontweight='bold', color=FG)
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([f'{2*k+1}n+1' for k in k_vals], fontsize=7, rotation=30)
    ax1.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor=FG,
               loc='best')

    # ── Top-center: Parity resolution ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    mod_levels = d2['mod_levels']
    x_parity = np.arange(len(mod_levels))
    bar_width = 0.18
    colors2 = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']

    for idx, km in enumerate(d2['ratio_metrics']):
        ratios = d2['ratio_data'][km]
        short = km.split(':')[1].replace('_', ' ')
        c = colors2[idx % len(colors2)]
        offset = (idx - len(d2['ratio_metrics']) / 2 + 0.5) * bar_width
        ax2.bar(x_parity + offset, ratios, bar_width, color=c, alpha=0.85,
                edgecolor='#333', label=short)

    ax2.axhline(y=1.0, color='#888', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xticks(x_parity)
    ax2.set_xticklabels([f'mod {m}' for m in mod_levels], fontsize=8, color=FG)
    ax2.set_ylabel('Real / Shuffled ratio', color=FG, fontsize=9)
    ax2.set_title('D2: Parity Resolution', fontsize=11, fontweight='bold', color=FG)
    ax2.legend(fontsize=6, facecolor='#222', edgecolor='#444', labelcolor=FG,
               loc='best')

    # ── Top-right: Delay embedding ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    tau_vals = d3['tau_values']
    sig_counts = [d3['tau_results'][t]['n_sig'] for t in tau_vals]

    bars3 = ax3.bar(range(len(tau_vals)), sig_counts, color='#4CAF50', alpha=0.85,
                    edgecolor='#333')
    # Highlight best
    best_idx = tau_vals.index(d3['best_tau'])
    bars3[best_idx].set_color('#FF5722')
    bars3[best_idx].set_alpha(1.0)

    ax3.set_xticks(range(len(tau_vals)))
    ax3.set_xticklabels([f'τ={t}' for t in tau_vals], fontsize=8, color=FG)
    ax3.set_ylabel('Significant metrics vs random', color=FG, fontsize=9)
    ax3.set_title('D3: Delay Embedding', fontsize=11, fontweight='bold', color=FG)
    for i, v in enumerate(sig_counts):
        ax3.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=8, fontweight='bold')

    # ── Bottom-left: Bitplane ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    planes = d4['planes']
    plane_sigs = [d4['plane_results'][p]['n_sig'] for p in planes]
    plane_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(planes)))

    ax4.bar(planes, plane_sigs, color=plane_colors, alpha=0.85, edgecolor='#333')
    ax4.set_xlabel('Bit plane (0=LSB, 7=MSB)', color=FG, fontsize=9)
    ax4.set_ylabel('Significant metrics vs random', color=FG, fontsize=9)
    ax4.set_title('D4: Bitplane Decomposition', fontsize=11, fontweight='bold', color=FG)
    ax4.set_xticks(planes)
    ax4.set_xticklabels([f'{p}\n{"LSB" if p==0 else "MSB" if p==7 else ""}' for p in planes],
                         fontsize=8)
    for i, v in enumerate(plane_sigs):
        ax4.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=8, fontweight='bold')

    # ── Bottom-center: Tropical slope histograms ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    for label, c in [('3n+1', '#E91E63'), ('5n+1', '#2196F3')]:
        odd_slopes = d5['slope_data'][label]['odd_slopes']
        # Filter to reasonable range for display
        odd_slopes = odd_slopes[(odd_slopes > -2) & (odd_slopes < 5)]
        ax5.hist(odd_slopes, bins=80, alpha=0.5, color=c, density=True,
                 label=f'{label} odd slopes')

    # Theoretical lines
    ax5.axvline(x=np.log2(3), color='#E91E63', linestyle='--', linewidth=2,
                label=f'log₂(3) = {np.log2(3):.3f}')
    ax5.axvline(x=np.log2(5), color='#2196F3', linestyle='--', linewidth=2,
                label=f'log₂(5) = {np.log2(5):.3f}')

    ax5.set_xlabel('log₂(step slope)', color=FG, fontsize=9)
    ax5.set_ylabel('Density', color=FG, fontsize=9)
    ax5.set_title('D5: Tropical Odd-Step Slopes', fontsize=11, fontweight='bold', color=FG)
    ax5.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)

    # ── Bottom-right: Summary text panel ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    lines = [
        "Deep Collatz — Key Findings",
        "",
        "D1: (2k+1)n+1 Family",
    ]

    # Phase transition summary
    k1_sig = d1['k_sig_vs_random'].get(1, 0)
    k2_sig = d1['k_sig_vs_random'].get(2, 0)
    lines.append(f"  3n+1: {k1_sig} sig | 5n+1: {k2_sig} sig metrics")
    lines.append(f"  Phase transition at k=1→2: {'SHARP' if abs(k1_sig - k2_sig) > 10 else 'gradual'}")

    lines.append("")
    lines.append("D2: Parity Resolution")
    mod_sigs = [d2['results_data'][m]['n_sig'] for m in d2['mod_levels']]
    best_mod = d2['mod_levels'][np.argmax(mod_sigs)]
    lines.append(f"  Strongest ordering: mod {best_mod} ({max(mod_sigs)} metrics)")

    lines.append("")
    lines.append("D3: Delay Embedding")
    lines.append(f"  Best τ = {d3['best_tau']} "
                 f"({d3['tau_results'][d3['best_tau']]['n_sig']} sig metrics)")

    lines.append("")
    lines.append("D4: Bitplane")
    best_plane = max(d4['planes'], key=lambda p: d4['plane_results'][p]['n_sig'])
    lines.append(f"  Most signal: plane {best_plane} "
                 f"({d4['plane_results'][best_plane]['n_sig']} sig metrics)")
    lsb_sig = d4['plane_results'][0]['n_sig']
    msb_sig = d4['plane_results'][7]['n_sig']
    lines.append(f"  LSB={lsb_sig}, MSB={msb_sig}")

    lines.append("")
    lines.append("D5: Tropical Slopes")
    for label in ['3n+1', '5n+1']:
        sd = d5['slope_data'][label]
        theo = d5['theoretical'][label]
        mu = np.mean(sd['odd_slopes'])
        lines.append(f"  {label}: μ={mu:.3f} (theory {theo:.3f})")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Deep Collatz Geometric Exploration',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'collatz_deep.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    d1, d2, d3, d4, d5 = main()
    make_figure(d1, d2, d3, d4, d5)
