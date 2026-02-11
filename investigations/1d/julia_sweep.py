#!/usr/bin/env python3
"""
Investigation: The Julia Sweep — Tunable Sensor Resonance
=========================================================

We test if different Julia set "landscapes" act as resonant filters for 
different types of structure.

DIRECTIONS:
D1: The Dendrite Resonance (c = i) — Infinitely thin lightning bolt.
D2: The Rabbit Resonance (c = -0.123 + 0.745i) — 3-fold symmetry.
D3: The San Marco Resonance (c = -0.75) — Real-axis dragon.
D4: The Siegel Disk Resonance (c = -0.391 - 0.587i) — Rotational islands.
D5: Best Fit Summary — Which c identifies which data best?
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))))
from exotic_geometry_framework import FractalJuliaGeometry
from tools.investigation_runner import Runner

# ==============================================================
# DATA GENERATORS
# ==============================================================

def gen_logistic_chaos(rng, size):
    x = 0.1 + 0.8 * rng.random()
    for _ in range(1000): x = 4.0 * x * (1 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 4.0 * x * (1 - x)
        vals[i] = int(x * 255)
    return vals

def _sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

_PRIMES = _sieve_primes(1_000_000)

def gen_prime_gaps(rng, size):
    max_start = len(_PRIMES) - size - 100
    start = rng.integers(0, max_start)
    gaps = np.diff(_PRIMES[start : start + size + 1])
    return np.clip(gaps, 0, 255).astype(np.uint8)

def gen_collatz(rng, size):
    n = int(rng.integers(10_000, 100_000_000))
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        vals[i] = n % 256
        if n % 2 == 0: n //= 2
        else: n = 3 * n + 1
    return vals

def gen_random(rng, size):
    return rng.integers(0, 256, size, dtype=np.uint8)

# ==============================================================
# INVESTIGATION LOGIC
# ==============================================================

def _compare_julia(data_a, data_b, n_metrics):
    """Compare two metric dicts with Bonferroni correction over n_metrics."""
    bonf_alpha = 0.05 / n_metrics
    findings = []
    for m in data_a:
        a = np.array(data_a[m])
        b = np.array(data_b[m])
        if len(a) < 3 or len(b) < 3:
            continue
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
        if not np.isfinite(d):
            continue
        _, p = sp_stats.ttest_ind(a, b, equal_var=False)
        if p < bonf_alpha and abs(d) > 0.8:
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return len(findings), findings


def run_julia_test(runner, c_real, c_imag, title):
    print(f"\nTesting {title} (c = {c_real:+.2f}{c_imag:+.2f}i)...")

    # Instantiate custom Julia geometry
    geom = FractalJuliaGeometry(c_real=c_real, c_imag=c_imag)

    conditions = {
        'Chaos': gen_logistic_chaos,
        'Primes': gen_prime_gaps,
        'Collatz': gen_collatz,
        'Random': gen_random
    }

    # Compute this specific geometry's metrics (bare keys, no runner prefix)
    results = {}
    for name, gen_fn in conditions.items():
        chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
        metrics_list = [geom.compute_metrics(c).metrics for c in chunks]
        pivoted = {k: [m[k] for m in metrics_list] for k in metrics_list[0].keys()}
        results[name] = pivoted

    n_metrics = len(results['Chaos'])  # 5 Julia metrics

    # Compare Chaos vs others
    comparison = {}
    print(f"  Means for {title}:")
    for name, m_dict in results.items():
        avg_esc = np.mean(m_dict['mean_escape_time'])
        avg_ent = np.mean(m_dict['escape_entropy'])
        print(f"    {name:10s}: escape={avg_esc:.2f}, entropy={avg_ent:.2f}")

    for other in ['Primes', 'Collatz', 'Random']:
        n_sig, findings = _compare_julia(results['Chaos'], results[other], n_metrics)
        comparison[other] = (n_sig, findings)
        print(f"  Chaos vs {other:7s}: {n_sig}/{n_metrics} sig metrics")
        if findings:
            print(f"    Best: {findings[0][0]} (d={findings[0][1]:.2f})")

    return results, comparison

def main():
    runner = Runner("Julia Sweep", mode="1d")
    
    # Direction Parameters
    PARAMS = [
        (0.0, 1.0, "D1: The Dendrite"),
        (-0.12, 0.75, "D2: The Rabbit"),
        (-0.75, 0.0, "D3: San Marco"),
        (-0.39, -0.59, "D4: Siegel Disk"),
        (0.3, 0.0, "D5: Cantor Dust")
    ]
    
    all_results = []
    
    for r, i, title in PARAMS:
        res, comp = run_julia_test(runner, r, i, title)
        all_results.append((title, res, comp))
        
    # FIGURE
    fig, axes = runner.create_figure(5, "The Julia Sweep: Multi-Resonant Sensors")
    
    for idx, (title, res, comp) in enumerate(all_results):
        # Plot bar chart of max d-value for Chaos vs Primes for this sensor
        p_res = comp.get('Primes', (0, []))
        if p_res[1]:
            # findings[0] is (metric, d, p)
            best_d = abs(p_res[1][0][1])
            best_m = p_res[1][0][0].split(':')[-1]
        else:
            best_d = 0
            best_m = "None"
            
        # Plot all 3 comparisons for this sensor
        names = ['Primes', 'Collatz', 'Random']
        d_vals = [abs(comp[n][1][0][1]) if comp[n][1] else 0 for n in names]
        runner.plot_bars(axes[idx], names, d_vals, f"{title}\nMax d (vs Chaos)")
        
    runner.save(fig, "julia_sweep")
    
    # Summary
    summary = {}
    for title, res, comp in all_results:
        summary[title] = f"Chaos/Primes d={abs(comp['Primes'][1][0][1]):.1f}" if comp['Primes'][1] else "No sig"
        
    runner.print_summary(summary)

if __name__ == "__main__":
    main()
