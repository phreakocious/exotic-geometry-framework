#!/usr/bin/env python3
"""
Investigation: Structure vs Chaos — The Mandelbrot Sensor Test
==============================================================

Can the new 'Fractal (Mandelbrot)' geometry distinguish between:
1. Logistic Chaos (deterministic, non-algebraic)
2. Collatz 3n+1 (deterministic, algebraic-ish)
3. Prime Gaps (deterministic, algebraic)
4. Random Noise (stochastic)

These categories were previously indistinguishable ("The Cluster").
We hypothesize the Mandelbrot sensor's sensitivity to initial conditions
will separate them.

DIRECTIONS:
D1: The Cluster Confusion Matrix — Baseline check with ALL geometries.
D2: Mandelbrot Specifics — Focus on 'Fractal (Mandelbrot)' metrics.
    Does 'mean_escape_time' or 'interior_fraction' separate them?
D3: Sensitivity Sweep — Does the separation improve if we change the
    Mandelbrot max_iter? (We can't easily change it runtime without hacking,
    so we'll just analyze the standard 64-iter results deeply).
D4: Transform Sweep — Diff/Sort transforms to see if they help the Fractal sensor.
D5: The Verdict — Is the cluster cracked?
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
SEED = 42
np.random.seed(SEED)

# ==============================================================
# DATA GENERATORS
# ==============================================================

def gen_logistic_chaos(rng, size):
    x = 0.1 + 0.8 * rng.random()
    for _ in range(1000): x = 4.0 * x * (1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 4.0 * x * (1.0 - x)
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
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Baseline Confusion Matrix"""
    print("\n" + "=" * 60)
    print("D1: BASELINE CONFUSION MATRIX")
    print("=" * 60)

    conditions = {
        'Logistic': gen_logistic_chaos,
        'PrimeGaps': gen_prime_gaps,
        'Collatz': gen_collatz,
        'Random': gen_random
    }
    
    metrics = {}
    for name, gen in conditions.items():
        chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
        metrics[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(metrics)
    return dict(matrix=matrix, names=names, metrics=metrics)

def direction_2(runner, d1_metrics):
    """D2: Mandelbrot Sensor Specifics"""
    print("\n" + "=" * 60)
    print("D2: FRACTAL (MANDELBROT) SENSOR ANALYSIS")
    print("=" * 60)
    
    # We want to see if 'Fractal (Mandelbrot)' metrics separate the hard pairs
    target_geom = "Fractal (Mandelbrot)"
    
    pairs = [('Logistic', 'PrimeGaps'), ('Logistic', 'Collatz'), ('PrimeGaps', 'Collatz')]
    
    results = {}
    
    for a, b in pairs:
        print(f"\nScanning {a} vs {b} for Mandelbrot signals...")
        met_a = d1_metrics[a]
        met_b = d1_metrics[b]
        
        # Use runner.compare to get stats for all metrics
        _, findings = runner.compare(met_a, met_b)
        
        # Filter findings for Mandelbrot metrics
        mand_findings = [f for f in findings if f[0].startswith(target_geom)]
        
        if not mand_findings:
             print("  No significant Mandelbrot metrics found.")
        
        sig_count = len(mand_findings)
        
        if sig_count > 0:
            # Findings are already sorted by |d| desc
            best_m_full, best_d, best_p = mand_findings[0]
            best_m = best_m_full.split(':')[1]
            print(f"  Found {sig_count} significant Mandelbrot metrics.")
            print(f"  Best discriminator: {best_m} (d={best_d:.2f})")
        else:
            best_d = 0
            best_m = "None"
            
        results[f"{a}_vs_{b}"] = (sig_count, best_m, best_d)
        
    return dict(results=results)

def direction_3(runner):
    """D3: Mandelbrot vs Entropy (Independence Check)"""
    print("\n" + "=" * 60)
    print("D3: INDEPENDENCE CHECK (Mandelbrot vs Entropy)")
    print("=" * 60)
    
    # Generate a mixed bag of data to check correlation
    # We'll use the data from D1, flattened
    
    # We need to re-collect raw data to compute entropy manually for correlation
    # Or just rely on the existing 'Wasserstein:entropy' or similar as a proxy?
    # Let's use 'Shannon Entropy' if available, otherwise compute it.
    
    # Let's generate new chunks to be sure
    entropies = []
    mand_means = []
    
    # Mix all types
    for gen in [gen_logistic_chaos, gen_prime_gaps, gen_random]:
        for rng in runner.trial_rngs()[:10]: # 10 trials each
            data = gen(rng, runner.data_size)
            
            # Compute entropy
            _, counts = np.unique(data, return_counts=True)
            p = counts / len(data)
            ent = -np.sum(p * np.log2(p))
            entropies.append(ent)
            
            # Compute Mandelbrot mean escape (manually to avoid full analyzer overhead)
            # This is a bit redundant but safe.
            # Actually, let's use the full analyzer on single chunks
            res = runner.collect([data])
            # Extract mean_escape_time
            # res is dict {metric: [val]}
            # keys are 'Fractal (Mandelbrot):mean_escape_time'
            k = 'Fractal (Mandelbrot):mean_escape_time'
            if k in res:
                mand_means.append(res[k][0])
            else:
                mand_means.append(0)
                
    if not mand_means or not entropies:
        return dict(corr=0)

    # Correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(entropies, mand_means)
    print(f"  Correlation (Entropy vs Mean Escape): r = {corr:.3f}")
    
    return dict(corr=corr, entropies=entropies, mand_means=mand_means)

def direction_4(runner):
    """D4: Sensitivity to Transform"""
    print("\n" + "=" * 60)
    print("D4: TRANSFORM SENSITIVITY")
    print("=" * 60)
    
    # Does 1st difference help the Mandelbrot sensor see structure?
    
    def apply_diff(gen_fn):
        def wrapper(rng, size):
            d = gen_fn(rng, size + 1)
            return np.abs(np.diff(d.astype(int))).astype(np.uint8)
        return wrapper
        
    log_diff = apply_diff(gen_logistic_chaos)
    prime_diff = apply_diff(gen_prime_gaps)
    
    chunks_log = [log_diff(rng, runner.data_size) for rng in runner.trial_rngs()]
    chunks_prime = [prime_diff(rng, runner.data_size) for rng in runner.trial_rngs()]
    
    m_log = runner.collect(chunks_log)
    m_prime = runner.collect(chunks_prime)
    
    k = 'Fractal (Mandelbrot):mean_escape_time'
    if k in m_log:
        d = runner.cohens_d(m_log[k], m_prime[k])
        print(f"  Logistic(Diff) vs Prime(Diff) [Mean Escape]: d = {d:.2f}")
    else:
        d = 0
        print("  Mandelbrot metric not found.")
        
    return dict(d_diff=d)

def direction_5(runner):
    """D5: Unused"""
    return {} # Placeholder

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(3, "Structure vs Chaos: Mandelbrot Test")

    # D1: Heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Confusion Matrix")
    
    # D2: Bar chart of d-values for Mandelbrot sensor
    labels = list(d2['results'].keys())
    d_vals = [abs(v[2]) for v in d2['results'].values()]
    runner.plot_bars(axes[1], labels, d_vals, "D2: Mandelbrot Sensor Sensitivity (Cohen's d)")
    
    # D3: Scatter Entropy vs Escape
    axes[2].scatter(d3['entropies'], d3['mand_means'], alpha=0.7, c='cyan')
    axes[2].set_xlabel("Shannon Entropy")
    axes[2].set_ylabel("Mandelbrot Mean Escape")
    axes[2].set_title(f"D3: Correlation r={d3['corr']:.2f}")
    
    runner.save(fig, "mandelbrot_test")

# ==============================================================
# MAIN
# ==============================================================
def main():
    runner = Runner("Mandelbrot Test", mode="1d")
    
    d1 = direction_1(runner)
    d2 = direction_2(runner, d1['metrics'])
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)
    
    make_figure(runner, d1, d2, d3, d4, d5)
    
    # Verdict
    log_prime_res = d2['results'].get('Logistic_vs_PrimeGaps', (0,0,0))
    runner.print_summary({
        'Verdict': 'Success' if log_prime_res[0] > 0 else 'Failure',
        'Log vs Prime': f"{log_prime_res[0]} sig metrics (Best d={log_prime_res[2]:.2f})",
        'Correlation': f"r={d3['corr']:.2f} (Independence check)"
    })

if __name__ == "__main__":
    main()
