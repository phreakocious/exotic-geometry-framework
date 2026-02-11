#!/usr/bin/env python3
"""
Investigation: Structure vs Chaos — Cracking the Indistinguishable Cluster
==========================================================================

The Meta-Investigation revealed a "blind spot": the framework cannot easily
distinguish between:
1. Logistic Chaos (deterministic, non-algebraic)
2. Collatz 3n+1 sequences (deterministic, algebraic-ish)
3. Prime Gaps (deterministic, algebraic)
4. Digits of Pi (deterministic, transcendental)

To the current "sensors," these all look like "structured aperiodic noise."
This investigation acts as a "sensor calibration" mission to find the specific
metric or transformation that separates them.

DIRECTIONS:
D1: The Confusion Matrix — Confirm the baseline indistinguishability.
D2: The "Rare Sensor" Hunt — Scan all metrics for *any* single feature
    that significantly separates these pairs (Cohen's d > 1.5).
D3: Transform Sweep — Apply transforms (Diff, Modulo, Parity) before analysis.
D4: Local Variance — Do they differ in how much their geometry fluctuates
    over time (stationarity)?
D5: Algebraic Detectors — Test specific candidates for the "missing sensor"
    (e.g., GCD-based metrics).
"""

import sys
import os
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))))
from tools.investigation_runner import Runner

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
    """D1: Confusion Matrix"""
    print("
" + "=" * 60)
    print("D1: CONFUSION MATRIX")
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
    """D2: Rare Sensor Hunt"""
    print("
" + "=" * 60)
    print("D2: THE 'RARE SENSOR' HUNT")
    print("=" * 60)
    
    pair = ('Logistic', 'PrimeGaps')
    met_a = d1_metrics[pair[0]]
    met_b = d1_metrics[pair[1]]
    
    _, findings = runner.compare(met_a, met_b)
    
    print(f"
Top 10 Sensors for {pair[0]} vs {pair[1]}:")
    for m, d, p in findings[:10]:
        print(f"  {m:40s} d={d:+.2f} (p={p:.1e})")
        
    return dict(findings=findings)

def direction_3(runner):
    """D3: Transform Sweep"""
    print("
" + "=" * 60)
    print("D3: TRANSFORM SWEEP")
    print("=" * 60)
    
    def apply_diff(gen_fn):
        def wrapper(rng, size):
            d = gen_fn(rng, size + 1)
            return np.abs(np.diff(d.astype(int))).astype(np.uint8)
        return wrapper

    results = {}
    for t_name, wrapper in [('Raw', lambda x: x), ('Diff', apply_diff)]:
        print(f"Testing {t_name}...")
        gen_a = wrapper(gen_logistic_chaos)
        gen_b = wrapper(gen_prime_gaps)
        
        m_a = runner.collect([gen_a(rng, runner.data_size) for rng in runner.trial_rngs()])
        m_b = runner.collect([gen_b(rng, runner.data_size) for rng in runner.trial_rngs()])
        
        ns, _ = runner.compare(m_a, m_b)
        results[t_name] = ns
        print(f"  {t_name} sig: {ns}")
        
    return dict(results=results)

def direction_4(runner):
    """D4: Stationarity (Local Entropy Variance)"""
    print("
" + "=" * 60)
    print("D4: STATIONARITY")
    print("=" * 60)
    
    def get_stationarity(data):
        chunks = np.array_split(data, 4)
        ents = []
        for c in chunks:
            _, cnts = np.unique(c, return_counts=True)
            p = cnts / len(c)
            ents.append(-np.sum(p * np.log2(p + 1e-9)))
        return np.var(ents)
        
    stats_res = {}
    for name, gen in [('Logistic', gen_logistic_chaos), ('PrimeGaps', gen_prime_gaps)]:
        scores = [get_stationarity(gen(rng, runner.data_size)) for rng in runner.trial_rngs()]
        stats_res[name] = scores
        
    d_val = runner.cohens_d(stats_res['Logistic'], stats_res['PrimeGaps'])
    print(f"Stationarity d-value: {d_val:.2f}")
    return dict(d_val=d_val, stats=stats_res)

def direction_5(runner):
    """D5: Algebraic Sensors (GCD)"""
    print("
" + "=" * 60)
    print("D5: ALGEBRAIC SENSOR (GCD)")
    print("=" * 60)
    
    def get_gcd_mean(data):
        d = np.maximum(data.astype(np.int64), 1)
        return np.mean(np.gcd(d[:-1], d[1:]))
        
    gcd_res = {}
    for name, gen in [('Logistic', gen_logistic_chaos), ('PrimeGaps', gen_prime_gaps), ('Collatz', gen_collatz)]:
        scores = [get_gcd_mean(gen(rng, runner.data_size)) for rng in runner.trial_rngs()]
        gcd_res[name] = scores
        
    print("Mean GCD per type:")
    for name, scores in gcd_res.items():
        print(f"  {name:10s}: {np.mean(scores):.2f}")
        
    return dict(gcd_res=gcd_res)

# ==============================================================
# MAIN
# ==============================================================
def main():
    runner = Runner("Structure vs Chaos (Original)", mode="1d")
    
    d1 = direction_1(runner)
    d2 = direction_2(runner, d1['metrics'])
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)
    
    fig, axes = runner.create_figure(5, "Structure vs Chaos: Sensor Calibration")
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Taxonomy")
    
    top_f = d2['findings'][:10]
    runner.plot_bars(axes[1], [x[0].split(':')[1][:10] for x in top_f], [abs(x[1]) for x in top_f], "D2: Top Sensors (d)")
    
    runner.plot_bars(axes[2], list(d3['results'].keys()), list(d3['results'].values()), "D3: Transform Efficacy")
    
    # D4 line plot of sorted stationarity
    axes[3].plot(sorted(d4['stats']['Logistic']), label='Chaos', color='cyan')
    axes[3].plot(sorted(d4['stats']['PrimeGaps']), label='Primes', color='magenta')
    axes[3].set_title("D4: Stationarity")
    axes[3].legend()
    
    # D5 GCD
    names5 = list(d5['gcd_res'].keys())
    vals5 = [np.mean(d5['gcd_res'][n]) for n in names5]
    runner.plot_bars(axes[4], names5, vals5, "D5: Mean GCD")
    
    runner.save(fig, "structure_vs_chaos_original")
    
    runner.print_summary({
        'D1': f"Pairwise mean: {d1['matrix'].mean():.1f} sig",
        'D2': f"Best Sensor: {d2['findings'][0][0].split(':')[1]} (d={d2['findings'][0][1]:.1f})",
        'D4': f"Stationarity d: {d4['d_val']:.2f}",
        'D5': f"GCD Chaos: {np.mean(d5['gcd_res']['Logistic']):.2f}"
    })

if __name__ == "__main__":
    main()
