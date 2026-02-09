#!/usr/bin/env python3
"""
Investigation: Deep Prime Gap Geometry — What Structure Does Cramér Miss?

primes.py found 55 significant metrics distinguishing real prime gaps from the
Cramér random model (each n is "prime" independently with prob 1/ln(n)). Real
prime gaps also had 97 sig metrics vs uniform random. The 55→97 gap means real
primes have detectable geometric structure *beyond* what the standard
probabilistic model predicts. This investigation probes what that structure is
and whether better models can close the gap.

Five directions:
  D5 (runs first): Metric Venn diagram — classify metrics as pure-primality,
      Cramér-specific, universal, or Cramér-captured.
  D1: Anatomy of the 55 — which geometry families detect Cramér-vs-real?
  D2: Hierarchy of random models — even-gap Cramér, sieved Cramér,
      distribution-matched.
  D3: Sequential correlation via delay embedding — amplify ordering effects.
  D4: Scale evolution — does the Cramér gap shrink for larger primes?

Execution order: D5 → D1 (free) → D2 → D3 → D4
Budget: ~601 analyzer calls, estimated ~4 minutes.
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
DATA_SIZE = 2000


# =============================================================================
# PRIME GENERATION
# =============================================================================

def sieve_primes(limit):
    """Sieve of Eratosthenes up to limit. Returns sorted array of primes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


PRIME_LIMIT = 2_000_000
ALL_PRIMES = sieve_primes(PRIME_LIMIT)
print(f"Sieved {len(ALL_PRIMES)} primes up to {PRIME_LIMIT:,}")


def get_primes_from(start_idx, count):
    """Get `count` consecutive primes starting at index `start_idx`."""
    end = start_idx + count
    if end > len(ALL_PRIMES):
        raise ValueError(f"Need {end} primes but only have {len(ALL_PRIMES)}")
    return ALL_PRIMES[start_idx:end]


def generate_prime_gaps(trial_seed, size=DATA_SIZE, start_idx=None):
    """Generate real prime gap sequence as uint8."""
    if start_idx is None:
        start_idx = 1000 + trial_seed * 137
    primes = get_primes_from(start_idx, size + 10)
    gaps = np.diff(primes)[:size]
    return np.clip(gaps, 0, 255).astype(np.uint8)


def generate_cramer_model(trial_seed, size=DATA_SIZE, start_value=10000):
    """Cramér random model: each n is 'prime' with probability 1/ln(n).
    Returns gap sequence as uint8.
    """
    rng = np.random.RandomState(trial_seed)
    gaps = []
    n = start_value
    last_prime = n
    while len(gaps) < size:
        n += 1
        if n > 10 * PRIME_LIMIT:
            break
        prob = 1.0 / np.log(n)
        if rng.random() < prob:
            gap = n - last_prime
            gaps.append(gap)
            last_prime = n
    gaps = np.array(gaps[:size])
    return np.clip(gaps, 0, 255).astype(np.uint8)


# =============================================================================
# NEW GAP MODELS
# =============================================================================

def generate_even_gap_cramer(trial_seed, size=DATA_SIZE, start_value=10000):
    """Even-gap Cramér: only test odd candidates (prob = 2/ln(n)).
    All gaps are guaranteed even, like real prime gaps for p > 2.
    """
    rng = np.random.RandomState(trial_seed)
    gaps = []
    # Start at an odd number
    n = start_value if start_value % 2 == 1 else start_value + 1
    last_prime = n
    while len(gaps) < size:
        n += 2  # only odd candidates
        if n > 10 * PRIME_LIMIT:
            break
        prob = 2.0 / np.log(n)
        if rng.random() < prob:
            gap = n - last_prime
            gaps.append(gap)
            last_prime = n
    gaps = np.array(gaps[:size])
    return np.clip(gaps, 0, 255).astype(np.uint8)


def generate_sieved_cramer(trial_seed, size=DATA_SIZE, start_value=10000):
    """Sieved Cramér: sieve out multiples of 2,3,5,7,11,13 then sample survivors.
    Respects local divisibility constraints that basic Cramér ignores.
    """
    rng = np.random.RandomState(trial_seed)
    small_primes = [2, 3, 5, 7, 11, 13]
    # Survival fraction: product of (1 - 1/p) for small primes
    survival_frac = 1.0
    for p in small_primes:
        survival_frac *= (1.0 - 1.0 / p)

    gaps = []
    n = start_value
    last_prime = n
    while len(gaps) < size:
        n += 1
        if n > 10 * PRIME_LIMIT:
            break
        # Check divisibility by small primes
        divisible = False
        for p in small_primes:
            if n % p == 0 and n != p:
                divisible = True
                break
        if divisible:
            continue
        # Survivor: "prime" with adjusted probability
        prob = min(1.0 / (np.log(n) * survival_frac), 1.0)
        if rng.random() < prob:
            gap = n - last_prime
            gaps.append(gap)
            last_prime = n
    gaps = np.array(gaps[:size])
    return np.clip(gaps, 0, 255).astype(np.uint8)


def generate_dist_matched(trial_seed, real_gaps, size=DATA_SIZE):
    """Distribution-matched: sample gaps from empirical distribution of real gaps,
    independently. Matches marginal distribution exactly but destroys sequential
    correlation.
    """
    rng = np.random.RandomState(trial_seed)
    sampled = rng.choice(real_gaps, size=size, replace=True)
    return np.clip(sampled, 0, 255).astype(np.uint8)


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
    """Count metrics significantly different (|d|>0.8, Bonferroni)."""
    alpha = 0.05 / max(n_total_tests, 1)
    sig = 0
    for km in metric_names:
        va = metrics_a.get(km, [])
        vb = metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb, equal_var=False)
        if abs(d) > 0.8 and p < alpha:
            sig += 1
    return sig


def per_metric_significance(metrics_a, metrics_b, metric_names, n_total_tests):
    """Per-metric Cohen's d, p-value, significance flag."""
    alpha = 0.05 / max(n_total_tests, 1)
    result = {}
    for km in metric_names:
        va = metrics_a.get(km, [])
        vb = metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            result[km] = (0.0, 1.0, False)
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb, equal_var=False)
        is_sig = abs(d) > 0.8 and p < alpha
        result[km] = (d, p, is_sig)
    return result


def collect_metrics(analyzer, data, target_dict):
    """Run analyzer on data and append results to target_dict."""
    results = analyzer.analyze(data)
    for r in results.results:
        for mn, mv in r.metrics.items():
            target_dict[f"{r.geometry_name}:{mn}"].append(mv)


# =============================================================================
# GEOMETRY FAMILY MAPPING
# =============================================================================

GEOMETRY_FAMILIES = {
    'Lattice/Discrete': ['E8 Lattice', 'Cantor (base 3)', '2-adic'],
    'Torus': ['Torus T^2', 'Clifford Torus'],
    'Curved': ['Hyperbolic (Poincaré)', 'Spherical S²'],
    'Heisenberg': ['Heisenberg (Nil)', 'Heisenberg (Nil) (centered)'],
    'Thurston': ['Sol (Thurston)', 'S² × ℝ (Thurston)',
                 'H² × ℝ (Thurston)', 'SL(2,ℝ) (Thurston)'],
    'Algebraic': ['Tropical', 'Projective ℙ²'],
    'Statistical': ['Wasserstein', 'Fisher Information', 'Persistent Homology'],
    'Physical': ['Lorentzian', 'Symplectic', 'Spiral (logarithmic)'],
    'Aperiodic': ['Penrose (Quasicrystal)', 'Ammann-Beenker (Octagonal)',
                  'Einstein (Hat Monotile)'],
    'Higher-Order': ['Higher-Order Statistics'],
}

# Reverse map: geometry name → family
GEOM_TO_FAMILY = {}
for fam, names in GEOMETRY_FAMILIES.items():
    for name in names:
        GEOM_TO_FAMILY[name] = fam


def metric_to_family(metric_name):
    """Extract geometry family from 'GeometryName:metric_name' string."""
    geom_name = metric_name.split(':')[0]
    return GEOM_TO_FAMILY.get(geom_name, 'Unknown')


# =============================================================================
# DIRECTION 5: METRIC VENN DIAGRAM (runs FIRST)
# =============================================================================

def direction5_venn(analyzer, metric_names):
    """Classify every metric: pure-primality, Cramér-specific, universal, Cramér-captured."""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Metric Venn Diagram")
    print("=" * 78)
    print(f"3 baselines × {N_TRIALS} trials = 75 calls")
    print(f"Question: which metrics detect structure specific to real primes?\n")

    n_total = len(metric_names)

    # Collect baselines
    real_metrics = defaultdict(list)
    cramer_metrics = defaultdict(list)
    random_metrics = defaultdict(list)

    print("  Real prime gaps...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_prime_gaps(trial)
        collect_metrics(analyzer, data, real_metrics)
    print("done")

    print("  Cramér model...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        data = generate_cramer_model(trial)
        collect_metrics(analyzer, data, cramer_metrics)
    print("done")

    print("  Random baseline...", end=" ", flush=True)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial + 5000)
        data = rng.randint(0, 256, DATA_SIZE, dtype=np.uint8)
        collect_metrics(analyzer, data, random_metrics)
    print("done")

    # Pairwise significance tests
    sig_real_cramer = per_metric_significance(real_metrics, cramer_metrics,
                                              metric_names, n_total)
    sig_real_random = per_metric_significance(real_metrics, random_metrics,
                                              metric_names, n_total)
    sig_cramer_random = per_metric_significance(cramer_metrics, random_metrics,
                                                metric_names, n_total)

    # Classify each metric
    categories = {'pure_primality': [], 'cramer_specific': [],
                  'universal': [], 'cramer_captured': [], 'unclassified': []}

    for km in metric_names:
        rc = sig_real_cramer[km][2]   # real vs Cramér significant?
        rr = sig_real_random[km][2]   # real vs random significant?
        cr = sig_cramer_random[km][2] # Cramér vs random significant?

        if rc and rr:
            categories['pure_primality'].append(km)
        elif rc and not cr:
            categories['cramer_specific'].append(km)
        elif rr and cr and not rc:
            categories['cramer_captured'].append(km)
        elif rr and cr:
            # Both distinguish from random, and real differs from Cramér
            # This is a subset of pure_primality already handled above
            categories['universal'].append(km)
        elif rr and not cr and not rc:
            categories['universal'].append(km)
        elif cr and not rr and not rc:
            categories['cramer_captured'].append(km)
        else:
            categories['unclassified'].append(km)

    # Print summary
    n_rc = sum(1 for km in metric_names if sig_real_cramer[km][2])
    n_rr = sum(1 for km in metric_names if sig_real_random[km][2])
    n_cr = sum(1 for km in metric_names if sig_cramer_random[km][2])

    print(f"\n  Pairwise significance counts:")
    print(f"    Real vs Cramér:  {n_rc}")
    print(f"    Real vs Random:  {n_rr}")
    print(f"    Cramér vs Random: {n_cr}")

    print(f"\n  Venn categories:")
    for cat, members in categories.items():
        if cat == 'unclassified':
            continue
        print(f"    {cat:<20} {len(members):>3} metrics")
    print(f"    {'unclassified':<20} {len(categories['unclassified']):>3} metrics")
    total_classified = sum(len(v) for v in categories.values())
    print(f"    {'TOTAL':<20} {total_classified:>3}")

    # Top 5 pure-primality metrics
    if categories['pure_primality']:
        print(f"\n  Top pure-primality metrics (structure Cramér misses AND differs from random):")
        scored = [(km, abs(sig_real_cramer[km][0])) for km in categories['pure_primality']]
        scored.sort(key=lambda x: -x[1])
        for km, d in scored[:8]:
            print(f"    {km:<50} |d|={d:.2f}")

    return {
        'real_metrics': real_metrics,
        'cramer_metrics': cramer_metrics,
        'random_metrics': random_metrics,
        'sig_real_cramer': sig_real_cramer,
        'sig_real_random': sig_real_random,
        'sig_cramer_random': sig_cramer_random,
        'categories': categories,
        'counts': {'real_vs_cramer': n_rc, 'real_vs_random': n_rr,
                   'cramer_vs_random': n_cr},
    }


# =============================================================================
# DIRECTION 1: ANATOMY OF THE 55 (0 extra calls)
# =============================================================================

def direction1_anatomy(metric_names, d5):
    """Group sig(real vs Cramér) metrics by geometry family."""
    print("\n" + "=" * 78)
    print("DIRECTION 1: Anatomy of Cramér-Distinguishing Metrics")
    print("=" * 78)
    print(f"0 extra calls — pure analysis of D5 results")
    print(f"Question: which geometry families detect what Cramér misses?\n")

    sig_rc = d5['sig_real_cramer']
    sig_rr = d5['sig_real_random']

    # Count per family: sig(real vs Cramér) and sig(real vs random)
    family_rc = defaultdict(int)
    family_rr = defaultdict(int)
    family_total = defaultdict(int)

    for km in metric_names:
        fam = metric_to_family(km)
        family_total[fam] += 1
        if sig_rc[km][2]:
            family_rc[fam] += 1
        if sig_rr[km][2]:
            family_rr[fam] += 1

    families = list(GEOMETRY_FAMILIES.keys())
    print(f"  {'Family':<20} {'vs Cramér':>10} {'vs Random':>10} {'Total':>7} {'Cramér%':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")
    for fam in families:
        rc = family_rc[fam]
        rr = family_rr[fam]
        tot = family_total[fam]
        pct = (rc / max(rr, 1)) * 100 if rr > 0 else 0
        print(f"  {fam:<20} {rc:>10} {rr:>10} {tot:>7} {pct:>7.0f}%")

    # Identify Cramér-specific detector families (high vs-Cramér / vs-random ratio)
    print(f"\n  Cramér-specific detectors (high vs-Cramér / vs-random ratio):")
    for fam in families:
        rc = family_rc[fam]
        rr = family_rr[fam]
        if rc > 0 and rr > 0:
            ratio = rc / rr
            print(f"    {fam:<20} ratio={ratio:.2f} ({rc}/{rr})")

    return {
        'family_rc': dict(family_rc),
        'family_rr': dict(family_rr),
        'family_total': dict(family_total),
        'families': families,
    }


# =============================================================================
# DIRECTION 2: HIERARCHY OF RANDOM MODELS (75 calls)
# =============================================================================

def direction2_hierarchy(analyzer, metric_names, d5):
    """Compare 4 gap models to real prime gaps: which is closest?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Hierarchy of Random Models")
    print("=" * 78)
    print(f"4 models × {N_TRIALS} trials (Cramér reused from D5, 75 new calls)")
    print(f"Question: can better models close the 55-metric gap?\n")

    real_metrics = d5['real_metrics']
    random_metrics = d5['random_metrics']
    cramer_metrics = d5['cramer_metrics']
    n_total = len(metric_names)

    # Build reference real gap array for distribution-matched model
    ref_gaps = []
    for trial in range(N_TRIALS):
        data = generate_prime_gaps(trial)
        ref_gaps.extend(data.tolist())
    ref_gaps = np.array(ref_gaps)

    # Collect metrics for new models
    model_metrics = {'Cramér basic': cramer_metrics}

    for model_name, gen_fn in [
        ('Even-gap Cramér', lambda t: generate_even_gap_cramer(t)),
        ('Sieved Cramér', lambda t: generate_sieved_cramer(t)),
        ('Dist-matched', lambda t: generate_dist_matched(t, ref_gaps)),
    ]:
        print(f"  {model_name}...", end=" ", flush=True)
        model_metrics[model_name] = defaultdict(list)
        for trial in range(N_TRIALS):
            data = gen_fn(trial)
            collect_metrics(analyzer, data, model_metrics[model_name])
        print("done")

    # Compare each model to real gaps and to random
    model_order = ['Cramér basic', 'Even-gap Cramér', 'Sieved Cramér', 'Dist-matched']
    print(f"\n  {'Model':<20} {'vs Real':>8} {'vs Random':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    model_vs_real = {}
    model_vs_random = {}
    for model_name in model_order:
        n_vs_real = count_significant(model_metrics[model_name], real_metrics,
                                      metric_names, n_total)
        n_vs_rand = count_significant(model_metrics[model_name], random_metrics,
                                      metric_names, n_total)
        model_vs_real[model_name] = n_vs_real
        model_vs_random[model_name] = n_vs_rand
        print(f"  {model_name:<20} {n_vs_real:>8} {n_vs_rand:>10}")

    # Top metrics that dist-matched still detects (pure ordering signal)
    pms = per_metric_significance(model_metrics['Dist-matched'], real_metrics,
                                  metric_names, n_total)
    sig_list = [(km, d, p) for km, (d, p, is_sig) in pms.items() if is_sig]
    sig_list.sort(key=lambda x: -abs(x[1]))
    if sig_list:
        print(f"\n  Dist-matched vs Real — top metrics (pure sequential correlation):")
        for km, d, p in sig_list[:5]:
            print(f"    {km:<50} d={d:+8.2f}")
    else:
        print(f"\n  Dist-matched vs Real: 0 sig — all structure is distributional!")

    return {
        'model_metrics': model_metrics,
        'model_vs_real': model_vs_real,
        'model_vs_random': model_vs_random,
        'model_order': model_order,
    }


# =============================================================================
# DIRECTION 3: SEQUENTIAL CORRELATION VIA DELAY EMBEDDING (250 calls)
# =============================================================================

def direction3_delay(analyzer, metric_names, d5):
    """Delay embed real and Cramér gaps at multiple τ. Does embedding amplify the gap?"""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Sequential Correlation via Delay Embedding")
    print("=" * 78)
    taus = [1, 2, 3, 5, 10]
    print(f"τ = {taus}, {N_TRIALS} trials × 2 (real + Cramér) = {len(taus)*N_TRIALS*2} calls")
    print(f"Question: does delay embedding amplify the real-vs-Cramér signal?\n")

    n_total = len(metric_names)
    tau_sig = {}

    for tau in taus:
        print(f"  τ={tau}...", end=" ", flush=True)
        real_emb = defaultdict(list)
        cramer_emb = defaultdict(list)

        for trial in range(N_TRIALS):
            # Generate with extra length to account for delay_embed reduction
            raw_real = generate_prime_gaps(trial, size=DATA_SIZE + tau + 100)
            raw_cramer = generate_cramer_model(trial, size=DATA_SIZE + tau + 100)

            emb_real = delay_embed(raw_real, tau)[:DATA_SIZE]
            emb_cramer = delay_embed(raw_cramer, tau)[:DATA_SIZE]

            collect_metrics(analyzer, emb_real, real_emb)
            collect_metrics(analyzer, emb_cramer, cramer_emb)

        n_sig = count_significant(real_emb, cramer_emb, metric_names, n_total)
        tau_sig[tau] = n_sig
        print(f"{n_sig} sig")

    # Baseline (raw, from D5)
    n_raw = d5['counts']['real_vs_cramer']
    print(f"\n  {'τ':<6} {'Sig metrics':>11}")
    print(f"  {'-'*6} {'-'*11}")
    print(f"  {'raw':<6} {n_raw:>11}  (D5 baseline)")
    for tau in taus:
        marker = " ← peak" if tau_sig[tau] == max(tau_sig.values()) else ""
        print(f"  {tau:<6} {tau_sig[tau]:>11}{marker}")

    return {'tau_sig': tau_sig, 'raw_sig': n_raw, 'taus': taus}


# =============================================================================
# DIRECTION 4: SCALE EVOLUTION (200 calls)
# =============================================================================

def direction4_scale(analyzer, metric_names):
    """At 4 prime ranges, compare real vs Cramér. Does the gap shrink?"""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Scale Evolution")
    print("=" * 78)
    print(f"4 scales × {N_TRIALS} trials × 2 = 200 calls")
    print(f"Question: does the Cramér gap shrink for larger primes?\n")

    # Prime ranges: index → approximate prime value
    ranges = {
        'near 1K': {'idx': 168, 'start_value': 1000},
        'near 10K': {'idx': 1229, 'start_value': 10000},
        'near 100K': {'idx': 9592, 'start_value': 100000},
        'near 1M': {'idx': 78498, 'start_value': 1000000},
    }

    n_total = len(metric_names)
    scale_results = {}
    all_real_m = {}
    all_cramer_m = {}

    for range_name, params in ranges.items():
        print(f"  {range_name}...", end=" ", flush=True)
        real_m = defaultdict(list)
        cramer_m = defaultdict(list)

        for trial in range(N_TRIALS):
            offset = trial * 50
            real_data = generate_prime_gaps(trial, start_idx=params['idx'] + offset)
            cramer_data = generate_cramer_model(trial, start_value=params['start_value'])

            collect_metrics(analyzer, real_data, real_m)
            collect_metrics(analyzer, cramer_data, cramer_m)

        n_sig = count_significant(real_m, cramer_m, metric_names, n_total)
        scale_results[range_name] = n_sig
        all_real_m[range_name] = real_m
        all_cramer_m[range_name] = cramer_m

        # Mean gap for context
        primes = get_primes_from(params['idx'], DATA_SIZE + 1)
        mean_gap = np.mean(np.diff(primes))
        print(f"{n_sig} sig (mean gap {mean_gap:.1f})")

    # Per-metric: which metrics are significant at ALL scales?
    per_scale_sig = {}
    for range_name in ranges:
        pms = per_metric_significance(all_real_m[range_name], all_cramer_m[range_name],
                                      metric_names, n_total)
        per_scale_sig[range_name] = {km for km, (d, p, is_sig) in pms.items() if is_sig}

    # Compute always-sig and scale-dependent
    range_names = list(ranges.keys())
    always_sig = per_scale_sig[range_names[0]]
    any_sig = set(per_scale_sig[range_names[0]])
    for rn in range_names[1:]:
        always_sig = always_sig & per_scale_sig[rn]
        any_sig = any_sig | per_scale_sig[rn]
    scale_dependent = any_sig - always_sig

    print(f"\n  Scale summary:")
    print(f"    Always significant (all 4 scales): {len(always_sig)}")
    print(f"    Scale-dependent (some but not all): {len(scale_dependent)}")
    print(f"    Ever significant (any scale):       {len(any_sig)}")

    if always_sig:
        print(f"\n  Always-sig metrics (top 5 by family):")
        by_family = defaultdict(list)
        for km in always_sig:
            by_family[metric_to_family(km)].append(km)
        for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
            print(f"    {fam}: {len(by_family[fam])} metrics")

    return {
        'scale_results': scale_results,
        'always_sig': always_sig,
        'scale_dependent': scale_dependent,
        'any_sig': any_sig,
        'range_names': range_names,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print("INVESTIGATION: Deep Prime Gap Geometry — What Structure Does Cramér Miss?")
    print("=" * 78)

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names from a test run
    test_data = np.random.RandomState(0).randint(0, 256, DATA_SIZE, dtype=np.uint8)
    test_result = analyzer.analyze(test_data)
    metric_names = []
    for r in test_result.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    print(f"Tracking {len(metric_names)} metrics across {len(test_result.results)} geometries\n")

    # Execution order: D5 → D1 → D2 → D3 → D4
    d5 = direction5_venn(analyzer, metric_names)
    d1 = direction1_anatomy(metric_names, d5)
    d2 = direction2_hierarchy(analyzer, metric_names, d5)
    d3 = direction3_delay(analyzer, metric_names, d5)
    d4 = direction4_scale(analyzer, metric_names)

    return metric_names, d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(metric_names, d1, d2, d3, d4, d5):
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

    # ── Panel 1 (top-left): D1 — Per-family sig counts ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)

    families = d1['families']
    y_pos = np.arange(len(families))
    bar_h = 0.35
    rc_vals = [d1['family_rc'].get(f, 0) for f in families]
    rr_vals = [d1['family_rr'].get(f, 0) for f in families]

    ax1.barh(y_pos - bar_h/2, rc_vals, bar_h, label='vs Cramér',
             color='#E91E63', alpha=0.85, edgecolor='#333')
    ax1.barh(y_pos + bar_h/2, rr_vals, bar_h, label='vs Random',
             color='#FFB300', alpha=0.85, edgecolor='#333')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(families, fontsize=7, color=FG)
    ax1.set_xlabel('Significant metrics', color=FG, fontsize=9)
    ax1.set_title('D1: Geometry Family Anatomy', fontsize=11,
                  fontweight='bold', color=FG)
    ax1.invert_yaxis()
    ax1.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG,
               loc='lower right')

    # ── Panel 2 (top-center): D2 — Model hierarchy ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)

    model_order = d2['model_order']
    x_pos = np.arange(len(model_order))
    bar_w = 0.35
    vs_real = [d2['model_vs_real'][m] for m in model_order]
    vs_rand = [d2['model_vs_random'][m] for m in model_order]

    ax2.bar(x_pos - bar_w/2, vs_real, bar_w, label='vs Real',
            color='#E91E63', alpha=0.85, edgecolor='#333')
    ax2.bar(x_pos + bar_w/2, vs_rand, bar_w, label='vs Random',
            color='#2196F3', alpha=0.85, edgecolor='#333')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in model_order],
                        fontsize=7, color=FG)
    ax2.set_ylabel('Significant metrics', color=FG, fontsize=9)
    ax2.set_title('D2: Model Hierarchy', fontsize=11, fontweight='bold', color=FG)
    ax2.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)
    for i, v in enumerate(vs_real):
        ax2.text(i - bar_w/2, v + 0.5, str(v), ha='center', color='#E91E63',
                 fontsize=8, fontweight='bold')
    for i, v in enumerate(vs_rand):
        ax2.text(i + bar_w/2, v + 0.5, str(v), ha='center', color='#2196F3',
                 fontsize=8, fontweight='bold')

    # ── Panel 3 (top-right): D3 — Delay embedding ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)

    taus = d3['taus']
    sig_vals = [d3['tau_sig'][t] for t in taus]
    raw_sig = d3['raw_sig']

    ax3.axhline(y=raw_sig, color='#FFB300', linestyle='--', alpha=0.7,
                label=f'Raw baseline ({raw_sig})')
    ax3.plot(range(len(taus)), sig_vals, 'o-', color='#E91E63',
             linewidth=2, markersize=8, alpha=0.85, label='Delay embedded')
    ax3.set_xticks(range(len(taus)))
    ax3.set_xticklabels([str(t) for t in taus], fontsize=9, color=FG)
    ax3.set_xlabel('Delay τ', color=FG, fontsize=9)
    ax3.set_ylabel('Sig metrics (real vs Cramér)', color=FG, fontsize=9)
    ax3.set_title('D3: Delay Embedding', fontsize=11, fontweight='bold', color=FG)
    ax3.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)
    for i, v in enumerate(sig_vals):
        ax3.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9,
                 fontweight='bold')

    # ── Panel 4 (bottom-left): D4 — Scale evolution ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)

    range_names = d4['range_names']
    scale_vals = [d4['scale_results'][rn] for rn in range_names]

    ax4.plot(range(len(range_names)), scale_vals, 's-', color='#E91E63',
             linewidth=2, markersize=10, alpha=0.85)
    ax4.set_xticks(range(len(range_names)))
    ax4.set_xticklabels(range_names, fontsize=8, color=FG)
    ax4.set_xlabel('Prime range', color=FG, fontsize=9)
    ax4.set_ylabel('Sig metrics (real vs Cramér)', color=FG, fontsize=9)
    ax4.set_title('D4: Scale Evolution', fontsize=11, fontweight='bold', color=FG)
    for i, v in enumerate(scale_vals):
        ax4.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9,
                 fontweight='bold')

    # Annotate always-sig / scale-dependent
    n_always = len(d4['always_sig'])
    n_scale_dep = len(d4['scale_dependent'])
    ax4.text(0.98, 0.98,
             f"Always-sig: {n_always}\nScale-dep: {n_scale_dep}",
             transform=ax4.transAxes, fontsize=8, va='top', ha='right',
             color='#aaa', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', edgecolor='#444'))

    # ── Panel 5 (bottom-center): D5 — Venn categories ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)

    cat_names = ['pure_primality', 'cramer_specific', 'universal', 'cramer_captured']
    cat_labels = ['Pure\nprimality', 'Cramér-\nspecific', 'Universal', 'Cramér-\ncaptured']
    cat_colors = ['#E91E63', '#FF5722', '#2196F3', '#4CAF50']
    cat_vals = [len(d5['categories'][c]) for c in cat_names]

    bars5 = ax5.bar(range(len(cat_names)), cat_vals, color=cat_colors,
                    alpha=0.85, edgecolor='#333')
    ax5.set_xticks(range(len(cat_names)))
    ax5.set_xticklabels(cat_labels, fontsize=8, color=FG)
    ax5.set_ylabel('Number of metrics', color=FG, fontsize=9)
    ax5.set_title('D5: Metric Venn Classification', fontsize=11,
                  fontweight='bold', color=FG)
    for i, v in enumerate(cat_vals):
        ax5.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=9,
                 fontweight='bold')

    # ── Panel 6 (bottom-right): Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    n_total = len(metric_names)
    lines = [
        "Deep Prime Gap Geometry — Summary",
        "",
        "D5: Metric Venn Diagram",
        f"  Real vs Cramér:  {d5['counts']['real_vs_cramer']:>3} / {n_total} sig",
        f"  Real vs Random:  {d5['counts']['real_vs_random']:>3} / {n_total} sig",
        f"  Cramér vs Rand:  {d5['counts']['cramer_vs_random']:>3} / {n_total} sig",
        "",
    ]
    for c, l in zip(cat_names, ['Pure primality', 'Cramér-specific',
                                'Universal', 'Cramér-captured']):
        lines.append(f"  {l:<18} {len(d5['categories'][c]):>3}")

    lines.append("")
    lines.append("D2: Model Hierarchy (vs Real)")
    for m in d2['model_order']:
        lines.append(f"  {m:<18} {d2['model_vs_real'][m]:>3} sig")

    lines.append("")
    lines.append("D3: Delay Embedding")
    lines.append(f"  Raw baseline:    {d3['raw_sig']:>3} sig")
    peak_tau = max(d3['taus'], key=lambda t: d3['tau_sig'][t])
    lines.append(f"  Peak τ={peak_tau}:       {d3['tau_sig'][peak_tau]:>3} sig")

    lines.append("")
    lines.append("D4: Scale Evolution")
    for rn in d4['range_names']:
        lines.append(f"  {rn:<12} {d4['scale_results'][rn]:>3} sig")
    lines.append(f"  Always-sig:   {len(d4['always_sig']):>3}")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor='#444'))

    fig.suptitle('Deep Prime Gap Geometry: What Structure Does Cramér Miss?',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'primes_deep.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved primes_deep.png")
    plt.close(fig)


if __name__ == '__main__':
    metric_names, d1, d2, d3, d4, d5 = main()
    make_figure(metric_names, d1, d2, d3, d4, d5)
