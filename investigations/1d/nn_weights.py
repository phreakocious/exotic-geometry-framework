#!/usr/bin/env python3
"""
Investigate Neural Network Weight Distributions with Exotic Geometries
======================================================================

Can the 21-geometry framework detect structural differences between
different types of neural network weight distributions?

We generate SYNTHETIC distributions that mimic the statistical properties
of real NN weights: Xavier init, Kaiming init, trained dense, trained conv,
pruned, quantized, lottery ticket, adversarial/backdoored, dropout-trained.

All converted to uint8 for byte-level geometric analysis.

Methodology:
- 25 trials x 2000 values each
- Cohen's d effect sizes
- Bonferroni correction for multiple comparisons
- Shuffle validation for all positive findings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from exotic_geometry_framework import GeometryAnalyzer

# =============================================================================
# PARAMETERS
# =============================================================================
N_TRIALS = 25
N_VALUES = 2000
SEED_BASE = 42
ALPHA = 0.05  # before Bonferroni

# =============================================================================
# WEIGHT DISTRIBUTION GENERATORS
# =============================================================================

def to_uint8(weights, clip_sigma=3.0):
    """Convert float weights to uint8 by clipping at +/-clip_sigma*std and scaling to [0,255]."""
    mu = np.mean(weights)
    sigma = np.std(weights) + 1e-10
    clipped = np.clip(weights, mu - clip_sigma * sigma, mu + clip_sigma * sigma)
    # Scale to [0, 255]
    lo, hi = clipped.min(), clipped.max()
    if hi - lo < 1e-12:
        return np.full(len(weights), 128, dtype=np.uint8)
    scaled = (clipped - lo) / (hi - lo) * 255
    return scaled.astype(np.uint8)


def gen_random_uniform(rng, n=N_VALUES):
    """Control: pure uniform random."""
    raw = rng.uniform(-1, 1, size=n)
    return to_uint8(raw)


def gen_xavier(rng, n=N_VALUES, fan_in=512, fan_out=256):
    """Xavier/Glorot initialization: Normal(0, sqrt(2/(fan_in+fan_out)))."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    raw = rng.normal(0, std, size=n)
    return to_uint8(raw)


def gen_kaiming(rng, n=N_VALUES, fan_in=512):
    """Kaiming/He initialization: Normal(0, sqrt(2/fan_in))."""
    std = np.sqrt(2.0 / fan_in)
    raw = rng.normal(0, std, size=n)
    return to_uint8(raw)


def gen_trained_dense(rng, n=N_VALUES, fan_in=512, fan_out=256):
    """Simulate trained dense weights: Xavier init + gradient-like updates.
    Creates sparsity, heavy tails, and local correlation structure."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    raw = rng.normal(0, std, size=n)

    # Apply many "gradient updates" - creates heavy tails
    for _ in range(50):
        grad = rng.normal(0, std * 0.1, size=n)
        # Simulate momentum: correlated updates on neighboring weights
        kernel = np.array([0.1, 0.3, 0.6, 0.3, 0.1])
        grad_smooth = np.convolve(grad, kernel / kernel.sum(), mode='same')
        raw += grad_smooth * 0.1

    # Push small weights toward zero (L1-like regularization effect)
    mask_small = np.abs(raw) < std * 0.3
    raw[mask_small] *= 0.1

    # Add a few large outlier weights (heavy tails)
    n_outliers = n // 50
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    raw[outlier_idx] *= rng.uniform(3, 8, size=n_outliers) * rng.choice([-1, 1], size=n_outliers)

    return to_uint8(raw)


def gen_trained_conv(rng, n=N_VALUES):
    """Simulate trained convolutional kernels.
    3x3 spatial filters, tiled, with edge-detector-like patterns."""
    n_filters = n // 9 + 1  # each filter is 3x3 = 9 values
    filters = []
    for _ in range(n_filters):
        ftype = rng.choice(['edge_h', 'edge_v', 'edge_d', 'blur', 'sharpen', 'random'])
        if ftype == 'edge_h':
            f = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
        elif ftype == 'edge_v':
            f = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
        elif ftype == 'edge_d':
            f = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=float)
        elif ftype == 'blur':
            f = np.ones((3, 3)) / 9.0
        elif ftype == 'sharpen':
            f = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
        else:
            f = rng.normal(0, 0.5, (3, 3))

        # Add noise to simulate training variability
        f += rng.normal(0, 0.1, (3, 3))
        filters.append(f.flatten())

    raw = np.concatenate(filters)[:n]
    return to_uint8(raw)


def gen_pruned(rng, n=N_VALUES, sparsity=0.9):
    """Pruned weights: trained-like distribution but 90% set to zero."""
    # Start with trained-like
    std = np.sqrt(2.0 / 512)
    raw = rng.normal(0, std, size=n)

    # Train a bit
    for _ in range(20):
        grad = rng.normal(0, std * 0.1, size=n)
        raw += grad * 0.05

    # Prune: zero out 90% of smallest magnitude weights
    threshold = np.percentile(np.abs(raw), sparsity * 100)
    raw[np.abs(raw) < threshold] = 0.0

    return to_uint8(raw)


def gen_quantized(rng, n=N_VALUES):
    """Simulated INT8 quantization: only ~256 distinct values from continuous range."""
    std = np.sqrt(2.0 / 512)
    raw = rng.normal(0, std, size=n)

    # Quantize to 256 levels (INT8-like)
    lo, hi = raw.min(), raw.max()
    step = (hi - lo) / 255
    raw = np.round((raw - lo) / step) * step + lo

    return to_uint8(raw)


def gen_lottery_ticket(rng, n=N_VALUES, sparsity=0.8):
    """Lottery ticket hypothesis: sparse but with power-law magnitude distribution."""
    # Create power-law distributed magnitudes
    magnitudes = rng.pareto(a=2.0, size=n) * 0.05  # heavy tail
    signs = rng.choice([-1, 1], size=n)
    raw = magnitudes * signs

    # Sparse: zero out a fraction
    mask = rng.random(n) > sparsity
    raw[~mask] = 0.0

    return to_uint8(raw)


def gen_adversarial_backdoor(rng, n=N_VALUES):
    """Adversarial/backdoored weights: mostly normal but with periodic spike pattern."""
    std = np.sqrt(2.0 / 512)
    raw = rng.normal(0, std, size=n)

    # Hidden periodic pattern: every 37th weight has a specific value
    period = 37
    spike_val = std * 2.5  # not huge, but consistent
    for i in range(0, n, period):
        # Alternate sign for subtlety
        raw[i] = spike_val * (1 if (i // period) % 2 == 0 else -1)
        # Also set a second weight nearby for a "signature"
        if i + 7 < n:
            raw[i + 7] = -raw[i] * 0.5

    return to_uint8(raw)


def gen_dropout_trained(rng, n=N_VALUES, dropout_rate=0.5):
    """Dropout-trained: subset of weights exactly zero (co-adapted neurons killed),
    remaining weights have larger magnitudes (scale compensation)."""
    std = np.sqrt(2.0 / 512)
    # Larger magnitudes to compensate for dropout
    raw = rng.normal(0, std / (1 - dropout_rate), size=n)

    # Train with dropout effect
    for _ in range(30):
        mask = rng.random(n) > dropout_rate
        grad = rng.normal(0, std * 0.1, size=n) * mask
        raw += grad * 0.05

    # Some weights become co-adapted (exactly zero after training)
    zero_frac = dropout_rate * 0.3  # not all dropped weights go to zero
    zero_idx = rng.choice(n, int(n * zero_frac), replace=False)
    raw[zero_idx] = 0.0

    return to_uint8(raw)


# =============================================================================
# ALL GENERATORS
# =============================================================================

GENERATORS = {
    'random_uniform':       gen_random_uniform,
    'xavier_init':          gen_xavier,
    'kaiming_init':         gen_kaiming,
    'trained_dense':        gen_trained_dense,
    'trained_conv':         gen_trained_conv,
    'pruned_90pct':         gen_pruned,
    'quantized_int8':       gen_quantized,
    'lottery_ticket':       gen_lottery_ticket,
    'adversarial_backdoor': gen_adversarial_backdoor,
    'dropout_trained':      gen_dropout_trained,
}

CATEGORIES = {
    'untrained':   ['random_uniform', 'xavier_init', 'kaiming_init'],
    'trained':     ['trained_dense', 'trained_conv', 'dropout_trained'],
    'compressed':  ['pruned_90pct', 'quantized_int8', 'lottery_ticket'],
    'anomalous':   ['adversarial_backdoor'],
}


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_label(d):
    """Label effect size magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "HUGE"


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("EXOTIC GEOMETRY ANALYSIS OF NEURAL NETWORK WEIGHT DISTRIBUTIONS")
    print("=" * 80)
    print(f"\nParameters: {N_TRIALS} trials x {N_VALUES} values each")
    print(f"Weight types: {len(GENERATORS)}")
    print(f"Geometries: 21 (full framework)")
    print()

    # -------------------------------------------------------------------------
    # Step 1: Generate all data and compute geometric metrics
    # -------------------------------------------------------------------------
    print("STEP 1: Generating synthetic weight distributions and computing metrics...")
    print("-" * 70)

    analyzer = GeometryAnalyzer().add_all_geometries()
    geom_names = [g.name for g in analyzer.geometries]
    print(f"Active geometries ({len(geom_names)}): {geom_names}")
    print()

    # all_metrics[weight_type][trial] = {geom_name: {metric: value}}
    all_metrics = defaultdict(list)

    for wtype, gen_func in GENERATORS.items():
        print(f"  Generating '{wtype}'...", end=" ", flush=True)
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(SEED_BASE + trial * 1000 + hash(wtype) % 10000)
            data = gen_func(rng)
            result = analyzer.analyze(data, f"{wtype}_t{trial}")
            metrics_dict = result.to_dict()
            all_metrics[wtype].append(metrics_dict)
        print(f"done ({N_TRIALS} trials)")

    print()

    # -------------------------------------------------------------------------
    # Step 2: Build fingerprint table (mean values across trials)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: GEOMETRIC FINGERPRINT TABLE (mean over 25 trials)")
    print("=" * 80)
    print()

    # Collect all (geometry, metric) pairs
    all_gm_pairs = []
    sample = all_metrics['random_uniform'][0]
    for geom_name, metrics in sample.items():
        for metric_name in metrics:
            all_gm_pairs.append((geom_name, metric_name))

    # Print fingerprint as a compact table
    # First, compute means for each weight type and geometry-metric pair
    fingerprints = {}
    for wtype in GENERATORS:
        fp = {}
        for gname, mname in all_gm_pairs:
            values = []
            for trial_data in all_metrics[wtype]:
                if gname in trial_data and mname in trial_data[gname]:
                    values.append(trial_data[gname][mname])
            if values:
                fp[(gname, mname)] = (np.mean(values), np.std(values))
        fingerprints[wtype] = fp

    # Select most discriminative metrics (highest coefficient of variation across types)
    metric_cv = {}
    for gm in all_gm_pairs:
        means = [fingerprints[wt].get(gm, (0, 0))[0] for wt in GENERATORS]
        overall_mean = np.mean(means)
        overall_std = np.std(means)
        if abs(overall_mean) > 1e-10:
            metric_cv[gm] = overall_std / abs(overall_mean)
        else:
            metric_cv[gm] = 0.0

    # Show top 20 most discriminative metrics
    top_metrics = sorted(metric_cv.items(), key=lambda x: -x[1])[:20]

    print(f"Top 20 most discriminative geometry-metric pairs (by CV across types):")
    print()

    # Header
    wtypes_short = {
        'random_uniform': 'RndUnif',
        'xavier_init': 'Xavier',
        'kaiming_init': 'Kaiming',
        'trained_dense': 'TrDense',
        'trained_conv': 'TrConv',
        'pruned_90pct': 'Pruned',
        'quantized_int8': 'Quant',
        'lottery_ticket': 'Lottery',
        'adversarial_backdoor': 'Backdoor',
        'dropout_trained': 'Dropout',
    }

    header = f"{'Geometry':<22} {'Metric':<20} " + " ".join(f"{wtypes_short[wt]:>8}" for wt in GENERATORS)
    print(header)
    print("-" * len(header))

    for (gname, mname), cv in top_metrics:
        row = f"{gname:<22} {mname:<20} "
        for wtype in GENERATORS:
            val, std = fingerprints[wtype].get((gname, mname), (0, 0))
            row += f"{val:>8.3f} "
        row += f"  CV={cv:.2f}"
        print(row)

    print()

    # -------------------------------------------------------------------------
    # Step 3: Pairwise comparisons vs random uniform baseline
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: EACH TYPE vs RANDOM UNIFORM BASELINE")
    print("=" * 80)
    print()

    baseline_type = 'random_uniform'
    n_tests = len(all_gm_pairs)
    bonferroni_alpha = ALPHA / n_tests
    print(f"Bonferroni-corrected alpha: {ALPHA} / {n_tests} = {bonferroni_alpha:.2e}")
    print()

    significant_findings = []

    for wtype in GENERATORS:
        if wtype == baseline_type:
            continue

        print(f"\n  --- {wtype} vs {baseline_type} ---")
        sig_count = 0

        for gname, mname in all_gm_pairs:
            baseline_vals = []
            test_vals = []
            for trial_data in all_metrics[baseline_type]:
                if gname in trial_data and mname in trial_data[gname]:
                    baseline_vals.append(trial_data[gname][mname])
            for trial_data in all_metrics[wtype]:
                if gname in trial_data and mname in trial_data[gname]:
                    test_vals.append(trial_data[gname][mname])

            if len(baseline_vals) < 5 or len(test_vals) < 5:
                continue

            # Check for zero variance
            if np.std(baseline_vals) < 1e-12 and np.std(test_vals) < 1e-12:
                continue

            t_stat, p_val = stats.ttest_ind(test_vals, baseline_vals, equal_var=False)
            d = cohens_d(test_vals, baseline_vals)

            if p_val < bonferroni_alpha and abs(d) >= 0.5:
                sig_count += 1
                label = effect_size_label(d)
                if sig_count <= 10:  # show top 10
                    print(f"    {gname:22s} {mname:20s} d={d:+7.2f} ({label:>10s})  p={p_val:.2e}")
                significant_findings.append({
                    'wtype': wtype,
                    'geometry': gname,
                    'metric': mname,
                    'cohens_d': d,
                    'p_value': p_val,
                    'label': label,
                })

        if sig_count > 10:
            print(f"    ... and {sig_count - 10} more significant metrics")
        print(f"    Total significant (d>=0.5, p<{bonferroni_alpha:.1e}): {sig_count}/{len(all_gm_pairs)}")

    print()

    # -------------------------------------------------------------------------
    # Step 4: Can we distinguish trained from untrained? Dense from conv?
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: CATEGORY-LEVEL DISTINCTIONS")
    print("=" * 80)

    def compare_categories(cat1_types, cat2_types, cat1_name, cat2_name):
        """Compare two categories of weight distributions."""
        print(f"\n  --- {cat1_name} vs {cat2_name} ---")
        results = []
        for gname, mname in all_gm_pairs:
            cat1_vals = []
            cat2_vals = []
            for wtype in cat1_types:
                for trial_data in all_metrics[wtype]:
                    if gname in trial_data and mname in trial_data[gname]:
                        cat1_vals.append(trial_data[gname][mname])
            for wtype in cat2_types:
                for trial_data in all_metrics[wtype]:
                    if gname in trial_data and mname in trial_data[gname]:
                        cat2_vals.append(trial_data[gname][mname])

            if len(cat1_vals) < 5 or len(cat2_vals) < 5:
                continue
            if np.std(cat1_vals) < 1e-12 and np.std(cat2_vals) < 1e-12:
                continue

            t_stat, p_val = stats.ttest_ind(cat1_vals, cat2_vals, equal_var=False)
            d = cohens_d(cat1_vals, cat2_vals)
            results.append((gname, mname, d, p_val))

        # Sort by absolute effect size
        results.sort(key=lambda x: -abs(x[2]))
        n_bonf = len(results)
        bonf_alpha = ALPHA / max(n_bonf, 1)

        shown = 0
        sig_total = 0
        for gname, mname, d, p_val in results:
            if p_val < bonf_alpha and abs(d) >= 0.5:
                sig_total += 1
                if shown < 8:
                    label = effect_size_label(d)
                    print(f"    {gname:22s} {mname:20s} d={d:+7.2f} ({label:>10s})  p={p_val:.2e}")
                    shown += 1
        if sig_total > 8:
            print(f"    ... and {sig_total - 8} more significant")
        print(f"    Total significant: {sig_total}/{n_bonf}")
        return results

    # Trained vs Untrained
    compare_categories(
        CATEGORIES['trained'], CATEGORIES['untrained'],
        'Trained (dense+conv+dropout)', 'Untrained (rand+xavier+kaiming)')

    # Dense vs Conv (within trained)
    compare_categories(
        ['trained_dense'], ['trained_conv'],
        'Trained Dense', 'Trained Conv')

    # Compressed vs Untrained
    compare_categories(
        CATEGORIES['compressed'], CATEGORIES['untrained'],
        'Compressed (pruned+quant+lottery)', 'Untrained')

    print()

    # -------------------------------------------------------------------------
    # Step 5: Can we detect the backdoor pattern?
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: BACKDOOR DETECTION")
    print("=" * 80)
    print()
    print("Question: Can geometric analysis detect the periodic spike pattern")
    print("hidden in adversarial_backdoor weights?")
    print()

    # Compare backdoor vs trained_dense (should look similar except for pattern)
    print("  Backdoor vs Trained Dense (same base distribution, different structure):")
    backdoor_results = []
    for gname, mname in all_gm_pairs:
        bd_vals = []
        td_vals = []
        for trial_data in all_metrics['adversarial_backdoor']:
            if gname in trial_data and mname in trial_data[gname]:
                bd_vals.append(trial_data[gname][mname])
        for trial_data in all_metrics['trained_dense']:
            if gname in trial_data and mname in trial_data[gname]:
                td_vals.append(trial_data[gname][mname])

        if len(bd_vals) < 5 or len(td_vals) < 5:
            continue
        if np.std(bd_vals) < 1e-12 and np.std(td_vals) < 1e-12:
            continue

        t_stat, p_val = stats.ttest_ind(bd_vals, td_vals, equal_var=False)
        d = cohens_d(bd_vals, td_vals)
        backdoor_results.append((gname, mname, d, p_val))

    backdoor_results.sort(key=lambda x: -abs(x[2]))
    n_bonf = len(backdoor_results)
    bonf_alpha = ALPHA / max(n_bonf, 1)

    sig_bd = 0
    print(f"  Top discriminating metrics (Bonferroni alpha={bonf_alpha:.2e}):")
    for gname, mname, d, p_val in backdoor_results[:15]:
        sig_marker = " ***" if (p_val < bonf_alpha and abs(d) >= 0.5) else ""
        if p_val < bonf_alpha and abs(d) >= 0.5:
            sig_bd += 1
        label = effect_size_label(d)
        print(f"    {gname:22s} {mname:20s} d={d:+7.2f} ({label:>10s})  p={p_val:.2e}{sig_marker}")

    print(f"\n  Total significant backdoor detections: {sig_bd}/{n_bonf}")

    # Also compare backdoor vs xavier (untrained baseline more similar in shape)
    print("\n  Backdoor vs Xavier Init:")
    for gname, mname in all_gm_pairs:
        bd_vals = []
        xv_vals = []
        for trial_data in all_metrics['adversarial_backdoor']:
            if gname in trial_data and mname in trial_data[gname]:
                bd_vals.append(trial_data[gname][mname])
        for trial_data in all_metrics['xavier_init']:
            if gname in trial_data and mname in trial_data[gname]:
                xv_vals.append(trial_data[gname][mname])

        if len(bd_vals) < 5 or len(xv_vals) < 5:
            continue
        if np.std(bd_vals) < 1e-12 and np.std(xv_vals) < 1e-12:
            continue

        t_stat, p_val = stats.ttest_ind(bd_vals, xv_vals, equal_var=False)
        d = cohens_d(bd_vals, xv_vals)

        if p_val < bonf_alpha and abs(d) >= 0.8:
            label = effect_size_label(d)
            print(f"    {gname:22s} {mname:20s} d={d:+7.2f} ({label:>10s})  p={p_val:.2e}")

    print()

    # -------------------------------------------------------------------------
    # Step 6: Shuffle validation for top findings
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: SHUFFLE VALIDATION")
    print("=" * 80)
    print()
    print("Testing whether top findings survive when we shuffle the byte order")
    print("(destroys sequential structure but preserves marginal distribution).")
    print()

    # Pick top 5 most discriminative (geometry, metric) pairs from Step 3
    # and the 3 most distinctive weight types
    top5_gm = sorted(significant_findings, key=lambda x: -abs(x['cohens_d']))[:5]
    if not top5_gm:
        print("  No significant findings to validate!")
    else:
        # Get unique (geometry, metric) and weight types
        gm_to_validate = list(set((f['geometry'], f['metric']) for f in top5_gm))[:5]
        wtypes_to_validate = list(set(f['wtype'] for f in top5_gm))[:3]

        print(f"  Validating {len(gm_to_validate)} metrics on {len(wtypes_to_validate)} weight types")
        print()

        for wtype in wtypes_to_validate:
            print(f"  Weight type: {wtype}")
            for gname, mname in gm_to_validate:
                # Original effect
                orig_vals = []
                base_vals = []
                for trial_data in all_metrics[wtype]:
                    if gname in trial_data and mname in trial_data[gname]:
                        orig_vals.append(trial_data[gname][mname])
                for trial_data in all_metrics[baseline_type]:
                    if gname in trial_data and mname in trial_data[gname]:
                        base_vals.append(trial_data[gname][mname])

                if len(orig_vals) < 5 or len(base_vals) < 5:
                    continue

                d_orig = cohens_d(orig_vals, base_vals)

                # Shuffled: regenerate the weight type data, shuffle bytes, re-analyze
                shuffled_vals = []
                for trial in range(N_TRIALS):
                    rng = np.random.default_rng(SEED_BASE + trial * 1000 + hash(wtype) % 10000)
                    data = GENERATORS[wtype](rng)
                    # Shuffle the byte order
                    rng2 = np.random.default_rng(SEED_BASE + trial * 7 + 999)
                    data_shuffled = data.copy()
                    rng2.shuffle(data_shuffled)
                    result = analyzer.analyze(data_shuffled, f"{wtype}_shuf_t{trial}")
                    mdict = result.to_dict()
                    if gname in mdict and mname in mdict[gname]:
                        shuffled_vals.append(mdict[gname][mname])

                if len(shuffled_vals) < 5:
                    continue

                d_shuffled = cohens_d(shuffled_vals, base_vals)

                # Did shuffling destroy the signal?
                destroyed = abs(d_shuffled) < abs(d_orig) * 0.5
                status = "SEQUENTIAL" if destroyed else "MARGINAL"

                print(f"    {gname:22s} {mname:18s}  d_orig={d_orig:+6.2f}  d_shuf={d_shuffled:+6.2f}  -> {status}")

            print()

    # -------------------------------------------------------------------------
    # Step 7: Which geometries detect which properties?
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 7: GEOMETRY-PROPERTY DETECTION MAP")
    print("=" * 80)
    print()

    # For each geometry, find which weight types it distinguishes from random_uniform
    # with large effect size
    print(f"{'Geometry':<26} {'Best detects (d>0.8 vs random_uniform)':}")
    print("-" * 80)

    for gname in set(gn for gn, _ in all_gm_pairs):
        detections = []
        for mname in set(mn for gn, mn in all_gm_pairs if gn == gname):
            for wtype in GENERATORS:
                if wtype == baseline_type:
                    continue
                base_vals = []
                test_vals = []
                for trial_data in all_metrics[baseline_type]:
                    if gname in trial_data and mname in trial_data[gname]:
                        base_vals.append(trial_data[gname][mname])
                for trial_data in all_metrics[wtype]:
                    if gname in trial_data and mname in trial_data[gname]:
                        test_vals.append(trial_data[gname][mname])

                if len(base_vals) < 5 or len(test_vals) < 5:
                    continue
                if np.std(base_vals) < 1e-12 and np.std(test_vals) < 1e-12:
                    continue

                d = cohens_d(test_vals, base_vals)
                if abs(d) > 0.8:
                    detections.append((wtype, mname, d))

        if detections:
            # Summarize: which weight types does this geometry catch?
            caught_types = set(wt for wt, _, _ in detections)
            best = max(detections, key=lambda x: abs(x[2]))
            det_str = ", ".join(sorted(caught_types))
            print(f"  {gname:<24} {det_str}")
            print(f"    {'':24} Best: {best[1]} on {best[0]} (d={best[2]:+.2f})")
        else:
            print(f"  {gname:<24} (no large effects)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Count significant findings per weight type
    findings_by_type = defaultdict(int)
    findings_by_geom = defaultdict(int)
    for f in significant_findings:
        findings_by_type[f['wtype']] += 1
        findings_by_geom[f['geometry']] += 1

    print("Significant geometric differences from random uniform (d>=0.5, Bonferroni-corrected):")
    print()
    print(f"  {'Weight Type':<25} {'# Sig Metrics':>15} {'Max |d|':>10}")
    print(f"  {'-'*25} {'-'*15} {'-'*10}")
    for wtype in GENERATORS:
        if wtype == baseline_type:
            continue
        n_sig = findings_by_type.get(wtype, 0)
        max_d = 0
        for f in significant_findings:
            if f['wtype'] == wtype:
                max_d = max(max_d, abs(f['cohens_d']))
        print(f"  {wtype:<25} {n_sig:>15} {max_d:>10.2f}")

    print()
    print("Most discriminative geometries (# significant findings):")
    print()
    for gname, count in sorted(findings_by_geom.items(), key=lambda x: -x[1])[:10]:
        print(f"  {gname:<26} {count:>5} significant findings")

    print()

    # Final assessment
    print("KEY FINDINGS:")
    print("-" * 60)

    # Which types are most easily distinguished?
    easy = [(wt, c) for wt, c in findings_by_type.items() if c > len(all_gm_pairs) * 0.3]
    hard = [(wt, c) for wt, c in findings_by_type.items() if c < len(all_gm_pairs) * 0.05]

    if easy:
        print(f"\n  Easily distinguished from random:")
        for wt, c in sorted(easy, key=lambda x: -x[1]):
            print(f"    - {wt} ({c} metrics)")

    if hard:
        print(f"\n  Hard to distinguish from random:")
        for wt, c in sorted(hard, key=lambda x: x[1]):
            print(f"    - {wt} ({c} metrics)")

    # Backdoor detection verdict
    print(f"\n  Backdoor detection: ", end="")
    if sig_bd > 0:
        print(f"YES - {sig_bd} metrics detect the periodic spike pattern")
    else:
        print("NO - periodic spike pattern not reliably detected")

    print()
    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
