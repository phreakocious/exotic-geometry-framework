#!/usr/bin/env python3
"""
Investigation: Metric Characterization — Calibrating the Sensors
==============================================================

The framework has ~131 metrics (sensors). Some might be "generalists" (like entropy,
firing for anything complex), while others might be "specialists" (firing only
for specific algebraic structures).

This investigation characterizes the *sensors themselves*, not the data. We treat
the metrics as the subjects of investigation.

DIRECTIONS:
D1: Specialist vs Generalist — Compute a "Selectivity Score" (Gini coefficient)
    for each metric across a diverse "Zoo" of data types.
D2: The Entropy Shadow — Identify metrics that are UNCORRELATED with Shannon
    entropy. These are the "Dark Sensors" providing unique information.
D3: Sensitivity / SNR — Which metrics have the highest signal-to-noise ratio
    (inter-class variance / intra-class trial variance)?
D4: Orthogonality — Cluster metrics to find unique, independent axes.
    (Redundancy check).
D5: Rare Signatures — Identify metrics that are "silent" (0 or constant) for
    90% of the zoo but "scream" (high value) for exactly one type.

The "Zoo":
- Noise (White)
- Periodic (Sine)
- Chaotic (Logistic)
- Bio (DNA-like)
- Text (English-like)
- Sparse (Many zeros)
- Number Theory (Primes)
"""

import sys
import os
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
SEED = 42
np.random.seed(SEED)

# ==============================================================
# DATA ZOO GENERATORS
# ==============================================================

def gen_white_noise(rng, size):
    return rng.integers(0, 256, size, dtype=np.uint8)

def gen_periodic_sine(rng, size):
    t = np.linspace(0, 8*np.pi, size)
    # Add slight phase jitter so trials aren't identical
    phase = rng.uniform(0, 2*np.pi)
    x = np.sin(t + phase)
    return ((x + 1) * 127.5).astype(np.uint8)

def gen_logistic_chaos(rng, size):
    x = 0.1 + 0.8 * rng.random()
    for _ in range(100): x = 4.0 * x * (1 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 4.0 * x * (1 - x)
        vals[i] = int(x * 255)
    return vals

def gen_dna_model(rng, size):
    # ATCG map to 4 values, spread out
    bases = np.array([0, 85, 170, 255], dtype=np.uint8)
    return rng.choice(bases, size=size)

def gen_sparse_bursts(rng, size):
    # Mostly zeros, occasional bursts of value
    arr = np.zeros(size, dtype=np.uint8)
    num_bursts = size // 50
    for _ in range(num_bursts):
        idx = rng.integers(0, size - 10)
        arr[idx:idx+10] = rng.integers(0, 256, 10, dtype=np.uint8)
    return arr

def gen_primes_proxy(rng, size):
    # Simple proxy: numbers with few factors (mocking prime gaps structure)
    # Just a stand-in for "algebraic structure"
    # Let's use a Linear Congruential Generator (LCG) with poor parameters
    # to create lattice structure
    a, c, m = 1664525, 1013904223, 2**32
    state = rng.integers(0, m)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        state = (a * state + c) % m
        vals[i] = (state >> 24) & 0xFF
    return vals

ZOO = {
    'Noise': gen_white_noise,
    'Sine': gen_periodic_sine,
    'Chaos': gen_logistic_chaos,
    'DNA': gen_dna_model,
    'Sparse': gen_sparse_bursts,
    'LCG': gen_primes_proxy
}

# ==============================================================
# HELPER: NORMALIZE METRICS
# ==============================================================
def normalize_metric_dict(all_data):
    """
    input: dict { type: { metric: [values] } }
    output: dict { metric: { type: mean_value } } (0-1 normalized across types)
    """
    # 1. Pivot to metric-first
    metric_raw = defaultdict(dict)
    for type_name, metrics in all_data.items():
        for m_name, vals in metrics.items():
            metric_raw[m_name][type_name] = np.mean(vals)
            
    # 2. Normalize each metric to 0-1 range across the types
    metric_norm = {}
    for m_name, type_vals in metric_raw.items():
        vals = list(type_vals.values())
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            # Constant metric (broken sensor?)
            continue
        
        metric_norm[m_name] = {
            t: (v - vmin) / (vmax - vmin) 
            for t, v in type_vals.items()
        }
    return metric_norm, metric_raw

# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Specialist vs Generalist (Gini Coeff)"""
    print("\n" + "=" * 60)
    print("D1: SPECIALIST VS GENERALIST SENSORS")
    print("=" * 60)
    
    # 1. Collect data for entire Zoo
    zoo_data = {}
    for name, gen in ZOO.items():
        chunks = [gen(rng, runner.data_size) for rng in runner.trial_rngs()]
        zoo_data[name] = runner.collect(chunks)
        
    # 2. Normalize
    norm_data, raw_data = normalize_metric_dict(zoo_data)
    
    # 3. Calculate Gini/Selectivity for each metric
    # A specialist has value 1.0 for one type and 0.0 for others (High Gini)
    # A generalist has 0.5 for everything (Low Gini)
    
    results = []
    for m_name, type_scores in norm_data.items():
        scores = np.array(list(type_scores.values()))
        # Gini coefficient calculation
        # Simple proxy: Max / Sum (if sum > 0)
        # Or standard deviation of the normalized scores
        selectivity = np.std(scores) # 0 to ~0.5
        
        # Determine "Favorite" type
        best_type = max(type_scores, key=type_scores.get)
        results.append((m_name, selectivity, best_type))
        
    # Sort by Selectivity
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 SPECIALIST Sensors (High Selectivity):")
    for m, s, t in results[:10]:
        print(f"  {m[:40]:<40} Score: {s:.3f} | Target: {t}")

    print("\nTop 10 GENERALIST Sensors (Low Selectivity):")
    for m, s, t in results[-10:]:
        print(f"  {m[:40]:<40} Score: {s:.3f} | Target: {t}")
        
    return dict(zoo_data=zoo_data, rankings=results, norm_data=norm_data)

def direction_2(runner, d1_data):
    """D2: The Entropy Shadow (Correlation vs Entropy)"""
    print("\n" + "=" * 60)
    print("D2: THE ENTROPY SHADOW")
    print("=" * 60)
    
    zoo_data = d1_data['zoo_data']
    
    # Flatten all data to compute correlation across the whole "universe"
    # We want correlation of (Metric_X) vs (Shannon_Entropy) across all trials of all types
    
    # 1. Calculate Shannon Entropy for every single chunk in the zoo
    # (Re-generate or use the metric if it exists. We assume 'shannon_entropy' isn't standard in exotic set?
    # Actually, let's calculate it manually for the reference)
    
    # Re-generating for exact alignment is safer or extracting if available.
    # Let's assume we need to compute it.
    
    # Use per-type mean values for correlation (avoids length mismatch
    # from collect() dropping non-finite values across trials)
    first_type = list(zoo_data.keys())[0]
    metric_names = list(zoo_data[first_type].keys())
    type_names = list(ZOO.keys())

    # Compute mean entropy per type
    type_entropies = {}
    for name, gen in ZOO.items():
        ents = []
        for rng in runner.trial_rngs():
            chunk = gen(rng, runner.data_size)
            _, counts = np.unique(chunk, return_counts=True)
            p = counts / len(chunk)
            ents.append(-np.sum(p * np.log2(p)))
        type_entropies[name] = ents

    # Correlate each metric with entropy across all trials of all types
    correlations = []

    for m in metric_names:
        x_metric = []
        y_entropy = []
        for name in type_names:
            m_vals = zoo_data[name].get(m, [])
            e_vals = type_entropies[name]
            # Use the shorter of the two lists
            n = min(len(m_vals), len(e_vals))
            if n < 3:
                continue
            x_metric.extend(m_vals[:n])
            y_entropy.extend(e_vals[:n])

        x_metric = np.array(x_metric)
        y_entropy = np.array(y_entropy)

        if len(x_metric) < 5 or np.std(x_metric) < 1e-9:
            r = 0.0
        else:
            r, _ = stats.pearsonr(x_metric, y_entropy)
            if not np.isfinite(r):
                r = 0.0

        correlations.append((m, r))
        
    # Sort by abs(correlation) ascending (We want LOW correlation)
    correlations.sort(key=lambda x: abs(x[1]))
    
    print("\nTop 10 'DARK SENSORS' (Uncorrelated with Entropy):")
    for m, r in correlations[:10]:
        print(f"  {m[:40]:<40} r={r:.3f}")
        
    print("\nTop 5 'ENTROPY PROXIES' (Highly Correlated):")
    for m, r in correlations[-5:]:
        print(f"  {m[:40]:<40} r={r:.3f}")
        
    return dict(correlations=correlations)

def direction_3(runner, d1_data):
    """D3: Sensitivity / SNR"""
    print("\n" + "=" * 60)
    print("D3: SENSITIVITY (SNR)")
    print("=" * 60)
    
    # SNR = Variance_Between_Classes / Mean_Variance_Within_Classes
    zoo_data = d1_data['zoo_data']
    metric_names = list(zoo_data['Noise'].keys())
    
    snr_scores = []
    
    for m in metric_names:
        # Gather all means and vars
        means = []
        vars_within = []
        
        for name in ZOO.keys():
            vals = zoo_data[name].get(m, [0])
            means.append(np.mean(vals))
            vars_within.append(np.var(vals))
            
        var_between = np.var(means)
        mean_var_within = np.mean(vars_within)
        
        if mean_var_within < 1e-12:
            snr = 0 # Avoid division by zero (or infinite if signal exists)
            if var_between > 1e-9:
                snr = 999999 # Perfect deterministic separator
        else:
            snr = var_between / mean_var_within
            
        snr_scores.append((m, snr))
        
    snr_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 'LOUD' SENSORS (High SNR):")
    for m, snr in snr_scores[:10]:
        print(f"  {m[:40]:<40} SNR: {snr:.1f}")
        
    return dict(snr_scores=snr_scores)

def direction_4(runner, d1_data):
    """D4: Orthogonality (Redundancy Check)"""
    print("\n" + "=" * 60)
    print("D4: ORTHOGONALITY")
    print("=" * 60)
    
    # We want to find metrics that are unique.
    # We can use the correlation matrix of metrics against each other.
    
    # 1. Build Data Matrix (Samples x Metrics)
    # Re-use the flattened arrays from D2 logic
    # rows = samples (25 * 6 = 150), cols = metrics (~130)
    
    zoo_data = d1_data['zoo_data']
    first_type = list(zoo_data.keys())[0]
    metric_names = list(zoo_data[first_type].keys())
    
    matrix = []
    valid_metrics = []
    
    for m in metric_names:
        vals = []
        for name in ZOO.keys():
            vals.extend(zoo_data[name][m])
        
        if np.std(vals) > 1e-9:
            matrix.append(vals)
            valid_metrics.append(m)
            
    matrix = np.array(matrix) # Shape (n_metrics, n_samples)
    
    # Correlation matrix
    corr_mat = np.corrcoef(matrix)
    
    # Find "Lonely" metrics (lowest max correlation with any other metric)
    loneliness = []
    for i in range(len(valid_metrics)):
        # correlation with self is 1.0, ignore that
        # set diagonal to 0 for search
        row = corr_mat[i].copy()
        row[i] = 0
        max_corr = np.max(np.abs(row))
        loneliness.append((valid_metrics[i], max_corr))
        
    loneliness.sort(key=lambda x: x[1]) # Ascending (lower max_corr = more unique)
    
    print("\nTop 10 UNIQUE SENSORS (Least correlated with others):")
    for m, c in loneliness[:10]:
        print(f"  {m[:40]:<40} MaxCorr: {c:.3f}")
        
    return dict(loneliness=loneliness)

def direction_5(runner, d1_data):
    """D5: Rare Signatures / Needle in Haystack"""
    print("\n" + "=" * 60)
    print("D5: RARE SIGNATURES")
    print("=" * 60)
    
    # Identify metrics that are essentially 0 for 5 out of 6 types, 
    # but high for 1 type.
    
    norm_data = d1_data['norm_data']
    
    special_cases = []
    
    for m, type_vals in norm_data.items():
        vals = list(type_vals.values())
        # Check if median is low (< 0.2) but max is high (> 0.8)
        if np.median(vals) < 0.2 and np.max(vals) > 0.8:
            target = max(type_vals, key=type_vals.get)
            special_cases.append((m, target))
            
    print(f"\nFound {len(special_cases)} 'Needle' Detectors:")
    for m, t in special_cases[:10]:
        print(f"  {m[:40]:<40} Detects: {t}")
        
    return dict(special_cases=special_cases)

# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(6, "Metric Characterization: Sensor Datasheet",
                                     rows=3, cols=2, figsize=(18, 16))

    # D1: Top Specialists — horizontal bars with full names
    top_spec = d1['rankings'][:15]
    names1 = [x[0] for x in top_spec]
    vals1 = [x[1] for x in top_spec]
    targets1 = [x[2] for x in top_spec]
    # Color by target type
    type_colors = {'Noise': '#95a5a6', 'Sine': '#2ecc71', 'Chaos': '#e74c3c',
                   'DNA': '#1abc9c', 'Sparse': '#e67e22', 'LCG': '#9b59b6'}
    colors1 = [type_colors.get(t, '#888888') for t in targets1]
    y_pos = np.arange(len(names1))
    axes[0].barh(y_pos, vals1, color=colors1, alpha=0.9, height=0.7)
    # Truncate names for readability
    short_names1 = [n if len(n) <= 35 else n[:33] + '..' for n in names1]
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(short_names1, fontsize=7)
    axes[0].set_xlabel('Selectivity (std of normalized scores)', fontsize=8)
    axes[0].set_title('D1: Top Specialist Sensors', fontsize=10, fontweight='bold')
    axes[0].invert_yaxis()
    for bar, val, tgt in zip(axes[0].patches, vals1, targets1):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f} [{tgt}]', va='center', fontsize=6.5, color='#cccccc')

    # D2: Entropy Correlation histogram
    corrs = [x[1] for x in d2['correlations']]
    axes[1].hist(corrs, bins=25, color='#3498db', alpha=0.8, edgecolor='#444444')
    axes[1].axvline(0, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Pearson r with Shannon entropy', fontsize=8)
    axes[1].set_ylabel('Count', fontsize=8)
    axes[1].set_title('D2: Entropy Correlation Distribution', fontsize=10,
                      fontweight='bold')
    n_dark = len([x for x in d2['correlations'] if abs(x[1]) < 0.1])
    axes[1].text(0.02, 0.95, f'{n_dark} dark sensors (|r| < 0.1)',
                 transform=axes[1].transAxes, fontsize=8, color='#e74c3c',
                 va='top', bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='#222222', edgecolor='#444444'))

    # D3: Top SNR — horizontal bars, log scale
    top_snr = d3['snr_scores'][:15]
    names3 = [x[0] for x in top_snr]
    vals3 = [np.log10(max(x[1], 1)) for x in top_snr]
    short_names3 = [n if len(n) <= 35 else n[:33] + '..' for n in names3]
    y_pos3 = np.arange(len(names3))
    axes[2].barh(y_pos3, vals3, color='#e67e22', alpha=0.9, height=0.7)
    axes[2].set_yticks(y_pos3)
    axes[2].set_yticklabels(short_names3, fontsize=7)
    axes[2].set_xlabel('log10(SNR)', fontsize=8)
    axes[2].set_title('D3: Loudest Sensors (SNR)', fontsize=10, fontweight='bold')
    axes[2].invert_yaxis()
    for bar, snr_raw in zip(axes[2].patches, [x[1] for x in top_snr]):
        axes[2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{snr_raw:.1e}', va='center', fontsize=6.5, color='#cccccc')

    # D4: Most Unique — horizontal bars
    top_unique = d4['loneliness'][:15]
    names4 = [x[0] for x in top_unique]
    vals4 = [x[1] for x in top_unique]
    short_names4 = [n if len(n) <= 35 else n[:33] + '..' for n in names4]
    y_pos4 = np.arange(len(names4))
    axes[3].barh(y_pos4, vals4, color='#2ecc71', alpha=0.9, height=0.7)
    axes[3].set_yticks(y_pos4)
    axes[3].set_yticklabels(short_names4, fontsize=7)
    axes[3].set_xlabel('Max |correlation| with any other metric', fontsize=8)
    axes[3].set_title('D4: Most Orthogonal Sensors', fontsize=10, fontweight='bold')
    axes[3].invert_yaxis()
    for bar, val in zip(axes[3].patches, vals4):
        axes[3].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=6.5, color='#cccccc')

    # D5: Rare Detectors by target type
    targets = [x[1] for x in d5['special_cases']]
    if targets:
        unique_t, counts = np.unique(targets, return_counts=True)
        colors5 = [type_colors.get(t, '#888888') for t in unique_t]
        bars5 = axes[4].bar(range(len(unique_t)), counts, color=colors5, alpha=0.9)
        axes[4].set_xticks(range(len(unique_t)))
        axes[4].set_xticklabels(unique_t, fontsize=8)
        axes[4].set_ylabel('Number of needle detectors', fontsize=8)
        for bar, c in zip(bars5, counts):
            axes[4].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         str(c), ha='center', fontsize=9, color='white')
    axes[4].set_title('D5: Needle Detectors per Type', fontsize=10, fontweight='bold')

    # Panel 6: Type legend + key stats
    axes[5].axis('off')
    lines = [
        "SENSOR DATASHEET SUMMARY",
        "",
        f"Zoo: {', '.join(ZOO.keys())} ({len(ZOO)} types, 25 trials each)",
        f"Metrics: {len(d1['rankings'])} characterized",
        "",
        f"Dark sensors (|r| < 0.1 with entropy): {n_dark}",
        f"  Darkest: {d2['correlations'][0][0]}",
        "",
        f"Loudest: {d3['snr_scores'][0][0]}",
        f"  SNR = {d3['snr_scores'][0][1]:.1e}",
        "",
        f"Most orthogonal: {d4['loneliness'][0][0]}",
        f"  Max corr = {d4['loneliness'][0][1]:.3f}",
        "",
        f"Needle detectors: {len(d5['special_cases'])}",
    ]
    for i, line in enumerate(lines):
        weight = 'bold' if i == 0 else 'normal'
        size = 10 if i == 0 else 7.5
        color = 'white' if i == 0 else '#cccccc'
        axes[5].text(0.05, 0.95 - i * 0.06, line, transform=axes[5].transAxes,
                     fontsize=size, color=color, fontweight=weight,
                     fontfamily='monospace', va='top')

    runner.save(fig, "metric_characterization")

# ==============================================================
# MAIN
# ==============================================================
def main():
    runner = Runner("Sensor Calibration", mode="1d")
    
    d1 = direction_1(runner)
    d2 = direction_2(runner, d1)
    d3 = direction_3(runner, d1)
    d4 = direction_4(runner, d1)
    d5 = direction_5(runner, d1)
    
    make_figure(runner, d1, d2, d3, d4, d5)
    
    runner.print_summary({
        'D1': f"Top Specialist: {d1['rankings'][0][0].split(':')[0]}",
        'D2': f"Dark Sensors found: {len([x for x in d2['correlations'] if abs(x[1]) < 0.1])}",
        'D3': f"Max SNR: {d3['snr_scores'][0][1]:.1e}",
        'D4': f"Most Unique: {d4['loneliness'][0][0].split(':')[0]}",
        'D5': f"Needle Detectors: {len(d5['special_cases'])}"
    })

if __name__ == "__main__":
    main()
