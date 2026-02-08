# Statistical Methodology

Every investigation in this framework follows the same rigorous protocol. This document explains why each step matters.

## The Protocol

1. **Generate data** from source and baseline (typically `os.urandom` or `np.random`)
2. **Run N trials** (typically 20-30) for each condition
3. **Compute geometric metrics** for every trial
4. **Compare conditions** using Welch's t-test
5. **Apply Bonferroni correction** for multiple comparisons
6. **Compute Cohen's d** effect size for every significant result
7. **Validate with shuffle baseline** to separate ordering from distribution effects

## Why Shuffled Baselines?

Early in this project, we found apparent "discoveries" that were actually artifacts of value distributions rather than sequential structure. A shuffled baseline is created by randomly permuting the same bytes — this preserves the distribution but destroys ordering.

If a metric is significant for real data but NOT for shuffled data, the detection is based on **sequential structure** (ordering matters).

If a metric is significant for BOTH real and shuffled data, the detection is based on **value distribution** (the particular bytes present, not their order).

Both are valid detections, but the distinction matters for interpretation. For example:
- ECB cipher mode has both ordering AND distribution effects (repeating blocks)
- DNA base composition is purely distributional (shuffling preserves GC content)
- Chaotic map attractors are ordering-dependent (shuffling destroys the attractor)

## Bonferroni Correction

With 24 geometries producing ~100 metrics, we perform ~100 statistical tests per comparison. Without correction, we'd expect ~5 false positives at alpha=0.05.

Bonferroni correction divides alpha by the number of tests:

```
alpha_corrected = 0.05 / n_tests
```

For 108 metrics: alpha = 0.000463. Only results surviving this threshold are reported.

This is conservative (some real effects may be missed), which is intentional — we prefer missed detections over false positives.

## Cohen's d Effect Size

P-values tell you whether an effect exists. Cohen's d tells you how large it is:

```
d = (mean_A - mean_B) / pooled_std
```

| d | Interpretation |
|---|---------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |
| 2.0+ | Very large |
| 10+ | Massive (common in this framework) |

We report d for every finding. Many of our effects are enormous (d > 10), reflecting that geometric embeddings amplify subtle structural differences.

## Standard Investigation Template

Every investigation script follows this pattern:

```python
N_TRIALS = 25
DATA_SIZE = 2000  # bytes per trial

# 1. Generate data
for trial in range(N_TRIALS):
    data_a = generate_condition_a(trial)
    data_b = generate_condition_b(trial)

    # 2. Compute metrics
    results_a = analyzer.analyze(data_a)
    results_b = analyzer.analyze(data_b)

# 3. Statistical comparison
alpha = 0.05 / n_metrics  # Bonferroni
for metric in all_metrics:
    t, p = scipy.stats.ttest_ind(values_a[metric], values_b[metric])
    d = cohens_d(values_a[metric], values_b[metric])
    if p < alpha:
        # 4. Report with effect size
        print(f"{metric}: d={d:.2f}, p={p:.2e}")

# 5. Shuffle validation
for significant_metric in significant_findings:
    shuffled_data = data_a.copy()
    np.random.shuffle(shuffled_data)
    # Compare real vs shuffled
```

## What "Significant" Means Here

A finding is reported as significant when ALL of these hold:
1. p-value survives Bonferroni correction (p < 0.05/n_tests)
2. Cohen's d indicates a meaningful effect size (typically |d| > 2)
3. The direction of the effect makes physical/mathematical sense
4. The finding is consistent across independent trials

## Negative Results

We report negative results with equal rigor. When no metric survives Bonferroni correction, that IS the finding — the data types are geometrically indistinguishable. The most important negative result: AES-CTR output is indistinguishable from true randomness across all 24 geometries, all preprocessing methods, and all combination strategies.
