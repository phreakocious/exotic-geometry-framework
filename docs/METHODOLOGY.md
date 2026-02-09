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

The framework has two geometry sets that are used independently:

- **1D (byte streams)**: 24 geometries, ~131 metrics — used for all `investigations/1d/` and the classifier
- **2D (spatial fields)**: 8 geometries, 80 metrics — used for `investigations/2d/` (images, cellular automata, etc.)

A given investigation uses one set or the other, so the number of statistical tests per comparison is either ~131 or ~80. Without correction, we'd expect ~7 or ~4 false positives at alpha=0.05.

**Important**: the Bonferroni divisor is the number of metrics, NOT the number of metrics times conditions. Each pairwise comparison is a separate family of tests.

Bonferroni correction divides alpha by the number of tests:

```
alpha_corrected = 0.05 / n_tests
```

For 1D (131 metrics): alpha = 0.000382. For 2D (80 metrics): alpha = 0.000625. Only results surviving the relevant threshold are reported.

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
    t, p = scipy.stats.ttest_ind(values_a[metric], values_b[metric],
                                  equal_var=False)  # Welch's t-test
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
2. Cohen's d indicates a large effect size (|d| > 0.8, per Cohen's convention)
3. The finding is consistent across independent trials

Many of our effects far exceed this threshold — d > 10 is common, and d > 100 is not unusual for well-separated phenomena.

## Negative Results

We report negative results with equal rigor. When no metric survives Bonferroni correction, that IS the finding — the data types are geometrically indistinguishable. The most important negative result: AES-CTR output is indistinguishable from true randomness across all 24 1D geometries, all preprocessing methods, and all combination strategies.

## Geometric Signature Classifier

Beyond pairwise statistical testing, the framework includes a signature-based classifier that identifies byte streams by their geometric fingerprint.

### Training

A signature is built by running the analysis pipeline across many trials (typically 50) of a system's output, recording the mean and standard deviation of every metric at multiple delay-embedding scales (tau = 1, 2, 5, 10). This produces a high-dimensional profile: 131 metrics x 4 scales = 524 dimensions.

### Classification

Given an unknown byte stream, the classifier:

1. Computes the same 524-dimensional metric vector
2. For each signature, computes per-metric z-scores: `|observed - mean| / std`
3. Filters out constant metrics (`std < 1e-8`) and non-finite values
4. Ranks signatures by **median z-score** (robust to outlier metrics)
5. Reports **match fraction** (`% of metrics within 2 sigma`) and **gap-based confidence** (`(second_best - best) / second_best`)

### Key findings

The classifier reliably distinguishes 37 systems at 90% top-1 accuracy and 100% top-3 on held-out data. Two fundamental limits emerge:

- **High-entropy boundary (~7.0 bits/byte)**: Strong compression and encryption algorithms produce output above this threshold where all geometric structure is destroyed. Gzip, bzip2, xz, zstd, AES-CTR, SHA-256, and Mersenne Twister are all indistinguishable from each other. Only LZ4 (which stays below ~6.0) and Brotli retain enough structure for identification.

- **Text cluster**: All ASCII-based formats (XML, JSON, HTML, JavaScript, source code) share the same byte-frequency geometry dominated by the printable ASCII range. The geometry captures content type (text vs machine code vs compressed data) rather than format.

The strongest signature in the library is Mach-O Binary (compiled machine code): 93-96% match fraction, 95% confidence, z-scores of 0.15-0.24 across binaries from 84KB to 45MB. This identifies executables purely from the statistical geometry of their instruction bytes — no magic-byte headers needed.
