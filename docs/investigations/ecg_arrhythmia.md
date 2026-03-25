# Investigation: ECG Arrhythmia Detection via Geometric Structure

**Date:** March 24, 2026
**Script:** `investigations/1d/ecg_arrhythmia.py`

## Objective

Apply the exotic geometry framework to **preprocessed ECG heartbeats** to determine whether geometric structure alone can detect and classify cardiac arrhythmias — without ML training, without ECG-specific feature engineering, and from short recordings (a few seconds of data).

The clinical question: can a purely geometric approach distinguish normal sinus rhythm from ventricular ectopy, supraventricular ectopy, and fusion beats? And how quickly — how few beats are needed for reliable detection?

## Data Source

**MIT-BIH Arrhythmia Database** (Kaggle preprocessed variant).

- **Beats:** 87,554 total, 187 samples each (normalized [0,1])
- **Classes:** Normal (72,471), Supraventricular (2,223), Ventricular (5,788), Fusion (641)
- **Sample rate:** 360 Hz (each beat spans ~0.5s)

**PTB Diagnostic ECG Database** (validation).

- **Normal:** 4,046 beats
- **Abnormal:** 10,506 beats

## Methodology

### Sequence Construction

For each class, 40 sequences are constructed by randomly concatenating N beats from the same class, then encoding the concatenation to uint8 via percentile clipping (p0.5, p99.5). This preserves inter-beat temporal dynamics while controlling sequence length.

### Statistical Comparison

Bonferroni-corrected Welch t-test (alpha = 0.05/200 = 2.5x10^-4) with |Cohen's d| > 0.8 threshold. All 200 framework metrics tested per comparison.

### Classification

Leave-one-out nearest-centroid on the top-K significant metrics. No model training — classification uses only geometric distances to class centroids in metric space.

## Findings

### 1. Detection Speed: Normal vs Ventricular

| Beats | Duration (at 360 Hz) | Significant metrics | Top \|d\| | Top-1 accuracy | Top-5 accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5s | 33 | 1.98 | 82.5% | 82.5% |
| 2 | 1.0s | 43 | 2.23 | 90.0% | 86.2% |
| 3 | 1.6s | 48 | 2.82 | 88.8% | 88.8% |
| 5 | 2.6s | 60 | 4.54 | 97.5% | 97.5% |
| 8 | 4.2s | 64 | 4.83 | 98.8% | 98.8% |
| 11 | 5.7s | 81 | 5.76 | 100% | 100% |

Even a single beat (0.5 seconds, 187 samples) produces 33 significant metrics with d=1.98 and 82.5% accuracy. The 95% accuracy threshold is crossed at 5 beats (2.6 seconds of ECG). At 11 beats (5.7 seconds), discrimination is perfect: 100% leave-one-out accuracy with 81 significant metrics.

Effect sizes grow monotonically with beat count — more beats average out beat-to-beat variability, sharpening the geometric separation.

### 2. All-Pairs Discrimination (11 beats)

| Pair | Significant metrics | Top \|d\| |
|:---|:---:|---:|
| Normal vs Supraventricular | 46 | 2.49 |
| Normal vs Ventricular | 81 | 5.76 |
| Normal vs Fusion | 121 | 5.75 |
| Supraventricular vs Ventricular | 76 | 5.46 |
| Supraventricular vs Fusion | 121 | 6.97 |
| Ventricular vs Fusion | 129 | 7.97 |

All six class pairs are separable. The hardest pair is Normal vs Supraventricular (46 significant metrics, d=2.49) — clinically expected, since supraventricular beats originate above the ventricles and can resemble normal morphology. The easiest pair is Ventricular vs Fusion (129 metrics, d=7.97), reflecting very different phase-space geometries.

### 3. Top Discriminating Metrics (Normal vs Ventricular, 11 beats)

| Metric | Cohen's d |
|:---|---:|
| p-Variation:var_p2 | +5.761 |
| Symplectic:total_area | +5.607 |
| Zariski:algebraic_residual | +3.769 |
| Symplectic:windowed_area_cv | -3.582 |
| Spectral Analysis:spectral_bandwidth | +3.362 |
| Spectral Analysis:spectral_r2 | -3.240 |
| Ordinal Partition:markov_mixing | +3.183 |
| Higher-Order Statistics:kurt_max | -3.086 |
| Holder Regularity:hurst_exponent | -2.982 |
| Hyperbolic (Poincare):mean_hyperbolic_radius | +2.791 |

The top two metrics (p-Variation `var_p2` and Symplectic `total_area`) both exceed d=5.5, reflecting the dominant morphological difference: ventricular beats have wider QRS complexes, producing larger phase-space excursions.

### 4. Best Single Metric

p-Variation `var_p2` achieves d=+5.761 with **zero distribution overlap** between classes:

- Normal: mean = 0.0097 +/- 0.0007
- Ventricular: mean = 0.0053 +/- 0.0008

A single threshold on this one number perfectly separates all 80 test sequences. p-variation at order 2 measures the total quadratic variation of the signal path — ventricular beats, with their broad, high-amplitude QRS deflections, trace paths with consistently different total variation than normal beats.

### 5. PTB Validation

The framework generalizes to a different ECG dataset without retuning:

- **96 significant metrics** (more than MIT-BIH's 81)
- **Top-5 LOO accuracy: 100%**
- **Top metrics differ from MIT-BIH:** Holder Regularity:holder_std (d=-6.393), Klein Bottle:orientation_coherence (+4.477), Ordinal Partition:forbidden_transitions (+4.209)

The fact that different metrics dominate in each dataset is informative: it means the framework is not overfitting to one morphological feature. MIT-BIH separates beat types within the same patients; PTB separates healthy patients from cardiac patients. The framework captures multiple independent geometric aspects of cardiac pathology — phase-space variation dominates for beat-type classification, while regularity and orientation structure dominate for patient-level classification.

### 6. Physical Interpretation

- **p-Variation and Symplectic total_area:** Ventricular ectopic beats originate below the His bundle, producing wide QRS complexes (>120ms). Wider deflections create larger phase-space excursions and higher quadratic variation. These are geometric proxies for QRS duration — the primary clinical discriminator.
- **Hurst exponent (lower for ventricular):** Ventricular ectopy is more irregular and less self-similar than normal sinus rhythm. Normal beats have more consistent morphology beat-to-beat.
- **Spectral bandwidth (higher for ventricular):** Wide QRS complexes spread energy across a broader frequency range than the sharp, narrow QRS of normal conduction.
- **Ordinal Partition markov_mixing:** Different temporal transition dynamics between beat morphologies — ventricular beats create different ordinal pattern sequences than normal beats, reflecting the altered depolarization pathway.
- **Klein Bottle orientation_coherence (PTB):** Abnormal beats have more consistent GF(2) orientation structure, potentially reflecting the stereotyped nature of pathological conduction patterns.

### 7. Interpretation Scale

| Significant metrics | Interpretation | This investigation |
|:---|:---|:---|
| 0-5 | No geometric difference (noise-equivalent) | |
| 5-30 | Mild structural difference | |
| 30-100 | Strong structural difference | **N vs S: 46, N vs V: 81** |
| 100+ | Very strong structural difference | **N vs F: 121, S vs F: 121, V vs F: 129** |

For reference:
- SETI 3I/ATLAS (real radio telescope, null result): 0 metrics
- Solar eclipse VLF (real ionospheric effect): 14 metrics
- ECG Normal vs Ventricular: 81 metrics
- ECG Ventricular vs Fusion: 129 metrics

ECG arrhythmia detection sits firmly in the strong-to-very-strong range, consistent with the fact that cardiac beat morphology differences are among the most robust biological signals available.

### 8. Caveats

- **Preprocessed data:** The Kaggle variant provides pre-segmented, normalized, fixed-length beats. Real clinical deployment requires beat detection and segmentation upstream of the framework.
- **Class imbalance:** Normal beats outnumber Fusion beats 113:1. The random sampling procedure equalizes class representation in the test sequences, but the Fusion class has only 641 source beats to draw from.
- **No Unknown class:** Class Q (Unknown/Unclassifiable) was excluded. These ambiguous beats would likely produce weaker geometric separation.
- **Concatenation artifacts:** Concatenating beats from different time points may introduce boundary discontinuities. The percentile encoding mitigates this, but beat-boundary effects are not explicitly modeled.
- **No patient-level split:** MIT-BIH analysis does not control for patient identity. Some geometric separation could reflect inter-patient variation rather than pure arrhythmia morphology. The PTB cross-dataset validation partially addresses this concern.

## Conclusion

The exotic geometry framework achieves **strong arrhythmia detection** (81 significant metrics, d=5.76) from short ECG recordings using geometric structure alone — no ML training, no ECG-specific features. Detection reaches 95% accuracy at 5 beats (2.6 seconds) and 100% at 11 beats (5.7 seconds). All four beat-type pairs are separable, with the hardest pair (Normal vs Supraventricular) still producing 46 significant metrics.

The best single metric, p-Variation `var_p2`, achieves zero distribution overlap between Normal and Ventricular classes — a single number perfectly classifies. This is a geometric proxy for QRS width, the primary clinical discriminator, discovered without any cardiac domain knowledge built into the framework.

Cross-dataset validation on PTB confirms generalization: 96 significant metrics and 100% accuracy, with different top metrics reflecting different aspects of cardiac pathology. The framework detects arrhythmias through multiple independent geometric lenses — phase-space variation, spectral structure, regularity, orientation — rather than relying on any single engineered feature.
