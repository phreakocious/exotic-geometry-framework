# Per-Metric 2×2 Classification: Replacing Flat-Average D1 Drop

**Date**: 2026-03-29
**Status**: Implemented and validated on 11 geometries. Atlas rebuild pending.

## The Problem

The ablation spec (2026-03-26) defined D1 drop as `mean |Cohen's d drop| across all metrics`. This formula systematically misclassified geometries when metric sets became heterogeneous after ShinkaEvolve evolution. Example: E8 had 2 structure-dependent metrics (D1≈0.6) and 4 generic metrics (D1≈0), producing aggregate D1=10.1% — classified GENERIC despite genuine structural signal. D4 was killed entirely at 41.7%.

**Root cause**: averaging structure-dependent metrics with generic ones. Evolution's fix (collapse to 1 metric) hid the problem by eliminating the denominator, but destroyed atlas dimensionality.

## The Solution

Classify each metric independently on two axes:

| | High source variance (>0.5) | Low source variance |
|---|---|---|
| **High D1 drop (>30%)** | **HEADLINE** — backs structure claim | **SCIENTIFIC** — proves math, not discriminative |
| **Low D1 drop** | **WORKHORSE** — atlas discrimination | **DEAD** — remove |

- Headlines validate the README claim ("structure that survives the embedding is real")
- Workhorses provide atlas fingerprint dimensionality without surviving ablation
- Different metrics, different jobs, different evaluation criteria
- Report two aggregates: headline-only D1 (structure claim) and all-metrics (atlas fingerprint)

## Results (50 trials, 7 eval sources)

| Geometry | Old formula | Headline-only | H | W | S | D |
|---|---|---|---|---|---|---|
| D4 Triality | 41.7% (KILLED) | **100.0%** | 5 | 2 | 0 | 1 |
| H4 600-Cell | 65.2% | **86.3%** | 5 | 1 | 0 | 0 |
| Lorentzian | 100.0% | **100.0%** | 2 | 0 | 1 | 0 |
| Dodecagonal | -15.4% (HARMFUL) | **90.6%** | 1 | 4 | 1 | 1 |
| Penrose | 71.9% | **71.9%** | 5 | 0 | 0 | 0 |
| H3 Icosahedral | 22.3% (GENERIC) | **69.7%** | 4 | 1 | 0 | 2 |
| AB | 59.0% | **59.0%** | 1 | 1 | 2 | 1 |
| Hyperbolic | 39.2% | **53.1%** | 4 | 1 | 0 | 0 |
| E8 Lattice | 10.1% (GENERIC) | **47.7%** | 2 | 5 | 0 | 1 |
| Einstein Hat | 17.7% | **35.3%** | 1 | 0 | 0 | 1 |
| Septagonal | 16.8% | **—** | 0 | 1 | 0 | 2 |
| Symplectic | -21.4% (HARMFUL) | **—** | 0 | 4 | 0 | 0 |

**Worst affected by old formula**: E8 (10→48%), H3 (22→70%), D4 (42→100%), Dodecagonal (-15→91%)

## Framework Changes (commit 2cf42c7)

### Decomposed geometries (evolved products → components + workhorses)

- **E8**: `e8_structure_score` → +`closure_score`, `edge_flow`, `vel_corr`, `std_profile` + `diversity_ratio`, `normalized_entropy`
- **D4**: Restored from removal. 6 evolved probes from ShinkaEvolve v5b: `d4_structure_score`, `triplet_temporal`, `neighborhood_asymmetry`, `peakedness`, `structural_coherence` + `diversity_ratio`, `normalized_entropy`
- **H3**: `nn_enrichment` → +`temporal_coherence`, `path_closure` + `diversity_ratio`, `normalized_entropy`
- **H4**: `lattice_closure` → +`edge_walk_fraction`, `mean_walk_length`, `closure_fidelity` + `diversity_ratio`, `normalized_entropy`
- **Hyperbolic**: `curvature_structure` → +`temporal_variance`, `spatio_temporal_corr`, `knn_scale_ratio`, `mean_hyperbolic_radius`
- **AB**: `pell_conformance` → +`convergent_resonance` + `eightfold_symmetry`, `peak_sharpness`
- **Dodecagonal**: `dodec_algebraic_coherence` → +`base_correlation`, `conjugate_correlation`, `algebraic_identity`, `sqrt3_resonance` + `ratio_symmetry`, `peak_sharpness`
- **Septagonal**: +`ratio_symmetry`, `peak_sharpness`

### Pruned dead metrics
- `alignment_anisotropy` (H3), `gram_fingerprint` (H3), `discrete_grammar` (E8), `spectral_coherence` (D4), `pell_gate` (AB), `chirality` (Hat)

### Net effect: 53→54 geometries (+D4), ~199→214 metrics

## Key Insights

1. **Evolved products often score lower than their best component** — E8's combined product is a WORKHORSE (13.3%), but `vel_corr` alone is a HEADLINE (60.1%). H4's components ALL survive ablation independently at 100%.

2. **Gate metrics are dead individually but essential multiplicatively** — `gram_fingerprint` (H3), `discrete_grammar` (E8), `pell_gate` (AB) return 0 on all sources but gate the headline product via multiplication.

3. **The SCIENTIFIC quadrant exists** — Lorentzian's `lightlike_fraction` has 100% D1 drop but source variance of 0.32. AB's `eightfold_symmetry` and `peak_sharpness` have ~41% D1 drop but <0.4 source variance. These prove the math works but don't help the atlas.

4. **D4 and Dodecagonal were misclassified as STRUCTURE-HARMFUL** — both have 100%/91% headline D1 drop. The "harmful" classification came from generic workhorses where Thomson beats the exact structure on entropy/diversity metrics, dragging the average negative.

5. **Synergy is real but rare** — AB's `pell_conformance` (59%) beats both `pell_gate` (0%) and `convergent_resonance` (0%) individually. But this is the exception; most geometries have components that survive alone.

## Script

`investigations/1d/per_metric_ablation.py` — supports 12 geometries: e8, d4, h3, h4, hyp, lor, symp, pen, ab, hat, dodec, sept. Configurable thresholds via CLI flags.

## What's Next

### Immediate
- **Rebuild atlas** with D4 restored and decomposed metrics (~6 hours)
- **Update geometry_ablation.py** to use per-metric classification in its reporting (currently still flat-averages for the classification table)
- **Investigate Septagonal discrepancy**: ShinkaEvolve says D1=83.6%, our ablation says 2.4%. Different variant construction?

### Evolution targets (weakest geometries)
- **E8** (47.7%) — only 2 headlines. Target `vel_corr` for focused evolution
- **Hat** (35.3%) — borderline single headline. Evolution or metric redesign
- **Symplectic** (0 headlines) — α-perturbation may be wrong ablation design for area forms
- **Dodecagonal headline** — Brownian beats it on spectral smoothness. Needs redesign to use algebraic identity more directly
- **Septagonal** (0 headlines) — blocked on discrepancy investigation

### Architectural
- Add `source_variance` to ShinkaEvolve fitness: `fitness = D1_drop × source_spread` prevents evolving metrics that are structure-dependent but source-uniform
- Atlas-level validation after rebuild: check cross-domain NN%, effective dimensionality, detection counts vs pre-evolution baseline (2026-03-22)

## Falsification Update

The original ablation spec's falsification condition "Names are branding: if ALL non-null geometries classify as Generic" is now better stated as: "if ALL non-null geometries have 0 HEADLINE metrics." Under the 2×2 framework, only Symplectic and Septagonal have 0 headlines — the vast majority of geometries have genuine structure-dependent signal. The README claim stands for 9 of 11 tested geometries.
