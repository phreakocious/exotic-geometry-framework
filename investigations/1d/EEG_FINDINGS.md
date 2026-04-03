# EEG Brain State Geometry: Findings

**Date**: 2026-04-03
**Investigations**: `eeg_deep.py`, `eeg_oiris.py`, `eeg_oiris_polar.py`
**Framework**: 54 geometries, 278 metrics, 8 parallel workers

## Datasets

| Dataset | Source | Subjects | Conditions | Format |
|---------|--------|----------|------------|--------|
| PhysioNet eegmmidb | OpenNeuro | 109 (22 cached with runs 1-3) | Eyes open/closed, motor real/imagined | EDF, 160 Hz, 64ch |
| Bonn epilepsy | Local CSV | 11,500 segments | Seizure, tumor EO, healthy EO, eyes closed/open | 178 samples/seg |
| CHB-MIT | Local EDF | 22 recordings | Seizure vs interictal | EDF, 256 Hz |
| O.IRIS Meditation | OpenNeuro ds001787 | 10 (of 24) | Meditation | BDF, 2048 Hz, 64ch |
| O.IRIS Sleep/Rest | OpenNeuro ds003768 | 10 (of 33) | Resting, sleep | BrainVision, 5000 Hz, 32ch |

All signals resampled to 250 Hz, bandpass filtered 1-55 Hz, channel-averaged,
min-max quantized to uint8. 25 trials per condition, 16,384 samples per trial
(~65 seconds at 250 Hz).

---

## Finding 1: Meditation is structurally unique — driven by temporal persistence

**Evidence**: `eeg_oiris.py` D2

| Comparison | Sig metrics (of 278) |
|-----------|---------------------|
| Meditation vs Rest | 186 |
| Meditation vs Sleep | 189 |
| Rest vs Sleep | 0 |

**Top discriminating metric**: Sol (Thurston) `dz_persistence`, Cohen's d = +36.05
(meditation > rest). Sol geometry measures dynamics in a Thurston geometry where
one axis scales exponentially relative to the others. What it detects: meditation
EEG has extraordinarily stable local dynamics — the signal persists in whatever
regime it occupies.

**Supporting metrics** (meditation > rest/sleep, all p < 0.001):

| Metric | d (Med vs Rest) | d (Med vs Sleep) | What it measures |
|--------|:-:|:-:|---|
| Sol:dz_persistence | +36.05 | +26.24 | Temporal persistence in exponential-scaling geometry |
| H²×R:radial_temporal_memory | +18.33 | +17.02 | Long autocorrelation in hyperbolic embedding |
| Cantor Set:jump_entropy | +16.63 | +10.98 | Diversity of gap transitions (high = rich, stable gaps) |
| SL(2,R):trace_autocorrelation | +13.70 | +12.85 | Persistence of Lie algebra trace dynamics |
| Logarithmic Spiral:radial_acf | +11.99 | +14.17 | Radial autocorrelation in spiral embedding |

None of these are the 3 metrics O.IRIS currently uses (spectral slope, ACF decay,
flicker). The framework identifies temporal persistence structure that is invisible
to spectral slope analysis.

**Within-session stability** (`eeg_oiris.py` D8): 0-1 significant metrics between
early, middle, and late thirds of a meditation session. The geometric signature
does not drift.

---

## Finding 2: Rest and sleep are geometrically indistinguishable

**Evidence**: `eeg_oiris.py` D2 — 0 significant metrics out of 278.

O.IRIS separates rest (median spectral slope m = 0.59) from sleep (m = 0.89) on
the pupil map because spectral slope is a primary axis. Our framework analyzes
temporal structure — entropy, recurrence, topology, dynamics — and finds no
difference.

**Implication**: The rest/sleep distinction is purely a frequency-domain phenomenon.
Rest and sleep have the same temporal dynamics, the same entropy structure, the
same topological signatures. The spectral slope difference is real but it measures
frequency content, not structural organization.

**Visual confirmation**: The polar comparison figure (`figures/eeg_oiris_polar.png`)
shows rest (blue) and sleep (purple) completely interleaved in the EGF panel while
separated in the O.IRIS panel.

**Caveat**: This result is conditional on our encoding (channel-averaged, uint8
quantized, 250 Hz). Sleep staging information may exist in per-channel or
cross-channel features that channel averaging destroys.

---

## Finding 3: Meditation is the least nonlinear brain state

**Evidence**: `eeg_oiris.py` D4 (IAAFT surrogates)

| State | Sig metrics surviving IAAFT |
|-------|:--:|
| Meditation | 17 |
| Rest | 38 |
| Sleep | 43 |

IAAFT surrogates preserve the power spectrum and amplitude distribution while
randomizing phase relationships. Metrics that still discriminate original from
surrogate detect genuinely nonlinear dynamics — structure that cannot be explained
by the frequency content alone.

Meditation has the **fewest** nonlinear signatures. Its structure is predominantly
linear and spectral. Rest and sleep have 2-2.5x more nonlinear dynamics:
forbidden ordinal patterns, bicoherence, nonstationarity, chaotic indicators.

**Interpretation**: The "coherent, integrated" meditation state is a simple,
stable, linear signal. The brain at rest is messier and more nonlinearly complex
than the brain meditating. This contradicts the intuition that coherence =
complexity. Meditation achieves coherence through simplicity.

**Comparison with seizure** (`eeg_deep.py` D8): Seizure EEG has 135 nonlinear sig,
resting EEG (eegmmidb) has 11. Seizure is a profoundly nonlinear state. The
spectrum for resting and meditation results is:

    Seizure >> Sleep > Rest >> Meditation > Resting (eegmmidb)

The meditation dataset (ds001787, experienced meditators) is even less nonlinear
than the eegmmidb resting baseline (naive subjects, eyes closed), suggesting
meditation actively suppresses nonlinear dynamics relative to ordinary rest.

---

## Finding 4: Seizure has profound nonlinear structure

**Evidence**: `eeg_deep.py` D8

| State | Sig metrics surviving IAAFT |
|-------|:--:|
| Seizure (Bonn) | 135 |
| Eyes Closed (eegmmidb) | 11 |

**Top nonlinear seizure signatures**:
- Tropical:slope_changes (d = -17.81)
- Higher-Order Statistics:perm_entropy (d = -16.91)
- Boltzmann:coupling_strength (d = +15.65)
- Ordinal Partition:time_irreversibility (d = +2.92)

Seizure EEG enters a dynamical regime where the relationships between time
points carry information far beyond what frequencies are present. This is genuine
nonlinear dynamics — not just a different spectrum, but a fundamentally different
kind of temporal organization.

**Seizure detection** (`eeg_deep.py` D6):
- Bonn (seizure vs healthy): 189 sig
- CHB-MIT (seizure vs interictal): 20 sig

The Bonn result is stronger because the segments are pre-selected and clean.
CHB-MIT uses real clinical recordings with noise, artifacts, and variable seizure
morphology, but still achieves 20 significant discriminators.

---

## Finding 5: Geometric structure degrades monotonically with frequency

**Evidence**: `eeg_deep.py` D4, `eeg_oiris.py` D5

### eeg_deep.py (eegmmidb, eyes closed)

| Band | Freq range | Sig metrics |
|------|-----------|:-----------:|
| Delta | 0.5-4 Hz | 194 |
| Theta | 4-8 Hz | 193 |
| Alpha | 8-13 Hz | 190 |
| Beta | 13-30 Hz | 165 |
| Gamma | 30-50 Hz | 131 |

### eeg_oiris.py (meditation)

| Band | Sig metrics |
|------|:-----------:|
| Delta | 189 |
| Theta | 173 |
| Alpha | 168 |
| Beta | 167 |
| Gamma | 145 |

Slow oscillations carry the most geometric structure across all brain states
and datasets. The pattern is consistent: delta > theta > alpha > beta > gamma.

---

## Finding 6: Sleep alpha is more structured than meditation alpha

**Evidence**: `eeg_oiris.py` D5

| Band | Meditation | Sleep |
|------|:----------:|:-----:|
| Delta | 189 | 198 |
| Theta | 173 | 186 |
| Alpha | 168 | **200** |
| Beta | 167 | 178 |
| Gamma | 145 | 158 |

Sleep has more geometric structure than meditation in **every** frequency band,
with the largest gap in alpha (200 vs 168). This is consistent with Finding 3:
sleep has richer nonlinear dynamics. Sleep alpha is nonlinearly structured
(contributing to sleep's higher IAAFT count) while meditation alpha is linearly
structured — coherent but simple.

---

## Finding 7: No regional geometric diversity at this encoding

**Evidence**: `eeg_deep.py` D5

| Comparison | Sig metrics |
|-----------|:-----------:|
| Frontal vs Occipital | 5 |
| Frontal vs Parietal | 3 |
| Occipital vs Temporal | 3 |
| All other pairs | 0 |

After channel-averaging within regions and uint8 quantization, the geometric
signature is the same everywhere on the scalp. Regional differences exist in
frequency content and amplitude (alpha is occipital, beta is frontal) but not
in temporal structure as measured by our framework.

**Caveat**: This tests single-channel-averaged temporal structure per region,
not cross-region connectivity or per-channel spatial patterns.

---

## Finding 8: Motor imagery is geometrically indistinguishable from execution

**Evidence**: `eeg_deep.py` D2, D7

| Comparison | Sig metrics |
|-----------|:-----------:|
| Motor Real vs Motor Imagined (eegmmidb) | 3 |
| Motor Real L/R vs Imagined L/R | 0 |

The BCI hard problem is confirmed from a geometric perspective. Real and imagined
movement produce essentially identical temporal structure. The differences that
BCI systems exploit (event-related desynchronization in mu/beta bands over motor
cortex) are subtle amplitude/frequency modulations in specific channels — not the
kind of broadband structural signatures the framework detects.

---

## Finding 9: The phi focal point is a spectral artifact

**Evidence**: `eeg_oiris.py` D4 + D6

O.IRIS proposes m = 2.18 (= ln(pi)/ln(phi)) as a theoretically motivated focal
point. Two tests:

**IAAFT test**: If the phi focal point were a nonlinear attractor, meditation
(nearest to phi) should have the most nonlinear structure. It has the least (17
vs 38/43). The signal is predominantly spectral.

**Quasicrystal test**: Penrose and Ammann-Beenker geometries detect golden-ratio
and silver-ratio structure in byte sequences.

| QC Metric | Med vs Rest (d) | Med vs Sleep (d) | Direction |
|-----------|:---:|:---:|---|
| convergent_resonance | +4.50*** | +6.24*** | Meditation higher |
| algebraic_tower | -0.61* | -1.06*** | Sleep/rest higher |
| convergent_profile | +0.04 | +0.19 | No difference |
| long_range_order | 0.00 | 0.00 | No difference |

Results are inconsistent. `convergent_resonance` favors meditation but
`algebraic_tower` goes the wrong direction. 2 of 4 QC metrics show no
difference. The phi signal is not consistent across quasicrystal metrics.

**Interpretation**: The spectral slope near 2.18 during meditation is real. The
claim that this reflects golden-ratio quasicrystalline brain organization is not
supported. The number ln(pi)/ln(phi) = 2.18 is a coincidence with the observed
spectral slope, not a mechanistic relationship.

---

## What O.IRIS should measure instead

The framework identifies metrics with 10-90x larger effect sizes than spectral
slope (O.IRIS Cohen's d ~= 0.40 for meditation vs working memory).

**Recommended primary metrics** (all available in the exotic geometry framework):

| Metric | d (Med vs Rest) | What it captures |
|--------|:---:|---|
| Sol:dz_persistence | +36.05 | Temporal persistence in exponential-scaling geometry |
| Navier-Stokes:ess_quality | -18.44 | Extended self-similarity (turbulence regularity) |
| H²×R:radial_temporal_memory | +18.33 | Long-range autocorrelation in hyperbolic space |
| Cantor Set:jump_entropy | +16.63 | Gap transition diversity |
| Tropical:slope_changes | -16.13 | Piecewise-linear dynamics (tropical algebra) |

**Recommended pupil map redesign**:
- **Radius**: Sol:dz_persistence (rank-normalized, inverted). Meditation clusters
  near center. Rest and sleep spread outward at similar radii.
- **Angle**: PCA of remaining top metrics (H²×R, Cantor, Tropical, SL(2,R)).
  Captures independent structural variation within the "dilated" zone.
- **Consequence**: The map becomes a 2-zone system (meditation vs everything else)
  rather than the current 5-zone system. Rest and sleep overlap because they are
  structurally identical.

---

## Methodological notes

- All results are conditional on: channel averaging, uint8 quantization, 250 Hz
  resampling, 1-55 Hz bandpass, ~65-second windows.
- The O.IRIS datasets come from different recording setups (ds001787: 64ch BDF
  2048 Hz; ds003768: 32ch BrainVision 5000 Hz). Cross-dataset comparisons assume
  that resampling and filtering normalize these differences. Systematic hardware
  artifacts could contribute to the meditation separation, though the eeg_deep.py
  within-dataset results (eegmmidb only) show similar patterns.
- The "rest vs sleep = 0 sig" finding should be verified with sleep-staged data.
  Our sleep segments are not stage-labeled; deep sleep (N3) may differ from light
  sleep (N1) in ways our unstaged analysis averages out.
- Spectral slope differences between our extraction (meditation median m = 1.97)
  and O.IRIS paper (m = 1.82) are likely due to continuous windowing vs
  epoch-locked extraction. Relative ordering is preserved.

## Reproducibility

```bash
# Download datasets (~20 GB)
python3 investigations/1d/download_oiris_data.py

# Run investigations
python3 investigations/1d/eeg_deep.py      # ~24 min, uses eegmmidb + Bonn + CHB-MIT
python3 investigations/1d/eeg_oiris.py     # ~36 min, uses ds001787 + ds003768
python3 investigations/1d/eeg_oiris_polar.py  # ~29 min, generates polar comparison
```

Requires: `mne`, `pyedflib`, `boto3`, `scikit-learn`, and the exotic geometry
framework with all dependencies.
