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

**Visual confirmation**: The polar comparison figure (`docs/figures/eeg_oiris_polar.png`)
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
  rather than the current 5-zone system. Rest and sleep overlap in single-channel
  temporal structure but separate in connectivity structure (see Finding 10).

---

## Rigorous follow-up (eeg_oiris_rigorous.py, 2026-04-03)

The initial findings identified 8 blind spots. The rigorous investigation
addresses each one. Investigation: `eeg_oiris_rigorous.py`, 11.4 minutes.

### Finding 10: Rest vs sleep separates in connectivity, not temporal dynamics

**Evidence**: `eeg_oiris_rigorous.py` D1, D2, D8

| Analysis | Rest vs Sleep sig |
|----------|:-:|
| 1D channel-averaged (eeg_oiris.py D2) | 0 |
| 1D within-dataset matched subjects (D1) | 1 |
| 2D channels×time spatial geometries (D2) | 3 |
| 1D coherence matrix geometry (D8) | **29** |

The rest/sleep distinction IS real but lives in the **connectivity pattern**
(who talks to whom), not in single-channel temporal dynamics (what each region
does). Channel averaging destroyed this signal. When we analyze the coherence
matrix as a geometric object, 29 metrics discriminate rest from sleep.

Top connectivity discriminators (rest vs sleep):
- Laplacian:cross_scale_curvature_coherence (d = +1.96)
- E8 Lattice:std_profile (d = -1.75)
- Sol (Thurston):path_length (d = +1.74)

**Cross-channel coherence** (D3) confirms: rest and sleep have nearly
identical coherence levels in all bands (theta p=0.79, alpha p=0.58,
beta p=0.82). The 29-sig connectivity difference is in the **pattern**
of who connects to whom, not the overall connection strength.

### Finding 11: Meditation decouples channels

**Evidence**: `eeg_oiris_rigorous.py` D3, D7

| Band | Meditation coherence | Rest coherence | Sleep coherence |
|------|:---:|:---:|:---:|
| Theta | 0.247 | 0.575 | 0.556 |
| Alpha | 0.364 | 0.526 | 0.555 |
| Beta | 0.266 | 0.531 | 0.542 |

Meditation has 2-2.3× lower cross-channel coherence than rest/sleep across
all bands (all p < 0.03). The brain **desynchronizes spatially** during
meditation while maintaining temporal persistence within each channel.

Per-channel metric diversity (D7) confirms: meditation has cross-channel CV
= 1.06 vs rest = 0.40 and sleep = 0.55. Different brain regions are doing
different things during meditation — each region is individually persistent
(Sol:dz_persistence) but spatially independent.

**Implication**: Meditation is not "global coherence" in the EEG connectivity
sense. It's local persistence with global independence. Each brain region
maintains its own stable dynamics without synchronizing to its neighbors.

### Finding 12: Phi is not the empirical attractor — m=2.0 is

**Evidence**: `eeg_oiris_rigorous.py` D6

Meditation spectral slopes (n=1,093 windows, 10-second):
- Mean: 1.975, Median: 1.971, SD: 0.070

| Reference value | Mean |m - ref| |
|:-:|:-:|
| m = 1.9 | 0.081 |
| **m = 2.0** | **0.053** (best fit) |
| m = 2.1 | 0.131 |
| m = 2.18 (phi) | 0.207 |
| m = 2.2 | 0.227 |

The empirical center of meditation spectral slopes is m = 2.00 (or more
precisely, 1.975). The O.IRIS focal point m = ln(π)/ln(φ) = 2.18 is 0.18
away from this center — 4× worse than the best-fit reference. Meditation
IS 7.1× closer to phi than rest (1.48 vs 0.21), but it's even closer to
the round number 2.0.

**Interpretation**: 1/f² is the spectral exponent of Brownian motion
(integrated white noise). Meditation EEG has a spectral slope consistent
with temporally integrated noise — a natural baseline for a persistent,
memory-rich process. The golden ratio is unnecessary; the physics of
temporal integration (m = 2) explains the observation more parsimoniously.

### Finding 13: Meditation separation survives at all timescales

**Evidence**: `eeg_oiris_rigorous.py` D5

| Window length | Samples | Med vs Rest sig |
|:-:|:-:|:-:|
| 5s | 1,250 | 141 |
| 10s | 2,500 | 160 |
| 30s | 7,500 | 171 |
| 65s | 16,250 | 177 |

The meditation signal is detectable even in 5-second windows (141 of 278
metrics). It scales monotonically with window length but is already strong
at O.IRIS's operational timescale (10s: 160 sig). This is not a long-window
artifact.

### Finding 14: Phase-amplitude coupling separates rest from sleep

**Evidence**: `eeg_oiris_rigorous.py` D4

| State | Theta→gamma PAC (MI) |
|---|:-:|
| Meditation | 0.00099 ± 0.0049 |
| Rest | 0.00014 ± 0.00017 |
| Sleep | 0.00023 ± 0.00021 |

Rest vs sleep: d = -0.476, p = 0.0002. Sleep has significantly more
theta-phase → gamma-amplitude coupling than rest. This cross-frequency
interaction is a known neural mechanism during memory consolidation
(hippocampal sharp-wave ripples nested in slow oscillations).

Meditation vs rest/sleep: borderline (p ≈ 0.06). The high variance in
meditation PAC (SD = 0.0049 vs rest SD = 0.00017, a 29× difference) suggests
meditation alternates between high-PAC and low-PAC states — consistent with
the reported mind-wandering/concentration cycling in the ds001787 protocol.

### Finding 15: Meditation connectivity is distinct from rest/sleep

**Evidence**: `eeg_oiris_rigorous.py` D8

When coherence matrices are analyzed as geometric objects through the
1D framework:

| Comparison | Sig metrics |
|-----------|:-:|
| Meditation vs Rest | 177 |
| Meditation vs Sleep | 184 |
| Rest vs Sleep | 29 |

Meditation connectivity is as distinct from rest/sleep as single-channel
temporal dynamics (177-184 sig in both analyses). The connectivity pattern
during meditation is fundamentally different.

Top connectivity discriminators (meditation vs rest):
- Visibility Graph:nvg_hvg_edge_divergence (d = +14.06)
- Visibility Graph:graph_density (d = +12.55)
- Hyperbolic:mean_hyperbolic_radius (d = +10.38)

---

## Revised conclusions (post-rigorous)

1. **Meditation is unique** — confirmed at every level of analysis (temporal,
   spatial, connectivity) and every timescale (5-65 seconds). The signature is
   local temporal persistence (Sol:dz_persistence) combined with spatial
   desynchronization (low cross-channel coherence, high per-channel diversity).

2. **Rest vs sleep** — partially overturned. They are identical in single-channel
   temporal dynamics (0-1 sig) but distinguishable in connectivity pattern
   (29 sig) and phase-amplitude coupling (p = 0.0002). The difference is in
   how regions coordinate, not in what they do individually.

3. **Phi is falsified as the attractor** — the empirical center is m = 2.0
   (Brownian noise exponent), not m = 2.18 (golden ratio). The golden ratio
   claim is not supported by the data.

4. **Meditation is local coherence + global independence** — not global
   coherence as commonly assumed. Each region maintains stable dynamics
   independently. This is a richer model than O.IRIS's single-radius
   "constricted" description.

---

## Methodological notes

- All 1D results are conditional on: channel averaging, uint8 quantization,
  250 Hz resampling, 1-55 Hz bandpass.
- The O.IRIS datasets come from different recording setups (ds001787: 64ch BDF;
  ds003768: 32ch BrainVision). D1 controls for this using within-dataset
  matched-subject comparisons.
- Sleep segments are not stage-labeled. Deep sleep (N3) may differ from light
  sleep (N1) in ways our unstaged analysis averages out. The ds003768 .vmrk
  files contain sync pulses and response markers but no polysomnographic
  sleep stage annotations.
- Spectral slope differences between our extraction (meditation median m = 1.97)
  and O.IRIS paper (m = 1.82) are likely due to continuous windowing vs
  epoch-locked extraction. Relative ordering is preserved.
- The cross-lab confound (different equipment for meditation vs rest/sleep) is
  only partially controlled. D1 shows rest/sleep comparisons are clean
  (within-dataset), but meditation vs rest/sleep still crosses labs. The
  within-dataset eegmmidb results (eeg_deep.py) show similar state separation
  patterns, providing indirect validation.
- D3 coherence analysis uses 3-5 files per state with channel subsampling
  (20 of available channels). Low N limits statistical power for coherence
  t-tests.

## Reproducibility

```bash
# Download datasets (~20 GB)
python3 investigations/1d/download_oiris_data.py

# Run investigations
python3 investigations/1d/eeg_deep.py            # ~24 min, eegmmidb + Bonn + CHB-MIT
python3 investigations/1d/eeg_oiris.py           # ~36 min, ds001787 + ds003768
python3 investigations/1d/eeg_oiris_polar.py     # ~29 min, polar comparison figure
python3 investigations/1d/eeg_oiris_rigorous.py  # ~12 min, blind spot controls
```

Requires: `mne`, `pyedflib`, `boto3`, `scikit-learn`, and the exotic geometry
framework with all dependencies.
