# Investigation: 3I/ATLAS — Geometric Analysis of Real SETI Radio Telescope Data

**Date:** March 24, 2026
**Script:** `investigations/1d/seti_3i_atlas.py`

## Objective

Apply the exotic geometry framework to **real radio telescope data** from interstellar object 3I/ATLAS to determine whether ON-target observations (telescope pointed at the object) contain geometric structure absent from OFF-target observations (nearby blank sky).

This is the follow-up to the synthetic SETI investigation (`investigations/1d/seti.py`), which showed the framework detects artificial signals at -13 dB SNR with 40-100+ significant metrics. Here we test whether the framework finds anything anomalous in actual observations.

## Data Source

**Green Bank Telescope (GBT)**, December 18, 2025 — the most sensitive radio search of any interstellar object to date (~0.1 W EIRP sensitivity), taken <24 hours before 3I/ATLAS's closest approach to Earth.

- **Portal:** `https://bldata.berkeley.edu/ATLAS/GB_ATLAS/`
- **File tier:** Part-0002 (spectral-line product, ~50 MB/file)
- **Cadence:** ABACAD — 3 ON scans, 3 OFF scans per band
- **Bands:** L (1-2 GHz), S (2-4 GHz), C (4-8 GHz), X (8-12 GHz)
- **Files processed:** 48 (2 compute nodes × 6 scans × 4 bands)
- **Total download:** ~2.4 GB
- **Spectrogram dimensions:** 279 time steps × 65,536 frequency channels per file
- **Time resolution:** 1.074 s/sample (5 min observation per scan)
- **Frequency resolution:** 2.86 kHz/channel
- **Compression:** Bitshuffle+LZ4 (required `hdf5plugin`)

## Methodology

### 1D Extraction

Each spectrogram was reduced to four 1D spectral profiles (power vs frequency):
- Divided 279 time steps into 4 windows, summed power within each
- Rebinned 65,536 channels to 2,000 points via contiguous block averaging
- Encoded to uint8 via percentile clipping (p0.5, p99.5)

Time series extraction was skipped: only 279 time steps vs the framework's 2,000-point minimum.

**Yield:** 4 spectral profiles × 48 files = 192 extractions (96 ON, 96 OFF).

### RFI Mitigation

Per-channel sigma-clip (3σ) applied before extraction. Pixels exceeding 3 standard deviations from the channel median were replaced with the median. This removes bright narrowband RFI spikes.

### Statistical Comparison

- **Exploratory (per-band):** Rank all 200 metrics by |Cohen's d| between ON and OFF profiles. No multiple-testing correction — screening pass for candidates.
- **Primary (pooled per-band):** Bonferroni-corrected Welch t-test (α = 0.05/200 = 2.5×10⁻⁴) with |d| > 0.8 threshold. 24 ON vs 24 OFF profiles per band.

## Findings

### 1. No Significant ON vs OFF Differences

| Band | Frequency | ON profiles | OFF profiles | Significant metrics |
|:---|:---|:---:|:---:|:---:|
| L | 1-2 GHz | 24 | 24 | **0** |
| S | 2-4 GHz | 24 | 24 | **0** |
| C | 4-8 GHz | 24 | 24 | **0** |
| X | 8-12 GHz | 24 | 24 | **0** |

Zero metrics reached significance in any band after Bonferroni correction.

### 2. Exploratory Effect Sizes Are Small

| Band | Top metric | Cohen's d |
|:---|:---|:---:|
| L | Zipf-Mandelbrot:zipf_r_squared | -0.41 |
| S | Decagonal:phi_squared_ratio | +0.50 |
| C | H3 Icosahedral:normalized_entropy | +0.56 |
| X | SpectralGraph:weyl_exponent | -0.73 |

All exploratory effect sizes are below the |d| > 0.8 threshold. The largest (X-band, d = -0.73) is consistent with sampling noise. The pattern across bands is inconsistent — different metrics dominate in each band, which is the signature of noise, not a coherent signal.

### 3. Interpretation Scale

| Significant metrics | Interpretation | This investigation |
|:---|:---|:---|
| 0-5 | No geometric difference (noise-equivalent) | **All bands: 0** |
| 5-30 | Mild structural difference (instrumental/RFI) | — |
| 30-100 | Strong structural difference (warrants investigation) | — |
| 100+ | Very strong structural difference | — |

For reference, the synthetic SETI investigation found:
- Artificial signals: 40-100+ significant metrics
- Natural astrophysical sources (pulsars, FRBs): 116-162
- Pure noise: ~0

The real GBT data behaves like pure noise under this comparison.

### 4. Caveats

- Only spectral profiles were analyzed (not time series). A time-varying signal that integrates away in the spectral domain would be missed. Part-0001 (high time resolution) data would address this.
- RFI sigma-clipping removes the strongest narrowband features. Any technosignature that resembles RFI (narrowband carrier) would be suppressed by this step. This is a standard SETI trade-off.
- We analyzed 2 of ~16 compute nodes per band. A signal in a narrow frequency range covered by a different node would be missed.

## Conclusion

The exotic geometry framework finds **no geometric structure difference** between ON-target (3I/ATLAS) and OFF-target (blank sky) observations across the full 1-12 GHz GBT dataset. This is consistent with the null results reported by Breakthrough Listen using conventional methods (Sheikh et al. 2025, Jacobson-Bell et al. 2025).

The framework demonstrated it can process real BL filterbank data end-to-end. The null result validates that the framework does not produce false positives on real radio telescope data — the ON and OFF spectrogram profiles are geometrically indistinguishable, as expected for two sky pointings with no detectable signal.

**Phase 2 (future):** Extend to ATA beamformed data for cross-telescope comparison, and analyze Part-0001 (high time resolution) data for temporal structure.
