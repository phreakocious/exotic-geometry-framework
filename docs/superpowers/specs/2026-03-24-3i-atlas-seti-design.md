# 3I/ATLAS SETI Investigation — Design Spec

## Objective

Apply the exotic geometry framework to real radio telescope data from interstellar object 3I/ATLAS to detect geometric structure that differs between ON-target (pointed at the object) and OFF-target (nearby blank sky) observations. This is the natural follow-up to the synthetic SETI investigation (`investigations/1d/seti.py`), which showed the framework detects artificial signals at -13 dB SNR.

## Data Source

**Green Bank Telescope (GBT)** observations from December 18, 2025 — the most sensitive radio search of any interstellar object to date (~0.1 W EIRP sensitivity), taken <24 hours before 3I/ATLAS's closest approach to Earth.

**Portal:** `https://bldata.berkeley.edu/ATLAS/GB_ATLAS/`

**File tier:** Part-0002 (spectral-line product), ~50 MB per file. Intermediate resolution — sufficient for geometric analysis without multi-GB downloads.

**Cadence structure:** Each band uses a standard BL 6-scan cadence (ABACAD pattern):
- ON scans (3): telescope pointed at 3I/ATLAS (A positions)
- OFF scans (3): telescope pointed at distinct nearby sky positions (B/C/D)
- Each scan observed by multiple compute nodes (blc20-blc35), each covering a frequency sub-range
- ON vs OFF filename distinction: ON files contain `3I_ATLAS`, OFF files contain `3I_ATLAS_OFF`

**Selection:** ~24 files covering:
- 4 bands: L (1-2 GHz), S (2-4 GHz), C (4-8 GHz), X (8-12 GHz)
- 1 ON + 1 OFF scan per band
- ~3 representative compute nodes per band (low/mid/high frequency within band)

**Total download:** ~1.2 GB

## 1D Extraction

Each spectrogram file is a 2D array: power(frequency_channel, time_step). The framework operates on 1D uint8 sequences. Two extraction families:

### A. Band-integrated time series

For each file, divide frequency channels into N sub-bands (4-8), sum power within each sub-band at each time step to produce power(t) curves. Each curve is a 1D source.

**Detects:** Temporal structure — pulsed signals, modulated carriers, time-varying anomalies.

### B. Time-integrated spectral profiles

For each file, divide time steps into M windows, sum power within each window at each frequency channel to produce power(f) curves. Each curve is a 1D source.

**Detects:** Spectral structure — narrowband emission, spectral features, frequency-domain patterns.

### Target extraction length

The framework standardizes on `data_size=2000`. Extractions must meet this minimum — delay embedding alone needs ~50+ points, and statistical tests need sufficient samples.

- **Spectral profiles** (power vs frequency): BL filterbank files typically have thousands of frequency channels. These will naturally exceed 2000 points. Truncate or resample to a standard length.
- **Time series** (power vs time): Part-0002 files may have limited time steps (spectral-line products trade time resolution for frequency resolution). If the time axis is too short, concatenate sub-band time series end-to-end, or use overlapping windows. If a time series extraction is shorter than 2000 points after all attempts, skip it and rely on spectral profiles for that file.

The exact spectrogram dimensions will be determined during implementation by inspecting the first downloaded file.

### Encoding

All 1D extractions are encoded to uint8 via percentile clipping (p0.5, p99.5) — the standard framework encoding for heavy-tailed data.

### Yield

~10-20 extractions per file × ~24 files = 240-480 individual 1D sequences to analyze.

## Analysis Pipeline

### Script

`investigations/1d/seti_3i_atlas.py`

### Pipeline stages

1. **Download** — fetch GBT part-0002 files from BL portal into `data/seti_3i/`. Cache locally; skip if already downloaded.
2. **Extract** — load each file with `h5py` (data at `/data` key, shape `(n_time, 1, n_freq)` for Stokes I), produce 1D extractions (time series + spectral profiles).
3. **RFI mitigation** — apply sigma-clip to spectrogram before extraction (reject pixels > 3σ from channel median). This removes bright RFI spikes. L-band (1-2 GHz) is especially contaminated. Also check for `/mask` dataset in HDF5 files and apply if present.
4. **Analyze** — run each extraction through `GeometryAnalyzer` with all geometries. Collect metric profiles.
5. **Compare** — for each band/node, compare ON vs OFF extractions. Per-file comparisons are exploratory (rank by effect size, no multiple-testing correction). Pooled comparison across bands is the primary analysis (Bonferroni + Cohen's d > 0.8).
5. **Aggregate** — pool results across bands for summary statistics.
6. **Report** — generate figures and findings.

### Caching

Per-file analysis results cached in `data/seti_3i/.cache/` as `.npz` files. Re-extraction and re-analysis only on cache miss.

## Comparison Strategy

### Per-file ON vs OFF (exploratory)

For each (band, node) pair, compare the geometric profiles of ON-target extractions against matched OFF-target extractions. No multiple-testing correction — rank metrics by Cohen's d to identify candidates. This is a screening pass: low statistical power (n~10-20 per group) means absence of significance is not meaningful, but large effect sizes are worth flagging.

### Pooled summary (primary analysis)

Aggregate all ON profiles and all OFF profiles per band. With all extractions pooled, sample sizes are large enough for Bonferroni-corrected testing. This is where the headline numbers come from.

### Interpretation scale

| Significant metrics (ON vs OFF) | Interpretation |
|---|---|
| 0-5 | No geometric difference. Noise-equivalent. |
| 5-30 | Mild structural difference. Likely instrumental or RFI. |
| 30-100 | Strong structural difference. Warrants investigation. |
| 100+ | Very strong structural difference. Compare against known astrophysical templates. |

Reference from synthetic SETI investigation: artificial signals produce 40-100+ significant metrics; natural astrophysical sources (pulsars, FRBs) produce 116-162; pure noise produces ~0. **Caveat:** Real GBT data has instrumental artifacts, quantization, and RFI that will produce structural differences unrelated to the target. These ranges are indicative, not thresholds — initial results will calibrate expectations for real data.

## Output

### Figures (saved to `figures/`)

1. **Metric significance heatmap** — band × metric, color = Cohen's d for ON vs OFF
2. **ON vs OFF comparison** — per-band violin or box plots of top discriminating metrics
3. **Anomaly highlights** — any extraction with unexpectedly high structure count, shown as spectrogram + metric profile
4. **Band summary** — bar chart of total significant metrics per band

### Findings document

`docs/investigations/seti_3i_atlas.md` — structured like existing investigation docs with objective, methodology, findings, conclusion.

## Dependencies

- `h5py` — already used in the framework for gravitational wave data. BL HDF5 filterbank format is straightforward: data at `/data` key with shape `(n_time, 1, n_freq)`, header metadata as HDF5 attributes. No need for `blimpy`.
- `requests` or `urllib` — for downloading files from the BL portal. Already used for Gutenberg downloads.
- Framework: `GeometryAnalyzer` used directly (not via `Runner`, since the data flow is file-based with variable-length extractions, similar to `gravitational_waves.py`)

## Scope boundaries

- **In scope:** GBT part-0002 data, ON/OFF comparison, geometric analysis, figures, findings doc.
- **Out of scope:** ATA/Parkes/MeerKAT data (future work), atlas source registration, turboSETI narrowband search (that's been done by BL already), 2D spatial analysis.
- **Phase 2 (if warranted):** Extend to ATA beamformed data for cross-telescope confirmation. Promote interesting signals to atlas sources.
