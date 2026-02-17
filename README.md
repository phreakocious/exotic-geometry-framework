# Exotic Geometry Framework

Embed byte sequences into 52 exotic geometric spaces — E8 lattices, Heisenberg groups, tropical semirings, Penrose quasicrystals, Mandelbrot/Julia fractal dynamics, Lie/Coxeter root systems, and 8 native 2D spatial geometries — and measure what comes out. Structure that survives the embedding is real. Structure that doesn't is noise.

This framework treats data analysis as a question of **geometry**: different mathematical spaces are sensitive to different kinds of hidden structure. A single data stream analyzed through 44 1D geometries (233 metrics) produces a geometric fingerprint that can distinguish chaotic maps, detect cipher weaknesses, identify DNA organisms, and find backdoors in neural network weights. A [Structure Atlas](#structure-atlas-how-the-framework-organizes-the-world) maps 179 data sources from 16 domains into a unified structure space, revealing cross-domain twins and dimensional organization. For 2D fields, 8 spatial geometries (80 metrics) span differential geometry, algebraic topology, conformal analysis, integral geometry, fractal scaling, Hodge theory, and spectral analysis.

**Key differentiator**: [Surrogate testing](#exotic-geometries-detect-nonlinear-structure-simple-features-cannot) proves these geometric embeddings capture genuinely nonlinear dynamical structure that no combination of entropy, autocorrelation, spectral slope, or other standard features can replicate. Simple features: **0 detections** against IAAFT surrogates. Full framework: **493 detections** across 6 signal types (279 from geometric embeddings, 214 from standard nonlinear features).

**[Explore the Structure Atlas](https://nullphase.net/sa)** — interactive 3D visualization of 179 data sources mapped through 233 geometric metrics.

## Headline Results

Every finding uses shuffled baselines, Bonferroni correction, and Cohen's d effect sizes. Zero false positives on validated random sources. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full protocol.

### Hash functions are indistinguishable from random (methodology validation)

All 6 hash functions tested (MD5, SHA-1, SHA-256, SHA-3, BLAKE2, SHA-512) produce 0 significant geometric differences from `os.urandom`. This is the first result for a reason: it proves the framework doesn't hallucinate structure.

→ `investigations/1d/hashes.py`

### Exotic geometries detect nonlinear structure simple features cannot

Using IAAFT surrogates (Schreiber & Schmitz, 1996) — which preserve both power spectrum and marginal distribution by construction — we prove the framework captures genuinely nonlinear structure. Simple features (entropy, autocorrelation, spectral slope, permutation entropy, kurtosis, etc.) score **0/11 detections** against IAAFT surrogates for all 7 test signals. This is mathematically guaranteed: the surrogates match all linear statistics exactly.

The full framework scores **493 total detections** across 6 of 7 signals (279 geometric, 214 standard nonlinear):

| Signal | Linear | Nonlinear | Framework (total) | Geometric only |
|--------|:-:|:-:|:-:|:-:|
| Hénon map | 0 | 11 | 119 | 72 |
| Logistic map | 0 | 11 | 107 | 63 |
| Collatz stopping times | 0 | 10 | 96 | 57 |
| Lorenz attractor | 0 | 9 | 78 | 38 |
| Heartbeat (ECG) | 0 | 7 | 73 | 42 |
| Prime gaps | 0 | 4 | 19 | 7 |
| Coupled AR (linear) | 0 | 1 | 1 | 0 |

The coupled AR system correctly returns 0 geometric detections — it's a linear system with no nonlinear structure for manifold embeddings to detect. Top geometric detectors: Torus T² (coverage d=-78), E8 Lattice, G2 Root System, D4 Triality. Geometric embeddings provide complementary detections via manifold topology.

An [ablation study](#ablation-233-metrics--40-effective-dimensions) confirms this: exotic geometries amplify detection but 233 metrics collapse to ~40 independent dimensions (151 effective after correlation clustering). The surrogate test is what proves they carry genuinely new information.

![Surrogate Testing](docs/figures/surrogate.png)

→ `investigations/1d/surrogate.py` · `investigations/1d/ablation.py`

### ECB cipher mode detection (d = 19-146)

ECB mode leaks plaintext structure catastrophically. E8 root diversity drops from 72 (random) to 2 (ECB). All block ciphers in ECB mode produce identical geometric signatures. Stream ciphers and CTR/CBC modes are invisible.

![ECB Detection](docs/figures/ecb_penguin_2d.png)

→ `investigations/1d/ciphers.py` · `investigations/2d/ecb_penguin.py`

### Reduced-round AES: cliff at R=4

AES with 1 round: 40 metrics detect weakness. 2 rounds: 38 metrics. 3 rounds: 8 metrics. **4 rounds: 0 metrics.** The transition from detectable to indistinguishable is sharp, matching AES design theory (4 rounds = full diffusion).

![Reduced AES](docs/figures/reduced_aes.png)

→ `investigations/1d/reduced_aes.py`

### Advanced steganography detection

Six embedding techniques tested — from naive LSB replacement to Hamming-coded matrix embedding. PVD (pixel value differencing) is the most visible at 42 significant metrics on raw bytes. Spread spectrum: 25 sig. Matrix embedding (syndrome coding) is truly invisible to all geometries. Bitplane extraction does not improve detection — the 8x sample size reduction destroys statistical power. Delay embedding amplifies the signal for detectable techniques.

→ `investigations/1d/stego.py` · `investigations/1d/stego_deep.py`

### Chaotic map fingerprinting (45/45 pairwise)

10 chaotic maps (logistic, Henon, tent, Lorenz, Rossler, baker's, and more), all detected as non-random, all distinguishable from each other. Lorenz uniquely destroys Penrose 5-fold symmetry (d=60). Rossler shows extreme Tropical linearity (d=266).

![Chaos Fingerprints](docs/figures/chaos.png)

→ `investigations/1d/chaos.py`

### Collatz sequences are massively non-random (7/7 encodings, 21/21 pairwise)

Seven encodings of Collatz (3n+1) dynamics — hailstone values, parity bits, stopping times, residues mod 7, high bits — all detected with 73-103 significant metrics. All 21 encoding pairs distinguished. Shuffle validation reveals hailstone structure is sequential (destroyed by shuffling) while parity is purely distributional. The 3n+1 and 5n+1 variants are geometrically distinguishable (71 metrics, d=12.8).

![Collatz Analysis](docs/figures/collatz.png)

→ `investigations/1d/collatz.py`

### Deep Collatz: why does 3n+1 converge?

Two follow-up investigations probe the convergence mechanism of 3n+1 from ten angles.

**Phase transition (collatz_deep)** — The (2k+1)n+1 family shows a sharp phase boundary: 3n+1 (k=1) has 86 significant metrics vs random, while 5n+1 (k=2) drops to 35. Tropical odd-step slopes match theory exactly (μ=1.5855 vs log₂(3)=1.5850). Optimal delay embedding is at τ=2 (78 sig, monotonically decreasing). Bitplane signal is U-shaped (LSB=77, MSB=68).

**Convergence anatomy (collatz_deep2)** — Of ~131 metrics, **45 are convergence-specific** (significant for 3n+1, vanish for 5n+1), 41 are universal, and only 3 are divergence-specific. Five geometry families — Fisher Information, Heisenberg, Sol, Spherical, Wasserstein — are 100% convergence-aware: every one of their significant metrics goes dark at k=2. The Syracuse encoding of the v₂ (2-adic valuation) sequence is the richest representation at 105 sig metrics. Transform composition order matters: delay-then-bitplane (86 sig) beats bitplane-then-delay (75 sig), exceeding both individual baselines. 37 metrics track drift rate across the family with |r| > 0.8.

![Deep Collatz I](docs/figures/collatz_deep.png)
![Deep Collatz II](docs/figures/collatz_deep2.png)

→ `investigations/1d/collatz_deep.py` · `investigations/1d/collatz_deep2.py`

### 2D spatial analysis: phase transitions and morphologies

Native 2D analysis with 8 spatial geometries providing 80 metrics. SpatialField (tension, curvature, basins), Surface (Gaussian/mean curvature, shape index), PersistentHomology2D (sublevel persistence), Conformal2D (Cauchy-Riemann, Riesz transform), MinkowskiFunctional (excursion sets), MultiscaleFractal (lacunarity, Hurst exponent), HodgeLaplacian (Dirichlet/biharmonic energy), and SpectralPower (spectral slope, anisotropy).

**Ising model** — Phase transitions detected at all temperatures. `multiscale_coherence` peaks sharply near T_c = 2.269, providing a geometric signature of criticality. Adjacent temperature pairs distinguished even at 0.1K resolution.

![Ising Phase Transition](docs/figures/ising.png)

**Reaction-diffusion** — All 6 Gray-Scott morphologies (spots, stripes, worms, coral, mazes, chaos) distinguished from each other (15/15 pairs). `curvature_std` is the strongest discriminator, capturing the difference between smooth blobs and jagged interfaces.

![Reaction-Diffusion](docs/figures/reaction_diffusion.png)

**Percolation** — All 28 probability pairs distinguished across the phase transition at p_c ≈ 0.593. `n_basins` tracks cluster fragmentation and is the single strongest metric, with effect sizes up to d=261.

![Percolation](docs/figures/percolation.png)

**Maze algorithms** — 6 generation algorithms (DFS, Prim, Kruskal, BinaryTree, Sidewinder, Aldous-Broder) fingerprinted with 15/15 pairs distinguished. `multiscale_coherence` at scale 2 separates BinaryTree's strong diagonal bias from the more uniform algorithms.

![Mazes](docs/figures/mazes.png)

**Cellular automata** — 6 rules classified with 14/15 pairs distinguished. GoL ≈ HighLife is the expected failure case (nearly identical rule tables). `anisotropy_mean` separates symmetric rules (DayNight, Anneal) from asymmetric ones.

![Cellular Automata](docs/figures/cellular_automata.png)

**Wave equation** — 6 source configurations (single, double, multi-5, plane, random-20, cavity) all distinguished (15/15). Plane waves produce extreme coherence (d=496), while cavity modes create unique basin structures.

![Wave Equation](docs/figures/wave_equation.png)

**More 2D investigations** — Voronoi point processes (10/10 pairs), growth models (DLA/Eden/random, 3/3), Lenia continuous CA (15/15 configs), and abelian sandpile SOC convergence.

![Voronoi](docs/figures/voronoi.png)
![Growth Models](docs/figures/growth_models.png)
![Lenia](docs/figures/lenia.png)
![Sandpile](docs/figures/sandpile.png)

→ See `investigations/2d/` for all scripts

### Prime number sequences (7/7 encodings, 21/21 pairwise, primes ≠ Cramér model)

Seven encodings of prime sequences — gaps, residues mod 256, last digits, binary expansion, gap pairs, mod 30#, second differences — all massively non-random (49-100 significant metrics). All 21 encoding pairs distinguished. Every encoding has ordering-dependent structure (destroyed by shuffling), but most also carry distributional structure that survives shuffling. Prime gaps are distinguishable from the Cramér probabilistic model (55 sig) and from semiprime gaps (80 sig) — there is geometric structure specific to primality. Gap geometry evolves with prime size: all 6 range pairs distinguished (68-82 sig), consistent with growing mean gap.

![Primes](docs/figures/primes.png)

→ `investigations/1d/primes.py`

### Deep primes: 52 metrics detect pure primality beyond Cramér's model

A follow-up investigation tests progressively refined probabilistic models of the primes. Model hierarchy (significant metrics vs real prime gaps): Cramér random model = 54, even-gap constrained = 37, sieved composites = 30, distribution-matched = 14. Those final 14 metrics — led by Lorentzian `causal_order` (d=10.5) — detect pure sequential correlation that survives even when the gap distribution is perfectly matched. The gap between real primes and Cramér shrinks with scale (75 sig at 1K primes → 54 sig at 1M primes), but 31 metrics remain significant at every scale tested.

→ `investigations/1d/primes_deep.py`

### Number theory: arithmetic functions have rich geometric structure

Eight arithmetic sequences tested — divisor count d(n), distinct prime factors Ω(n), Euler totient ratio φ(n)/n, Mertens function M(n), Riemann zeta zero spacings, Möbius μ(n), Liouville λ(n), and totient mod 256. All 8 detected as non-random (50-105 sig metrics), all 28 pairwise pairs distinguished. Even degenerate sequences — μ(n) takes only 3 values, λ(n) only 2 — are detected at 78 and 71 significant metrics.

Geometry detects structure theoretical models miss: Mertens function has 37 significant metrics beyond a random walk model. Zeta zero spacings vs the GUE/Wigner-Dyson prediction show 90 significant metrics — massive structure beyond what random matrix theory predicts. A second investigation adds continued fractions (π's CF geometric mean = 2.663 ≈ Khinchin's constant K = 2.685), the partition function (107 sig vs Hardy-Ramanujan asymptotics, but only 1 sig vs shuffled — purely distributional), and σ(n)/n ratios (96 sig, 71 ordering-dependent).

→ `investigations/1d/number_theory.py` · `investigations/1d/number_theory_deep.py`

### Time series: process class fingerprinting and Hurst detection

Nine signal types — Brownian motion, pink noise, fractional Brownian motion (H=0.3, H=0.7), ARMA(2,1), Ornstein-Uhlenbeck, regime-switching, heartbeat ECG, and network traffic — all detected as non-random (74-94 sig metrics). All process class pairs distinguished (62-73 sig). Tropical `slope_changes` is the best Hurst parameter discriminator (H=0.3 vs H=0.7: d=15.5). Brownian motion signatures are invariant to 2x temporal subsampling (only 5 sig metrics) but break at 4x+ (38 sig).

→ `investigations/1d/time_series.py`

### Elementary cellular automata: Wolfram classes from spacetime geometry

Twelve elementary CA rules spanning all four Wolfram complexity classes, analyzed as 2D spacetime fields with 80 spatial metrics. Class I (homogeneous) = 79 avg sig vs random, Class II (periodic) = 74, Class IV (complex) = 70, Class III (chaotic) = 65. All 6 class pairs distinguished (66-75 sig). Within Class IV, Rules 110 and 124 are nearly identical (only 2 sig metrics) — they are computationally equivalent via left-right reflection. `PersistentHomology2D:persistence_asymmetry` is the Class IV signature (d = 331-2013). Detection is scale-robust down to 32×32 fields.

![Elementary CA](docs/figures/elementary_ca.png)

→ `investigations/2d/elementary_ca.py`

### Structure Atlas: how the framework organizes the world

179 data sources from 16 domains — chaos, number theory, noise, waveforms, bearings, binary, DNA, medical (ECG/EEG), financial, motion, astronomy, climate, speech, quantum, geophysics, exotic — mapped into 233-metric structure space. At 12,000 bytes per chunk, effective dimensionality = 8.9 (PC1+2 = 40.0%). Key findings:

- **Cross-domain twins**: EEG Eyes Closed ↔ Bearing Outer (d=0.084), EEG Seizure ↔ Speech (d=0.18), financial returns ↔ accelerometer (NASDAQ ↔ Accel Stairs d=0.16)
- **DNA is unique**: synthetic and real DNA cluster tightly (Synthetic DNA ↔ DNA Chimp d=0.001), isolated from everything else in its own cluster
- **Surrogate decomposition**: ECG Supraventricular is the most sequential source (87/233 metrics disrupted by shuffling). White Noise/AES = 0 disrupted. Pi (base 256) has more structure disrupted by rolling than by shuffling — positional structure beyond autocorrelation. Chaos maps are time-asymmetric, financial returns too
- **Multi-scale robustness**: ECG Normal detectable at all scales (92-101 sig at 256-8192 bytes). DNA Human rock-solid (90-98). NYSE Returns jump from 50 (256 bytes) to 87 (8192 bytes) — financial data needs longer windows

![Structure Atlas](docs/figures/structure_atlas.png)
![Structure Atlas 3D](docs/figures/structure_atlas_3d.png)
![Structure Atlas Techniques](docs/figures/structure_atlas_techniques.png)

→ `investigations/1d/structure_atlas.py` · [Interactive 3D Atlas](https://nullphase.net/sa)

### Bearing fault diagnosis (CWRU dataset)

Real vibration data from CWRU bearing fault dataset. 4 conditions (Normal, Ball, Inner, Outer race faults) all distinguished. Fractal geometries (Mandelbrot/Julia) are inner-race specialists: 9/10 fractal metrics significant for inner race (d=1.9-5.5), only 3/10 for outer race.

![Bearing Faults](docs/figures/bearing_fault.png)

→ `investigations/1d/bearing_fault.py`

### Mathematical constants: fingerprints of transcendence

Base-256 digits of Pi, e, Phi, Sqrt(2) are indistinguishable from each other (0 sig) but all differ from white noise (84-85 sig). Continued fraction taxonomy reveals deep structure: CF(e) vs CF(Pi) = 94 sig, CF(ln2) vs algebraics = 118 sig. Pi's CF has only 5 sig vs Gauss-Kuzmin i.i.d. — barely detectable sequential structure. Same constant (Pi) in 4 representations (base-256, base-10, binary, CF) produces wildly different fingerprints (52-106 sig pairwise).

![Math Constants](docs/figures/math_constants.png)

→ `investigations/1d/math_constants.py`

### RNG quality testing: geometric fingerprints of PRNG weakness

Ten generators spanning a quality gradient — from cryptographic (os.urandom, SHA-256 CTR) through good (PCG64, MT19937, SFC64) to weak (XorShift128, MINSTD) to bad (RANDU, LFSR-16, Middle-Square) — tested against an os.urandom reference. Self-check (urandom vs urandom) returns 0 significant metrics, validating the methodology.

| Generator | Category | Sig metrics | Top detector |
|-----------|----------|:-----------:|:-------------|
| os.urandom | CRYPTO | 0 | — |
| SHA-256 CTR | CRYPTO | 0 | — |
| PCG64 | GOOD | 0 | — |
| SFC64 | GOOD | 0 | — |
| MT19937 | GOOD | 1 | S² × ℝ (borderline) |
| XorShift128 | WEAK | 0 (raw), 2 (DE2) | Thurston height_drift |
| MINSTD | WEAK | 1 | S² × ℝ |
| RANDU | BAD | **44** | Penrose index_diversity d=-37 |
| LFSR-16 | BAD | 2 | Higher-Order Stats |
| Middle-Square | BAD | **78** | Penrose index_diversity d=-35 |

Geometric metrics outperform standard statistical tests: RANDU has 44 geometric detections vs 4 standard, Middle-Square has 78 vs 7. Delay embedding (τ=2) newly reveals XorShift128 (0→2 sig) and amplifies Middle-Square (78→82). RANDU is detectable at just 500 bytes (41 sig). Different PRNG weaknesses have distinct geometric fingerprints — E8 Lattice, Higher-Order Statistics, Torus T², and Ammann-Beenker are the top weakness detectors.

![RNG Quality](docs/figures/rng.png)

→ `investigations/1d/rng.py`

### Ablation: 233 metrics → 20 effective dimensions

An honest self-assessment: a 14-feature simple baseline (entropy, autocorrelation, spectral slope, permutation entropy, kurtosis) detects the same phenomena as the full framework — 9-12 significant features across all test cases. The 233 framework metrics collapse to ~20 independent dimensions at 95% explained variance, with 212 metrics >95% redundant with another metric. But exotic geometries contribute 90-95% of all detections (e.g., heartbeat: simple=11, framework=182). Greedy geometry selection reaches 90% of max detections at 33 geometries. The framework is an amplifier — with one critical exception: [surrogate testing](#exotic-geometries-detect-nonlinear-structure-simple-features-cannot) proves geometric embeddings capture nonlinear structure simple features provably cannot.

→ `investigations/1d/ablation.py`

## Quick Start

```bash
git clone <repo-url>
cd exotic-geometry-framework
pip install -r requirements.txt
python quickstart.py
```

Output:
```
EXOTIC GEOMETRY FRAMEWORK - QUICKSTART
[OK] Loaded 44 geometries

Data                 |  E8 roots |  Heis. twist |    Sol aniso |   Penrose 5f
Random               |        73 |          0.1 |         3.52 |       0.9891
Chaos (logistic)     |        54 |          0.4 |         5.83 |       0.9889
Periodic (sine)      |         8 |         10.7 |         0.12 |       0.1234
Fibonacci            |         6 |          2.3 |         0.08 |       0.4567
```

Run an investigation:
```bash
python investigations/1d/chaos.py       # ~30 seconds
python investigations/2d/ising.py       # ~60 seconds
```

## API

### 1D Analysis (byte streams)

```python
from exotic_geometry_framework import GeometryAnalyzer
import numpy as np

data = np.random.randint(0, 256, 2000, dtype=np.uint8)

# All 44 geometries
analyzer = GeometryAnalyzer().add_all_geometries()
results = analyzer.analyze(data)

for name, result in results.items():
    for metric, value in result.metrics.items():
        print(f"{name}.{metric} = {value:.4f}")
```

### Caching and Parallel Processing

```python
# Per-geometry disk cache (324x speedup on warm cache)
analyzer = GeometryAnalyzer(cache_dir=".cache").add_all_geometries()
results = analyzer.analyze(data)  # cold: computes all, stores to cache
results = analyzer.analyze(data)  # warm: loads from cache instantly
analyzer.clear_cache()             # wipe cache

# Parallel chunk processing via Runner (2-3x speedup)
from tools.investigation_runner import Runner
runner = Runner("my_investigation", n_workers=4, cache=True)
metrics = runner.collect(chunks)   # distributes across 4 processes
```

### 2D Analysis (spatial fields)

```python
from exotic_geometry_framework import GeometryAnalyzer, SpatialFieldGeometry

field = np.random.rand(64, 64)

# All 8 spatial geometries (80 metrics)
analyzer = GeometryAnalyzer().add_spatial_geometries()
results = analyzer.analyze(field)

# Or standalone
geom = SpatialFieldGeometry()
result = geom.compute_metrics(field)
print(result.metrics)  # 15 spatial metrics
```

### Preprocessing

```python
from exotic_geometry_framework import delay_embed, spectral_preprocess, bitplane_extract

# Delay embedding: pair byte[i] with byte[i+tau]
delayed = delay_embed(data, tau=5)  # 52x improvement for lag detection

# Spectral: FFT magnitude → uint8
spectral = spectral_preprocess(data)  # catches chirp vs sine

# Bitplane: extract single bit plane
lsb = bitplane_extract(data, plane=0)  # isolate bit plane for analysis
```

### Float data

```python
from exotic_geometry_framework import GeometryAnalyzer, encode_float_to_unit

# Auto mode handles float data internally
analyzer = GeometryAnalyzer().add_all_geometries(data_mode='auto')
results = analyzer.analyze(float_array)

# Or manually encode
encoded = encode_float_to_unit(float_array)
```

## Geometries at a Glance

| Geometry | Key Metric | What It Detects | Strongest Finding |
|----------|------------|-----------------|-------------------|
| **E8 Lattice** | `unique_roots` | Algebraic constraints | RANDU d=-20 |
| **Fisher Information** | `trace_fisher` | Transition statistics | Universal workhorse |
| **Tropical** | `linearity` | Piecewise-linear structure | Rossler d=266 |
| **Heisenberg** (centered) | `twist_rate` | Autocorrelation | Period detection 1000x |
| **Cantor** | `coverage` | Fractal/binary patterns | Pruned NN d=1.3B |
| **Penrose** | `fivefold_balance` | Quasiperiodic/chaotic | Lorenz d=60 |
| **SpatialField** (2D) | `n_basins`, `coherence` | Spatial structure | Percolation d=261 |
| **Surface** (2D) | `gaussian_curvature` | Height map geometry | Stego curvedness d=-6.1 |
| **Conformal2D** (2D) | `cauchy_riemann_residual` | Angle preservation | Riesz amplitude d=+7.0 |
| **PersistentHomology2D** (2D) | `sub_total_persistence` | Component lifetimes | 9.0 sig/pair avg |
| **SpectralPower** (2D) | `spectral_slope` | Frequency structure | 1/f^β power law |
| **Fractal (Mandelbrot)** | `interior_fraction` | Escape dynamics, set membership | Bearing inner race d=5.5 |
| **Fractal (Julia)** | `connectedness` | Julia set structure, stability | Bearing inner race d=4.2 |

Full catalog of all 52 geometries (44 1D + 8 2D): [docs/GEOMETRY_CATALOG.md](docs/GEOMETRY_CATALOG.md)

## Investigations (51 1D + 19 2D = 70 scripts)

### 1D (byte stream analysis)

| Script | Domain | Key Result |
|--------|--------|------------|
| [hashes.py](investigations/1d/hashes.py) | Hash functions | 0 sig (validates methodology) |
| [rng.py](investigations/1d/rng.py) | RNG quality testing | 10 generators: RANDU=44 sig, Middle-Square=78, CRYPTO/GOOD=0 |
| [prng.py](investigations/1d/prng.py) | PRNG weakness (early) | RANDU d=-19.89 |
| [ciphers.py](investigations/1d/ciphers.py) | Cipher modes | ECB d=19-146 |
| [reduced_aes.py](investigations/1d/reduced_aes.py) | Reduced-round AES | Cliff at R=4 |
| [stego.py](investigations/1d/stego.py) | Steganography | LSB correlation d=1.06 (Fisher) |
| [stego_deep.py](investigations/1d/stego_deep.py) | Advanced stego | PVD d=42 sig, matrix embed invisible |
| [chaos.py](investigations/1d/chaos.py) | Chaotic maps | 45/45 pairwise |
| [dna.py](investigations/1d/dna.py) | DNA sequences | 291 findings |
| [nn_weights.py](investigations/1d/nn_weights.py) | Neural network weights | Backdoor d=7.12 |
| [compression_algos.py](investigations/1d/compression_algos.py) | Compressed data | bz2 vs zlib d=7.75 |
| [collatz.py](investigations/1d/collatz.py) | Collatz sequences | 7/7 encodings detected, 3n+1 vs 5n+1 d=12.8 |
| [collatz_deep.py](investigations/1d/collatz_deep.py) | Deep Collatz I | Sharp k=1→2 phase boundary, tropical slopes match theory |
| [collatz_deep2.py](investigations/1d/collatz_deep2.py) | Deep Collatz II | 45 convergence-specific metrics, 5 geometry families go dark at k=2 |
| [primes.py](investigations/1d/primes.py) | Prime numbers | 7/7 encodings, 21/21 pairwise. Primes vs Cramér model: 55 sig |
| [primes_deep.py](investigations/1d/primes_deep.py) | Deep primes | 52 pure-primality metrics, model hierarchy, scale dependence |
| [number_theory.py](investigations/1d/number_theory.py) | Arithmetic functions | All 8 detected (50-105 sig), zeta spacings vs GUE: 90 sig |
| [number_theory_deep.py](investigations/1d/number_theory_deep.py) | Additive number theory | CFs, partitions, σ(n)/n. π CF geo_mean ≈ Khinchin K |
| [time_series.py](investigations/1d/time_series.py) | Time series | 9 processes, Hurst detection, process class fingerprinting |
| [surrogate.py](investigations/1d/surrogate.py) | Surrogate testing | Linear=0, framework=493, geometric=279 vs IAAFT |
| [ablation.py](investigations/1d/ablation.py) | Ablation study | 233 metrics → 20 dimensions, exotic = 90-95% of detections |
| [bearing_fault.py](investigations/1d/bearing_fault.py) | Bearing faults (CWRU) | 4 conditions distinguished, fractals = inner race specialists |
| [bearing_fault_baseline.py](investigations/1d/bearing_fault_baseline.py) | Bearing baseline | Exotic geometries vs standard features comparison |
| [bearing_fault_ml.py](investigations/1d/bearing_fault_ml.py) | Bearing ML | SVM classifier on geometric features |
| [structure_atlas.py](investigations/1d/structure_atlas.py) | Structure Atlas | 179 sources, 16 domains, 8.9D structure space |
| [math_constants.py](investigations/1d/math_constants.py) | Math constants | CF taxonomy, Pi CF ≈ Gauss-Kuzmin, representation fingerprints |
| [binary_anatomy.py](investigations/1d/binary_anatomy.py) | ISA taxonomy | x86/ARM/WASM/JVM distinguishable (85-101 sig) |
| [quantum_geometry.py](investigations/1d/quantum_geometry.py) | Quantum wavefunctions | Coherence d≈4000, energy levels fingerprinted |
| [memory_geometry.py](investigations/1d/memory_geometry.py) | Data structures | Arrays/lists/trees/hash tables distinguished (77 avg sig) |
| [esoteric_code.py](investigations/1d/esoteric_code.py) | Esolangs | Brainfuck/Whitespace/Malbolge/Zalgo (92 avg sig) |
| [sorting_algorithms.py](investigations/1d/sorting_algorithms.py) | Sorting traces | Memory access fingerprints of sort algorithms |
| [music_theory.py](investigations/1d/music_theory.py) | Music theory | Harmonic/melodic structure detection |
| [network_protocols.py](investigations/1d/network_protocols.py) | Network protocols | Protocol fingerprinting |
| [julia_sweep.py](investigations/1d/julia_sweep.py) | Julia set sweep | Parameter space geometry |
| [proteins.py](investigations/1d/proteins.py) | Protein sequences | Globular vs IDP: 53 sig (compositional, not sequential) |
| [unsolved.py](investigations/1d/unsolved.py) | Goldbach's comet | g(2n) vs Hardy-Littlewood: 17 sig beyond HL |
| [noise_robustness.py](investigations/1d/noise_robustness.py) | Noise robustness | 2-4x more noise-tolerant than standard tests |
| [noise_deep.py](investigations/1d/noise_deep.py) | Deep noise | IAAFT surrogate decomposition |
| [seti.py](investigations/1d/seti.py) | SETI | Chaotic modulation at -20dB, 7-17dB advantage |
| [ai_text.py](investigations/1d/ai_text.py) | AI text detection | Byte-level geometry of text |
| [text_geometry.py](investigations/1d/text_geometry.py) | Text structure | Order-2 Markov fools geometry |
| [chaos_deep.py](investigations/1d/chaos_deep.py) | Deep chaos | Attractor geometry, Lyapunov detection |
| [continued_fractions.py](investigations/1d/continued_fractions.py) | Continued fractions | CF coefficients carry geometric fingerprints |
| [prng_deep.py](investigations/1d/prng_deep.py) | Deep PRNG | Multi-scale PRNG weakness analysis |
| [primes_deep2.py](investigations/1d/primes_deep2.py) | Deep primes II | Primality residuals at extreme scales |
| [mandelbrot_sensor_test.py](investigations/1d/mandelbrot_sensor_test.py) | Mandelbrot sensors | Log vs prime orbit sampling |
| [signature_space.py](investigations/1d/signature_space.py) | Signature space | Geometry of the metric space itself |
| [metric_characterization.py](investigations/1d/metric_characterization.py) | Metric analysis | Per-metric sensitivity and redundancy |
| [structure_vs_chaos.py](investigations/1d/structure_vs_chaos.py) | Structure vs chaos | Confusion matrix across source types |

### 2D (spatial field analysis)

| Script | Domain | Key Result |
|--------|--------|------------|
| [ising.py](investigations/2d/ising.py) | Ising model | Phase transition at T_c |
| [reaction_diffusion.py](investigations/2d/reaction_diffusion.py) | Gray-Scott | 15/15 morphologies |
| [percolation.py](investigations/2d/percolation.py) | Site percolation | 28/28 pairs |
| [cellular_automata.py](investigations/2d/cellular_automata.py) | CA rules | 14/15 (GoL ≈ HighLife) |
| [ecb_penguin.py](investigations/2d/ecb_penguin.py) | ECB in 2D | ECB detected, CBC/CTR invisible |
| [mazes.py](investigations/2d/mazes.py) | Maze algorithms | 15/15 pairs |
| [wave_equation.py](investigations/2d/wave_equation.py) | Wave equation | 15/15 configs |
| [voronoi.py](investigations/2d/voronoi.py) | Point processes | 10/10 pairs |
| [growth_models.py](investigations/2d/growth_models.py) | DLA/Eden/random | 3/3 pairs |
| [sandpile.py](investigations/2d/sandpile.py) | Self-organized criticality | SOC convergence |
| [lenia.py](investigations/2d/lenia.py) | Continuous CA | 15/15 configs |
| [stego_bitmatrix.py](investigations/2d/stego_bitmatrix.py) | 2D stego detection | Co-occurrence SS=14/15, diff grid LSBMR=14 sig |
| [elementary_ca.py](investigations/2d/elementary_ca.py) | Elementary CA | Wolfram classes I-IV, R110≈R124, scale-robust to 32×32 |
| [meta_geometry.py](investigations/2d/meta_geometry.py) | Meta-investigation | Metric correlation field, 7D signature space |
| [rule_space.py](investigations/2d/rule_space.py) | CA rule space | Rule parameter sweep |
| [strange_attractors.py](investigations/2d/strange_attractors.py) | Strange attractors | 2D attractor geometry |
| [mandelbrot.py](investigations/2d/mandelbrot.py) | Mandelbrot set | 2D fractal field analysis |
| [reaction_diffusion_patterns.py](investigations/2d/reaction_diffusion_patterns.py) | RD patterns | Extended reaction-diffusion morphologies |
| [stego_bitplane_delta.py](investigations/2d/stego_bitplane_delta.py) | Bitplane stego | Delta-based 2D steganographic analysis |

## Statistical Methodology

Every comparison uses:
- **Shuffled baselines** to separate ordering effects from distribution effects
- **Bonferroni correction** for multiple testing (typically alpha = 0.05/233)
- **Cohen's d** effect sizes (not just p-values)
- **20-30 trials** per condition

The framework produces zero false positives on validated random sources and correctly identifies AES-CTR as indistinguishable from random across all approaches tested. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## What Doesn't Work

Equally important: the framework's known limits.

- **AES-CTR vs random**: Indistinguishable across all 44 geometries, preprocessings, and combination strategies. This is the fundamental limit.
- **XorShift128**: Undetected raw (0 sig), but delay embedding reveals 2 sig metrics via Thurston height_drift — borderline
- **MT19937**: 1 borderline detection (S² × ℝ sphere_concentration). Essentially passes at byte level
- **Standard map, Arnold cat map**: Uniformly mixing → look random
- **Matrix embedding (Hamming syndrome coding)**: Invisible to all geometries at all rates — too few pixel changes
- **LSB replacement/matching at sub-100% rates**: Detectable only at 100% embedding via raw bytes; bitplane extraction does not help

See [docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md).

## Complete Results

All validated findings and negative results across 70 investigations: [docs/FINDINGS.md](docs/FINDINGS.md)

## Dependencies

**Core** (required):
- numpy
- scipy
- matplotlib

**Optional**:
- pycryptodome — for `ciphers.py`, `reduced_aes.py`, `ecb_penguin.py`
- mpmath — for `number_theory.py` (Riemann zeta zeros), `math_constants.py`
- h5py — for LIGO data in `structure_atlas.py`
- scikit-learn — for `bearing_fault_ml.py`

## License

MIT
