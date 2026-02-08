# Exotic Geometry Framework

Embed byte sequences into 24 exotic geometric spaces — E8 lattices, Heisenberg groups, tropical semirings, Penrose quasicrystals — and measure what comes out. Structure that survives the embedding is real. Structure that doesn't is noise.

This framework treats data analysis as a question of **geometry**: different mathematical spaces are sensitive to different kinds of hidden structure. A single data stream analyzed through 24 independent lenses produces a geometric fingerprint that can distinguish chaotic maps, detect cipher weaknesses, identify DNA organisms, and find backdoors in neural network weights.

## Headline Results

Every finding uses shuffled baselines, Bonferroni correction, and Cohen's d effect sizes. Zero false positives on validated random sources. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full protocol.

### Hash functions are indistinguishable from random (methodology validation)

All 6 hash functions tested (MD5, SHA-1, SHA-256, SHA-3, BLAKE2, SHA-512) produce 0 significant geometric differences from `os.urandom`. This is the first result for a reason: it proves the framework doesn't hallucinate structure.

→ `investigations/1d/hashes.py`

### ECB cipher mode detection (d = 19-146)

ECB mode leaks plaintext structure catastrophically. E8 root diversity drops from 72 (random) to 2 (ECB). All block ciphers in ECB mode produce identical geometric signatures. Stream ciphers and CTR/CBC modes are invisible.

![ECB Detection](docs/figures/ecb_penguin_2d.png)

→ `investigations/1d/ciphers.py` · `investigations/2d/ecb_penguin.py`

### Reduced-round AES: cliff at R=4

AES with 1 round: 40 metrics detect weakness. 2 rounds: 38 metrics. 3 rounds: 8 metrics. **4 rounds: 0 metrics.** The transition from detectable to indistinguishable is sharp, matching AES design theory (4 rounds = full diffusion).

![Reduced AES](docs/figures/reduced_aes.png)

→ `investigations/1d/reduced_aes.py`

### Steganography cracked via bitplane decomposition (d = 1166)

LSB steganography is invisible at the byte level (our original negative result). Extracting the LSB plane and analyzing the bit stream makes it massively detectable. Fisher Information catches stego at just 10% embedding rate.

![Stego Detection](docs/figures/stego.png)

→ `investigations/1d/stego.py`

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

Native 2D analysis with 15 spatial metrics (tension, curvature, anisotropy, criticality, basin structure, multiscale coherence). Each metric captures a different aspect of spatial organization.

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

→ See `investigations/2d/` for all 11 scripts

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
[OK] Loaded 24 geometries

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

# All 24 geometries
analyzer = GeometryAnalyzer().add_all_geometries()
results = analyzer.analyze(data)

for name, result in results.items():
    for metric, value in result.metrics.items():
        print(f"{name}.{metric} = {value:.4f}")
```

### 2D Analysis (spatial fields)

```python
from exotic_geometry_framework import GeometryAnalyzer, SpatialFieldGeometry

field = np.random.rand(64, 64)

# Via analyzer
analyzer = GeometryAnalyzer().add_spatial_geometries()

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
lsb = bitplane_extract(data, plane=0)  # cracks LSB steganography
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

Full catalog of all 25 geometries: [docs/GEOMETRY_CATALOG.md](docs/GEOMETRY_CATALOG.md)

## Investigations

### 1D (byte stream analysis)

| Script | Domain | Key Result |
|--------|--------|------------|
| [hashes.py](investigations/1d/hashes.py) | Hash functions | 0 sig (validates methodology) |
| [prng.py](investigations/1d/prng.py) | PRNG weakness | RANDU d=-19.89 |
| [ciphers.py](investigations/1d/ciphers.py) | Cipher modes | ECB d=19-146 |
| [reduced_aes.py](investigations/1d/reduced_aes.py) | Reduced-round AES | Cliff at R=4 |
| [stego.py](investigations/1d/stego.py) | Steganography | Cracked at bitplane (d=1166) |
| [chaos.py](investigations/1d/chaos.py) | Chaotic maps | 45/45 pairwise |
| [dna.py](investigations/1d/dna.py) | DNA sequences | 291 findings |
| [nn_weights.py](investigations/1d/nn_weights.py) | Neural network weights | Backdoor d=7.12 |
| [compression_algos.py](investigations/1d/compression_algos.py) | Compressed data | bz2 vs zlib d=7.75 |
| [collatz.py](investigations/1d/collatz.py) | Collatz sequences | 7/7 encodings detected, 3n+1 vs 5n+1 d=12.8 |
| [collatz_deep.py](investigations/1d/collatz_deep.py) | Deep Collatz I | Sharp k=1→2 phase boundary, tropical slopes match theory |
| [collatz_deep2.py](investigations/1d/collatz_deep2.py) | Deep Collatz II | 45 convergence-specific metrics, 5 geometry families go dark at k=2 |

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

## Statistical Methodology

Every comparison uses:
- **Shuffled baselines** to separate ordering effects from distribution effects
- **Bonferroni correction** for multiple testing (typically alpha = 0.05/108)
- **Cohen's d** effect sizes (not just p-values)
- **20-30 trials** per condition

The framework produces zero false positives on validated random sources and correctly identifies AES-CTR as indistinguishable from random across all approaches tested. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## What Doesn't Work

Equally important: the framework's known limits.

- **AES-CTR vs random**: Indistinguishable across all 24 geometries, preprocessings, and combination strategies. This is the fundamental limit.
- **MT19937, XorShift32, MINSTD**: Pass all geometric tests (weaknesses are in higher dimensions than byte-level)
- **Standard map, Arnold cat map**: Uniformly mixing → look random
- **LSB steganography at byte level**: Must use bitplane decomposition

See [docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md).

## Complete Results

All 62 validated findings and 17 negative results from 32 investigations: [docs/FINDINGS.md](docs/FINDINGS.md)

## Dependencies

**Core** (required):
- numpy
- scipy
- matplotlib

**Optional** (3 of 20 investigations):
- pycryptodome — for `ciphers.py`, `reduced_aes.py`, `ecb_penguin.py`

## License

MIT
