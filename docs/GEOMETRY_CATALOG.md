# Geometry Catalog

The framework implements 24 one-dimensional geometries and 1 two-dimensional spatial field geometry. Each embeds byte sequences into a different mathematical space and computes structure-sensitive metrics.

## 1D Geometries (24 total)

### Tier 1: Core Geometries (6 independent)

These capture most of the framework's discriminative power. The remaining 18 are largely redundant with these.

| # | Geometry | Space | Key Metric | What It Detects | Best d |
|---|----------|-------|------------|-----------------|--------|
| 1 | **E8 Lattice** | 8D root lattice (240 roots) | `unique_roots` | Algebraic constraints, entropy | RANDU d=-20 |
| 2 | **Fisher Information** | Statistical manifold | `trace_fisher` | Transition probability concentration | Universal workhorse |
| 3 | **Tropical** | (min,+) semiring | `linearity` | Piecewise-linear structure | Rossler d=266 |
| 4 | **Heisenberg** (centered) | Nil geometry | `twist_rate` | Autocorrelation, periodicity | Period detection 1000x |
| 5 | **Cantor** | Fractal (dim ≈ 0.631) | `coverage` | Hierarchical binary patterns | Pruned NN d=1.3B |
| 6 | **Penrose** | 5-fold quasicrystal | `fivefold_balance` | Aperiodic/chaotic structure | Chaos d=60 |

### Tier 2: Useful Specialists

| # | Geometry | Space | Key Metric | What It Detects |
|---|----------|-------|------------|-----------------|
| 7 | Torus | n-dimensional torus T^n | `coverage`, `entropy` | Uniformity, modular patterns |
| 8 | Hyperbolic | Poincare disk | `mean_radius` | Hierarchical depth |
| 9 | Sol | Thurston geometry | `anisotropy` | Directional asymmetry |
| 10 | Spherical S2 | Unit sphere | `concentration` | Directional clustering |
| 11 | Lorentzian | Minkowski spacetime | `causal_order` | Temporal ordering |
| 12 | Wasserstein | Optimal transport | `dist_from_uniform` | Distribution divergence |
| 13 | Heisenberg (bias) | Nil geometry (uncentered) | `path_length` | Raw correlation accumulation |
| 14 | AmmannBeenker | 8-fold quasicrystal | `square_diag_ratio` | Octagonal symmetry |
| 15 | HigherOrder | 3rd/4th moment space | `kurt_mean` | Non-Gaussian tails |

### Tier 3: Redundant (included for completeness)

| # | Geometry | Redundant With | Correlation |
|---|----------|---------------|-------------|
| 16 | Symplectic | Cantor | r > 0.85 |
| 17 | H2xR | Hyperbolic | r > 0.95 |
| 18 | S2xR | Spherical | r = 0.999 |
| 19 | SL(2,R) | Hyperbolic | r > 0.90 |
| 20 | Projective | Tropical | r = 0.980 |
| 21 | 2-adic | E8/Fisher | r > 0.85 |
| 22 | Persistent Homology | Torus | r = 0.964 |
| 23 | Spiral | Fisher | r > 0.85 |
| 24 | EinsteinHat | Penrose | balance r = 1.000 |

### Redundancy Summary

The 24 geometries collapse to ~6 independent dimensions of information. The correlation structure was established by computing all metrics across 8 diverse data types and measuring Pearson correlations. Key clusters:

- **Uniformity cluster**: E8, Fisher, Lorentzian, Torus, 2-adic, Persistent Homology
- **Value spread cluster**: Heisenberg, Sol, Hyperbolic, Spherical
- **Scale cluster**: H2xR, S2xR, Symplectic, Cantor
- **Linearity cluster**: Projective, Tropical
- **Aperiodic cluster**: Penrose, AmmannBeenker, EinsteinHat

Note: High aggregate correlation does NOT mean identical metrics. Adversarial examples can break r=0.998 correlations (e.g., sorting data drops E8 alignment by d=-73 while Fisher trace is unchanged). Keep both when the domain warrants it.

## 2D Spatial Field Geometry

`SpatialFieldGeometry` analyzes 2D arrays (fields, images, simulation grids) natively rather than flattening to 1D.

### Metrics (15 total)

| Metric | What It Measures |
|--------|-----------------|
| `tension_mean`, `tension_std` | Gradient magnitude (local change rate) |
| `curvature_mean`, `curvature_std` | Laplacian (local concavity) |
| `anisotropy_mean`, `anisotropy_std` | Hessian eigenvalue ratio (directional preference) |
| `criticality_extrema_frac` | Fraction of local extrema |
| `criticality_saddle_frac` | Fraction of saddle points |
| `n_basins` | Number of distinct basins (connected components after thresholding) |
| `coherence_score` | Spatial autocorrelation (Moran's I analog) |
| `multiscale_coherence_1` | Coherence at scale 1 (3x3 blocks) |
| `multiscale_coherence_2` | Coherence at scale 2 (5x5 blocks) |
| `multiscale_coherence_4` | Coherence at scale 4 (9x9 blocks) |
| `multiscale_coherence_8` | Coherence at scale 8 (17x17 blocks) |

### Best Metrics by Domain

| Domain | Best Metric | Why |
|--------|------------|-----|
| Percolation | `n_basins` (d=261) | Cluster count tracks connectivity |
| Voronoi | `n_basins` (d=-154) | Cell count is the defining feature |
| Mazes | `multiscale_coherence_2` (d=109) | Corridor width matches scale 2 |
| Wave equation | `coherence_score` (d=-496) | Wave coherence is literally what it measures |
| Reaction-diffusion | `tension_std` (d=97) | Pattern sharpness varies by morphology |
| Cellular automata | `anisotropy_mean` (d=78) | Rule symmetry affects directional structure |
| Growth models | `anisotropy_mean` (d=-178) | DLA branching vs Eden compactness |
| Ising model | `multiscale_coherence_4` | Peaks near T_c (scale-free structure) |

## Preprocessing Utilities

Three preprocessing functions extend the framework's reach:

| Function | What It Does | Best For |
|----------|-------------|----------|
| `delay_embed(data, tau)` | Pairs byte[i] with byte[i+tau] | Lag-specific structure (52x improvement for AR processes) |
| `spectral_preprocess(data)` | FFT magnitude spectrum → uint8 | Spectral structure invisible in raw bytes |
| `bitplane_extract(data, plane)` | Extract single bit plane (0=LSB, 7=MSB) | Steganography detection (d=1166) |

## Usage

```python
from exotic_geometry_framework import (
    GeometryAnalyzer, SpatialFieldGeometry,
    delay_embed, spectral_preprocess, bitplane_extract
)

# All 1D geometries
analyzer = GeometryAnalyzer().add_all_geometries()
results = analyzer.analyze(data_uint8)

# 2D spatial analysis
analyzer_2d = GeometryAnalyzer().add_spatial_geometries()
# or standalone:
geom = SpatialFieldGeometry()
result = geom.compute_metrics(field_2d)

# Preprocessing
delayed = delay_embed(data, tau=5)        # for lag detection
spectral = spectral_preprocess(data)       # for spectral analysis
lsb = bitplane_extract(data, plane=0)      # for stego detection
```
