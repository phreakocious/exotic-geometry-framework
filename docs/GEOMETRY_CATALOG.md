# Geometry Catalog

The framework implements 26 one-dimensional geometries and 8 two-dimensional spatial geometries (80 metrics). Each embeds byte sequences into a different mathematical space and computes structure-sensitive metrics.

## 1D Geometries (26 total)

### Tier 1: Core Geometries (6 independent)

These capture most of the framework's discriminative power. The remaining 20 are largely redundant with these.

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
| 16 | Fractal (Mandelbrot) | Mandelbrot set escape dynamics | `interior_fraction` | Escape behavior, set boundary structure |
| 17 | Fractal (Julia) | Julia set stability | `connectedness` | Julia set topology, stability |

### Tier 3: Redundant (included for completeness)

| # | Geometry | Redundant With | Correlation |
|---|----------|---------------|-------------|
| 18 | Symplectic | Cantor | r > 0.85 |
| 19 | H2xR | Hyperbolic | r > 0.95 |
| 20 | S2xR | Spherical | r = 0.999 |
| 21 | SL(2,R) | Hyperbolic | r > 0.90 |
| 22 | Projective | Tropical | r = 0.980 |
| 23 | 2-adic | E8/Fisher | r > 0.85 |
| 24 | Persistent Homology | Torus | r = 0.964 |
| 25 | Spiral | Fisher | r > 0.85 |
| 26 | EinsteinHat | Penrose | balance r = 1.000 |

### Redundancy Summary

The 26 geometries collapse to ~7 independent dimensions of information (participation ratio = 7.0 from meta-investigation). The correlation structure was established by computing all metrics across 8 diverse data types and measuring Pearson correlations. Key clusters:

- **Uniformity cluster**: E8, Fisher, Lorentzian, Torus, 2-adic, Persistent Homology
- **Value spread cluster**: Heisenberg, Sol, Hyperbolic, Spherical
- **Scale cluster**: H2xR, S2xR, Symplectic, Cantor
- **Linearity cluster**: Projective, Tropical
- **Aperiodic cluster**: Penrose, AmmannBeenker, EinsteinHat

Note: High aggregate correlation does NOT mean identical metrics. Adversarial examples can break r=0.998 correlations (e.g., sorting data drops E8 alignment by d=-73 while Fisher trace is unchanged). Keep both when the domain warrants it.

## 2D Spatial Geometries (8 total, 80 metrics)

`SpatialFieldGeometry` analyzes 2D arrays (fields, images, simulation grids) natively rather than flattening to 1D. Seven additional 2D geometries provide complementary perspectives.

### SpatialFieldGeometry Metrics (15)

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

### Additional 2D Geometries (7 more, 65 additional metrics)

| Geometry | Metrics | What It Measures |
|----------|:-------:|-----------------|
| **Surface** | 9 | Gaussian/mean curvature, shape index, Gauss-Bonnet theorem, metric area |
| **PersistentHomology2D** | 10 | Sublevel/superlevel persistence via union-find (33ms@64×64) |
| **Conformal2D** | 10 | Cauchy-Riemann residual, Riesz transform (FFT), isotropy, Liouville energy |
| **MinkowskiFunctional** | 10 | Excursion set area, boundary length, Euler characteristic at multiple thresholds |
| **MultiscaleFractal** | 9 | Box-counting dimension, lacunarity, Hurst exponent (68ms@64×64) |
| **HodgeLaplacian** | 9 | Laplacian/Dirichlet/biharmonic energy, Poisson recovery, gradient coherence |
| **SpectralPower** | 8 | Spectral slope β, centroid, entropy, anisotropy, kurtosis |

All 2D geometries use max_field_size=128, validate_data reshapes 1D→square, downsample via block averaging.

## Preprocessing Utilities

Three preprocessing functions extend the framework's reach:

| Function | What It Does | Best For |
|----------|-------------|----------|
| `delay_embed(data, tau)` | Pairs byte[i] with byte[i+tau] | Lag-specific structure (52x improvement for AR processes) |
| `spectral_preprocess(data)` | FFT magnitude spectrum → uint8 | Spectral structure invisible in raw bytes |
| `bitplane_extract(data, plane)` | Extract single bit plane (0=LSB, 7=MSB) | Bit-level structure (Collatz LSB=77 sig) |

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
