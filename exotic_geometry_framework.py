"""
Exotic Geometry Framework for Data Analysis

A reusable framework for applying exotic geometries to detect hidden structure in data.
Each geometry provides a consistent API for embedding data and computing metrics.

AVAILABLE GEOMETRIES (31 total):

  Lattice/Discrete:
    - E8Geometry: 240 roots in 8D, detects algebraic constraints (validated d=2.34)
    - CantorGeometry: fractal embedding, detects binary/hierarchical structure
    - UltrametricGeometry: p-adic distances, detects tree structure

  Curved Spaces:
    - HyperbolicGeometry: Poincaré disk, detects hierarchies
    - SphericalGeometry: S², detects cyclic/directional patterns
    - TorusGeometry: T², detects periodic structure (validated)

  Thurston's Geometries:
    - SolGeometry: exponential stretch/shrink, dynamical systems
    - ProductS2RGeometry: S² × ℝ, spherical with drift
    - ProductH2RGeometry: H² × ℝ, hyperbolic with drift
    - SL2RGeometry: SL̃(2,ℝ), projective transformations

  Algebraic:
    - TropicalGeometry: min-plus algebra, detects piecewise-linear structure
    - ProjectiveGeometry: ℙ², scale-invariant analysis

  Statistical:
    - WassersteinGeometry: optimal transport, compares distributions
    - FisherGeometry: information geometry, statistical manifolds
    - PersistentHomologyGeometry: TDA, topological features

  Physical/Dynamical:
    - HeisenbergGeometry: Nil group, detects correlations (2 modes: bias/correlation)
    - LorentzianGeometry: spacetime structure, causal ordering
    - SymplecticGeometry: phase space, Hamiltonian dynamics
    - SpiralGeometry: growth patterns, self-similarity

  Aperiodic:
    - PenroseGeometry: quasicrystals, 5-fold symmetry, golden ratio φ
    - AmmannBeenkerGeometry: octagonal, 8-fold symmetry, silver ratio
    - EinsteinHatGeometry: chiral aperiodic monotile, chirality detection

  Higher-Order:
    - HigherOrderGeometry: 3rd/4th order statistics (bispectrum, kurtosis,
      permutation entropy, C3 autocorrelation) - independent from all
      covariance-based geometries

  Spatial (opt-in, not in add_all_geometries):
    - SpatialFieldGeometry: native 2D field analysis — gradient tension,
      Hessian anisotropy, basin detection, multi-scale coherence
    - SurfaceGeometry: differential geometry of height maps — Gaussian/mean
      curvature, shape index, curvedness, Gauss-Bonnet integral
    - PersistentHomology2DGeometry: sublevel/superlevel persistence on 2D
      grids — component lifetimes, persistence entropy, asymmetry
    - ConformalGeometry2D: conformal analysis — Cauchy-Riemann residual,
      Riesz transform, structure isotropy, Liouville curvature
    - MinkowskiFunctionalGeometry: integral geometry — area, boundary, Euler
      characteristic of excursion sets at multiple thresholds (Hadwiger)
    - MultiscaleFractalGeometry: fractal/scaling analysis — box-counting
      dimension, lacunarity, Hurst exponent, fluctuation scaling
    - HodgeLaplacianGeometry: Hodge-Laplacian analysis — Dirichlet/biharmonic
      energy, Poisson recovery, gradient coherence, spectral gap
    - SpectralPowerGeometry: 2D FFT power spectrum — spectral slope β,
      centroid, entropy, anisotropy, high-frequency ratio

PREPROCESSING UTILITIES:
    - delay_embed(data, tau): Takens delay embedding, pairs byte[i] with byte[i+τ]
    - spectral_preprocess(data): FFT magnitude spectrum, normalized to uint8
    - bitplane_extract(data, plane): Extract single bit plane, repack to uint8

ENCODING UTILITIES (for non-byte data):
    - encode_symbolic(symbols, alphabet): Map symbolic sequences to [0,1] floats
    - encode_float_to_unit(data): Normalize float array to [0,1]
    - DNA_ALPHABET, AMINO_ALPHABET: Pre-defined alphabets

Usage:
    from exotic_geometry_framework import GeometryAnalyzer, E8Geometry, TorusGeometry

    # Quick analysis with defaults (uint8 byte data)
    analyzer = GeometryAnalyzer().add_default_geometries()
    results = analyzer.analyze(data)
    print(results.summary())

    # Full analysis with all geometries
    analyzer = GeometryAnalyzer().add_all_geometries()
    results = analyzer.analyze(data)

    # Float data (any range) — use data_mode='auto'
    analyzer = GeometryAnalyzer().add_all_geometries(data_mode='auto')
    results = analyzer.analyze(float_array)

    # Data already in [0,1] — use data_mode='unit'
    analyzer = GeometryAnalyzer().add_all_geometries(data_mode='unit')
    results = analyzer.analyze(unit_data)

    # Symbolic data (DNA, amino acids, etc.)
    from exotic_geometry_framework import encode_symbolic, DNA_ALPHABET
    encoded = encode_symbolic("ACGTACGT", DNA_ALPHABET)
    analyzer = GeometryAnalyzer().add_all_geometries(data_mode='unit')
    results = analyzer.analyze(encoded)

    # Individual geometry with auto-scaling
    geom = HyperbolicGeometry(input_scale='auto')
    result = geom.compute_metrics(float_array)

    # Custom selection
    analyzer = GeometryAnalyzer()
    analyzer.add_geometry(E8Geometry())
    analyzer.add_geometry(TorusGeometry(bins=16))
    results = analyzer.analyze(data)

    # Compare two datasets
    comparison = analyzer.compare(data1, data2, "Good", "Bad")
"""

import math
import os
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import combinations, product
from collections import Counter
import warnings


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class GeometryResult:
    """Result from a single geometry analysis."""
    geometry_name: str
    metrics: Dict[str, float]
    raw_data: Optional[Dict[str, Any]] = None

    def __repr__(self):
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"{self.geometry_name}: {metrics_str}"


@dataclass
class AnalysisResult:
    """Combined results from all geometries."""
    results: List[GeometryResult] = field(default_factory=list)
    data_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "EXOTIC GEOMETRY ANALYSIS RESULTS", "=" * 60]
        if self.data_info:
            lines.append(f"Data: {self.data_info}")
        lines.append("")
        for r in self.results:
            lines.append(str(r))
        return "\n".join(lines)

    def get_metric(self, geometry: str, metric: str) -> Optional[float]:
        for r in self.results:
            if r.geometry_name == geometry:
                return r.metrics.get(metric)
        return None

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {r.geometry_name: r.metrics for r in self.results}


class ExoticGeometry(ABC):
    """Base class for exotic geometries."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this geometry."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> Union[int, str]:
        """Dimension of the geometry (or 'variable', 'fractal', etc.)."""
        pass

    @abstractmethod
    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data into this geometry's space."""
        pass

    @abstractmethod
    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute all metrics for this geometry on the given data."""
        pass

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        """Validate and preprocess input data."""
        data = np.asarray(data)
        if data.ndim == 1:
            return data
        return data.flatten()

    @staticmethod
    def _normalize_to_unit(data: np.ndarray, input_scale='auto') -> np.ndarray:
        """Normalize data to [0, 1] range.

        Parameters
        ----------
        data : np.ndarray
            Input data (any numeric type).
        input_scale : float or 'auto'
            If 'auto', normalize by data range. If a number, divide by that value.

        Returns
        -------
        np.ndarray of float in [0, 1]
        """
        data = data.astype(float)
        if input_scale == 'auto':
            lo, hi = data.min(), data.max()
            return (data - lo) / (hi - lo + 1e-10) if hi > lo else np.zeros_like(data)
        return data / input_scale


# =============================================================================
# E8 LATTICE GEOMETRY
# =============================================================================

class E8Geometry(ExoticGeometry):
    """
    E8 Lattice Geometry - detects algebraic constraints.

    The E8 lattice has 240 roots that uniformly cover the 8-sphere.
    Low root diversity indicates data is constrained to a subspace.

    Validated for:
    - S-box analysis: d=2.34 distinguishing good from weak
    - PRNG analysis: works with domain-specific transformations
    """

    def __init__(self, window_size: int = 8, normalize: bool = True):
        self.window_size = window_size
        self.normalize = normalize
        self._roots = None
        self._roots_normalized = None

    @property
    def name(self) -> str:
        return "E8 Lattice"

    @property
    def dimension(self) -> int:
        return 8

    @property
    def roots(self) -> np.ndarray:
        """Lazily compute E8 roots (240 vectors)."""
        if self._roots is None:
            self._roots = self._compute_e8_roots()
            norms = np.linalg.norm(self._roots, axis=1, keepdims=True)
            self._roots_normalized = self._roots / norms
        return self._roots

    @property
    def roots_normalized(self) -> np.ndarray:
        """Normalized E8 roots on the 7-sphere."""
        if self._roots_normalized is None:
            _ = self.roots  # Trigger computation
        return self._roots_normalized

    def _compute_e8_roots(self) -> np.ndarray:
        """Generate the 240 roots of E8."""
        roots = []

        # Type 1: ±eᵢ ± eⱼ (112 roots)
        for pos in combinations(range(8), 2):
            for signs in product([1, -1], repeat=2):
                root = np.zeros(8)
                root[pos[0]] = signs[0]
                root[pos[1]] = signs[1]
                roots.append(root)

        # Type 2: (±1/2, ..., ±1/2) with even number of minus signs (128 roots)
        for signs in product([0.5, -0.5], repeat=8):
            if sum(1 for s in signs if s < 0) % 2 == 0:
                roots.append(np.array(signs))

        return np.array(roots)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data as 8-dimensional windows."""
        data = self.validate_data(data)
        n_windows = len(data) // self.window_size

        if n_windows == 0:
            raise ValueError(f"Data too short for window size {self.window_size}")

        windows = []
        for i in range(n_windows):
            window = data[i * self.window_size:(i + 1) * self.window_size].astype(float)
            if self.normalize:
                # Normalize to unit sphere
                window = window / (np.max(data) + 1e-10)  # Scale to [0, 1]
                mean = np.mean(window)
                std = np.std(window) + 1e-10
                window = (window - mean) / std
            windows.append(window)

        return np.array(windows)

    def find_closest_roots(self, embedded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find closest E8 root for each embedded point."""
        dots = embedded @ self.roots_normalized.T
        best_indices = np.argmax(np.abs(dots), axis=1)
        best_alignments = np.array([np.abs(dots[i, best_indices[i]])
                                    for i in range(len(dots))])
        return best_indices, best_alignments

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute E8 metrics: diversity, alignment stats, entropy."""
        embedded = self.embed(data)
        root_indices, alignments = self.find_closest_roots(embedded)

        # Core metrics
        unique_roots = len(set(root_indices))
        diversity_ratio = unique_roots / 240  # Fraction of roots used

        # Alignment statistics
        align_mean = np.mean(alignments)
        align_std = np.std(alignments)

        # Root usage entropy
        counts = Counter(root_indices)
        probs = np.array(list(counts.values())) / len(root_indices)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(min(len(root_indices), 240))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "unique_roots": unique_roots,
                "diversity_ratio": diversity_ratio,
                "alignment_mean": align_mean,
                "alignment_std": align_std,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
            },
            raw_data={
                "root_indices": root_indices,
                "alignments": alignments,
                "root_counts": dict(counts),
            }
        )


# =============================================================================
# TORUS GEOMETRY
# =============================================================================

class TorusGeometry(ExoticGeometry):
    """
    Torus (𝕋²) Geometry - detects periodic/cyclic structure.

    Maps consecutive pairs to a 2D torus and measures coverage.
    Non-uniform coverage indicates periodic or constrained structure.

    Validated for:
    - S-box analysis: AES ≈ 0.90 coverage, weak ≈ 0.50
    """

    def __init__(self, bins: int = 16, dimension: int = 2):
        self.bins = bins
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"Torus T^{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data as points on torus (consecutive pairs/tuples)."""
        data = self.validate_data(data)
        data = data.astype(float)

        # Normalize to [0, 1)
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min + 1e-10)

        # Create tuples
        n_points = len(data) // self._dimension
        points = []
        for i in range(n_points):
            point = data[i * self._dimension:(i + 1) * self._dimension]
            # Wrap to [0, 1) for torus
            point = point % 1.0
            points.append(point)

        return np.array(points)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute torus metrics: coverage, entropy, uniformity."""
        embedded = self.embed(data)

        # Bin the points
        bins_per_dim = [self.bins] * self._dimension
        hist, _ = np.histogramdd(embedded, bins=bins_per_dim,
                                  range=[(0, 1)] * self._dimension)

        # Coverage: fraction of bins with at least one point
        n_occupied = np.sum(hist > 0)
        n_total = np.prod(bins_per_dim)
        coverage = n_occupied / n_total

        # Entropy of bin distribution
        hist_flat = hist.flatten()
        probs = hist_flat / (np.sum(hist_flat) + 1e-10)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(n_total)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Uniformity: chi-square statistic vs uniform
        expected = len(embedded) / n_total
        chi2 = np.sum((hist_flat - expected) ** 2 / (expected + 1e-10))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "coverage": coverage,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "chi2_uniformity": chi2,
                "occupied_bins": n_occupied,
            },
            raw_data={
                "histogram": hist,
                "embedded_points": embedded,
            }
        )


# =============================================================================
# HYPERBOLIC GEOMETRY (POINCARÉ DISK)
# =============================================================================

class HyperbolicGeometry(ExoticGeometry):
    """
    Hyperbolic Geometry (Poincaré Disk Model) - detects hierarchical structure.

    Maps data to the Poincaré disk and measures dispersion from origin.
    Hierarchical data tends to cluster near the boundary.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Hyperbolic (Poincaré)"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed pairs to Poincaré disk."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        points = []
        for i in range(n_points):
            x, y = data[2*i], data[2*i + 1]
            # Map to disk (ensure |z| < 1)
            z = complex(x - 0.5, y - 0.5) * 1.8  # Scale to use more of disk
            r = abs(z)
            if r >= 1:
                z = z / (r + 0.01) * 0.99
            points.append([z.real, z.imag])

        return np.array(points)

    def hyperbolic_distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Compute hyperbolic distance in Poincaré disk."""
        z1, z2 = complex(z1[0], z1[1]), complex(z2[0], z2[1])
        num = abs(z1 - z2)
        denom = abs(1 - np.conj(z1) * z2)
        arg = num / (denom + 1e-10)
        arg = min(arg, 0.9999)  # Clamp for numerical stability
        return 2 * np.arctanh(arg)

    def mobius_add(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Möbius addition in Poincaré disk."""
        z1c = complex(z1[0], z1[1])
        z2c = complex(z2[0], z2[1])
        num = z1c + z2c
        denom = 1 + np.conj(z1c) * z2c
        result = num / denom
        return np.array([result.real, result.imag])

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute hyperbolic metrics: centroid, dispersion, boundary proximity."""
        embedded = self.embed(data)

        # Centroid (Euclidean approximation - proper would use Fréchet mean)
        centroid = np.mean(embedded, axis=0)
        centroid_norm = np.linalg.norm(centroid)

        # Distances from origin
        radii = np.linalg.norm(embedded, axis=1)
        mean_radius = np.mean(radii)

        # Hyperbolic distances from origin
        hyp_radii = 2 * np.arctanh(np.clip(radii, 0, 0.9999))
        mean_hyp_radius = np.mean(hyp_radii)

        # Boundary proximity (how close to edge)
        boundary_proximity = np.mean(radii > 0.8)

        # Dispersion (variance of hyperbolic distances)
        dispersion = np.std(hyp_radii)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "centroid_norm": centroid_norm,
                "mean_radius": mean_radius,
                "mean_hyperbolic_radius": mean_hyp_radius,
                "boundary_proximity": boundary_proximity,
                "dispersion": dispersion,
            },
            raw_data={
                "embedded_points": embedded,
                "centroid": centroid,
            }
        )


# =============================================================================
# HEISENBERG (NIL) GEOMETRY
# =============================================================================

class HeisenbergGeometry(ExoticGeometry):
    """
    Heisenberg/Nil Geometry - detects correlations and area accumulation.

    The z-coordinate tracks accumulated "twist" between x and y.

    IMPORTANT: There are two modes:
    - center_data=False (default): Detects MEAN BIAS. Data far from 127.5 will
      show high twist because all products have the same sign.
    - center_data=True: Detects CORRELATION. Centers data around its mean first,
      so twist measures actual dependency between consecutive values.

    Use center_data=True for correlation detection, False for mean bias detection.
    """

    def __init__(self, input_scale: float = 255.0, center_data: bool = False):
        self.input_scale = input_scale
        self.center_data = center_data

    @property
    def name(self) -> str:
        suffix = " (centered)" if self.center_data else ""
        return f"Heisenberg (Nil){suffix}"

    @property
    def dimension(self) -> int:
        return 3

    def heisenberg_multiply(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """Heisenberg group multiplication: (x1,y1,z1)·(x2,y2,z2) = (x1+x2, y1+y2, z1+z2+x1*y2)"""
        return np.array([
            g1[0] + g2[0],
            g1[1] + g2[1],
            g1[2] + g2[2] + g1[0] * g2[1]
        ])

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data as path in Heisenberg group."""
        data = self.validate_data(data)
        data = data.astype(float)

        if self.center_data:
            # Center around actual mean - detects CORRELATION
            data = (data - data.mean()) / (data.std() + 1e-10)
        else:
            # Center around midpoint - detects MEAN BIAS
            data = self._normalize_to_unit(data, self.input_scale) - 0.5

        # Build path: each pair (x,y) becomes a group element
        n_steps = len(data) // 2
        path = [np.array([0.0, 0.0, 0.0])]

        for i in range(n_steps):
            step = np.array([data[2*i], data[2*i + 1], 0.0])
            new_point = self.heisenberg_multiply(path[-1], step)
            path.append(new_point)

        return np.array(path)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Heisenberg metrics: final z, path length, twist rate."""
        path = self.embed(data)

        # Final position
        final = path[-1]
        final_z = abs(final[2])  # Accumulated twist

        # Path length (Euclidean approximation)
        diffs = np.diff(path, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))

        # Twist rate (z per step)
        n_steps = len(path) - 1
        twist_rate = final_z / n_steps if n_steps > 0 else 0

        # XY spread
        xy_std = np.std(path[:, :2])

        # Z accumulation profile
        z_values = path[:, 2]
        z_variance = np.var(z_values)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "final_z": final_z,
                "path_length": path_length,
                "twist_rate": twist_rate,
                "xy_spread": xy_std,
                "z_variance": z_variance,
            },
            raw_data={
                "path": path,
                "final_position": final,
            }
        )


# =============================================================================
# SPHERICAL GEOMETRY
# =============================================================================

class SphericalGeometry(ExoticGeometry):
    """
    Spherical Geometry (S²) - detects directional/cyclic structure.

    Maps data to points on a sphere and measures distribution.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Spherical S²"

    @property
    def dimension(self) -> int:
        return 2  # Surface dimension

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed pairs as points on S² using spherical coordinates."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        points = []
        for i in range(n_points):
            theta = data[2*i] * np.pi  # [0, π]
            phi = data[2*i + 1] * 2 * np.pi  # [0, 2π]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])

        return np.array(points)

    def spherical_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Great circle distance on S²."""
        dot = np.clip(np.dot(p1, p2), -1, 1)
        return np.arccos(dot)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute spherical metrics."""
        embedded = self.embed(data)

        # Mean direction (vector sum normalized)
        mean_vec = np.mean(embedded, axis=0)
        mean_length = np.linalg.norm(mean_vec)  # Resultant length

        if mean_length > 1e-10:
            mean_dir = mean_vec / mean_length
        else:
            mean_dir = np.array([0, 0, 1])

        # Concentration (1 - variance)
        concentration = mean_length

        # Distances from mean direction
        dots = embedded @ mean_dir
        angles = np.arccos(np.clip(dots, -1, 1))
        angular_spread = np.std(angles)

        # Hemisphere balance
        north_frac = np.mean(embedded[:, 2] > 0)
        hemisphere_balance = 1 - 2 * abs(north_frac - 0.5)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "concentration": concentration,
                "angular_spread": angular_spread,
                "hemisphere_balance": hemisphere_balance,
                "mean_z": np.mean(embedded[:, 2]),
            },
            raw_data={
                "embedded_points": embedded,
                "mean_direction": mean_dir,
            }
        )


# =============================================================================
# FRACTAL (CANTOR) GEOMETRY
# =============================================================================

class CantorGeometry(ExoticGeometry):
    """
    Cantor Set Embedding - analyzes binary/ternary structure.

    Maps bytes to Cantor set coordinates and measures dimension.
    """

    def __init__(self, base: int = 3):
        self.base = base

    @property
    def name(self) -> str:
        return f"Cantor (base {self.base})"

    @property
    def dimension(self) -> str:
        return f"fractal (~{np.log(2)/np.log(self.base):.3f})"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed bytes as Cantor set coordinates."""
        data = self.validate_data(data)

        coords = []
        for byte in data:
            # Map byte to [0, 1] via ternary expansion
            x = 0
            val = byte
            for i in range(8):
                bit = (val >> (7 - i)) & 1
                x += (2 * bit) / (self.base ** (i + 1))
            coords.append(x)

        return np.array(coords)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute fractal metrics: box-counting dimension estimate, gaps."""
        embedded = self.embed(data)

        # Box-counting dimension estimate
        dimensions = []
        for scale in [0.1, 0.05, 0.02, 0.01]:
            bins = int(1 / scale)
            hist, _ = np.histogram(embedded, bins=bins, range=(0, 1))
            n_occupied = np.sum(hist > 0)
            if n_occupied > 0:
                dimensions.append(np.log(n_occupied) / np.log(bins))

        est_dimension = np.mean(dimensions) if dimensions else 1.0

        # Gap analysis
        sorted_coords = np.sort(embedded)
        gaps = np.diff(sorted_coords)
        mean_gap = np.mean(gaps)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0

        # Coverage
        coverage = len(set(np.round(embedded, 4))) / len(embedded)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "estimated_dimension": est_dimension,
                "mean_gap": mean_gap,
                "max_gap": max_gap,
                "coverage": coverage,
            },
            raw_data={
                "embedded_coords": embedded,
            }
        )


# =============================================================================
# ULTRAMETRIC (P-ADIC) GEOMETRY
# =============================================================================

class UltrametricGeometry(ExoticGeometry):
    """
    Ultrametric/p-adic Geometry - detects hierarchical tree structure.

    Uses ultrametric distance where d(x,z) ≤ max(d(x,y), d(y,z)).
    """

    def __init__(self, p: int = 2):
        self.p = p

    @property
    def name(self) -> str:
        return f"{self.p}-adic"

    @property
    def dimension(self) -> str:
        return "ultrametric"

    def p_adic_distance(self, a: int, b: int) -> float:
        """Compute p-adic distance between integers."""
        if a == b:
            return 0
        diff = abs(int(a) - int(b))
        # Count powers of p in diff
        v = 0
        while diff % self.p == 0:
            diff //= self.p
            v += 1
        return float(self.p ** (-v))

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Return data as-is (p-adic metric is on integers)."""
        return self.validate_data(data).astype(int)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute ultrametric metrics."""
        embedded = self.embed(data)

        # Sample pairwise distances
        n = min(len(embedded), 500)
        indices = np.random.choice(len(embedded), n, replace=False)
        sample = embedded[indices]

        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, min(i + 50, len(sample))):
                distances.append(self.p_adic_distance(sample[i], sample[j]))

        distances = np.array(distances)

        # Statistics
        mean_dist = np.mean(distances)
        dist_entropy = -np.sum([
            (np.sum(distances == d) / len(distances)) *
            np.log2(np.sum(distances == d) / len(distances) + 1e-10)
            for d in set(distances)
        ])

        # Check ultrametric property violations (should be 0 for tree-like)
        violations = 0
        total_checks = 0
        for i in range(min(100, len(sample) - 2)):
            d_ab = self.p_adic_distance(sample[i], sample[i+1])
            d_bc = self.p_adic_distance(sample[i+1], sample[i+2])
            d_ac = self.p_adic_distance(sample[i], sample[i+2])
            if d_ac > max(d_ab, d_bc) * 1.001:  # Small tolerance
                violations += 1
            total_checks += 1

        violation_rate = violations / total_checks if total_checks > 0 else 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "mean_distance": mean_dist,
                "distance_entropy": dist_entropy,
                "ultrametric_violation_rate": violation_rate,
            },
            raw_data={
                "sample_distances": distances,
            }
        )


# =============================================================================
# TROPICAL GEOMETRY
# =============================================================================

class TropicalGeometry(ExoticGeometry):
    """
    Tropical Geometry - detects piecewise-linear structure.

    In tropical algebra: ⊕ = min, ⊗ = +
    Tropical polynomials become piecewise-linear functions.
    Useful for optimization and detecting linear constraints.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Tropical"

    @property
    def dimension(self) -> str:
        return "piecewise-linear"

    def tropical_add(self, a: float, b: float) -> float:
        """Tropical addition: a ⊕ b = min(a, b)"""
        return min(a, b)

    def tropical_mult(self, a: float, b: float) -> float:
        """Tropical multiplication: a ⊗ b = a + b"""
        return a + b

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data for tropical analysis."""
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute tropical metrics: linearity detection, tropical convexity."""
        embedded = self.embed(data)

        # Tropical metric: d(x,y) = max|xᵢ - yᵢ| (L∞ norm)
        # Measure how "tropical-linear" the sequence is
        n = len(embedded)

        # Piecewise linearity: count slope changes
        if n < 3:
            slope_changes = 0
            linearity = 1.0
        else:
            diffs = np.diff(embedded)
            slope_changes = np.sum(np.abs(np.diff(diffs)) > 0.01)
            # Linearity: fraction without slope change
            linearity = 1 - slope_changes / (n - 2)

        # Tropical convex hull approximation (min-plus)
        # For 1D, this is just the running minimum
        running_min = np.minimum.accumulate(embedded)
        tropical_hull_area = np.sum(embedded - running_min)

        # Check for tropical polynomial structure
        # A tropical polynomial has at most k+1 linear pieces for degree k
        # Count distinct "slopes" in windows
        window_size = 10
        slopes = []
        for i in range(0, n - window_size, window_size):
            window = embedded[i:i + window_size]
            slope = (window[-1] - window[0]) / window_size
            slopes.append(round(slope, 2))
        unique_slopes = len(set(slopes)) if slopes else 1

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "linearity": linearity,
                "slope_changes": slope_changes,
                "tropical_hull_area": tropical_hull_area,
                "unique_slopes": unique_slopes,
            },
            raw_data={
                "embedded": embedded,
                "running_min": running_min,
            }
        )


# =============================================================================
# WASSERSTEIN / OPTIMAL TRANSPORT GEOMETRY
# =============================================================================

class WassersteinGeometry(ExoticGeometry):
    """
    Wasserstein (Optimal Transport) Geometry - compares distributions.

    Measures the "earth mover's distance" between empirical distributions.
    Good for detecting distributional shifts and comparing histograms.
    """

    def __init__(self, n_bins: int = 32, input_scale: float = 255.0):
        self.n_bins = n_bins
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Wasserstein"

    @property
    def dimension(self) -> str:
        return "distribution space"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Convert to histogram (empirical distribution)."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)
        hist, _ = np.histogram(data, bins=self.n_bins, range=(0, 1), density=True)
        return hist / (np.sum(hist) + 1e-10)  # Normalize to probability

    def wasserstein_1d(self, p: np.ndarray, q: np.ndarray) -> float:
        """1D Wasserstein distance (earth mover's distance)."""
        # For 1D, W₁ = integral of |CDF_p - CDF_q|
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return np.sum(np.abs(cdf_p - cdf_q)) / len(p)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Wasserstein metrics: distance from uniform, self-similarity."""
        hist = self.embed(data)

        # Distance from uniform distribution
        uniform = np.ones(self.n_bins) / self.n_bins
        dist_from_uniform = self.wasserstein_1d(hist, uniform)

        # Self-similarity: compare first half to second half
        data = self.validate_data(data)
        mid = len(data) // 2
        hist1 = self.embed(data[:mid])
        hist2 = self.embed(data[mid:])
        self_similarity = 1 - self.wasserstein_1d(hist1, hist2)

        # Entropy of distribution
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(self.n_bins)
        normalized_entropy = entropy / max_entropy

        # Concentration: how peaked is the distribution
        concentration = np.max(hist) * self.n_bins  # 1 = uniform, >1 = concentrated

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "dist_from_uniform": dist_from_uniform,
                "self_similarity": self_similarity,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "concentration": concentration,
            },
            raw_data={
                "histogram": hist,
            }
        )


# =============================================================================
# PERSISTENT HOMOLOGY (TDA)
# =============================================================================

class PersistentHomologyGeometry(ExoticGeometry):
    """
    Persistent Homology - topological data analysis.

    Tracks birth/death of topological features (components, loops, voids)
    across scales. Good for detecting robust structural features.

    Note: This is a simplified implementation. For full TDA, use giotto-tda or ripser.
    """

    def __init__(self, n_points: int = 100, max_dim: int = 1):
        self.n_points = n_points
        self.max_dim = max_dim

    @property
    def name(self) -> str:
        return "Persistent Homology"

    @property
    def dimension(self) -> str:
        return "topological"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as 2D point cloud using delay embedding."""
        data = self.validate_data(data)
        data = data.astype(float)

        # Subsample if too large
        if len(data) > self.n_points * 2:
            indices = np.linspace(0, len(data) - 2, self.n_points, dtype=int)
        else:
            indices = np.arange(0, len(data) - 1)

        # Delay embedding: (x[i], x[i+1])
        points = np.array([[data[i], data[i + 1]] for i in indices])
        return points

    def compute_rips_homology_h0(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """
        Simplified H0 (connected components) persistent homology.
        Uses union-find to track component merges.
        """
        n = len(points)
        if n == 0:
            return []

        # Compute all pairwise distances
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(points))

        # Get sorted edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dists[i, j], i, j))
        edges.sort()

        # Union-find
        parent = list(range(n))
        birth = [0.0] * n  # All components born at scale 0

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        persistence_pairs = []
        for dist, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                # Merge: younger component dies
                # (In practice, we merge the one with higher index)
                if ri > rj:
                    ri, rj = rj, ri
                parent[rj] = ri
                death = dist
                persistence_pairs.append((birth[rj], death))

        # One component lives forever (we cap at max distance)
        max_dist = edges[-1][0] if edges else 1.0
        persistence_pairs.append((0.0, max_dist))

        return persistence_pairs

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute persistence metrics."""
        points = self.embed(data)

        # Normalize points to [0,1]
        points = points - points.min()
        points = points / (points.max() + 1e-10)

        # Compute H0 persistence
        h0_pairs = self.compute_rips_homology_h0(points)

        # Persistence statistics
        lifetimes = [death - birth for birth, death in h0_pairs]
        lifetimes = sorted(lifetimes, reverse=True)

        # Total persistence
        total_persistence = sum(lifetimes)

        # Number of significant features (lifetime > threshold)
        threshold = 0.1
        n_significant = sum(1 for l in lifetimes if l > threshold)

        # Persistence entropy
        lifetimes_arr = np.array(lifetimes)
        probs = lifetimes_arr / (total_persistence + 1e-10)
        persistence_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Betti curve summary (number of components at different scales)
        # Simplified: just report max and final
        max_components = len(h0_pairs)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "total_persistence": total_persistence,
                "n_significant_features": n_significant,
                "persistence_entropy": persistence_entropy,
                "max_components": max_components,
                "max_lifetime": lifetimes[0] if lifetimes else 0,
            },
            raw_data={
                "h0_pairs": h0_pairs,
                "lifetimes": lifetimes,
                "points": points,
            }
        )


# =============================================================================
# LORENTZIAN / MINKOWSKI GEOMETRY
# =============================================================================

class LorentzianGeometry(ExoticGeometry):
    """
    Lorentzian (Minkowski) Geometry - spacetime structure.

    Uses metric ds² = -dt² + dx² (signature -,+,+,...)
    Points can be timelike, spacelike, or lightlike separated.
    Good for detecting causal/sequential structure.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Lorentzian"

    @property
    def dimension(self) -> str:
        return "1+1 spacetime"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as spacetime events (t, x)."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        # t = index (time), x = value (space)
        events = np.array([[i / len(data), data[i]] for i in range(len(data))])
        return events

    def minkowski_interval(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Compute Minkowski interval: s² = -(Δt)² + (Δx)²
        s² < 0: timelike (causal)
        s² > 0: spacelike (acausal)
        s² = 0: lightlike (null)
        """
        dt = e2[0] - e1[0]
        dx = e2[1] - e1[1]
        return -dt**2 + dx**2

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Lorentzian metrics: causal structure, light cone analysis."""
        events = self.embed(data)
        n = len(events)

        # Sample intervals
        n_samples = min(500, n * (n - 1) // 2)
        intervals = []

        for _ in range(n_samples):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                s2 = self.minkowski_interval(events[i], events[j])
                intervals.append(s2)

        intervals = np.array(intervals)

        # Classify separations
        timelike_frac = np.mean(intervals < -1e-10)
        spacelike_frac = np.mean(intervals > 1e-10)
        lightlike_frac = np.mean(np.abs(intervals) < 1e-10)

        # Causal ordering: for consecutive events
        consecutive_intervals = []
        for i in range(n - 1):
            s2 = self.minkowski_interval(events[i], events[i + 1])
            consecutive_intervals.append(s2)

        consecutive_intervals = np.array(consecutive_intervals)
        causal_order_preserved = np.mean(consecutive_intervals < 0)

        # Light cone analysis: how fast does the signal propagate?
        # dx/dt for consecutive events
        velocities = []
        for i in range(n - 1):
            dt = events[i + 1, 0] - events[i, 0]
            dx = events[i + 1, 1] - events[i, 1]
            if dt > 1e-10:
                velocities.append(abs(dx / dt))

        mean_velocity = np.mean(velocities) if velocities else 0
        superluminal_frac = np.mean(np.array(velocities) > 1) if velocities else 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "timelike_fraction": timelike_frac,
                "spacelike_fraction": spacelike_frac,
                "lightlike_fraction": lightlike_frac,
                "causal_order_preserved": causal_order_preserved,
                "mean_velocity": mean_velocity,
                "superluminal_fraction": superluminal_frac,
            },
            raw_data={
                "events": events,
                "intervals": intervals,
            }
        )


# =============================================================================
# SPIRAL GEOMETRY
# =============================================================================

class SpiralGeometry(ExoticGeometry):
    """
    Spiral Geometry - detects growth patterns and self-similarity.

    Maps data to logarithmic spiral and measures winding/growth.
    Good for detecting exponential/multiplicative structure.
    """

    def __init__(self, input_scale: float = 255.0, spiral_type: str = "logarithmic"):
        self.input_scale = input_scale
        self.spiral_type = spiral_type

    @property
    def name(self) -> str:
        return f"Spiral ({self.spiral_type})"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data on spiral."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        points = []
        for i, val in enumerate(data):
            theta = i * 0.1  # Angle increases with index

            if self.spiral_type == "logarithmic":
                # r = a * e^(b*θ), with r modulated by value
                r = (0.1 + val) * np.exp(0.1 * theta)
            elif self.spiral_type == "archimedean":
                # r = a + b*θ
                r = (0.1 + val) + 0.05 * theta
            else:  # fermat
                # r² = a²*θ
                r = np.sqrt((0.1 + val) * theta)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])

        return np.array(points)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute spiral metrics: growth rate, winding uniformity."""
        points = self.embed(data)

        # Radial distances from origin
        radii = np.linalg.norm(points, axis=1)

        # Growth rate (exponential fit)
        if len(radii) > 10:
            log_radii = np.log(radii + 1e-10)
            indices = np.arange(len(radii))
            growth_rate = np.polyfit(indices, log_radii, 1)[0]
        else:
            growth_rate = 0

        # Angular positions
        angles = np.arctan2(points[:, 1], points[:, 0])

        # Winding number
        unwrapped = np.unwrap(angles)
        total_winding = (unwrapped[-1] - unwrapped[0]) / (2 * np.pi) if len(unwrapped) > 1 else 0

        # Angular uniformity (should increase steadily)
        if len(unwrapped) > 2:
            angular_diffs = np.diff(unwrapped)
            angular_uniformity = 1 - np.std(angular_diffs) / (np.mean(np.abs(angular_diffs)) + 1e-10)
        else:
            angular_uniformity = 1

        # Spiral tightness (ratio of radial to angular change)
        if len(radii) > 1:
            radial_change = radii[-1] - radii[0]
            tightness = radial_change / (total_winding + 1e-10)
        else:
            tightness = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "growth_rate": growth_rate,
                "total_winding": total_winding,
                "angular_uniformity": angular_uniformity,
                "tightness": tightness,
                "final_radius": radii[-1] if len(radii) > 0 else 0,
            },
            raw_data={
                "points": points,
                "radii": radii,
            }
        )


# =============================================================================
# PROJECTIVE GEOMETRY
# =============================================================================

class ProjectiveGeometry(ExoticGeometry):
    """
    Projective Geometry (ℙ²) - scale-invariant analysis.

    Points are equivalence classes [x:y:z] with (x,y,z) ~ (λx,λy,λz).
    Good for detecting relationships that are scale-invariant.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Projective ℙ²"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed triples as projective points."""
        data = self.validate_data(data)
        data = data.astype(float) + 1  # Avoid zeros

        n_points = len(data) // 3
        points = []

        for i in range(n_points):
            x, y, z = data[3*i], data[3*i + 1], data[3*i + 2]
            # Normalize to projective coordinates (on unit sphere)
            norm = np.sqrt(x**2 + y**2 + z**2)
            points.append([x/norm, y/norm, z/norm])

        return np.array(points)

    def projective_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Fubini-Study distance on ℙ².
        d(p,q) = arccos(|p·q|)
        """
        dot = np.abs(np.dot(p1, p2))
        dot = np.clip(dot, -1, 1)
        return np.arccos(dot)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute projective metrics."""
        points = self.embed(data)
        n = len(points)

        if n < 2:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": n},
                raw_data={"points": points}
            )

        # Sample pairwise distances
        distances = []
        for i in range(min(200, n)):
            for j in range(i + 1, min(i + 20, n)):
                distances.append(self.projective_distance(points[i], points[j]))

        distances = np.array(distances)

        # Cross-ratio invariant (projective invariant)
        # For 4 collinear points, cross-ratio is preserved
        cross_ratios = []
        for _ in range(min(100, n // 4)):
            idx = np.random.choice(n, 4, replace=False)
            p = points[idx]
            # Simplified: compute ratio of distance products
            d01 = self.projective_distance(p[0], p[1])
            d23 = self.projective_distance(p[2], p[3])
            d02 = self.projective_distance(p[0], p[2])
            d13 = self.projective_distance(p[1], p[3])
            if d02 * d13 > 1e-10:
                cr = (d01 * d23) / (d02 * d13)
                cross_ratios.append(cr)

        cross_ratio_std = np.std(cross_ratios) if cross_ratios else 0

        # Collinearity: how often are 3 points nearly collinear?
        collinear_count = 0
        total_triples = 0
        for _ in range(min(100, n // 3)):
            idx = np.random.choice(n, 3, replace=False)
            p = points[idx]
            # Volume of parallelepiped (0 = collinear in ℙ²)
            vol = abs(np.linalg.det(p))
            if vol < 0.1:
                collinear_count += 1
            total_triples += 1

        collinearity = collinear_count / total_triples if total_triples > 0 else 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "mean_distance": np.mean(distances),
                "distance_std": np.std(distances),
                "cross_ratio_std": cross_ratio_std,
                "collinearity": collinearity,
            },
            raw_data={
                "points": points,
                "distances": distances,
            }
        )


# =============================================================================
# INFORMATION GEOMETRY (FISHER)
# =============================================================================

class FisherGeometry(ExoticGeometry):
    """
    Fisher Information Geometry - statistical manifold analysis.

    Treats data as samples from a distribution and analyzes
    the geometry of the parameter space.
    """

    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "Fisher Information"

    @property
    def dimension(self) -> str:
        return "statistical manifold"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as histogram (empirical distribution parameters)."""
        data = self.validate_data(data)
        data = data.astype(float)
        data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        hist, _ = np.histogram(data, bins=self.n_bins, range=(0, 1))
        return hist.astype(float) / (len(data) + 1e-10)

    def fisher_information(self, p: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information matrix for multinomial.
        For multinomial, F_ij = δ_ij/p_i (diagonal)
        """
        p = np.clip(p, 1e-10, 1)
        return np.diag(1.0 / p)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Fisher geometry metrics."""
        p = self.embed(data)

        # Fisher information matrix
        F = self.fisher_information(p)

        # Scalar curvature proxy: trace and determinant
        trace_F = np.trace(F)
        det_F = np.linalg.det(F)

        # Effective dimension (number of significant parameters)
        eigenvalues = np.linalg.eigvalsh(F)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            # Participation ratio
            eff_dim = (np.sum(eigenvalues) ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)
        else:
            eff_dim = 0

        # KL divergence from uniform (information content)
        uniform = np.ones(self.n_bins) / self.n_bins
        kl_div = np.sum(p * np.log((p + 1e-10) / uniform))

        # Jeffreys prior volume (sqrt(det(F)))
        jeffreys_volume = np.sqrt(abs(det_F)) if det_F > 0 else 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "trace_fisher": trace_F,
                "det_fisher": det_F,
                "effective_dimension": eff_dim,
                "kl_from_uniform": kl_div,
                "jeffreys_volume": jeffreys_volume,
            },
            raw_data={
                "distribution": p,
                "fisher_matrix": F,
            }
        )


# =============================================================================
# SYMPLECTIC GEOMETRY
# =============================================================================

class SymplecticGeometry(ExoticGeometry):
    """
    Symplectic Geometry - phase space analysis.

    Pairs data as (position, momentum) and checks symplectic structure.
    Good for detecting Hamiltonian/conservative dynamics.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Symplectic"

    @property
    def dimension(self) -> int:
        return 2  # 2D phase space

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as phase space points (q, p) = (x, dx/dt)."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        # q = value, p = discrete derivative
        points = []
        for i in range(len(data) - 1):
            q = data[i]
            p = data[i + 1] - data[i]  # Momentum ~ velocity
            points.append([q, p])

        return np.array(points)

    def symplectic_area(self, points: np.ndarray) -> float:
        """Compute symplectic area (integral of dp ∧ dq)."""
        if len(points) < 3:
            return 0

        # Shoelace formula for signed area
        area = 0
        for i in range(len(points) - 1):
            area += points[i, 0] * points[i + 1, 1] - points[i + 1, 0] * points[i, 1]

        return abs(area) / 2

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute symplectic metrics."""
        points = self.embed(data)

        if len(points) < 3:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(points)},
                raw_data={"points": points}
            )

        # Phase space area
        total_area = self.symplectic_area(points)

        # Liouville measure preservation (area should be constant for Hamiltonian)
        window_size = min(50, len(points) // 4)
        areas = []
        for i in range(0, len(points) - window_size, window_size):
            window = points[i:i + window_size]
            areas.append(self.symplectic_area(window))

        area_variation = np.std(areas) / (np.mean(areas) + 1e-10) if areas else 0

        # Poincaré recurrence: does the trajectory return near starting point?
        start = points[0]
        distances_from_start = np.linalg.norm(points - start, axis=1)
        min_return_dist = np.min(distances_from_start[len(points)//4:]) if len(points) > 4 else 1

        # Phase space spread
        q_spread = np.std(points[:, 0])
        p_spread = np.std(points[:, 1])

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "total_area": total_area,
                "area_variation": area_variation,
                "min_return_distance": min_return_dist,
                "q_spread": q_spread,
                "p_spread": p_spread,
            },
            raw_data={
                "points": points,
                "window_areas": areas if areas else [],
            }
        )


# =============================================================================
# THURSTON GEOMETRIES
# =============================================================================
# The 8 Thurston geometries classify 3-manifolds:
# 1. E³ (Euclidean) - implicit in other tools
# 2. S³ (Spherical) - see SphericalGeometry
# 3. H³ (Hyperbolic) - see HyperbolicGeometry
# 4. S² × ℝ - product geometry
# 5. H² × ℝ - product geometry
# 6. Nil (Heisenberg) - see HeisenbergGeometry
# 7. Sol - solvable geometry
# 8. SL̃(2,ℝ) - universal cover

class SolGeometry(ExoticGeometry):
    """
    Sol Geometry - one of Thurston's 8 geometries.

    Sol has exponential stretch in one direction and shrink in another:
    The metric is ds² = e^(2z)dx² + e^(-2z)dy² + dz²

    This creates hyperbolic-like behavior in 3D, good for detecting
    exponential growth/decay patterns.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Sol (Thurston)"

    @property
    def dimension(self) -> int:
        return 3

    def sol_multiply(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """
        Sol group multiplication.
        (x₁,y₁,z₁) · (x₂,y₂,z₂) = (x₁ + e^z₁·x₂, y₁ + e^(-z₁)·y₂, z₁ + z₂)
        """
        return np.array([
            g1[0] + np.exp(g1[2]) * g2[0],
            g1[1] + np.exp(-g1[2]) * g2[1],
            g1[2] + g2[2]
        ])

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data as path in Sol geometry."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale) - 0.5

        # Build path: each triple (x,y,z) becomes a group element
        n_steps = len(data) // 3
        path = [np.array([0.0, 0.0, 0.0])]

        for i in range(n_steps):
            # Scale z smaller to avoid numerical overflow
            step = np.array([data[3*i], data[3*i + 1], data[3*i + 2] * 0.1])
            new_point = self.sol_multiply(path[-1], step)
            # Clamp to avoid overflow
            new_point = np.clip(new_point, -100, 100)
            path.append(new_point)

        return np.array(path)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Sol geometry metrics."""
        path = self.embed(data)

        # Final position
        final = path[-1]

        # Exponential stretch ratio
        x_range = np.max(path[:, 0]) - np.min(path[:, 0])
        y_range = np.max(path[:, 1]) - np.min(path[:, 1])
        stretch_ratio = (x_range + 1e-10) / (y_range + 1e-10)

        # Z drift (exponential control parameter)
        z_drift = final[2]
        z_variance = np.var(path[:, 2])

        # Path length
        diffs = np.diff(path, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))

        # Anisotropy: ratio of spreads in x vs y
        anisotropy = np.std(path[:, 0]) / (np.std(path[:, 1]) + 1e-10)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "stretch_ratio": stretch_ratio,
                "z_drift": z_drift,
                "z_variance": z_variance,
                "path_length": path_length,
                "anisotropy": anisotropy,
            },
            raw_data={
                "path": path,
                "final_position": final,
            }
        )


class ProductS2RGeometry(ExoticGeometry):
    """
    S² × ℝ Geometry - product of sphere and real line.

    Points live on a sphere with an additional "height" coordinate.
    Good for detecting cyclical patterns with drift.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "S² × ℝ (Thurston)"

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data on S² × ℝ."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 3
        points = []

        for i in range(n_points):
            # First two values → point on S²
            theta = data[3*i] * np.pi
            phi = data[3*i + 1] * 2 * np.pi
            # Third value → height in ℝ
            h = data[3*i + 2] - 0.5  # Center around 0

            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z, h])

        return np.array(points)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute S² × ℝ metrics."""
        points = self.embed(data)

        if len(points) < 2:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(points)},
                raw_data={"points": points}
            )

        # Spherical spread (on S²)
        sphere_points = points[:, :3]
        mean_vec = np.mean(sphere_points, axis=0)
        sphere_concentration = np.linalg.norm(mean_vec)

        # Height statistics (ℝ component)
        heights = points[:, 3]
        height_drift = heights[-1] - heights[0]
        height_variance = np.var(heights)

        # Correlation between sphere position and height
        z_coords = sphere_points[:, 2]  # Use z-coord of sphere
        if np.std(z_coords) > 1e-10 and np.std(heights) > 1e-10:
            sphere_height_corr = np.corrcoef(z_coords, heights)[0, 1]
        else:
            sphere_height_corr = 0

        # Winding number on sphere (approximate)
        angles = np.arctan2(sphere_points[:, 1], sphere_points[:, 0])
        unwrapped = np.unwrap(angles)
        winding = (unwrapped[-1] - unwrapped[0]) / (2 * np.pi)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "sphere_concentration": sphere_concentration,
                "height_drift": height_drift,
                "height_variance": height_variance,
                "sphere_height_corr": sphere_height_corr,
                "winding_number": winding,
            },
            raw_data={
                "points": points,
            }
        )


class ProductH2RGeometry(ExoticGeometry):
    """
    H² × ℝ Geometry - product of hyperbolic plane and real line.

    Points live in the Poincaré disk with an additional "height" coordinate.
    Good for detecting hierarchical patterns with drift.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "H² × ℝ (Thurston)"

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data on H² × ℝ (Poincaré disk × ℝ)."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 3
        points = []

        for i in range(n_points):
            # First two values → point in Poincaré disk
            x = (data[3*i] - 0.5) * 1.8
            y = (data[3*i + 1] - 0.5) * 1.8
            # Ensure inside disk
            r = np.sqrt(x**2 + y**2)
            if r >= 1:
                x, y = x / (r + 0.01) * 0.99, y / (r + 0.01) * 0.99
            # Third value → height in ℝ
            h = data[3*i + 2] - 0.5

            points.append([x, y, h])

        return np.array(points)

    def hyperbolic_distance_from_origin(self, x: float, y: float) -> float:
        """Distance from origin in Poincaré disk."""
        r = np.sqrt(x**2 + y**2)
        r = min(r, 0.9999)
        return 2 * np.arctanh(r)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute H² × ℝ metrics."""
        points = self.embed(data)

        if len(points) < 2:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(points)},
                raw_data={"points": points}
            )

        # Hyperbolic distances from origin
        hyp_dists = [self.hyperbolic_distance_from_origin(p[0], p[1]) for p in points]
        mean_hyp_dist = np.mean(hyp_dists)
        hyp_dist_variance = np.var(hyp_dists)

        # Boundary proximity (how close to edge of disk)
        radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        boundary_proximity = np.mean(radii > 0.8)

        # Height statistics
        heights = points[:, 2]
        height_drift = heights[-1] - heights[0]
        height_variance = np.var(heights)

        # Correlation between hyperbolic depth and height
        if np.std(hyp_dists) > 1e-10 and np.std(heights) > 1e-10:
            depth_height_corr = np.corrcoef(hyp_dists, heights)[0, 1]
        else:
            depth_height_corr = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "mean_hyperbolic_distance": mean_hyp_dist,
                "hyperbolic_variance": hyp_dist_variance,
                "boundary_proximity": boundary_proximity,
                "height_drift": height_drift,
                "height_variance": height_variance,
                "depth_height_corr": depth_height_corr,
            },
            raw_data={
                "points": points,
                "hyperbolic_distances": hyp_dists,
            }
        )


class SL2RGeometry(ExoticGeometry):
    """
    SL(2,ℝ) Geometry - 2x2 matrices with determinant 1.

    This is the universal cover of PSL(2,ℝ), related to hyperbolic isometries.
    Good for detecting projective/Möbius transformation patterns.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "SL(2,ℝ) (Thurston)"

    @property
    def dimension(self) -> int:
        return 3  # SL(2,R) is 3-dimensional

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed data as elements of SL(2,ℝ).
        Parametrize as: [[a, b], [c, d]] with ad - bc = 1
        Use (a, b, c) with d = (1 + bc) / a
        """
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_matrices = len(data) // 3
        matrices = []

        for i in range(n_matrices):
            # Parametrize SL(2,R) element
            a = data[3*i] + 0.5  # Avoid a=0
            b = data[3*i + 1] - 0.5
            c = data[3*i + 2] - 0.5

            if abs(a) < 0.01:
                a = 0.01

            d = (1 + b * c) / a  # Ensures det = 1

            matrices.append(np.array([[a, b], [c, d]]))

        return matrices

    def sl2_distance(self, M1: np.ndarray, M2: np.ndarray) -> float:
        """
        Distance in SL(2,ℝ) based on Frobenius norm of difference.
        (Approximate - true metric involves geodesics)
        """
        return np.linalg.norm(M1 - M2, 'fro')

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute SL(2,ℝ) metrics."""
        matrices = self.embed(data)

        if len(matrices) < 2:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_matrices": len(matrices)},
                raw_data={"matrices": matrices}
            )

        # Trace statistics (trace² - 2 classifies: <2 elliptic, =2 parabolic, >2 hyperbolic)
        traces = [np.trace(M) for M in matrices]
        trace_sq_minus_2 = [t**2 - 2 for t in traces]

        elliptic_frac = np.mean([x < 0 for x in trace_sq_minus_2])
        parabolic_frac = np.mean([abs(x) < 0.1 for x in trace_sq_minus_2])
        hyperbolic_frac = np.mean([x > 0.1 for x in trace_sq_minus_2])

        # Product of consecutive matrices (composition of transformations)
        products = [np.eye(2)]
        for M in matrices:
            products.append(products[-1] @ M)

        # Final transformation
        final = products[-1]
        final_trace = np.trace(final)

        # Spectral radius (largest eigenvalue magnitude)
        eigenvalues = [np.max(np.abs(np.linalg.eigvals(M))) for M in matrices]
        mean_spectral_radius = np.mean(eigenvalues)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "elliptic_fraction": elliptic_frac,
                "parabolic_fraction": parabolic_frac,
                "hyperbolic_fraction": hyperbolic_frac,
                "mean_trace": np.mean(traces),
                "final_trace": final_trace,
                "mean_spectral_radius": mean_spectral_radius,
            },
            raw_data={
                "matrices": matrices,
                "traces": traces,
            }
        )


# =============================================================================
# CLIFFORD TORUS (ELLIPTIC CURVE GEOMETRY)
# =============================================================================

class CliffordTorusGeometry(ExoticGeometry):
    """
    Clifford Torus Geometry - reveals elliptic curve structure.

    The Clifford torus is S¹ × S¹ embedded in S³.
    Elliptic curves ARE tori topologically, and their group law
    becomes LINEAR TRANSLATION in this embedding!

    Key insight: With the right embedding, EC addition = angle addition.
    This "straightens out" the algebraic complexity.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Clifford Torus"

    @property
    def dimension(self) -> int:
        return 4  # Lives in S³ ⊂ ℝ⁴

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed data pairs as points on Clifford torus.

        Clifford torus: (cos θ, sin θ, cos φ, sin φ) / √2

        For data pairs (x, y), use:
        θ = 2πx / scale (first circle)
        φ = 2πy / scale (second circle)
        """
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        points = []

        for i in range(n_points):
            theta = 2 * np.pi * data[2*i]
            phi = 2 * np.pi * data[2*i + 1]

            # Clifford torus embedding
            x1 = np.cos(theta) / np.sqrt(2)
            x2 = np.sin(theta) / np.sqrt(2)
            x3 = np.cos(phi) / np.sqrt(2)
            x4 = np.sin(phi) / np.sqrt(2)

            points.append([x1, x2, x3, x4, theta, phi])

        return np.array(points)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Clifford torus metrics: linearity, winding, torus coverage."""
        embedded = self.embed(data)

        if len(embedded) < 3:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(embedded)},
                raw_data={"embedded": embedded}
            )

        thetas = embedded[:, 4]
        phis = embedded[:, 5]

        # Angle progression linearity (key for EC detection!)
        indices = np.arange(len(thetas))
        theta_coeffs = np.polyfit(indices, np.unwrap(thetas), 1)
        theta_predicted = np.polyval(theta_coeffs, indices)
        theta_residuals = np.unwrap(thetas) - theta_predicted
        theta_r2 = 1 - np.var(theta_residuals) / (np.var(np.unwrap(thetas)) + 1e-10)

        phi_coeffs = np.polyfit(indices, np.unwrap(phis), 1)
        phi_predicted = np.polyval(phi_coeffs, indices)
        phi_residuals = np.unwrap(phis) - phi_predicted
        phi_r2 = 1 - np.var(phi_residuals) / (np.var(np.unwrap(phis)) + 1e-10)

        # Winding numbers
        theta_winding = (np.unwrap(thetas)[-1] - np.unwrap(thetas)[0]) / (2 * np.pi)
        phi_winding = (np.unwrap(phis)[-1] - np.unwrap(phis)[0]) / (2 * np.pi)

        # Torus coverage (how much of the torus is visited)
        theta_bins = np.histogram(thetas % (2*np.pi), bins=16, range=(0, 2*np.pi))[0]
        phi_bins = np.histogram(phis % (2*np.pi), bins=16, range=(0, 2*np.pi))[0]
        theta_coverage = np.sum(theta_bins > 0) / 16
        phi_coverage = np.sum(phi_bins > 0) / 16

        # Path regularity on torus
        diffs = np.diff(embedded[:, :4], axis=0)
        step_sizes = np.linalg.norm(diffs, axis=1)
        regularity = np.mean(step_sizes) / (np.std(step_sizes) + 1e-10)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "theta_linearity_r2": theta_r2,
                "phi_linearity_r2": phi_r2,
                "theta_winding": theta_winding,
                "phi_winding": phi_winding,
                "theta_coverage": theta_coverage,
                "phi_coverage": phi_coverage,
                "path_regularity": regularity,
                "theta_slope": theta_coeffs[0],
                "phi_slope": phi_coeffs[0],
            },
            raw_data={
                "embedded": embedded,
                "thetas": thetas,
                "phis": phis,
            }
        )


# =============================================================================
# PENROSE GEOMETRY (QUASICRYSTALS)
# =============================================================================

class PenroseGeometry(ExoticGeometry):
    """
    Penrose Geometry - detects quasicrystalline structure.

    Penrose tilings are aperiodic but have long-range order:
    - 5-fold rotational symmetry (forbidden in periodic crystals)
    - Self-similar at all scales
    - Connected to golden ratio φ = (1+√5)/2

    Good for detecting structure that's between periodic and random.
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Penrose (Quasicrystal)"

    @property
    def dimension(self) -> str:
        return "2D with 5-fold"

    def penrose_projection(self, x: float, y: float) -> Tuple[float, ...]:
        """Project 2D point onto 5 Penrose directions."""
        angles = [2 * np.pi * k / 5 for k in range(5)]
        return tuple(x * np.cos(a) + y * np.sin(a) for a in angles)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data pairs in 5-direction Penrose space."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        embedded = []

        for i in range(n_points):
            x = data[2*i] * 10 - 5  # Scale to [-5, 5]
            y = data[2*i + 1] * 10 - 5
            proj = self.penrose_projection(x, y)
            embedded.append(list(proj) + [x, y])

        return np.array(embedded)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Penrose/quasicrystal metrics."""
        embedded = self.embed(data)

        if len(embedded) < 10:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(embedded)},
                raw_data={"embedded": embedded}
            )

        # 5-fold symmetry: check if projections onto 5 directions are balanced
        projections = embedded[:, :5]
        proj_stds = np.std(projections, axis=0)
        fivefold_balance = 1 - np.std(proj_stds) / (np.mean(proj_stds) + 1e-10)

        # Golden ratio test: check spacing ratios
        x_coords = embedded[:, 5]
        diffs = np.abs(np.diff(np.sort(x_coords)))
        diffs = diffs[diffs > 0.1]  # Filter tiny differences

        if len(diffs) > 1:
            ratios = diffs[1:] / (diffs[:-1] + 1e-10)
            # Count ratios close to φ or 1/φ
            phi_matches = np.sum((np.abs(ratios - self.PHI) < 0.2) |
                                (np.abs(ratios - 1/self.PHI) < 0.2))
            golden_ratio_score = phi_matches / len(ratios)
        else:
            golden_ratio_score = 0

        # Penrose index diversity (quantize to lattice)
        indices = set()
        for pt in embedded[:, :5]:
            idx = tuple(int(np.round(p)) for p in pt)
            indices.add(idx)
        index_diversity = len(indices) / len(embedded)

        # Quasiperiodicity: autocorrelation structure
        x_centered = x_coords - np.mean(x_coords)
        if len(x_centered) > 50:
            autocorr = np.correlate(x_centered, x_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Long-range order: correlation at distance
            n = len(autocorr)
            long_range = np.mean(np.abs(autocorr[n//4:n//2])) if n > 4 else 0
        else:
            long_range = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "fivefold_balance": fivefold_balance,
                "golden_ratio_score": golden_ratio_score,
                "index_diversity": index_diversity,
                "long_range_order": long_range,
            },
            raw_data={
                "embedded": embedded,
                "projections": projections,
            }
        )


# =============================================================================
# AMMANN-BEENKER GEOMETRY (8-fold aperiodic tiling)
# =============================================================================

class AmmannBeenkerGeometry(ExoticGeometry):
    """
    Ammann-Beenker Geometry - detects octagonal quasicrystalline structure.

    Ammann-Beenker tilings have 8-fold rotational symmetry and are connected
    to the silver ratio δ = 1 + √2. They tile the plane with squares and
    45-degree rhombi. Complementary to Penrose's 5-fold symmetry.
    """

    SILVER = 1 + np.sqrt(2)  # Silver ratio

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Ammann-Beenker (Octagonal)"

    @property
    def dimension(self) -> str:
        return "2D with 8-fold"

    def octagonal_projection(self, x: float, y: float) -> Tuple[float, ...]:
        """Project 2D point onto 8 directions (octagonal symmetry)."""
        angles = [2 * np.pi * k / 8 for k in range(8)]
        return tuple(x * np.cos(a) + y * np.sin(a) for a in angles)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data pairs in 8-direction octagonal space."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        embedded = []

        for i in range(n_points):
            x = data[2*i] * 10 - 5
            y = data[2*i + 1] * 10 - 5
            proj = self.octagonal_projection(x, y)
            embedded.append(list(proj) + [x, y])

        return np.array(embedded)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute octagonal quasicrystal metrics."""
        embedded = self.embed(data)

        if len(embedded) < 10:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(embedded)},
                raw_data={"embedded": embedded}
            )

        # 8-fold symmetry balance
        projections = embedded[:, :8]
        proj_stds = np.std(projections, axis=0)
        eightfold_balance = 1 - np.std(proj_stds) / (np.mean(proj_stds) + 1e-10)

        # Silver ratio test (δ = 1+√2 ≈ 2.414)
        x_coords = embedded[:, 8]
        diffs = np.abs(np.diff(np.sort(x_coords)))
        diffs = diffs[diffs > 0.1]

        if len(diffs) > 1:
            ratios = diffs[1:] / (diffs[:-1] + 1e-10)
            silver_matches = np.sum((np.abs(ratios - self.SILVER) < 0.3) |
                                   (np.abs(ratios - 1/self.SILVER) < 0.15))
            silver_ratio_score = silver_matches / len(ratios)
        else:
            silver_ratio_score = 0

        # Octagonal index diversity
        indices = set()
        for pt in embedded[:, :8]:
            idx = tuple(int(np.round(p)) for p in pt)
            indices.add(idx)
        index_diversity = len(indices) / len(embedded)

        # 4-fold vs 8-fold: compare variance across 90-degree vs 45-degree axes
        # Directions 0,2,4,6 are the square axes; 1,3,5,7 are the diagonal axes
        square_stds = proj_stds[[0, 2, 4, 6]]
        diag_stds = proj_stds[[1, 3, 5, 7]]
        square_diag_ratio = np.mean(square_stds) / (np.mean(diag_stds) + 1e-10)

        # Long-range order (same method as Penrose)
        x_centered = x_coords - np.mean(x_coords)
        if len(x_centered) > 50:
            autocorr = np.correlate(x_centered, x_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            n = len(autocorr)
            long_range = np.mean(np.abs(autocorr[n//4:n//2])) if n > 4 else 0
        else:
            long_range = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "eightfold_balance": eightfold_balance,
                "silver_ratio_score": silver_ratio_score,
                "index_diversity": index_diversity,
                "square_diag_ratio": square_diag_ratio,
                "long_range_order": long_range,
            },
            raw_data={
                "embedded": embedded,
                "projections": projections,
            }
        )


# =============================================================================
# EINSTEIN (HAT) GEOMETRY (chiral aperiodic monotile)
# =============================================================================

class EinsteinHatGeometry(ExoticGeometry):
    """
    Einstein Hat Geometry - detects chiral aperiodic structure.

    Based on the 2023 discovery of the "hat" tile - a single shape that
    tiles the plane aperiodically. Unlike Penrose (5-fold) and
    Ammann-Beenker (8-fold), the hat has NO rotational symmetry requirement.

    Uses a hexagonal grid basis with 13-vertex structure. Projects data onto
    the 6 hexagonal directions plus a chirality-breaking asymmetric direction.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Einstein (Hat Monotile)"

    @property
    def dimension(self) -> str:
        return "2D with hex+chiral"

    def hat_projection(self, x: float, y: float) -> Tuple[float, ...]:
        """Project onto hexagonal grid directions plus chirality axis.

        6 hexagonal directions (60-degree spacing) plus an asymmetric
        "chirality" direction at 15 degrees (breaks pure 6-fold symmetry,
        capturing the hat tile's fundamental asymmetry).
        """
        hex_angles = [2 * np.pi * k / 6 for k in range(6)]
        # Chirality-breaking direction (not aligned with any hex axis)
        chiral_angle = np.pi / 12  # 15 degrees

        projections = [x * np.cos(a) + y * np.sin(a) for a in hex_angles]
        projections.append(x * np.cos(chiral_angle) + y * np.sin(chiral_angle))
        return tuple(projections)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data pairs in hex+chiral projection space."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        embedded = []

        for i in range(n_points):
            x = data[2*i] * 10 - 5
            y = data[2*i + 1] * 10 - 5
            proj = self.hat_projection(x, y)
            embedded.append(list(proj) + [x, y])

        return np.array(embedded)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute hat/einstein tile metrics."""
        embedded = self.embed(data)

        if len(embedded) < 10:
            return GeometryResult(
                geometry_name=self.name,
                metrics={"n_points": len(embedded)},
                raw_data={"embedded": embedded}
            )

        # Hexagonal balance (6-fold, using first 6 projections)
        hex_projections = embedded[:, :6]
        hex_stds = np.std(hex_projections, axis=0)
        hexagonal_balance = 1 - np.std(hex_stds) / (np.mean(hex_stds) + 1e-10)

        # Chirality: sequential signed area of consecutive point triples.
        # For (x_i,y_i), (x_{i+1},y_{i+1}), (x_{i+2},y_{i+2}), the signed
        # area (cross product) has a preferred sign for chiral data.
        # Random data → mean ≈ 0; chiral patterns → nonzero.
        x_coords = embedded[:, 7]
        y_coords = embedded[:, 8]
        if len(x_coords) > 2:
            dx1 = x_coords[1:-1] - x_coords[:-2]
            dy1 = y_coords[1:-1] - y_coords[:-2]
            dx2 = x_coords[2:] - x_coords[1:-1]
            dy2 = y_coords[2:] - y_coords[1:-1]
            signed_areas = dx1 * dy2 - dy1 * dx2
            mean_abs = np.mean(np.abs(signed_areas))
            chirality = np.mean(signed_areas) / (mean_abs + 1e-10) if mean_abs > 1e-10 else 0.0
        else:
            chirality = 0.0

        # Hex index diversity
        indices = set()
        for pt in embedded[:, :6]:
            idx = tuple(int(np.round(p)) for p in pt)
            indices.add(idx)
        index_diversity = len(indices) / len(embedded)

        # Hex skewness asymmetry: compare 3rd moments across hex directions.
        # Skewness(proj_k) depends on data's 3rd-order joint distribution,
        # which CAN differ across directions even when variance is identical.
        # For isotropic data, all skewnesses are equal; asymmetric data breaks this.
        from scipy.stats import skew as scipy_skew
        hex_skews = np.array([scipy_skew(hex_projections[:, k]) for k in range(6)])
        hex_asymmetry = np.std(hex_skews) / (np.mean(np.abs(hex_skews)) + 1e-10)

        # Long-range order
        x_coords = embedded[:, 7]
        x_centered = x_coords - np.mean(x_coords)
        if len(x_centered) > 50:
            autocorr = np.correlate(x_centered, x_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            n = len(autocorr)
            long_range = np.mean(np.abs(autocorr[n//4:n//2])) if n > 4 else 0
        else:
            long_range = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "hexagonal_balance": hexagonal_balance,
                "chirality": chirality,
                "index_diversity": index_diversity,
                "hex_asymmetry": hex_asymmetry,
                "long_range_order": long_range,
            },
            raw_data={
                "embedded": embedded,
            }
        )


# =============================================================================
# HIGHER-ORDER GEOMETRY (3rd/4th order statistics)
# =============================================================================

class HigherOrderGeometry(ExoticGeometry):
    """
    Higher-order statistics geometry - genuinely independent from all
    covariance-based geometries.

    Computes 3rd and 4th order metrics:
    - Projection kurtosis (4th order): max|r|=0.22 with existing metrics
    - Bicoherence (normalized bispectrum): zero for Gaussian by definition
    - Third-order autocorrelation C3: temporal nonlinearity
    - Projection skewness (3rd order): directional asymmetry
    - Permutation entropy: ordinal pattern analysis (33x better for recurrence)
    """

    def __init__(self, n_directions: int = 20, perm_order: int = 5,
                 c3_max_tau: int = 10, bispec_max_freq: int = 64):
        self.n_directions = n_directions
        self.perm_order = perm_order
        self.c3_max_tau = c3_max_tau
        self.bispec_max_freq = bispec_max_freq

    @property
    def name(self) -> str:
        return "Higher-Order Statistics"

    @property
    def dimension(self) -> str:
        return "mixed (3rd/4th order)"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """No single embedding - metrics are computed directly."""
        return np.asarray(data, dtype=float)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute all higher-order metrics."""
        from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

        data = self.validate_data(data)
        fdata = data.astype(float)
        metrics = {}

        # --- Projection kurtosis & skewness (3rd/4th order directional) ---
        n_points = len(fdata) // 2
        if n_points >= 10:
            points = fdata[:2*n_points].reshape(-1, 2)
            mu = points.mean(axis=0)
            std = points.std(axis=0)
            std[std < 1e-10] = 1.0
            points = (points - mu) / std

            angles = np.linspace(0, np.pi, self.n_directions, endpoint=False)
            skewnesses = []
            kurtoses = []
            for angle in angles:
                direction = np.array([np.cos(angle), np.sin(angle)])
                proj = points @ direction
                if np.std(proj) > 1e-10:
                    skewnesses.append(scipy_skew(proj))
                    kurtoses.append(scipy_kurtosis(proj, fisher=True))
                else:
                    skewnesses.append(0.0)
                    kurtoses.append(0.0)

            skewnesses = np.array(skewnesses)
            kurtoses = np.array(kurtoses)

            metrics['skew_max'] = float(np.max(np.abs(skewnesses)))
            metrics['skew_mean'] = float(np.mean(np.abs(skewnesses)))
            metrics['kurt_max'] = float(np.max(np.abs(kurtoses)))
            metrics['kurt_mean'] = float(np.mean(kurtoses))
        else:
            metrics.update({'skew_max': 0, 'skew_mean': 0,
                          'kurt_max': 0, 'kurt_mean': 0})

        # --- Permutation entropy (ordinal patterns) ---
        n = len(fdata)
        order = self.perm_order
        if n >= order + 1:
            n_patterns = n - order + 1
            pattern_counts = Counter()
            for i in range(n_patterns):
                window = fdata[i:i+order]
                pattern = tuple(int(x) for x in np.argsort(np.argsort(window)))
                pattern_counts[pattern] += 1

            total_possible = math.factorial(order)
            n_observed = len(pattern_counts)
            probs = np.array(list(pattern_counts.values())) / n_patterns
            entropy = -np.sum(probs * np.log2(probs + 1e-15))
            max_entropy = np.log2(total_possible)

            metrics['perm_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0
            metrics['perm_forbidden'] = float((total_possible - n_observed) / total_possible)
        else:
            metrics.update({'perm_entropy': 0, 'perm_forbidden': 0})

        # --- Third-order autocorrelation C3 ---
        centered = fdata - fdata.mean()
        std_val = np.std(centered)
        if std_val > 1e-10:
            centered = centered / std_val
        c3_values = []
        max_tau = min(self.c3_max_tau, len(centered) // 3)
        for tau1 in range(1, max_tau + 1):
            for tau2 in range(tau1, max_tau + 1):
                valid_n = len(centered) - tau2
                if valid_n >= 10:
                    c3 = np.mean(centered[:valid_n] *
                                centered[tau1:valid_n+tau1] *
                                centered[tau2:valid_n+tau2])
                    c3_values.append(c3)

        if c3_values:
            c3_arr = np.array(c3_values)
            metrics['c3_energy'] = float(np.sum(c3_arr**2))
            metrics['c3_mean'] = float(np.mean(np.abs(c3_arr)))
        else:
            metrics.update({'c3_energy': 0, 'c3_mean': 0})

        # --- Bicoherence (segment-averaged normalized bispectrum) ---
        # Single-FFT bicoherence is trivially 1. Must average over segments.
        n = len(fdata)
        seg_len = 256
        n_segs = n // seg_len
        max_freq = min(seg_len // 2, self.bispec_max_freq)

        if n_segs >= 4 and max_freq > 2:
            # Accumulate bispectrum and power across segments
            bispec_sum = {}
            power_sums = np.zeros(seg_len // 2)
            for s in range(n_segs):
                seg = centered[s*seg_len:(s+1)*seg_len]
                X = np.fft.fft(seg)
                power_sums += np.abs(X[:seg_len//2])**2
                for f1 in range(1, max_freq):
                    for f2 in range(f1, max_freq):
                        f3 = f1 + f2
                        if f3 < seg_len // 2:
                            B = X[f1] * X[f2] * np.conj(X[f3])
                            key = (f1, f2)
                            if key not in bispec_sum:
                                bispec_sum[key] = 0.0
                            bispec_sum[key] += B

            power_avg = power_sums / n_segs
            bicoherence_vals = []
            for (f1, f2), B_sum in bispec_sum.items():
                f3 = f1 + f2
                B_avg = B_sum / n_segs
                denom = np.sqrt(power_avg[f1] * power_avg[f2] *
                               power_avg[f3] + 1e-10)
                bicoherence_vals.append(np.abs(B_avg) / denom)

            bic = np.array(bicoherence_vals)
            metrics['bicoherence_mean'] = float(np.mean(bic))
            metrics['bicoherence_max'] = float(np.max(bic))
        else:
            metrics.update({'bicoherence_mean': 0, 'bicoherence_max': 0})

        return GeometryResult(
            geometry_name=self.name,
            metrics=metrics,
            raw_data={}
        )


class SpatialFieldGeometry(ExoticGeometry):
    """
    Native 2D field analysis via differential geometry on scalar fields.

    Unlike all other geometries which flatten input to 1D, this preserves
    2D spatial structure and computes gradient, curvature, Hessian, and
    basin-of-attraction metrics directly on the field.

    Detects: edge density, directional structure, saddle/extrema balance,
    attractor topology, multi-scale coherence.

    Parameters
    ----------
    max_basin_iter : int
        Maximum iterations for gradient-descent basin mapping (default 200).
    max_field_size : int
        Fields larger than this (per side) are downsampled (default 256).
    """

    def __init__(self, max_basin_iter: int = 200, max_field_size: int = 256):
        self._max_basin_iter = max_basin_iter
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "SpatialField"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        """Preserve 2D shape. Reshape 1D input to nearest square, pad with mean."""
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Return the 2D field itself (identity embedding)."""
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        """Block-average downsample if field exceeds max_field_size."""
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        factor_h = int(np.ceil(H / self._max_field_size))
        factor_w = int(np.ceil(W / self._max_field_size))
        # Trim to multiple of factor
        H_trim = (H // factor_h) * factor_h
        W_trim = (W // factor_w) * factor_w
        trimmed = field[:H_trim, :W_trim]
        return trimmed.reshape(H_trim // factor_h, factor_h,
                               W_trim // factor_w, factor_w).mean(axis=(1, 3))

    def _compute_differentials(self, field: np.ndarray):
        """
        Compute all differential quantities in one pass using finite differences.

        Returns: gx, gy, grad_mag_sq, laplacian, Hxx, Hyy, Hxy
        """
        # Pad with edge values for boundary handling
        padded = np.pad(field, 1, mode='edge')
        H, W = field.shape

        # First derivatives (central differences)
        gx = (padded[1:H+1, 2:W+2] - padded[1:H+1, 0:W]) / 2.0
        gy = (padded[2:H+2, 1:W+1] - padded[0:H, 1:W+1]) / 2.0
        grad_mag_sq = gx**2 + gy**2

        # Laplacian (5-point stencil)
        laplacian = (padded[1:H+1, 2:W+2] + padded[1:H+1, 0:W] +
                     padded[2:H+2, 1:W+1] + padded[0:H, 1:W+1] -
                     4.0 * field)

        # Second derivatives for Hessian
        Hxx = padded[1:H+1, 2:W+2] - 2.0 * field + padded[1:H+1, 0:W]
        Hyy = padded[2:H+2, 1:W+1] - 2.0 * field + padded[0:H, 1:W+1]
        # Mixed partial (average of 4 cross-differences)
        Hxy = (padded[2:H+2, 2:W+2] - padded[2:H+2, 0:W] -
               padded[0:H, 2:W+2] + padded[0:H, 0:W]) / 4.0

        return gx, gy, grad_mag_sq, laplacian, Hxx, Hyy, Hxy

    def _find_basins(self, field: np.ndarray, gx: np.ndarray, gy: np.ndarray):
        """
        Map each cell to its basin of attraction via steepest descent.

        Uses 8-neighbor descent with path caching.

        Returns: basin_labels (H×W int array), n_basins
        """
        H, W = field.shape
        labels = -np.ones((H, W), dtype=np.int32)
        current_label = 0

        # 8 neighbors: dx, dy offsets
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        for start_r in range(H):
            for start_c in range(W):
                if labels[start_r, start_c] >= 0:
                    continue

                # Trace path to minimum
                path = []
                r, c = start_r, start_c
                visited = set()

                for _ in range(self._max_basin_iter):
                    if labels[r, c] >= 0:
                        # Hit a cell already assigned
                        target_label = labels[r, c]
                        for pr, pc in path:
                            labels[pr, pc] = target_label
                        break

                    if (r, c) in visited:
                        # Cycle or local minimum — assign new label
                        for pr, pc in path:
                            labels[pr, pc] = current_label
                        current_label += 1
                        break

                    visited.add((r, c))
                    path.append((r, c))

                    # Find steepest descent neighbor
                    best_val = field[r, c]
                    best_r, best_c = r, c
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if field[nr, nc] < best_val:
                                best_val = field[nr, nc]
                                best_r, best_c = nr, nc

                    if best_r == r and best_c == c:
                        # Local minimum — assign new label
                        for pr, pc in path:
                            labels[pr, pc] = current_label
                        current_label += 1
                        break

                    r, c = best_r, best_c
                else:
                    # max_basin_iter exceeded — assign new label
                    for pr, pc in path:
                        labels[pr, pc] = current_label
                    current_label += 1

        return labels, current_label

    def _block_average(self, field: np.ndarray, block_size: int) -> np.ndarray:
        """Downsample field by block averaging."""
        H, W = field.shape
        bh = (H // block_size) * block_size
        bw = (W // block_size) * block_size
        if bh < block_size or bw < block_size:
            return field  # Too small to downsample
        trimmed = field[:bh, :bw]
        return trimmed.reshape(bh // block_size, block_size,
                               bw // block_size, block_size).mean(axis=(1, 3))

    def _coherence(self, field: np.ndarray) -> float:
        """
        Compute coherence score for a field.

        Composite: depth × curvature × tension balance.
        High for organized fields (smooth basins), low for noise.
        """
        if field.shape[0] < 4 or field.shape[1] < 4:
            return float('nan')
        _, _, grad_mag_sq, laplacian, _, _, _ = self._compute_differentials(field)
        tension = float(np.mean(grad_mag_sq))
        curvature = float(np.mean(np.abs(laplacian)))
        depth = float(np.max(field) - np.min(field))
        # Normalized product — high when all three are substantial
        denom = (1.0 + tension) * (1.0 + curvature)
        return depth * curvature / denom

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute spatial field metrics on 2D data."""
        field = self.validate_data(data)
        field = self._downsample(field)
        H, W = field.shape

        metrics = {}

        # Minimum field size check
        if H < 4 or W < 4:
            nan_keys = [
                'tension_mean', 'tension_std', 'curvature_mean', 'curvature_std',
                'anisotropy_mean', 'criticality_saddle_frac', 'criticality_extrema_frac',
                'n_basins', 'basin_size_entropy', 'basin_depth_cv',
                'coherence_score', 'multiscale_coherence_1',
                'multiscale_coherence_2', 'multiscale_coherence_4',
                'multiscale_coherence_8',
            ]
            metrics = {k: float('nan') for k in nan_keys}
            return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})

        # Compute all differentials
        gx, gy, grad_mag_sq, laplacian, Hxx, Hyy, Hxy = self._compute_differentials(field)

        # Tension (gradient energy)
        metrics['tension_mean'] = float(np.mean(grad_mag_sq))
        metrics['tension_std'] = float(np.std(grad_mag_sq))

        # Curvature (Laplacian magnitude)
        abs_lap = np.abs(laplacian)
        metrics['curvature_mean'] = float(np.mean(abs_lap))
        metrics['curvature_std'] = float(np.std(abs_lap))

        # Anisotropy from Hessian eigenvalues
        # For 2×2 Hessian [[Hxx, Hxy],[Hxy, Hyy]]:
        # eigenvalues = (Hxx+Hyy)/2 ± sqrt(((Hxx-Hyy)/2)^2 + Hxy^2)
        trace = Hxx + Hyy
        disc = np.sqrt(((Hxx - Hyy) / 2.0)**2 + Hxy**2)
        lam1 = trace / 2.0 + disc
        lam2 = trace / 2.0 - disc
        aniso = np.abs(lam1 - lam2) / (np.abs(lam1) + np.abs(lam2) + 1e-10)
        metrics['anisotropy_mean'] = float(np.mean(aniso))

        # Criticality from Hessian determinant
        det_H = Hxx * Hyy - Hxy**2
        n_total = H * W
        metrics['criticality_saddle_frac'] = float(np.sum(det_H < 0) / n_total)
        metrics['criticality_extrema_frac'] = float(np.sum(det_H > 0) / n_total)

        # Basin analysis
        basin_labels, n_basins = self._find_basins(field, gx, gy)
        metrics['n_basins'] = int(n_basins)

        if n_basins > 1:
            # Basin size entropy (normalized)
            sizes = np.bincount(basin_labels.ravel())
            sizes = sizes[sizes > 0].astype(float)
            probs = sizes / sizes.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-15))
            max_entropy = np.log(n_basins)
            metrics['basin_size_entropy'] = float(entropy / (max_entropy + 1e-15))

            # Basin depth CV
            depths = []
            for b in range(n_basins):
                mask = basin_labels == b
                if np.any(mask):
                    vals = field[mask]
                    depths.append(float(np.max(vals) - np.min(vals)))
            depths = np.array(depths)
            mean_depth = np.mean(depths)
            metrics['basin_depth_cv'] = float(np.std(depths) / (mean_depth + 1e-10))
        else:
            metrics['basin_size_entropy'] = 0.0
            metrics['basin_depth_cv'] = 0.0

        # Coherence score
        metrics['coherence_score'] = self._coherence(field)

        # Multi-scale coherence at block sizes 1, 2, 4, 8
        for scale in [1, 2, 4, 8]:
            if scale == 1:
                metrics['multiscale_coherence_1'] = metrics['coherence_score']
            else:
                coarsened = self._block_average(field, scale)
                if coarsened.shape[0] >= 4 and coarsened.shape[1] >= 4:
                    metrics[f'multiscale_coherence_{scale}'] = self._coherence(coarsened)
                else:
                    metrics[f'multiscale_coherence_{scale}'] = float('nan')

        return GeometryResult(
            geometry_name=self.name,
            metrics=metrics,
            raw_data={}
        )


class SurfaceGeometry(ExoticGeometry):
    """Differential geometry of 2D fields as height maps z = f(x,y).

    Computes Gaussian/mean curvature, shape index, curvedness, and the
    Gauss-Bonnet integral — invariants from classical surface theory.

    Gaussian curvature K is intrinsic (preserved by bending); mean curvature H
    is extrinsic (depends on embedding). Shape index classifies local shape
    from cup (-1) through saddle (0) to cap (+1). The Gauss-Bonnet integral
    ∫∫ K dA is a topological invariant related to Euler characteristic.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    """

    def __init__(self, max_field_size: int = 128):
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "Surface"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        metrics: Dict[str, float] = {}

        # First derivatives
        fy, fx = np.gradient(field)
        # Second derivatives
        fyy, fxy = np.gradient(fy)
        _, fxx = np.gradient(fx)

        # Metric tensor quantities
        fx2 = fx ** 2
        fy2 = fy ** 2
        denom_sq = 1.0 + fx2 + fy2  # (1 + |∇f|²)

        # Gaussian curvature: K = (fxx·fyy - fxy²) / (1 + fx² + fy²)²
        K = (fxx * fyy - fxy ** 2) / (denom_sq ** 2 + 1e-12)
        metrics['gaussian_curvature_mean'] = float(np.mean(K))
        metrics['gaussian_curvature_std'] = float(np.std(K))

        # Mean curvature: H = ((1+fy²)fxx - 2·fx·fy·fxy + (1+fx²)fyy) / (2·(1+|∇f|²)^(3/2))
        H_num = (1 + fy2) * fxx - 2 * fx * fy * fxy + (1 + fx2) * fyy
        H = H_num / (2.0 * denom_sq ** 1.5 + 1e-12)
        metrics['mean_curvature_mean'] = float(np.mean(H))
        metrics['mean_curvature_std'] = float(np.std(H))

        # Principal curvatures: κ₁,₂ = H ± √(H² - K)
        discriminant = np.maximum(H ** 2 - K, 0.0)
        sqrt_disc = np.sqrt(discriminant)
        kappa1 = H + sqrt_disc
        kappa2 = H - sqrt_disc

        # Shape index: S = (2/π)·arctan((κ₁+κ₂)/(κ₁-κ₂))
        ksum = kappa1 + kappa2
        kdiff = kappa1 - kappa2
        with np.errstate(divide='ignore', invalid='ignore'):
            S = (2.0 / np.pi) * np.arctan2(ksum, kdiff + 1e-12)
        S = np.nan_to_num(S, nan=0.0)
        metrics['shape_index_mean'] = float(np.mean(S))
        metrics['shape_index_std'] = float(np.std(S))

        # Curvedness: C = √((κ₁² + κ₂²) / 2)
        C = np.sqrt((kappa1 ** 2 + kappa2 ** 2) / 2.0)
        metrics['curvedness_mean'] = float(np.mean(C))

        # Gauss-Bonnet integral: ∫∫ K √(1+|∇f|²) dx dy
        dA = np.sqrt(denom_sq)
        metrics['gauss_bonnet_integral'] = float(np.sum(K * dA))

        # Total metric area: ∫∫ √(1+|∇f|²) dx dy
        metrics['total_metric_area'] = float(np.sum(dA))

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class PersistentHomology2DGeometry(ExoticGeometry):
    """Sublevel/superlevel set persistence on 2D grids via union-find.

    Computes H₀ (connected component) persistence for both sublevel sets
    (low→high threshold) and superlevel sets (high→low). Metrics capture
    the birth-death structure of topological features.

    Unlike the 1D PersistentHomologyGeometry (which builds a VR complex on
    byte pairs), this works natively on the grid's 4-connectivity.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    """

    def __init__(self, max_field_size: int = 128):
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "PersistentHomology2D"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def _persistence_h0(self, values: np.ndarray, H: int, W: int) -> list:
        """Compute H₀ persistence via union-find on a flat array of pixel values.

        Parameters
        ----------
        values : 1D array of pixel values (flat view of 2D grid)
        H, W : grid dimensions

        Returns
        -------
        List of (birth, death) persistence pairs.
        """
        n = H * W
        parent = np.arange(n)
        rank = np.zeros(n, dtype=np.int32)
        birth = values.copy()  # birth value of each component
        alive = np.zeros(n, dtype=bool)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b, threshold):
            ra, rb = find(a), find(b)
            if ra == rb:
                return None
            # Elder rule: component born earlier (lower value) survives
            if birth[ra] > birth[rb]:
                ra, rb = rb, ra
            # rb dies (younger component)
            pair = (float(birth[rb]), float(threshold))
            if rank[ra] < rank[rb]:
                rank[ra], rank[rb] = rank[rb], rank[ra]
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return pair

        # Process pixels in order of increasing value (sublevel filtration)
        order = np.argsort(values)
        pairs = []
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

        for idx in order:
            alive[idx] = True
            r, c = divmod(int(idx), W)
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nidx = nr * W + nc
                    if alive[nidx]:
                        pair = union(idx, nidx, values[idx])
                        if pair is not None:
                            pairs.append(pair)

        return pairs

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        H, W = field.shape
        flat = field.ravel()
        metrics: Dict[str, float] = {}

        # Sublevel persistence (low → high)
        sub_pairs = self._persistence_h0(flat, H, W)
        sub_lifetimes = np.array([d - b for b, d in sub_pairs]) if sub_pairs else np.array([0.0])
        sub_lifetimes = sub_lifetimes[sub_lifetimes > 0]
        if len(sub_lifetimes) == 0:
            sub_lifetimes = np.array([0.0])

        metrics['sub_total_persistence'] = float(np.sum(sub_lifetimes))
        metrics['sub_max_persistence'] = float(np.max(sub_lifetimes))
        metrics['sub_n_components'] = float(len(sub_lifetimes))

        # Superlevel persistence (high → low): negate values
        super_pairs = self._persistence_h0(-flat, H, W)
        super_lifetimes = np.array([d - b for b, d in super_pairs]) if super_pairs else np.array([0.0])
        super_lifetimes = super_lifetimes[super_lifetimes > 0]
        if len(super_lifetimes) == 0:
            super_lifetimes = np.array([0.0])

        metrics['super_total_persistence'] = float(np.sum(super_lifetimes))
        metrics['super_max_persistence'] = float(np.max(super_lifetimes))
        metrics['super_n_components'] = float(len(super_lifetimes))

        # Persistence entropy (sublevel)
        total = np.sum(sub_lifetimes)
        if total > 0:
            probs = sub_lifetimes / total
            probs = probs[probs > 0]
            metrics['persistence_entropy'] = float(-np.sum(probs * np.log(probs)))
        else:
            metrics['persistence_entropy'] = 0.0

        # Asymmetry between sublevel and superlevel
        s_total = metrics['sub_total_persistence']
        u_total = metrics['super_total_persistence']
        if s_total + u_total > 0:
            metrics['persistence_asymmetry'] = float(abs(s_total - u_total) / (s_total + u_total))
        else:
            metrics['persistence_asymmetry'] = 0.0

        # Number of persistent features (lifetime > median)
        all_lifetimes = np.concatenate([sub_lifetimes, super_lifetimes])
        med = np.median(all_lifetimes)
        metrics['n_persistent_features'] = float(np.sum(all_lifetimes > med))

        # Range ratio
        total_all = np.sum(all_lifetimes)
        if total_all > 0:
            metrics['persistence_range_ratio'] = float(np.max(all_lifetimes) / total_all)
        else:
            metrics['persistence_range_ratio'] = 0.0

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class ConformalGeometry2D(ExoticGeometry):
    """Conformal analysis of 2D fields.

    Measures departure from conformal (angle-preserving) structure using
    Cauchy-Riemann residuals, the Riesz transform (2D Hilbert transform via FFT),
    structure tensor isotropy, and Liouville curvature.

    Random fields have high conformal distortion and uniform orientation entropy.
    Structured fields exhibit lower distortion and orientation bias.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    """

    def __init__(self, max_field_size: int = 128):
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "Conformal2D"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        H, W = field.shape
        metrics: Dict[str, float] = {}

        # Gradients
        fy, fx = np.gradient(field)

        # --- Structure tensor ---
        # J = [[Σfx², Σfxfy], [Σfxfy, Σfy²]]  (locally averaged)
        Jxx = fx ** 2
        Jxy = fx * fy
        Jyy = fy ** 2

        # Local averaging via 3x3 box filter
        from scipy.ndimage import uniform_filter
        Jxx_s = uniform_filter(Jxx, size=3)
        Jxy_s = uniform_filter(Jxy, size=3)
        Jyy_s = uniform_filter(Jyy, size=3)

        # Eigenvalues of structure tensor
        trace = Jxx_s + Jyy_s
        det = Jxx_s * Jyy_s - Jxy_s ** 2
        disc = np.maximum(trace ** 2 - 4 * det, 0.0)
        sqrt_disc = np.sqrt(disc)
        lam1 = (trace + sqrt_disc) / 2.0
        lam2 = (trace - sqrt_disc) / 2.0

        # Structure isotropy: 1 - (λ₁ - λ₂)/(λ₁ + λ₂ + ε)
        isotropy = 1.0 - (lam1 - lam2) / (lam1 + lam2 + 1e-12)
        metrics['structure_isotropy'] = float(np.mean(isotropy))

        # Conformal distortion: log(λ₁ / (λ₂ + ε)) — log-ratio of max to min stretch
        with np.errstate(divide='ignore', invalid='ignore'):
            distortion = np.log(lam1 / (lam2 + 1e-12) + 1.0)
        distortion = np.clip(distortion, 0.0, 20.0)
        metrics['conformal_distortion_mean'] = float(np.mean(distortion))
        metrics['conformal_distortion_std'] = float(np.std(distortion))

        # Liouville curvature: Δ(log(conformal_factor))
        # conformal factor ≈ √(λ₁) (dominant stretch)
        log_cf = np.log(np.sqrt(np.maximum(lam1, 1e-12)) + 1e-12)
        laplacian_log_cf = (
            np.gradient(np.gradient(log_cf, axis=0), axis=0)
            + np.gradient(np.gradient(log_cf, axis=1), axis=1)
        )
        metrics['liouville_curvature_mean'] = float(np.mean(laplacian_log_cf))
        metrics['liouville_curvature_std'] = float(np.std(laplacian_log_cf))

        # Harmonic residual: ||Δf|| / ||f||
        laplacian_f = (
            np.gradient(np.gradient(field, axis=0), axis=0)
            + np.gradient(np.gradient(field, axis=1), axis=1)
        )
        f_norm = np.sqrt(np.mean(field ** 2)) + 1e-12
        metrics['harmonic_residual'] = float(np.sqrt(np.mean(laplacian_f ** 2)) / f_norm)

        # --- Riesz transform via FFT ---
        F = np.fft.fft2(field)
        freqs_y = np.fft.fftfreq(H).reshape(-1, 1)
        freqs_x = np.fft.fftfreq(W).reshape(1, -1)
        mag_freq = np.sqrt(freqs_x ** 2 + freqs_y ** 2)
        mag_freq[0, 0] = 1.0  # avoid division by zero at DC

        # Riesz kernels: -i * k_j / |k|
        Rx = np.fft.ifft2(F * (-1j * freqs_x / mag_freq)).real
        Ry = np.fft.ifft2(F * (-1j * freqs_y / mag_freq)).real

        riesz_amp = np.sqrt(Rx ** 2 + Ry ** 2)
        metrics['riesz_amplitude_mean'] = float(np.mean(riesz_amp))

        # Riesz orientation entropy
        riesz_angle = np.arctan2(Ry, Rx + 1e-12)
        n_bins = 36
        hist, _ = np.histogram(riesz_angle, bins=n_bins, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-12)
        hist = hist[hist > 0]
        max_entropy = np.log(n_bins)
        metrics['riesz_orientation_entropy'] = float(-np.sum(hist * np.log(hist)) / max_entropy) if max_entropy > 0 else 0.0

        # Cauchy-Riemann residual: treat (field, Ry) as (u, v) of complex function
        # C-R equations: ∂u/∂x = ∂v/∂y, ∂u/∂y = -∂v/∂x
        vy, vx = np.gradient(Ry)
        cr1 = fx - vy  # ∂u/∂x - ∂v/∂y
        cr2 = fy + vx  # ∂u/∂y + ∂v/∂x
        cr_residual = np.sqrt(cr1 ** 2 + cr2 ** 2)
        grad_norm_mean = np.mean(np.sqrt(fx ** 2 + fy ** 2)) + 1e-12
        metrics['cauchy_riemann_residual'] = float(np.mean(cr_residual) / grad_norm_mean)

        # Analytic energy fraction: energy in "positive frequency" half
        total_energy = np.sum(np.abs(F) ** 2)
        # Analytic signal: zero out negative frequencies
        F_analytic = F.copy()
        # For 2D, "analytic" = keep only positive kx half
        half = W // 2
        F_analytic[:, half + 1:] = 0
        analytic_energy = np.sum(np.abs(F_analytic) ** 2)
        metrics['analytic_energy_fraction'] = float(analytic_energy / (total_energy + 1e-12))

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class MinkowskiFunctionalGeometry(ExoticGeometry):
    """Integral geometry of 2D fields via Minkowski functionals.

    From Hadwiger's theorem: any additive, motion-invariant, continuous
    valuation on convex bodies in ℝ² is a linear combination of three
    Minkowski functionals — area (V₀), perimeter (V₁), and Euler
    characteristic (V₂).

    We threshold the field at multiple levels and compute all three
    functionals for each excursion set {f > t}. The resulting functional
    curves encode the complete morphological structure of the field.

    Used in cosmology (CMB analysis), materials science, and stochastic
    geometry. Captures level-set topology that curvature and persistence miss.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    n_thresholds : int
        Number of threshold levels to sweep.
    """

    def __init__(self, max_field_size: int = 128, n_thresholds: int = 11):
        self._max_field_size = max_field_size
        self._n_thresholds = n_thresholds

    @property
    def name(self) -> str:
        return "MinkowskiFunctional"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def _excursion_functionals(self, binary: np.ndarray) -> tuple:
        """Compute Minkowski functionals for a binary excursion set.

        Returns (area_fraction, boundary_density, euler_density).
        Uses the marching squares approach for boundary and Euler characteristic.
        """
        H, W = binary.shape
        n_pixels = H * W

        # V₀: area fraction
        area = float(np.sum(binary)) / n_pixels

        # V₁: boundary density — count edges between 0 and 1
        h_edges = np.sum(binary[:, :-1] != binary[:, 1:])
        v_edges = np.sum(binary[:-1, :] != binary[1:, :])
        boundary = float(h_edges + v_edges) / n_pixels

        # V₂: Euler characteristic density via 2×2 quad counting
        # For each 2×2 block, count vertices in the excursion set
        # χ = (n₁ - n₃ + 2·n_diag) / 4  where n_k = quads with k corners set
        # Simpler: χ = V - E + F for the excursion boundary
        # Use the efficient formula: χ = Σ q₁ - Σ q₃ + 2·Σ q_d
        if H < 2 or W < 2:
            euler = 0.0
        else:
            quads = (binary[:-1, :-1].astype(int) + binary[:-1, 1:].astype(int) +
                     binary[1:, :-1].astype(int) + binary[1:, 1:].astype(int))
            # Count quads by vertex count
            n1 = np.sum(quads == 1)
            n3 = np.sum(quads == 3)
            # Diagonal configurations (checkerboard 2×2): both diagonals set
            diag1 = (binary[:-1, :-1] == 1) & (binary[1:, 1:] == 1) & \
                     (binary[:-1, 1:] == 0) & (binary[1:, :-1] == 0)
            diag2 = (binary[:-1, 1:] == 1) & (binary[1:, :-1] == 1) & \
                     (binary[:-1, :-1] == 0) & (binary[1:, 1:] == 0)
            n_diag = np.sum(diag1) + np.sum(diag2)
            euler = float(n1 - n3 + 2 * n_diag) / (4.0 * n_pixels)

        return area, boundary, euler

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        metrics: Dict[str, float] = {}

        fmin, fmax = float(np.min(field)), float(np.max(field))
        if fmax - fmin < 1e-12:
            # Constant field — all functionals are trivial
            for prefix in ['area', 'boundary', 'euler']:
                metrics[f'{prefix}_mean'] = 0.0
                metrics[f'{prefix}_std'] = 0.0
            metrics['euler_max'] = 0.0
            metrics['euler_max_threshold'] = 0.5
            metrics['boundary_max'] = 0.0
            metrics['area_auc'] = 0.0
            return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})

        # Sweep thresholds from 5th to 95th percentile
        thresholds = np.linspace(
            np.percentile(field, 5), np.percentile(field, 95), self._n_thresholds
        )

        areas, boundaries, eulers = [], [], []
        for t in thresholds:
            binary = (field > t).astype(np.int8)
            a, b, e = self._excursion_functionals(binary)
            areas.append(a)
            boundaries.append(b)
            eulers.append(e)

        areas = np.array(areas)
        boundaries = np.array(boundaries)
        eulers = np.array(eulers)

        # Area functional statistics
        metrics['area_mean'] = float(np.mean(areas))
        metrics['area_std'] = float(np.std(areas))
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        metrics['area_auc'] = float(_trapz(areas, np.linspace(0, 1, len(areas))))

        # Boundary functional statistics
        metrics['boundary_mean'] = float(np.mean(boundaries))
        metrics['boundary_std'] = float(np.std(boundaries))
        metrics['boundary_max'] = float(np.max(boundaries))

        # Euler characteristic statistics
        metrics['euler_mean'] = float(np.mean(eulers))
        metrics['euler_std'] = float(np.std(eulers))
        metrics['euler_max'] = float(np.max(np.abs(eulers)))
        # Threshold at which Euler characteristic is maximized (topology most complex)
        idx_max = int(np.argmax(np.abs(eulers)))
        metrics['euler_max_threshold'] = float(
            (thresholds[idx_max] - fmin) / (fmax - fmin + 1e-12)
        )

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class MultiscaleFractalGeometry(ExoticGeometry):
    """Multiscale fractal analysis of 2D fields.

    Computes fractal dimension via 2D box-counting, lacunarity (gap structure
    at multiple scales), and the Hurst exponent (long-range dependence via
    rescaled range analysis on rows/columns).

    Stego subtly breaks self-similarity across scales — the fractal
    dimension may stay similar, but lacunarity (how "gappy" the structure is)
    and the Hurst exponent shift detectably.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    """

    def __init__(self, max_field_size: int = 128):
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "MultiscaleFractal"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def _box_counting_dim(self, field: np.ndarray, threshold: float) -> float:
        """2D box-counting dimension of the excursion set {f > threshold}."""
        binary = (field > threshold).astype(np.int8)
        H, W = binary.shape
        min_dim = min(H, W)
        if min_dim < 4:
            return 2.0

        sizes = []
        counts = []
        box_size = 2
        while box_size <= min_dim // 2:
            # Count non-empty boxes
            nH, nW = H // box_size, W // box_size
            trimmed = binary[:nH * box_size, :nW * box_size]
            blocks = trimmed.reshape(nH, box_size, nW, box_size)
            occupied = np.any(blocks, axis=(1, 3)).sum()
            if occupied > 0:
                sizes.append(box_size)
                counts.append(int(occupied))
            box_size *= 2

        if len(sizes) < 2:
            return 2.0

        log_sizes = np.log(1.0 / np.array(sizes, dtype=np.float64))
        log_counts = np.log(np.array(counts, dtype=np.float64))
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return float(np.clip(coeffs[0], 0.0, 3.0))

    def _lacunarity(self, field: np.ndarray, box_sizes: list) -> np.ndarray:
        """Gliding-box lacunarity at multiple scales.

        Λ(r) = <M²>/<M>² where M is the mass (sum) in each box of size r.
        Λ=1 for perfectly uniform, Λ>1 for gappy/clustered structure.
        """
        H, W = field.shape
        lacs = []
        for r in box_sizes:
            if r > min(H, W):
                lacs.append(1.0)
                continue
            # Compute box sums via cumulative sum
            cumsum = np.cumsum(np.cumsum(field, axis=0), axis=1)
            # Pad with zeros for easy box sum computation
            padded = np.zeros((H + 1, W + 1))
            padded[1:, 1:] = cumsum
            # Box sums for all positions
            box_sums = (padded[r:, r:] - padded[r:, :-r] -
                        padded[:-r, r:] + padded[:-r, :-r])
            if box_sums.size == 0:
                lacs.append(1.0)
                continue
            mean_m = np.mean(box_sums)
            mean_m2 = np.mean(box_sums ** 2)
            if mean_m > 1e-12:
                lacs.append(float(mean_m2 / (mean_m ** 2)))
            else:
                lacs.append(1.0)
        return np.array(lacs)

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """Rescaled range (R/S) estimate of Hurst exponent."""
        n = len(series)
        if n < 16:
            return 0.5

        sizes = []
        rs_values = []
        size = 8
        while size <= n // 2:
            n_chunks = n // size
            rs_chunk = []
            for i in range(n_chunks):
                chunk = series[i * size:(i + 1) * size]
                mean = np.mean(chunk)
                deviations = np.cumsum(chunk - mean)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(chunk, ddof=1)
                if S > 1e-12:
                    rs_chunk.append(R / S)
            if rs_chunk:
                sizes.append(size)
                rs_values.append(np.mean(rs_chunk))
            size *= 2

        if len(sizes) < 2:
            return 0.5

        log_s = np.log(np.array(sizes, dtype=np.float64))
        log_rs = np.log(np.array(rs_values, dtype=np.float64))
        coeffs = np.polyfit(log_s, log_rs, 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        H, W = field.shape
        metrics: Dict[str, float] = {}

        fmin, fmax = float(np.min(field)), float(np.max(field))

        # Box-counting dimension at median threshold
        median_val = float(np.median(field))
        metrics['box_counting_dim'] = self._box_counting_dim(field, median_val)

        # Box-counting at 25th and 75th percentile — dimension spread
        d25 = self._box_counting_dim(field, float(np.percentile(field, 25)))
        d75 = self._box_counting_dim(field, float(np.percentile(field, 75)))
        metrics['dim_spread'] = float(abs(d75 - d25))

        # Lacunarity at multiple scales
        box_sizes = [2, 4, 8, 16]
        box_sizes = [s for s in box_sizes if s <= min(H, W)]
        if box_sizes:
            lacs = self._lacunarity(field, box_sizes)
            metrics['lacunarity_small'] = float(lacs[0]) if len(lacs) > 0 else 1.0
            metrics['lacunarity_large'] = float(lacs[-1]) if len(lacs) > 0 else 1.0
            metrics['lacunarity_slope'] = float(
                (np.log(lacs[-1] + 1e-12) - np.log(lacs[0] + 1e-12)) /
                (np.log(box_sizes[-1]) - np.log(box_sizes[0]) + 1e-12)
            ) if len(lacs) > 1 else 0.0
        else:
            metrics['lacunarity_small'] = 1.0
            metrics['lacunarity_large'] = 1.0
            metrics['lacunarity_slope'] = 0.0

        # Hurst exponent — average over rows and columns
        row_hursts = [self._hurst_exponent(field[i, :]) for i in range(H)]
        col_hursts = [self._hurst_exponent(field[:, j]) for j in range(W)]
        all_hursts = row_hursts + col_hursts
        metrics['hurst_mean'] = float(np.mean(all_hursts))
        metrics['hurst_std'] = float(np.std(all_hursts))
        metrics['hurst_anisotropy'] = float(
            abs(np.mean(row_hursts) - np.mean(col_hursts))
        )

        # Fluctuation function: variance at multiple scales (DFA-like)
        if min(H, W) >= 8:
            variances = []
            for scale in [1, 2, 4, 8]:
                if scale > min(H, W) // 2:
                    break
                bh = scale
                nH, nW = H // bh, W // bh
                if nH < 2 or nW < 2:
                    break
                trimmed = field[:nH * bh, :nW * bh]
                blocks = trimmed.reshape(nH, bh, nW, bh)
                block_means = blocks.mean(axis=(1, 3))
                variances.append(float(np.var(block_means)))

            if len(variances) >= 2:
                log_scales = np.log(np.array([1, 2, 4, 8][:len(variances)], dtype=np.float64))
                log_vars = np.log(np.array(variances) + 1e-12)
                coeffs = np.polyfit(log_scales, log_vars, 1)
                metrics['fluctuation_exponent'] = float(coeffs[0])
            else:
                metrics['fluctuation_exponent'] = 0.0
        else:
            metrics['fluctuation_exponent'] = 0.0

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class HodgeLaplacianGeometry(ExoticGeometry):
    """Hodge-Laplacian analysis of 2D fields via FFT Poisson solvers.

    Analyzes the field through the lens of the Laplacian operator and its
    iterated applications: the Laplacian Δf (source/sink density), the
    biharmonic Δ²f (curvature of curvature), and the Poisson recovery
    error (non-periodic boundary content).

    The FFT Poisson solver recovers f from Δf assuming periodic boundaries.
    The recovery error measures how much structure lives at the boundary
    (non-periodic content). Stego disrupts fine-scale Laplacian statistics
    and shifts the energy partition between Dirichlet and biharmonic terms.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    """

    def __init__(self, max_field_size: int = 128):
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "HodgeLaplacian"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def _solve_poisson_fft(self, rhs: np.ndarray) -> np.ndarray:
        """Solve Δu = rhs with periodic boundary conditions via FFT."""
        H, W = rhs.shape
        ky = np.fft.fftfreq(H).reshape(-1, 1) * 2 * np.pi
        kx = np.fft.fftfreq(W).reshape(1, -1) * 2 * np.pi
        k_sq = kx ** 2 + ky ** 2
        k_sq[0, 0] = 1.0  # avoid division by zero
        rhs_hat = np.fft.fft2(rhs)
        u_hat = rhs_hat / (-k_sq)
        u_hat[0, 0] = 0.0  # zero mean
        return np.fft.ifft2(u_hat).real

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        metrics: Dict[str, float] = {}

        f_centered = field - np.mean(field)
        f_norm_sq = np.sum(f_centered ** 2) + 1e-12

        # Gradient and Laplacian
        fy, fx = np.gradient(field)
        laplacian = (np.gradient(np.gradient(field, axis=0), axis=0) +
                     np.gradient(np.gradient(field, axis=1), axis=1))

        # Laplacian statistics (= divergence of gradient = Δf)
        metrics['laplacian_mean'] = float(np.mean(laplacian))
        metrics['laplacian_std'] = float(np.std(laplacian))

        # Laplacian energy ratio: ||Δf||² / ||f||²
        lap_energy = np.sum(laplacian ** 2)
        metrics['laplacian_energy'] = float(lap_energy / f_norm_sq)

        # Dirichlet energy: ||∇f||² / ||f||²
        dirichlet = np.sum(fx ** 2 + fy ** 2)
        metrics['dirichlet_energy'] = float(dirichlet / f_norm_sq)

        # Biharmonic: Δ²f (Laplacian of Laplacian)
        biharm = (np.gradient(np.gradient(laplacian, axis=0), axis=0) +
                  np.gradient(np.gradient(laplacian, axis=1), axis=1))
        biharm_energy = np.sum(biharm ** 2)
        metrics['biharmonic_energy'] = float(biharm_energy / (lap_energy + 1e-12))

        # Poisson recovery error: solve Δu = Δf, compare u to f
        recovered = self._solve_poisson_fft(laplacian)
        recovery_error = np.sum((recovered - f_centered) ** 2)
        metrics['poisson_recovery_error'] = float(recovery_error / f_norm_sq)

        # Source/sink balance: fraction of field where Δf > 0
        metrics['source_fraction'] = float(np.mean(laplacian > 0))

        # Gradient coherence: mean cosine similarity between adjacent gradient vectors
        # Measures how "aligned" the gradient field is locally
        grad_mag = np.sqrt(fx ** 2 + fy ** 2) + 1e-12
        ux, uy = fx / grad_mag, fy / grad_mag  # unit gradient
        # Dot product with rightward and downward neighbors
        coh_x = ux[:, :-1] * ux[:, 1:] + uy[:, :-1] * uy[:, 1:]
        coh_y = ux[:-1, :] * ux[1:, :] + uy[:-1, :] * uy[1:, :]
        metrics['gradient_coherence'] = float(
            (np.mean(coh_x) + np.mean(coh_y)) / 2.0
        )

        # Spectral gap ratio: energy in lowest vs highest frequency bands of Laplacian
        lap_fft = np.abs(np.fft.fftshift(np.fft.fft2(laplacian))) ** 2
        H, W = lap_fft.shape
        center = (H // 2, W // 2)
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        r_max = min(H, W) // 2
        low_mask = r < r_max * 0.25
        high_mask = r > r_max * 0.75
        low_e = np.sum(lap_fft[low_mask])
        high_e = np.sum(lap_fft[high_mask])
        metrics['laplacian_spectral_ratio'] = float(
            low_e / (high_e + 1e-12)
        )

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class SpectralPowerGeometry(ExoticGeometry):
    """Spectral power analysis of 2D fields via FFT.

    Natural images/textures follow characteristic power-law spectra
    P(k) ∝ k^(-β) where β ≈ 2 for natural scenes. Steganographic
    embedding disrupts this scaling, particularly at high frequencies.

    Computes the radial power spectrum P(k), spectral slope β, spectral
    centroid, spectral entropy, and directional anisotropy.

    Parameters
    ----------
    max_field_size : int
        Fields larger than this are block-averaged down.
    """

    def __init__(self, max_field_size: int = 128):
        self._max_field_size = max_field_size

    @property
    def name(self) -> str:
        return "SpectralPower"

    @property
    def dimension(self) -> Union[int, str]:
        return "2D"

    def validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            return data
        data = data.flatten()
        n = len(data)
        side = int(np.ceil(np.sqrt(n)))
        padded = np.full(side * side, np.mean(data))
        padded[:n] = data
        return padded.reshape(side, side)

    def embed(self, data: np.ndarray) -> np.ndarray:
        return self.validate_data(data)

    def _downsample(self, field: np.ndarray) -> np.ndarray:
        H, W = field.shape
        if H <= self._max_field_size and W <= self._max_field_size:
            return field
        scale = max(H, W) / self._max_field_size
        bh = max(1, int(np.ceil(scale)))
        new_H, new_W = H // bh, W // bh
        if new_H < 4 or new_W < 4:
            return field[:4, :4]
        trimmed = field[:new_H * bh, :new_W * bh]
        return trimmed.reshape(new_H, bh, new_W, bh).mean(axis=(1, 3))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        field = self._downsample(self.validate_data(data))
        H, W = field.shape
        metrics: Dict[str, float] = {}

        # 2D FFT power spectrum
        F = np.fft.fft2(field - np.mean(field))  # subtract mean (remove DC)
        power = np.abs(np.fft.fftshift(F)) ** 2

        # Frequency coordinates (integer spatial frequencies)
        ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
        kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)
        angles = np.arctan2(KY, KX)

        # Radial power spectrum P(k) via binning
        k_max = min(H, W) // 2
        k_bins = np.arange(1, k_max + 1, dtype=np.float64)
        radial_power = np.zeros(len(k_bins))
        for i, k in enumerate(k_bins):
            mask = (K >= k - 0.5) & (K < k + 0.5)
            if np.any(mask):
                radial_power[i] = np.mean(power[mask])

        # Filter out zeros for log fitting
        valid = radial_power > 0
        if np.sum(valid) >= 3:
            log_k = np.log(k_bins[valid])
            log_p = np.log(radial_power[valid])
            coeffs = np.polyfit(log_k, log_p, 1)
            metrics['spectral_slope'] = float(coeffs[0])

            # Residual from power law (goodness of fit)
            predicted = np.polyval(coeffs, log_k)
            ss_res = np.sum((log_p - predicted) ** 2)
            ss_tot = np.sum((log_p - np.mean(log_p)) ** 2) + 1e-12
            metrics['spectral_r_squared'] = float(1.0 - ss_res / ss_tot)
        else:
            metrics['spectral_slope'] = 0.0
            metrics['spectral_r_squared'] = 0.0

        # Spectral centroid: <k> = Σ k·P(k) / Σ P(k)
        total_power = np.sum(radial_power) + 1e-12
        metrics['spectral_centroid'] = float(np.sum(k_bins * radial_power) / total_power)

        # Spectral entropy
        p_norm = radial_power / total_power
        p_norm = p_norm[p_norm > 0]
        max_entropy = np.log(len(k_bins)) if len(k_bins) > 1 else 1.0
        metrics['spectral_entropy'] = float(
            -np.sum(p_norm * np.log(p_norm)) / max_entropy
        ) if max_entropy > 0 else 0.0

        # High-frequency energy ratio: energy above k_max/2 vs total
        hf_mask = K > k_max / 2
        metrics['high_freq_ratio'] = float(np.sum(power[hf_mask]) / (np.sum(power) + 1e-12))

        # Directional anisotropy: compare power in 4 angular sectors
        sector_powers = []
        for angle_start in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            sector_mask = (angles >= angle_start - np.pi / 8) & \
                          (angles < angle_start + np.pi / 8) & (K > 1)
            if np.any(sector_mask):
                sector_powers.append(float(np.mean(power[sector_mask])))
            else:
                sector_powers.append(0.0)
        sector_powers = np.array(sector_powers)
        if np.sum(sector_powers) > 0:
            sector_norm = sector_powers / (np.sum(sector_powers) + 1e-12)
            # Anisotropy: 0 = isotropic, 1 = all power in one direction
            metrics['spectral_anisotropy'] = float(
                1.0 - 4.0 * np.min(sector_norm)
            )
        else:
            metrics['spectral_anisotropy'] = 0.0

        # Spectral kurtosis: peakedness of radial spectrum
        spec_mean = np.mean(radial_power)
        spec_std = np.std(radial_power) + 1e-12
        metrics['spectral_kurtosis'] = float(
            np.mean(((radial_power - spec_mean) / spec_std) ** 4) - 3.0
        )

        # Mid-frequency concentration: power in k_max/4 to k_max/2 band
        mid_mask = (K > k_max / 4) & (K <= k_max / 2)
        metrics['mid_freq_ratio'] = float(np.sum(power[mid_mask]) / (np.sum(power) + 1e-12))

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================

def delay_embed(data: np.ndarray, tau: int) -> np.ndarray:
    """
    Takens delay embedding: restructure data so geometries pair byte[i] with byte[i+τ].

    For tau=1 this is equivalent to the original data (consecutive pairs).
    For tau>1, the geometry sees correlations at lag τ instead of lag 1.

    Heisenberg twist_rate at delay τ IS the lag-τ autocorrelation.
    Fisher is tau-invariant (measures marginals, not pairs).

    Parameters
    ----------
    data : np.ndarray of uint8
    tau : int, delay parameter (1 = no change)

    Returns
    -------
    np.ndarray of uint8, length 2*(N-tau)
    """
    data = np.asarray(data, dtype=np.uint8)
    N = len(data)
    n_pairs = N - tau
    if n_pairs <= 0:
        raise ValueError(f"Data length {N} too short for delay tau={tau}")
    result = np.empty(2 * n_pairs, dtype=np.uint8)
    result[0::2] = data[:n_pairs]
    result[1::2] = data[tau:tau + n_pairs]
    return result


def spectral_preprocess(data: np.ndarray) -> np.ndarray:
    """
    FFT magnitude spectrum preprocessing: converts data to frequency domain.

    Spectral and raw are complementary (not substitutes):
    - FFT catches spectrally-distinct signals with similar raw bytes
    - Raw catches sequential/dynamical structure FFT destroys
    - Cepstrum is mostly useless after uint8 quantization

    Parameters
    ----------
    data : np.ndarray of uint8

    Returns
    -------
    np.ndarray of uint8 (FFT magnitude, normalized to 0-255)
    """
    sig = np.asarray(data, dtype=np.float64) - 128.0
    spectrum = np.abs(np.fft.rfft(sig))
    spectrum = spectrum[1:]  # drop DC
    if len(spectrum) == 0:
        return data
    mx = np.max(spectrum)
    if mx > 0:
        spectrum = spectrum / mx * 255.0
    return spectrum.astype(np.uint8)


def bitplane_extract(data: np.ndarray, plane: int) -> np.ndarray:
    """
    Extract a single bit plane and repack into uint8 bytes.

    Useful for isolating bit-level structure (e.g., Collatz bitplane analysis
    yields 77 sig metrics at plane 0). Note: for LSB steganography on
    realistic carriers, raw byte analysis outperforms bitplane extraction —
    PVD and spread spectrum are detectable at the byte level (d > 20) while
    bitplane extraction reduces sample size 8x, destroying statistical power.

    Parameters
    ----------
    data : np.ndarray of uint8
    plane : int, 0 (LSB) to 7 (MSB)

    Returns
    -------
    np.ndarray of uint8, length N//8
    """
    assert 0 <= plane <= 7, f"Plane must be 0-7, got {plane}"
    data = np.asarray(data, dtype=np.uint8)
    bits = (data >> plane) & 1
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    bits_reshaped = bits.reshape(-1, 8)
    powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    return np.sum(bits_reshaped * powers, axis=1).astype(np.uint8)


# =============================================================================
# ENCODING UTILITIES
# =============================================================================

# Standard alphabets for symbolic encoding
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
AMINO_ALPHABET = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
    'T': 16, 'W': 17, 'Y': 18, 'V': 19,
}


def encode_symbolic(symbols, alphabet=None):
    """Map a symbolic sequence to a float array in [0, 1].

    Parameters
    ----------
    symbols : iterable of hashable
        The symbolic sequence (e.g. list of chars, strings, ints).
    alphabet : dict or None
        Mapping from symbol -> int index. If None, auto-discovered from data.

    Returns
    -------
    np.ndarray of float in [0, 1]

    Examples
    --------
    >>> encode_symbolic("ACGT", DNA_ALPHABET)
    array([0.  , 0.333, 0.667, 1.  ])
    >>> encode_symbolic([1, 2, 3, 1])  # auto-alphabet
    array([0. , 0.5, 1. , 0. ])
    """
    symbols = list(symbols)
    if alphabet is None:
        unique = sorted(set(symbols))
        alphabet = {s: i for i, s in enumerate(unique)}
    n_symbols = max(alphabet.values()) + 1
    indices = np.array([alphabet[s] for s in symbols], dtype=float)
    if n_symbols <= 1:
        return np.zeros(len(symbols))
    return indices / (n_symbols - 1)


def encode_float_to_unit(data):
    """Normalize a float array to [0, 1] range.

    Parameters
    ----------
    data : array-like
        Any numeric sequence.

    Returns
    -------
    np.ndarray of float in [0, 1]
    """
    data = np.asarray(data, dtype=float)
    lo, hi = data.min(), data.max()
    if hi > lo:
        return (data - lo) / (hi - lo)
    return np.zeros_like(data)


# =============================================================================
# GEOMETRY ANALYZER (MAIN INTERFACE)
# =============================================================================

class GeometryAnalyzer:
    """
    Main interface for analyzing data with multiple geometries.

    Usage:
        analyzer = GeometryAnalyzer()
        analyzer.add_geometry(E8Geometry())
        analyzer.add_geometry(TorusGeometry())
        results = analyzer.analyze(data)
    """

    def __init__(self):
        self.geometries: List[ExoticGeometry] = []

    def add_geometry(self, geometry: ExoticGeometry) -> 'GeometryAnalyzer':
        """Add a geometry to the analyzer (chainable)."""
        self.geometries.append(geometry)
        return self

    def _resolve_input_scale(self, data_mode: str) -> Union[float, str]:
        """Convert data_mode to input_scale value for geometry constructors."""
        if data_mode == 'bytes':
            return 255.0
        elif data_mode == 'auto':
            return 'auto'
        elif data_mode == 'unit':
            return 1.0
        else:
            raise ValueError(f"Unknown data_mode '{data_mode}'. Use 'bytes', 'auto', or 'unit'.")

    def add_default_geometries(self, data_mode: str = 'bytes') -> 'GeometryAnalyzer':
        """Add core validated geometries (fast, good discriminators).

        Parameters
        ----------
        data_mode : str
            'bytes' (default) — input is uint8 [0,255], backward-compatible.
            'auto'  — auto-detect range from data.
            'unit'  — data already in [0,1].
        """
        s = self._resolve_input_scale(data_mode)
        self.geometries = [
            E8Geometry(),
            TorusGeometry(bins=16),
            HyperbolicGeometry(input_scale=s),
            HeisenbergGeometry(input_scale=s),
            WassersteinGeometry(input_scale=s),
        ]
        return self

    def add_all_geometries(self, data_mode: str = 'bytes') -> 'GeometryAnalyzer':
        """Add ALL available geometries (comprehensive but slower).

        Parameters
        ----------
        data_mode : str
            'bytes' (default) — input is uint8 [0,255], backward-compatible.
            'auto'  — auto-detect range from data.
            'unit'  — data already in [0,1].
        """
        s = self._resolve_input_scale(data_mode)
        self.geometries = [
            # Validated / core
            E8Geometry(),
            TorusGeometry(bins=16),
            HyperbolicGeometry(input_scale=s),
            HeisenbergGeometry(input_scale=s),
            HeisenbergGeometry(input_scale=s, center_data=True),  # Correlation mode
            # Classical
            SphericalGeometry(input_scale=s),
            # Thurston's 8 geometries (we have E³, S³, H³, Nil above)
            SolGeometry(input_scale=s),
            ProductS2RGeometry(input_scale=s),
            ProductH2RGeometry(input_scale=s),
            SL2RGeometry(input_scale=s),
            # Algebraic
            UltrametricGeometry(p=2),
            TropicalGeometry(input_scale=s),
            ProjectiveGeometry(input_scale=s),
            # Statistical
            WassersteinGeometry(input_scale=s),
            FisherGeometry(),
            PersistentHomologyGeometry(),
            # Physical
            LorentzianGeometry(input_scale=s),
            SymplecticGeometry(input_scale=s),
            SpiralGeometry(input_scale=s),
            # Aperiodic
            PenroseGeometry(input_scale=s),
            AmmannBeenkerGeometry(input_scale=s),
            EinsteinHatGeometry(input_scale=s),
            # Clifford Torus
            CliffordTorusGeometry(input_scale=s),
            # Higher-order (3rd/4th order, independent from all above)
            HigherOrderGeometry(),
        ]
        # Cantor uses bit-extraction: only meaningful for integer/byte data
        if data_mode == 'bytes':
            self.geometries.insert(11, CantorGeometry())
        return self

    def add_thurston_geometries(self, data_mode: str = 'bytes') -> 'GeometryAnalyzer':
        """Add Thurston's 8 geometries for 3-manifold analysis."""
        s = self._resolve_input_scale(data_mode)
        self.geometries = [
            # E³ (Euclidean) - implicit
            SphericalGeometry(input_scale=s),      # S³
            HyperbolicGeometry(input_scale=s),      # H³
            HeisenbergGeometry(input_scale=s),      # Nil
            SolGeometry(input_scale=s),              # Sol
            ProductS2RGeometry(input_scale=s),      # S² × ℝ
            ProductH2RGeometry(input_scale=s),      # H² × ℝ
            SL2RGeometry(input_scale=s),             # SL̃(2,ℝ)
        ]
        return self

    def add_spatial_geometries(self, data_mode: str = 'bytes') -> 'GeometryAnalyzer':
        """Add SpatialFieldGeometry for native 2D field analysis.

        NOT included in add_all_geometries() — explicit opt-in only.
        Can be chained: analyzer.add_all_geometries().add_spatial_geometries()

        Parameters
        ----------
        data_mode : str
            'bytes' (default), 'auto', or 'unit'. Spatial geometry works on
            float64 internally regardless, but this controls other geometries
            if chained.
        """
        self.geometries.append(SpatialFieldGeometry())
        self.geometries.append(SurfaceGeometry())
        self.geometries.append(PersistentHomology2DGeometry())
        self.geometries.append(ConformalGeometry2D())
        self.geometries.append(MinkowskiFunctionalGeometry())
        self.geometries.append(MultiscaleFractalGeometry())
        self.geometries.append(HodgeLaplacianGeometry())
        self.geometries.append(SpectralPowerGeometry())
        return self

    def analyze(self, data: np.ndarray, data_label: str = "") -> AnalysisResult:
        """Run all geometries on the data."""
        result = AnalysisResult()
        result.data_info = {
            "label": data_label,
            "length": len(data),
            "dtype": str(data.dtype) if hasattr(data, 'dtype') else type(data).__name__,
        }

        for geom in self.geometries:
            try:
                geom_result = geom.compute_metrics(data)
                result.results.append(geom_result)
            except Exception as e:
                warnings.warn(f"Geometry {geom.name} failed: {e}")

        return result

    def compare(self, data1: np.ndarray, data2: np.ndarray,
                label1: str = "Data 1", label2: str = "Data 2") -> Dict[str, Dict[str, float]]:
        """Compare two datasets across all geometries."""
        results1 = self.analyze(data1, label1)
        results2 = self.analyze(data2, label2)

        comparison = {}
        for r1 in results1.results:
            for r2 in results2.results:
                if r1.geometry_name == r2.geometry_name:
                    diffs = {}
                    for metric in r1.metrics:
                        if metric in r2.metrics:
                            v1, v2 = r1.metrics[metric], r2.metrics[metric]
                            diffs[metric] = {
                                label1: v1,
                                label2: v2,
                                "diff": v1 - v2,
                                "ratio": v1 / (v2 + 1e-10),
                            }
                    comparison[r1.geometry_name] = diffs

        return comparison

    def classify(self, data: np.ndarray, signature_dir: str = "signatures") -> List[Dict[str, Any]]:
        """
        Classify a data stream by comparing its geometric signature to a library.
        
        Returns a ranked list of potential system matches with confidence scores.
        """
        classifier = GeometricClassifier(signature_dir)
        if not classifier.has_signatures:
            warnings.warn("No signatures found in directory. Use train_signature.py to create them.")
            return []
            
        return classifier.classify(self, data)


# =============================================================================
# CLASSIFICATION ENGINE
# =============================================================================

class GeometricClassifier:
    """
    Engine for identifying systems via high-dimensional geometric signatures.

    Matches input data against a library of pre-trained signatures using
    Z-score distance across multiple scales. Uses median z-score (robust to
    outlier metrics) and match fraction for interpretable quality.
    """

    def __init__(self, signature_dir: str = "signatures"):
        self.signature_dir = signature_dir
        self.signatures = self._load_signatures()
        self.has_signatures = len(self.signatures) > 0

    def _load_signatures(self) -> List[Dict[str, Any]]:
        """Load all JSON signatures from the directory."""
        sigs = []
        if not os.path.exists(self.signature_dir):
            return []

        import json
        for filename in sorted(os.listdir(self.signature_dir)):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.signature_dir, filename), 'r') as f:
                        sigs.append(json.load(f))
                except Exception as e:
                    warnings.warn(f"Failed to load signature {filename}: {e}")
        return sigs

    def classify(self, analyzer: 'GeometryAnalyzer', data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Classify data using the provided analyzer.

        Returns ranked list of matches. Each entry contains:
          - system: signature name
          - avg_z_score: median absolute z-score (backward compat key name)
          - median_z: same as avg_z_score
          - match_fraction: fraction of metrics within 2 sigma
          - confidence: gap-based confidence (only meaningful for rank #1)
          - metric_details: list of (metric_name, z_value) sorted by |z|
        """
        if not self.signatures:
            return []

        # 1. Compute input signature (must match library scales)
        target_scales = self.signatures[0].get("scales", [1])

        input_metrics = []
        for tau in target_scales:
            if tau == 1:
                scaled_data = data
            else:
                scaled_data = delay_embed(data, tau=tau)
                if len(scaled_data) > 2000:
                    scaled_data = scaled_data[:2000]

            results = analyzer.analyze(scaled_data)
            for r in results.results:
                for mname in sorted(r.metrics.keys()):
                    input_metrics.append(r.metrics[mname])

        input_vec = np.array(input_metrics)

        # 2. Compare against all signatures
        rankings = []
        for sig in self.signatures:
            means = np.array(sig["means"])
            stds = np.array(sig["stds"])
            metric_names = sig.get("metrics", [])

            # Ensure vector lengths match
            if len(input_vec) != len(means):
                min_len = min(len(input_vec), len(means))
                iv = input_vec[:min_len]
                m = means[:min_len]
                s = stds[:min_len]
                names = metric_names[:min_len] if metric_names else []
            else:
                iv, m, s = input_vec, means, stds
                names = metric_names

            # Filter out constant metrics (std < 1e-8) and NaN/Inf values
            mask = (s >= 1e-8) & np.isfinite(iv) & np.isfinite(m) & np.isfinite(s)
            if not np.any(mask):
                # All metrics constant — can't classify
                rankings.append({
                    "system": sig["name"],
                    "avg_z_score": float('inf'),
                    "median_z": float('inf'),
                    "match_fraction": 0.0,
                    "confidence": 0.0,
                    "metric_details": []
                })
                continue

            iv_f, m_f, s_f = iv[mask], m[mask], s[mask]
            names_f = [names[i] for i in range(len(names)) if mask[i]] if names else []

            # Z-score distance
            z_scores = np.abs((iv_f - m_f) / s_f)
            z_scores = np.nan_to_num(z_scores, nan=10.0, posinf=10.0, neginf=10.0)

            median_z = float(np.median(z_scores))
            match_frac = float(np.mean(z_scores < 2.0))

            # Per-metric diagnostics
            if names_f:
                details = sorted(zip(names_f, z_scores.tolist()), key=lambda x: x[1])
            else:
                details = sorted(enumerate(z_scores.tolist()), key=lambda x: x[1])

            rankings.append({
                "system": sig["name"],
                "avg_z_score": median_z,  # backward compat
                "median_z": median_z,
                "match_fraction": match_frac,
                "confidence": 0.0,  # computed after sorting
                "metric_details": details,
                "attributes": sig.get("attributes", {}),
            })

        # Sort by median z-score (lowest first = best match)
        rankings.sort(key=lambda x: x["avg_z_score"])

        # Gap-based confidence for top match
        if len(rankings) >= 2:
            best_z = rankings[0]["median_z"]
            second_z = rankings[1]["median_z"]
            if second_z > 0:
                rankings[0]["confidence"] = float((second_z - best_z) / second_z)
            else:
                rankings[0]["confidence"] = 0.0
        elif len(rankings) == 1:
            rankings[0]["confidence"] = 1.0 if rankings[0]["median_z"] < 2.0 else 0.0

        return rankings


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_analyze(data: np.ndarray, label: str = "") -> AnalysisResult:
    """Quick analysis with default geometries."""
    analyzer = GeometryAnalyzer().add_default_geometries()
    return analyzer.analyze(data, label)


def sbox_quality_check(sbox: List[int]) -> Dict[str, Any]:
    """
    Quick S-box quality check using validated metrics.

    Returns assessment based on E8 diversity and torus coverage.

    Thresholds (empirically validated):
    - E8 unique roots: good S-boxes use 20-30+, weak use <15
    - Torus coverage: relative metric (compare to random permutation baseline)
    """
    data = np.array(sbox, dtype=np.uint8)

    e8 = E8Geometry()
    torus = TorusGeometry(bins=16)

    e8_result = e8.compute_metrics(data)
    torus_result = torus.compute_metrics(data)

    e8_unique = e8_result.metrics["unique_roots"]
    coverage = torus_result.metrics["coverage"]

    # Thresholds from validation
    issues = []
    if e8_unique < 15:
        issues.append(f"Low E8 diversity ({e8_unique} < 15): algebraic structure likely")
    if e8_unique < 10:
        issues.append(f"Very low E8 diversity ({e8_unique} < 10): definitely constrained")

    # For 256-byte S-boxes, 128 pairs can cover at most ~50% of 256 bins
    # Good S-boxes: ~0.4 coverage, weak affine: ~0.12
    if coverage < 0.25:
        issues.append(f"Very low torus coverage ({coverage:.2f} < 0.25): constrained structure")

    # Fixed points
    fixed_points = sum(1 for i, v in enumerate(sbox) if i == v)
    if fixed_points > 0:
        issues.append(f"Has {fixed_points} fixed points")

    assessment = "LIKELY SECURE" if not issues else "POTENTIAL WEAKNESS"

    return {
        "assessment": assessment,
        "e8_unique_roots": e8_unique,
        "torus_coverage": coverage,
        "fixed_points": fixed_points,
        "issues": issues,
    }


# =============================================================================
# DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXOTIC GEOMETRY FRAMEWORK - DEMO")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 2000, dtype=np.uint8)
    structured_data = np.array([(i * 17 + 31) % 256 for i in range(2000)], dtype=np.uint8)

    # Create analyzer with all geometries
    analyzer = GeometryAnalyzer()
    analyzer.add_geometry(E8Geometry())
    analyzer.add_geometry(TorusGeometry(bins=16))
    analyzer.add_geometry(HyperbolicGeometry())
    analyzer.add_geometry(HeisenbergGeometry())
    analyzer.add_geometry(SphericalGeometry())
    analyzer.add_geometry(CantorGeometry())
    analyzer.add_geometry(UltrametricGeometry(p=2))

    print("\n--- Random Data ---")
    random_results = analyzer.analyze(random_data, "Random")
    print(random_results.summary())

    print("\n--- Structured Data (LCG-like) ---")
    struct_results = analyzer.analyze(structured_data, "Structured")
    print(struct_results.summary())

    print("\n--- Comparison ---")
    comparison = analyzer.compare(random_data, structured_data, "Random", "Structured")

    print("\nKey differences:")
    for geom, metrics in comparison.items():
        for metric, values in metrics.items():
            diff = values["diff"]
            if abs(diff) > 0.1 * max(abs(values["Random"]), abs(values["Structured"]), 0.1):
                print(f"  {geom}.{metric}: Random={values['Random']:.3f}, "
                      f"Structured={values['Structured']:.3f} (diff={diff:+.3f})")

    # S-box demo
    print("\n" + "=" * 70)
    print("S-BOX QUALITY CHECK DEMO")
    print("=" * 70)

    # Full AES S-box (256 bytes)
    aes_sbox = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]

    # Random permutation (full 256 bytes)
    random_perm = list(np.random.permutation(256))

    # Weak: affine S-box (full 256 bytes)
    affine_sbox = [(3 * i + 17) % 256 for i in range(256)]

    for name, sbox in [("AES (secure)", aes_sbox),
                       ("Random perm", random_perm),
                       ("Affine (weak)", affine_sbox)]:
        result = sbox_quality_check(sbox)
        print(f"\n{name}:")
        print(f"  Assessment: {result['assessment']}")
        print(f"  E8 roots: {result['e8_unique_roots']}")
        print(f"  Torus coverage: {result['torus_coverage']:.3f}")
        if result['issues']:
            for issue in result['issues']:
                print(f"  Warning: {issue}")

    print("\n" + "=" * 70)
    print("Framework ready for use!")
    print("=" * 70)
