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
    - ZariskiGeometry: non-Hausdorff algebraic structure, polynomial recurrences
    - CayleyGeometry: geometric group theory — growth, hyperbolicity, spectral gap

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

import hashlib
import inspect
import math
import os
import pickle
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import combinations, permutations, product
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

    @property
    def description(self) -> str:
        """What this geometry detects / its mathematical basis."""
        return ""

    @property
    def view(self) -> str:
        """Geometric lens family: distributional, topological, dynamical,
        symmetry, scale, quasicrystal, or other."""
        return "other"

    @property
    def detects(self) -> str:
        """Short phrase: what phenomena this geometry reveals."""
        return ""

    @property
    def encoding_invariant(self) -> bool:
        """True if this geometry's metrics depend only on rank order, not byte values."""
        return False

    @abstractmethod
    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data into this geometry's space."""
        pass

    @abstractmethod
    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute all metrics for this geometry on the given data."""
        pass

    @property
    def atlas_exclude(self) -> set:
        """Metrics to exclude from atlas (kept in compute_metrics for direct use)."""
        return set()

    def metadata(self) -> dict:
        """Export geometry metadata for atlas/viewer."""
        # Derive metric names by running on dummy data
        try:
            dummy = np.arange(256, dtype=np.uint8)
            result = self.compute_metrics(dummy)
            exclude = self.atlas_exclude
            metrics = [m for m in result.metrics.keys() if m not in exclude]
        except Exception:
            metrics = []
        return {
            "class": type(self).__name__,
            "name": self.name,
            "dimension": self.dimension,
            "description": self.description,
            "view": self.view,
            "detects": self.detects,
            "encoding_invariant": self.encoding_invariant,
            "metrics": metrics,
        }

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


def _effective_root_directions(roots: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute the effective root→direction map after per-window centering.

    Per-window centering projects data onto the hyperplane sum=0.  On this
    hyperplane, some roots become indistinguishable (they project to the
    same direction, or to antiparallel directions which abs-dot equates).

    Returns
    -------
    root_to_dir : ndarray of int, shape (n_roots,)
        Maps each root index to its effective direction index.
    n_dirs : int
        Number of unique effective directions (the honest codebook size).
    """
    dim = roots.shape[1]
    ones = np.ones(dim) / np.sqrt(dim)
    projected = roots - np.outer(roots @ ones, ones)
    norms = np.linalg.norm(projected, axis=1)

    root_to_dir = np.full(len(roots), -1, dtype=int)
    dir_reps = []  # representative normalized vectors

    for i in range(len(roots)):
        if norms[i] < 1e-10:
            # Degenerate (e.g. (½,…,½) with sum=±dim/2) — assign to dir 0
            # if no other match found below
            continue
        v = projected[i] / norms[i]
        matched = False
        for d, rep in enumerate(dir_reps):
            if abs(v @ rep) > 0.9999:
                root_to_dir[i] = d
                matched = True
                break
        if not matched:
            root_to_dir[i] = len(dir_reps)
            dir_reps.append(v)

    # Assign degenerate roots to a catch-all direction
    if np.any(root_to_dir == -1):
        catch_all = len(dir_reps)
        dir_reps.append(np.zeros(dim))
        root_to_dir[root_to_dir == -1] = catch_all

    return root_to_dir, len(dir_reps)


def _find_closest_roots_canonical(embedded: np.ndarray,
                                  roots_normalized: np.ndarray):
    """Find closest roots with deterministic tie-breaking.

    Standard argmax on abs-dot products breaks ties by array position,
    making results sensitive to root ordering. This matters for Weyl
    invariance: co-directional roots (±r pairs) always tie, and roots
    from different effective directions can also tie exactly when data
    sits on a Voronoi boundary (e.g. H3 icosahedral has 4-way ties
    between direction pairs for certain RANDU data vectors).

    Fix: add a tiny canonical offset based on the root vector itself
    (via irrational-weighted inner product). Since the offset depends
    only on the root vector, not its array position, the argmax result
    is the same regardless of how the root array is permuted.

    Returns (idx, alignments) matching the find_closest_roots interface.
    """
    dots = embedded @ roots_normalized.T
    abs_dots = np.abs(dots)
    dim = roots_normalized.shape[1]
    _WEIGHTS = np.array([1.0, np.pi, np.e, np.sqrt(2), np.sqrt(3),
                         np.sqrt(5), np.sqrt(7), np.sqrt(11)])[:dim]
    tiebreak = roots_normalized @ _WEIGHTS
    tiebreak = tiebreak / (np.max(np.abs(tiebreak)) + 1e-15) * 1e-10
    idx = np.argmax(abs_dots + tiebreak[None, :], axis=1)
    return idx, abs_dots[np.arange(len(embedded)), idx]


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
        self._root_to_dir = None
        self._n_effective_dirs = None

    @property
    def name(self) -> str:
        return "E8 Lattice"


    @property

    def description(self) -> str:

        return "Projects 8-byte windows onto the 240 roots of E8, the densest lattice sphere-packing in 8 dimensions. Root usage diversity and alignment measure algebraic constraint."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Lattice alignment, algebraic constraint"

    @property
    def dimension(self) -> int:
        return 8

    @property
    def atlas_exclude(self) -> set:
        # e8_structure_score: multiplicative product of components (var 3.3e11)
        # closure_score, vel_corr, trajectory_curvature: Thomson-wins + IQR < 0.06
        # edge_flow: Thomson-wins -104%, anti-structural spread
        return {"e8_structure_score", "closure_score", "vel_corr",
                "trajectory_curvature", "edge_flow"}

    @property
    def roots(self) -> np.ndarray:
        """Lazily compute E8 roots (240 vectors)."""
        if self._roots is None:
            self._roots = self._compute_e8_roots()
            norms = np.linalg.norm(self._roots, axis=1, keepdims=True)
            self._roots_normalized = self._roots / norms
            self._root_to_dir, self._n_effective_dirs = _effective_root_directions(self._roots)
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
        """Embed data as 8-dimensional windows (vectorized)."""
        data = self.validate_data(data)
        n_windows = len(data) // self.window_size

        if n_windows == 0:
            raise ValueError(f"Data too short for window size {self.window_size}")

        windows = data[:n_windows * self.window_size].astype(float).reshape(n_windows, self.window_size)
        if self.normalize:
            windows = windows / (np.max(data) + 1e-10)
            means = windows.mean(axis=1, keepdims=True)
            stds = windows.std(axis=1, keepdims=True) + 1e-10
            windows = (windows - means) / stds

        return windows

    def find_closest_roots(self, embedded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find closest E8 root for each embedded point (vectorized)."""
        dots = embedded @ self.roots_normalized.T
        abs_dots = np.abs(dots)
        best_indices = np.argmax(abs_dots, axis=1)
        best_alignments = np.max(abs_dots, axis=1)
        return best_indices, best_alignments

    def _nearest_e8_lattice_points(self, embedded: np.ndarray):
        """Vectorized nearest-E8-lattice-point via parity-snap.

        E8 = D8 ∪ (D8 + [½,...,½]) where D8 = {x ∈ Z⁸ : Σxᵢ even}.
        For each coset, round to nearest integer/half-integer coords,
        then fix parity by flipping the coordinate with largest rounding
        error. Return whichever coset is closer.

        Returns: (distances, coset_ids) where coset 0=integer, 1=half-integer.
        """
        # Coset 0: integer coordinates with even sum
        f0 = np.round(embedded).copy()
        res0 = np.abs(embedded - f0)
        sums0 = np.sum(f0, axis=1).astype(np.int64)
        odd0 = (sums0 % 2 != 0)
        if np.any(odd0):
            odd_rows = np.where(odd0)[0]
            flip_cols = np.argmax(res0[odd0], axis=1)
            signs = np.sign(embedded[odd_rows, flip_cols] - f0[odd_rows, flip_cols])
            signs[signs == 0] = 1
            f0[odd_rows, flip_cols] += signs
        d0 = np.sum((embedded - f0) ** 2, axis=1)

        # Coset 1: half-integer coordinates with even sum
        f1 = np.floor(embedded) + 0.5
        f1 = f1.copy()
        res1 = np.abs(embedded - f1)
        # sum(f1) = sum(floor(v)) + 4, so parity = parity(sum(floor(v)))
        floor_sums = np.sum(np.floor(embedded), axis=1).astype(np.int64)
        odd1 = (floor_sums % 2 != 0)
        if np.any(odd1):
            odd_rows = np.where(odd1)[0]
            flip_cols = np.argmax(res1[odd1], axis=1)
            signs = np.sign(embedded[odd_rows, flip_cols] - f1[odd_rows, flip_cols])
            signs[signs == 0] = 1
            f1[odd_rows, flip_cols] += signs
        d1 = np.sum((embedded - f1) ** 2, axis=1)

        # Pick closer coset
        use_half = d1 < d0
        distances = np.sqrt(np.where(use_half, d1, d0))
        coset = use_half.astype(int)
        return distances, coset

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute E8 metrics: evolved headline + decomposed components + generic workhorses."""
        embedded = self.embed(data)
        root_indices, alignments = self.find_closest_roots(embedded)

        # Evolved components (ShinkaEvolve v3 gen 21).
        components = self._evolved_structural_components(embedded)

        # Generic root-counting metrics (workhorses for atlas discrimination)
        n_eff = self._n_effective_dirs or len(self.roots_normalized)
        dir_idx = np.array([self._root_to_dir[i] for i in root_indices]) if self._root_to_dir is not None else root_indices
        unique = len(set(dir_idx))
        counts = Counter(dir_idx)
        probs = np.array(list(counts.values())) / len(dir_idx)
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(min(len(dir_idx), n_eff))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "e8_structure_score": components['e8_structure_score'],
                "closure_score": components['closure_score'],
                "edge_flow": components['edge_flow'],
                "vel_corr": components['vel_corr'],
                "std_profile": components['std_profile'],
                "coset_transition": components['coset_transition'],
                "trajectory_curvature": components['trajectory_curvature'],
                "diversity_ratio": unique / n_eff,
                "normalized_entropy": ent / max_ent if max_ent > 0 else 0,
            },
            raw_data={
                "root_indices": root_indices,
                "alignments": alignments,
                "root_counts": dict(Counter(root_indices)),
            }
        )

    def _evolved_structural_components(self, embedded: np.ndarray):
        """E8 structure score (ShinkaEvolve v5 gen 28, D1=87.0%).

        Exploits rotation-invariant algebraic properties of E8: discrete Gram
        values {0, ±0.5, ±1}, lattice closure under root differences,
        nearest-neighbor graph connectivity, coset sparsity preference,
        phase coherence (sign prediction from gram matrix), coset-conditioned
        transition quantization, and trajectory curvature closure.

        Uses signed dot products throughout for higher fidelity.

        Returns dict of individual components AND the combined headline product.
        """
        roots_n = self.roots_normalized
        n_roots = len(roots_n)
        roots_raw = self.roots

        # Signed dot products
        dots = embedded @ roots_n.T
        rows = np.arange(len(embedded))[:, None]

        # --- Intra-window local geometry (top-8 roots) ---
        k_local = min(8, n_roots)
        top_k_idx = np.argpartition(-np.abs(dots), k_local, axis=1)[:, :k_local]
        top_k_align = dots[rows, top_k_idx]
        order = np.argsort(-np.abs(top_k_align), axis=1)
        top_k_idx = np.take_along_axis(top_k_idx, order, axis=1)
        top_k_align = np.take_along_axis(top_k_align, order, axis=1)

        std_profile = float(np.sum(np.std(np.abs(top_k_align), axis=0)))
        U_all = roots_n[top_k_idx]  # (N, k_local, 8)

        # Discrete grammar: distance of local Gram entries to E8's {0, ±0.5, ±1}
        intra_gram = np.matmul(U_all, U_all.transpose(0, 2, 1))
        vals = intra_gram.flatten()
        intra_dp = float(np.mean(np.minimum.reduce([
            np.abs(vals), np.abs(vals - 0.5), np.abs(vals + 0.5),
            np.abs(vals - 1.0), np.abs(vals + 1.0)])))

        # Local eigenvalue variance
        eigvals = np.linalg.eigvalsh(intra_gram)
        eig_var = float(np.mean(np.var(eigvals, axis=0)))

        # --- Lattice closure + edge flow ---
        k_trans = min(4, k_local)
        U = U_all[:-1, :k_trans]
        V = U_all[1:, :k_trans]
        cross_dots = np.sum(
            U[:, :, None, :] * V[:, None, :, :], axis=-1).reshape(-1)
        changes = np.abs(cross_dots) < 0.999

        if not np.any(changes):
            closure_score = 1.0
            edge_flow = 0.0
            inter_dp = 0.0
        else:
            diffs = (V[:, None, :, :] - U[:, :, None, :]).reshape(-1, 8)
            diffs_c = diffs[changes]
            cross_dots_c = cross_dots[changes]

            norms = np.linalg.norm(diffs_c, axis=1, keepdims=True) + 1e-15
            diff_dirs = diffs_c / norms
            max_align = np.max(np.abs(diff_dirs @ roots_n.T), axis=1)
            closure_score = float(max(0.0, np.mean(max_align) - np.std(max_align)))

            abs_cd = np.abs(cross_dots_c)
            inter_dp = float(np.mean(np.minimum.reduce([
                np.abs(abs_cd), np.abs(abs_cd - 0.5), np.abs(abs_cd + 0.5),
                np.abs(abs_cd - 1.0), np.abs(abs_cd + 1.0)])))

            edge_hits = np.abs(abs_cd - 0.5) < 1e-3
            ortho_hits = abs_cd < 1e-3
            edge_flow = float(np.sum(edge_hits)) / (float(np.sum(ortho_hits)) + 1.0)

        # --- Velocity correlation ---
        d_diffs = embedded[1:] - embedded[:-1]
        r_diffs = U_all[1:, 0, :] - U_all[:-1, 0, :]
        d_norms = np.linalg.norm(d_diffs, axis=1, keepdims=True) + 1e-15
        r_norms = np.linalg.norm(r_diffs, axis=1, keepdims=True) + 1e-15
        valid_vel = (r_norms.flatten() > 1e-5) & (d_norms.flatten() > 1e-5)
        if np.any(valid_vel):
            vel_corr = np.sum(
                (d_diffs[valid_vel] / d_norms[valid_vel]) *
                (r_diffs[valid_vel] / r_norms[valid_vel]), axis=1)
            mean_vel_corr = float(np.mean(np.abs(vel_corr)))
        else:
            mean_vel_corr = 0.0

        # --- Coset sparsity boost ---
        l0_norms = np.sum(np.abs(roots_raw) > 1e-5, axis=1)
        mask_sparse = l0_norms <= 4   # integer-coord roots
        mask_dense = l0_norms >= 6    # half-integer roots
        sparsity_boost = 1.0
        if np.sum(mask_sparse) > 10 and np.sum(mask_dense) > 10:
            aligned_energy = np.abs(dots) ** 6
            pref_sparse = np.sum(aligned_energy[:, mask_sparse], axis=1) / np.sum(mask_sparse)
            pref_dense = np.sum(aligned_energy[:, mask_dense], axis=1) / np.sum(mask_dense)
            sparsity_boost = 1.0 + 10.0 * float(np.var(pref_sparse) + np.var(pref_dense))

        # --- Phase coherence boost (v4 gen 29) ---
        # Checks whether signs of secondary alignments are predicted by the
        # gram matrix. E8's discrete gram values {-1,-0.5,0,0.5,1} make sign
        # prediction more reliable than Thomson's continuous gram.
        gram = roots_n @ roots_n.T
        phase_coherence_boost = 1.0
        k_phase = 16
        if n_roots >= k_phase:
            pk_idx = np.argpartition(-np.abs(dots), k_phase, axis=1)[:, :k_phase]
            pk_align = dots[rows, pk_idx]
            pk_order = np.argsort(-np.abs(pk_align), axis=1)
            pk_idx = np.take_along_axis(pk_idx, pk_order, axis=1)
            pk_align = np.take_along_axis(pk_align, pk_order, axis=1)

            pk_signs = np.sign(pk_align)
            primary_idx, primary_sign = pk_idx[:, 0], pk_signs[:, 0]
            secondary_idx, secondary_signs = pk_idx[:, 1:], pk_signs[:, 1:]

            gram_entries = gram[primary_idx[:, None], secondary_idx]
            predicted_signs = primary_sign[:, None] * np.sign(gram_entries)
            valid = np.abs(gram_entries) > 1e-9
            matches = (predicted_signs == secondary_signs)
            weights = np.abs(pk_align[:, 1:])
            weighted_matches = np.sum(matches * valid * weights)
            total_weight = np.sum(valid * weights)
            coherence = weighted_matches / (total_weight + 1e-9)
            phase_coherence_boost = 1.0 + 2.0 * coherence

        # --- Coset-conditioned transition quantization (v5 gen 28) ---
        # E8 roots split into two cosets: sparse (L0≤4, integer-coord) and
        # dense (L0≥6, half-integer). Intra-coset gram values quantize to
        # {0, ±1}, inter-coset to {±0.5}. Thomson has no coset structure.
        coset_transition_score = 0.0
        if np.sum(mask_sparse) > 10 and np.sum(mask_dense) > 10:
            best_indices, _ = self.find_closest_roots(embedded)
            valid_trans = best_indices[:-1] != best_indices[1:]
            if np.sum(valid_trans) > 10:
                from_idx = best_indices[:-1][valid_trans]
                to_idx = best_indices[1:][valid_trans]
                coset_from = mask_dense[from_idx]
                coset_to = mask_dense[to_idx]
                trans_grams = gram[from_idx, to_idx]

                intra_mask = (coset_from == coset_to)
                inter_mask = (coset_from != coset_to)

                q_intra = 0.5
                if np.any(intra_mask):
                    g_intra = trans_grams[intra_mask]
                    q_intra = float(np.mean(np.minimum.reduce([
                        np.abs(g_intra), np.abs(g_intra - 1.0), np.abs(g_intra + 1.0)])))
                q_inter = 0.5
                if np.any(inter_mask):
                    g_inter = trans_grams[inter_mask]
                    q_inter = float(np.mean(np.minimum(
                        np.abs(g_inter - 0.5), np.abs(g_inter + 0.5))))

                total_penalty = 0.5 * (q_intra + q_inter)
                coset_transition_score = float(np.exp(-15.0 * total_penalty))
        coset_transition_boost = 1.0 + 3.0 * coset_transition_score

        # --- Trajectory curvature closure (v5 gen 28) ---
        # Discrete second derivative (acceleration) of root path should align
        # with roots under E8 lattice closure. Thomson has no such constraint.
        trajectory_curvature_score = 0.0
        best_indices, _ = self.find_closest_roots(embedded)
        if len(best_indices) >= 3:
            idx_prev = best_indices[:-2]
            idx_curr = best_indices[1:-1]
            idx_next = best_indices[2:]
            is_turn = (idx_prev != idx_curr) & (idx_curr != idx_next)
            if np.sum(is_turn) > 5:
                r_prev = roots_n[idx_prev[is_turn]]
                r_curr = roots_n[idx_curr[is_turn]]
                r_next = roots_n[idx_next[is_turn]]
                accel = r_next - 2.0 * r_curr + r_prev
                anorms = np.linalg.norm(accel, axis=1)
                valid_accel = anorms > 1e-9
                if np.sum(valid_accel) > 0:
                    accel_dirs = accel[valid_accel] / anorms[valid_accel, None]
                    max_aligns = np.max(np.abs(accel_dirs @ roots_n.T), axis=1)
                    trajectory_curvature_score = float(np.mean(max_aligns))
        curvature_boost = 1.0 + 1.5 * trajectory_curvature_score

        # --- Combined structure score ---
        structure_factor = 1.0 / (1.0 + 5.0 * eig_var
                                  + 20.0 * intra_dp + 20.0 * inter_dp)
        lattice_boost = 1.0 + 2.0 * edge_flow
        e8_structure_score = float(
            std_profile * closure_score * structure_factor
            * lattice_boost * sparsity_boost
            * (1.0 + mean_vel_corr) * phase_coherence_boost
            * coset_transition_boost * curvature_boost)

        return {
            'e8_structure_score': e8_structure_score,
            'closure_score': closure_score,
            'edge_flow': edge_flow,
            'vel_corr': mean_vel_corr,
            'std_profile': std_profile,
            'coset_transition': coset_transition_score,
            'trajectory_curvature': trajectory_curvature_score,
        }


# =============================================================================
# G2 ROOT SYSTEM GEOMETRY
# =============================================================================

def _perm_sign(perm):
    """Return +1 for even permutation, -1 for odd."""
    perm = list(perm)
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if not visited[i]:
            j, cycle_len = i, 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len % 2 == 0:
                sign *= -1
    return sign


class G2Geometry(ExoticGeometry):
    """
    G2 Root System — hexagonal byte-pair structure.

    12 roots in 2D: 6 short at 60° intervals, 6 long at 30° offsets (length √3).
    Window size 2 probes consecutive byte-pair correlations — complementary to
    E8's 8-byte windows.
    """

    def __init__(self, window_size: int = 2, normalize: bool = True):
        self.window_size = window_size
        self.normalize = normalize
        self._roots = None
        self._roots_normalized = None

    @property
    def name(self) -> str:
        return "G2 Root System"

    @property
    def description(self) -> str:
        return "Projects byte pairs onto the 12 roots of G2. Detects hexagonal symmetry in consecutive-byte correlations."

    @property
    def view(self) -> str:
        return "symmetry"

    @property
    def detects(self) -> str:
        return "Hexagonal byte-pair symmetry"

    @property
    def dimension(self) -> int:
        return 2

    @property
    def atlas_exclude(self) -> set:
        # diversity_ratio: IQR 0.13, 12 roots saturate quickly
        # kurtosis_raw: r=0.995 with kurtosis_angular, lower F-stat
        return {"diversity_ratio", "kurtosis_raw"}

    @property
    def roots(self) -> np.ndarray:
        if self._roots is None:
            self._roots = self._compute_roots()
            norms = np.linalg.norm(self._roots, axis=1, keepdims=True)
            self._roots_normalized = self._roots / norms
        return self._roots

    @property
    def roots_normalized(self) -> np.ndarray:
        if self._roots_normalized is None:
            _ = self.roots
        return self._roots_normalized

    def _compute_roots(self) -> np.ndarray:
        """12 roots of G2: 6 short + 6 long."""
        roots = []
        for k in range(6):
            a = k * np.pi / 3
            roots.append([np.cos(a), np.sin(a)])
        for k in range(6):
            a = np.pi / 6 + k * np.pi / 3
            r = np.sqrt(3)
            roots.append([r * np.cos(a), r * np.sin(a)])
        return np.array(roots)

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        n = len(data) // 2
        if n == 0:
            raise ValueError("Data too short for window size 2")
        w = data[:n * 2].astype(float).reshape(n, 2)
        if self.normalize:
            # Global normalization — per-window centering collapses 2D to ±1
            mean = np.mean(w)
            std = np.std(w) + 1e-10
            w = (w - mean) / std
        return w

    def find_closest_roots(self, embedded):
        """Closest root by angle (normalized). Returns (indices, alignments)."""
        dots = embedded @ self.roots_normalized.T
        abs_dots = np.abs(dots)
        return np.argmax(abs_dots, axis=1), np.max(abs_dots, axis=1)

    def find_closest_roots_raw(self, embedded):
        """Closest root by raw dot product (length-weighted). Returns (indices, values)."""
        dots = embedded @ self.roots.T
        abs_dots = np.abs(dots)
        return np.argmax(abs_dots, axis=1), np.max(abs_dots, axis=1)

    @staticmethod
    def _fisher_kurtosis(vals):
        """Fisher's kurtosis (excess kurtosis; normal = 0)."""
        std = np.std(vals)
        if std < 1e-9:
            return 0.0
        centered = (vals - np.mean(vals)) / std
        return float(np.mean(centered**4) - 3.0)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        embedded = self.embed(data)
        idx, aligns = self.find_closest_roots(embedded)
        raw_idx, raw_vals = self.find_closest_roots_raw(embedded)
        n_roots = 12
        unique = len(set(idx))
        counts = Counter(idx)
        probs = np.array(list(counts.values())) / len(idx)
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(min(len(idx), n_roots))
        short = sum(counts.get(i, 0) for i in range(6))

        # Kurtosis differential (evolved via ShinkaEvolve v1 gen 36, D1=0.977)
        # G2's bimodal root lengths make raw alignments bimodal (negative kurtosis)
        # while angular alignments are unimodal. Thomson has raw ≈ angular.
        kurt_angular = self._fisher_kurtosis(aligns)
        kurt_raw = self._fisher_kurtosis(raw_vals)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "diversity_ratio": unique / n_roots,
                "alignment_mean": float(np.mean(aligns)),
                "alignment_std": float(np.std(aligns)),
                "normalized_entropy": ent / max_ent if max_ent > 0 else 0,
                "short_long_ratio": short / (len(idx) + 1e-10),
                "kurtosis_angular": kurt_angular,
                "kurtosis_raw": kurt_raw,
                "kurtosis_differential": kurt_angular - kurt_raw,
            },
            raw_data={"root_indices": idx, "alignments": aligns,
                      "raw_indices": raw_idx, "raw_values": raw_vals}
        )


# =============================================================================
# D4 COXETER GEOMETRY (TRIALITY)
# =============================================================================

class D4Geometry(ExoticGeometry):
    """
    D4 Root System — 4-byte structural symmetry.

    24 roots (±eᵢ ± eⱼ) in 4D. D4 is the only Lie algebra with triality:
    an order-3 outer automorphism permuting vector, spinor+, and spinor−
    representations of Spin(8). Six rotation-invariant structural probes
    (ShinkaEvolve v5b) detect D4's edge graph, spectral eigenspaces, and
    lattice closure properties.
    """

    def __init__(self, window_size: int = 4, normalize: bool = True):
        self.window_size = window_size
        self.normalize = normalize
        self._roots = None
        self._roots_normalized = None
        self._root_to_dir = None
        self._n_effective_dirs = None
        self._is_edge = None
        self._is_ortho = None
        self._edge_degree = None

    @property
    def name(self) -> str:
        return "D4 Triality"

    @property
    def description(self) -> str:
        return "Projects 4-byte windows onto the 24 roots of D4 and measures structural probes — edge graph, spectral coherence, lattice closure."

    @property
    def view(self) -> str:
        return "symmetry"

    @property
    def detects(self) -> str:
        return "D4 lattice structure, 4-byte symmetry constraints"

    @property
    def dimension(self) -> int:
        return 4

    @property
    def atlas_exclude(self) -> set:
        # d4_structure_score: additive headline of components (var 1.95)
        return {"d4_structure_score"}

    @property
    def roots(self) -> np.ndarray:
        if self._roots is None:
            self._roots = self._compute_roots()
            norms = np.linalg.norm(self._roots, axis=1, keepdims=True)
            self._roots_normalized = self._roots / norms
            self._root_to_dir, self._n_effective_dirs = _effective_root_directions(self._roots)
            gram = self._roots_normalized @ self._roots_normalized.T
            self._is_edge = np.abs(np.abs(gram) - 0.5) < 1e-4
            self._is_ortho = np.abs(gram) < 1e-4
            self._edge_degree = np.sum(self._is_edge, axis=1)
        return self._roots

    @property
    def roots_normalized(self) -> np.ndarray:
        if self._roots_normalized is None:
            _ = self.roots
        return self._roots_normalized

    def _compute_roots(self) -> np.ndarray:
        """D4 roots: ±eᵢ ± eⱼ for i < j. 24 roots in 4D."""
        roots = []
        for i, j in combinations(range(4), 2):
            for si, sj in product([1, -1], repeat=2):
                r = np.zeros(4)
                r[i] = si
                r[j] = sj
                roots.append(r)
        return np.array(roots)

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        n = len(data) // 4
        if n == 0:
            raise ValueError("Data too short for window size 4")
        w = data[:n * 4].astype(float).reshape(n, 4)
        if self.normalize:
            w = w / (np.max(data) + 1e-10)
            means = w.mean(axis=1, keepdims=True)
            stds = w.std(axis=1, keepdims=True) + 1e-10
            w = (w - means) / stds
        return w

    def find_closest_roots(self, embedded):
        dots = embedded @ self.roots_normalized.T
        abs_dots = np.abs(dots)
        return np.argmax(abs_dots, axis=1), np.max(abs_dots, axis=1)

    def _spectral_transition(self, idx):
        """Spectral transition metric (ShinkaEvolve atlas v1 gen 6, score=1.184).

        Builds root-to-root transition matrix from data, projects each row onto
        D4's edge-adjacency eigenspaces {-8, 0, 16}. Returns weighted mean of
        (E_16 - E_-8) / E_total — whether transitions align with D4's clustering
        vs anti-clustering eigenspace.
        """
        n_roots = len(self.roots_normalized)
        adj = self._is_edge.astype(np.float64)

        # Edge adjacency eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
        except np.linalg.LinAlgError:
            return 0.0

        projectors = {}
        for val in [-8.0, 0.0, 16.0]:
            mask = np.isclose(eigvals, val, atol=1e-3)
            if np.any(mask):
                vecs = eigvecs[:, mask]
                projectors[val] = vecs @ vecs.T
            else:
                projectors[val] = np.zeros((n_roots, n_roots))

        # Transition matrix
        T = np.zeros((n_roots, n_roots), dtype=np.float64)
        np.add.at(T, (idx[:-1], idx[1:]), 1)
        row_sums = T.sum(axis=1)
        active = np.where(row_sums > 0)[0]
        if len(active) == 0:
            return 0.0

        metrics = []
        weights = []
        for i in active:
            tv = T[i, :] / row_sums[i]
            e16 = tv @ projectors[16.0] @ tv
            e0 = tv @ projectors[0.0] @ tv
            en8 = tv @ projectors[-8.0] @ tv
            total = e16 + e0 + en8
            if total > 1e-9:
                metrics.append((e16 - en8) / total)
                weights.append(row_sums[i])

        if not metrics:
            return 0.0
        return float(np.average(metrics, weights=weights))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """D4 metrics: structural probes + spectral transition + generic workhorses."""
        embedded = self.embed(data)
        idx, aligns = self.find_closest_roots(embedded)
        _ = self.roots  # ensure edge/ortho computed

        components = self._structural_probes(embedded, idx, aligns)

        # Spectral transition (evolved atlas metric)
        spectral_trans = self._spectral_transition(idx)

        # Generic workhorses
        n_eff = self._n_effective_dirs or len(self.roots_normalized)
        dir_idx = np.array([self._root_to_dir[i] for i in idx]) if self._root_to_dir is not None else idx
        unique = len(set(dir_idx))
        counts = Counter(dir_idx)
        probs = np.array(list(counts.values())) / len(dir_idx)
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(min(len(dir_idx), n_eff))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "d4_structure_score": components['d4_structure_score'],
                "triplet_temporal": components['triplet_temporal'],
                "neighborhood_asymmetry": components['neighborhood_asymmetry'],
                "structural_coherence": components['structural_coherence'],
                "spectral_transition": spectral_trans,
                "diversity_ratio": unique / n_eff,
                "normalized_entropy": ent / max_ent if max_ent > 0 else 0,
            },
            raw_data={"root_indices": idx, "alignments": aligns}
        )

    def _structural_probes(self, embedded, best_idx, best_align) -> dict:
        """Six rotation-invariant structural probes (ShinkaEvolve v5b).

        Returns individual components AND the additive headline.
        """
        n_windows = len(embedded)
        n_roots = len(self.roots_normalized)
        zero = {'d4_structure_score': 0.0, 'triplet_temporal': 0.0,
                'neighborhood_asymmetry': 0.0,
                'structural_coherence': 0.0, 'spectral_coherence': 0.0}

        # Recompute edge/ortho structure from current roots (may differ from
        # __init__ if roots were swapped by ablation variant factory)
        gram = self.roots_normalized @ self.roots_normalized.T
        is_edge = np.abs(np.abs(gram) - 0.5) < 1e-4
        is_ortho = np.abs(gram) < 1e-4
        edge_degree = np.sum(is_edge, axis=1)

        if n_windows < 10 or np.mean(edge_degree) < 1.0:
            return zero

        dots = np.abs(embedded @ self.roots_normalized.T)

        # --- TEMPORAL: triplet graph traversal ---
        temporal = 0.0
        if n_windows >= 3:
            idx_A, idx_B, idx_C = best_idx[:-2], best_idx[1:-1], best_idx[2:]
            is_step1 = is_edge[idx_A, idx_B]
            is_step2 = is_edge[idx_B, idx_C]
            is_walk = is_step1 & is_step2
            scores = np.zeros(len(idx_A))
            scores[is_walk & is_edge[idx_A, idx_C]] = 2.0
            scores[is_walk & is_ortho[idx_A, idx_C]] = 1.0
            scores[is_walk & (idx_A == idx_C)] = -2.0
            weights = np.minimum.reduce([best_align[:-2], best_align[1:-1], best_align[2:]])
            temporal = float(np.sum(scores * weights) / (np.sum(weights) + 1e-9))

        # --- STATIC: top-k fuzzy neighborhood asymmetry ---
        k = min(3, n_roots)
        top_k_idx = np.argpartition(-dots, k, axis=1)[:, :k]
        rows = np.arange(n_windows)[:, None]
        top_k_align = dots[rows, top_k_idx]
        order = np.argsort(-top_k_align, axis=1)
        top_k_idx = np.take_along_axis(top_k_idx, order, axis=1)
        top_k_align = np.take_along_axis(top_k_align, order, axis=1)

        coh_energy = dots @ is_edge.astype(float).T
        incoh_energy = dots @ is_ortho.astype(float).T
        tk_coh = coh_energy[rows, top_k_idx]
        tk_incoh = incoh_energy[rows, top_k_idx]
        asym = (tk_coh - tk_incoh) / (tk_coh + tk_incoh + 1e-9)
        fuzzy = np.sum(top_k_align * asym, axis=1) / (np.sum(top_k_align, axis=1) + 1e-9)
        static = float(np.mean(fuzzy))

        # --- PEAKEDNESS: alignment vector kurtosis ---
        l2_sq = np.sum(dots**2, axis=1)
        l4 = np.sum(dots**4, axis=1)
        pr = l4 / (l2_sq**2 + 1e-9)
        pk_scores = pr * n_roots - 1.0
        peakedness = float(np.sum(np.maximum(0, pk_scores) * l2_sq) / (np.sum(l2_sq) + 1e-9))

        # --- ALIGNMENT VARIANCE ---
        align_var = float(np.std(best_align) * np.sqrt(n_roots))

        # --- STRUCTURAL COHERENCE: covariance vs Gram structure ---
        signed_dots = embedded @ self.roots_normalized.T
        centered = signed_dots - np.mean(signed_dots, axis=0, keepdims=True)
        cov = (centered.T @ centered) / n_windows
        gram = self.roots_normalized @ self.roots_normalized.T
        is_anti = np.isclose(gram, -1.0)
        is_struct = is_edge | is_anti
        struct_num = np.sum(gram * cov * is_struct)
        struct_den = np.sum(is_struct) + 1e-9
        ortho_incoh = np.sum(np.abs(cov * is_ortho)) / (np.sum(is_ortho) + 1e-9)
        struct_coh = float((struct_num / struct_den) / (ortho_incoh + 1e-9))

        # --- SPECTRAL COHERENCE: eigenspace energy ratio ---
        spectral = 0.0
        adj = is_edge.astype(float)
        if n_roots > 20 and np.all(edge_degree == 16):
            try:
                eigvals, eigvecs = np.linalg.eigh(adj)
                coeffs = centered @ eigvecs
                energy = np.mean(coeffs**2, axis=0)
                neg_space = np.isclose(eigvals, -8.0, atol=1e-3)
                pos_space = np.isclose(eigvals, 16.0, atol=1e-3)
                zero_space = np.isclose(eigvals, 0.0, atol=1e-3)
                e_struct = np.sum(energy[neg_space]) + np.sum(energy[pos_space])
                e_unstruct = np.sum(energy[zero_space])
                spectral = float(e_struct / (e_unstruct + 1e-9))
            except np.linalg.LinAlgError:
                pass

        headline = temporal + static + peakedness + align_var + struct_coh + spectral

        return {
            'd4_structure_score': float(headline),
            'triplet_temporal': temporal,
            'neighborhood_asymmetry': static,
            'structural_coherence': struct_coh,
            'spectral_coherence': spectral,
        }


# =============================================================================
# H3 COXETER GEOMETRY (ICOSAHEDRAL)
# =============================================================================

class H3CoxeterGeometry(ExoticGeometry):
    """
    H3 Coxeter root system — icosahedral symmetry in 3-byte windows.

    30 roots in 3D: the vertices of the icosidodecahedron. H3 is the
    symmetry group of the icosahedron/dodecahedron, featuring 5-fold
    rotational symmetry not achievable by any crystallographic group.
    """

    def __init__(self, window_size: int = 3, normalize: bool = True):
        self.window_size = window_size
        self.normalize = normalize
        self._roots = None
        self._roots_normalized = None
        self._root_to_dir = None
        self._n_effective_dirs = None

    @property
    def name(self) -> str:
        return "H3 Icosahedral"

    @property
    def description(self) -> str:
        return "Projects 3-byte windows onto the 30 roots of H3 (icosidodecahedron vertices). Detects non-crystallographic 5-fold symmetry."

    @property
    def view(self) -> str:
        return "symmetry"

    @property
    def detects(self) -> str:
        return "Icosahedral symmetry, 5-fold rotational order"

    @property
    def dimension(self) -> int:
        return 3

    @property
    def atlas_exclude(self) -> set:
        # diversity_ratio: IQR 0.03, 12 effective dirs saturate for most sources
        # path_closure: r>0.9 with nn_enrichment (higher F), redundant
        return {"diversity_ratio", "path_closure"}

    @property
    def roots(self) -> np.ndarray:
        if self._roots is None:
            self._roots = self._compute_roots()
            self._roots_normalized = self._roots.copy()  # already unit length
            self._root_to_dir, self._n_effective_dirs = _effective_root_directions(self._roots)
        return self._roots

    @property
    def roots_normalized(self) -> np.ndarray:
        if self._roots_normalized is None:
            _ = self.roots
        return self._roots_normalized

    def _compute_roots(self) -> np.ndarray:
        """30 roots of H3: icosidodecahedron vertices (all unit length)."""
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        roots = []
        # 6 axis-aligned: permutations of (±1, 0, 0)
        for i in range(3):
            for s in [1, -1]:
                r = [0.0, 0.0, 0.0]
                r[i] = s
                roots.append(r)
        # 24 golden: even permutations of (±½, ±φ/2, ±1/(2φ))
        base = [0.5, phi / 2, inv_phi / 2]
        even_perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        for p in even_perms:
            for signs in product([1, -1], repeat=3):
                roots.append([signs[k] * base[p[k]] for k in range(3)])
        return np.array(roots)

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        n = len(data) // 3
        if n == 0:
            raise ValueError("Data too short for window size 3")
        w = data[:n * 3].astype(float).reshape(n, 3)
        if self.normalize:
            w = w / (np.max(data) + 1e-10)
            means = w.mean(axis=1, keepdims=True)
            stds = w.std(axis=1, keepdims=True) + 1e-10
            w = (w - means) / stds
        return w

    def find_closest_roots(self, embedded):
        return _find_closest_roots_canonical(embedded, self.roots_normalized)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        embedded = self.embed(data)
        idx, aligns = self.find_closest_roots(embedded)

        # Decomposed components + headline product
        components = self._nn_enrichment_components(embedded, idx)

        # Generic root-counting metrics (workhorses for atlas discrimination)
        n_eff = self._n_effective_dirs or len(self.roots_normalized)
        dir_idx = np.array([self._root_to_dir[i] for i in idx]) if self._root_to_dir is not None else idx
        unique = len(set(dir_idx))
        counts = Counter(dir_idx)
        probs = np.array(list(counts.values())) / len(dir_idx)
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(min(len(dir_idx), n_eff))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "nn_enrichment": components['nn_enrichment'],
                "temporal_coherence": components['temporal_coherence'],
                "path_closure": components['path_closure'],
                "diversity_ratio": unique / n_eff,
                "normalized_entropy": ent / max_ent if max_ent > 0 else 0,
            },
            raw_data={"root_indices": idx, "alignments": aligns}
        )

    def _nn_enrichment_components(self, embedded, best_idx) -> dict:
        """Icosahedral coherence: anisotropic graph dynamics.

        Rotation-invariant metric (ShinkaEvolve v3 gen 44, D1=0.918).
        Returns individual components AND the combined headline product.
        """
        n_roots = len(self.roots_normalized)
        zero = {'nn_enrichment': 0.0, 'gram_fingerprint': 0.0,
                'temporal_coherence': 0.0, 'path_closure': 0.0,
                'alignment_anisotropy': 0.0}
        if len(best_idx) < 10:
            return zero

        phi = (1 + np.sqrt(5)) / 2.0
        gram = self.roots_normalized @ self.roots_normalized.T

        # Multi-lag gram fingerprint (geometric mean of lag-1 and lag-2).
        h3_canonical = np.array([0.0, 1.0 / (2 * phi), 0.5, phi / 2.0, 1.0])
        grams_l1 = np.abs(gram[best_idx[:-1], best_idx[1:]])
        err_l1 = np.mean(np.min(np.abs(grams_l1[:, None] - h3_canonical), axis=1))
        score_l1 = np.exp(-50.0 * err_l1)
        if len(best_idx) >= 3:
            grams_l2 = np.abs(gram[best_idx[:-2], best_idx[2:]])
            err_l2 = np.mean(np.min(np.abs(grams_l2[:, None] - h3_canonical), axis=1))
            score_l2 = np.exp(-50.0 * err_l2)
            gram_score = np.sqrt(score_l1 * score_l2)
        else:
            gram_score = score_l1

        # Adaptive adjacency + temporal coherence + path closure.
        if gram_score > 0.5:
            adj = np.isclose(np.abs(gram), 0.5, atol=1e-4)
        else:
            k_nn = min(4, n_roots - 1)
            adj = np.zeros_like(gram, dtype=bool)
            gram_abs = np.abs(gram.copy())
            np.fill_diagonal(gram_abs, -np.inf)
            top_idx = np.argpartition(-gram_abs, k_nn, axis=1)[:, :k_nn]
            rows_arr = np.arange(n_roots)[:, None]
            adj[rows_arr, top_idx] = True
        np.fill_diagonal(adj, False)

        k_avg = np.mean(np.sum(adj, axis=1))
        baseline = k_avg / (n_roots - 1.0) if n_roots > 1 else 0
        temporal = max(0.0, np.mean(adj[best_idx[:-1], best_idx[1:]]) - baseline)
        closure = max(0.0, np.mean(adj[best_idx[:-2], best_idx[2:]]) - baseline)

        # Alignment anisotropy: variance of a2/a1 ratios.
        dots = np.abs(embedded @ self.roots_normalized.T)
        k = min(2, dots.shape[1])
        if k < 2:
            anisotropy = 0.0
        else:
            top2_idx = np.argpartition(-dots, k, axis=1)[:, :k]
            rows_arr = np.arange(len(embedded))[:, None]
            top2_align = dots[rows_arr, top2_idx]
            a1 = np.max(top2_align, axis=1)
            a2 = np.min(top2_align, axis=1)
            valid = a1 > 1e-7
            anisotropy = float(np.var(a2[valid] / a1[valid])) if np.any(valid) else 0.0

        base_signal = temporal + 0.5 * closure + 2.5 * anisotropy
        headline = float(base_signal * (1.0 + 6.0 * gram_score))

        return {
            'nn_enrichment': headline,
            'gram_fingerprint': float(gram_score),
            'temporal_coherence': float(temporal),
            'path_closure': float(closure),
            'alignment_anisotropy': float(anisotropy),
        }


# =============================================================================
# H4 COXETER GEOMETRY (600-CELL)
# =============================================================================

class H4CoxeterGeometry(ExoticGeometry):
    """
    H4 Coxeter root system — 600-cell symmetry in 4-byte windows.

    120 roots in 4D: the vertices of a 600-cell. H4 is the largest
    non-crystallographic Coxeter group, governing the symmetry of
    4D polytopes with icosahedral cross-sections.
    """

    def __init__(self, window_size: int = 4, normalize: bool = True):
        self.window_size = window_size
        self.normalize = normalize
        self._roots = None
        self._roots_normalized = None
        self._root_to_dir = None
        self._n_effective_dirs = None
        self._is_edge = None

    @property
    def name(self) -> str:
        return "H4 600-Cell"

    @property
    def description(self) -> str:
        return "Projects 4-byte windows onto the 120 roots of H4 (600-cell vertices). Detects non-crystallographic symmetry in 4D."

    @property
    def view(self) -> str:
        return "symmetry"

    @property
    def detects(self) -> str:
        return "600-cell alignment, non-crystallographic 4D symmetry"

    @property
    def atlas_exclude(self) -> set:
        # closure_fidelity: r=0.99 with lattice_closure, redundant
        # mean_walk_length: F=0.43, effectively dead
        return {"closure_fidelity", "mean_walk_length"}

    @property
    def dimension(self) -> int:
        return 4

    @property
    def roots(self) -> np.ndarray:
        if self._roots is None:
            self._roots = self._compute_roots()
            self._roots_normalized = self._roots.copy()  # already unit length
            self._root_to_dir, self._n_effective_dirs = _effective_root_directions(self._roots)
        return self._roots

    @property
    def roots_normalized(self) -> np.ndarray:
        if self._roots_normalized is None:
            _ = self.roots
        return self._roots_normalized

    @property
    def is_edge(self) -> np.ndarray:
        """600-cell edge adjacency: |gram| ≈ φ/2 ≈ 0.809."""
        if self._is_edge is None:
            phi = (1 + np.sqrt(5)) / 2
            gram = self.roots_normalized @ self.roots_normalized.T
            self._is_edge = np.abs(np.abs(gram) - phi / 2) < 1e-4
        return self._is_edge

    def _compute_roots(self) -> np.ndarray:
        """120 roots of H4: vertices of the 600-cell (all unit length)."""
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        roots = []
        # 8 axis-aligned: permutations of (±1, 0, 0, 0)
        for i in range(4):
            for s in [1, -1]:
                r = [0.0, 0.0, 0.0, 0.0]
                r[i] = s
                roots.append(r)
        # 16 half-integer: all (±½, ±½, ±½, ±½)
        for signs in product([0.5, -0.5], repeat=4):
            roots.append(list(signs))
        # 96 golden: even permutations of (0, ±½, ±1/(2φ), ±φ/2)
        base_vals = [0.0, 0.5, inv_phi / 2, phi / 2]
        all_perms = list(permutations(range(4)))
        even_perms = [p for p in all_perms if _perm_sign(p) == 1]
        for p in even_perms:
            perm_vals = [base_vals[p[k]] for k in range(4)]
            nz = [i for i in range(4) if perm_vals[i] != 0.0]
            for signs in product([1, -1], repeat=len(nz)):
                r = list(perm_vals)
                for si, pos in enumerate(signs):
                    r[nz[si]] *= pos
                roots.append(r)
        return np.array(roots)

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        n = len(data) // 4
        if n == 0:
            raise ValueError("Data too short for window size 4")
        w = data[:n * 4].astype(float).reshape(n, 4)
        if self.normalize:
            w = w / (np.max(data) + 1e-10)
            means = w.mean(axis=1, keepdims=True)
            stds = w.std(axis=1, keepdims=True) + 1e-10
            w = (w - means) / stds
        return w

    def find_closest_roots(self, embedded):
        return _find_closest_roots_canonical(embedded, self.roots_normalized)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        embedded = self.embed(data)
        idx, aligns = self.find_closest_roots(embedded)

        # Decomposed components + headline product
        components = self._lattice_closure_components(embedded, idx)

        # Generic root-counting metrics (workhorses for atlas discrimination)
        n_eff = self._n_effective_dirs or len(self.roots_normalized)
        dir_idx = np.array([self._root_to_dir[i] for i in idx]) if self._root_to_dir is not None else idx
        unique = len(set(dir_idx))
        counts = Counter(dir_idx)
        probs = np.array(list(counts.values())) / len(dir_idx)
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(min(len(dir_idx), n_eff))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "lattice_closure": components['lattice_closure'],
                "edge_walk_fraction": components['edge_walk_fraction'],
                "mean_walk_length": components['mean_walk_length'],
                "closure_fidelity": components['closure_fidelity'],
                "diversity_ratio": unique / n_eff,
                "normalized_entropy": ent / max_ent if max_ent > 0 else 0,
            },
            raw_data={"root_indices": idx, "alignments": aligns}
        )

    def _lattice_closure_components(self, embedded, best_idx) -> dict:
        """Walk-length-weighted closure fidelity (ShinkaEvolve v3 gen 40).

        Combines 600-cell edge walks with lattice closure (2I group property),
        power-weighted by walk length². Rotation-invariant.
        Returns individual components AND the combined headline.
        """
        zero = {'lattice_closure': 0.0, 'edge_walk_fraction': 0.0,
                'mean_walk_length': 0.0, 'closure_fidelity': 0.0}
        if len(best_idx) < 10:
            return zero

        # Quantize to signed roots
        dots = embedded @ self.roots_normalized.T
        rows = np.arange(len(embedded))
        signs = np.sign(dots[rows, best_idx])
        signed_roots = self.roots_normalized[best_idx] * signs[:, None]

        # Identify edge transitions and contiguous walks
        edge_transitions = self.is_edge[best_idx[:-1], best_idx[1:]]
        edge_frac = float(np.mean(edge_transitions)) if len(edge_transitions) > 0 else 0.0
        if not np.any(edge_transitions):
            return {**zero, 'edge_walk_fraction': edge_frac}

        padded_edges = np.concatenate(([False], edge_transitions, [False]))
        diffs = np.diff(padded_edges.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(starts) == 0:
            return {**zero, 'edge_walk_fraction': edge_frac}

        walk_lengths = []
        walk_closures = []

        # Closure fidelity per walk
        for s, e in zip(starts, ends):
            v1_walk = signed_roots[s:e]
            v2_walk = signed_roots[s + 1:e + 1]

            diff_vecs = v1_walk - v2_walk
            norms = np.linalg.norm(diff_vecs, axis=1, keepdims=True)

            valid_mask = (norms > 1e-9).flatten()
            if not np.any(valid_mask):
                continue

            diff_vecs_norm = diff_vecs[valid_mask] / norms[valid_mask]
            closure_dots = np.abs(diff_vecs_norm @ self.roots_normalized.T)
            avg_walk_closure = np.mean(np.max(closure_dots, axis=1))

            walk_lengths.append(e - s)
            walk_closures.append(avg_walk_closure)

        if not walk_lengths:
            return {**zero, 'edge_walk_fraction': edge_frac}

        walk_lengths = np.array(walk_lengths, dtype=np.float64)
        walk_closures = np.array(walk_closures, dtype=np.float64)

        weights = np.power(walk_lengths, 2.0)
        headline = float(np.sum(weights * walk_closures) / (np.sum(weights) + 1e-9))
        mean_wl = float(np.mean(walk_lengths))
        mean_cf = float(np.mean(walk_closures))

        return {
            'lattice_closure': headline,
            'edge_walk_fraction': edge_frac,
            'mean_walk_length': mean_wl,
            'closure_fidelity': mean_cf,
        }


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

    def description(self) -> str:

        return "Maps consecutive byte pairs to a flat torus (quotient R^2/Z^2). Coverage and nearest-neighbor distance reveal periodic or constrained structure."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Periodicity, cyclic coverage, uniformity"

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
        """Compute torus metrics: coverage, entropy, uniformity, toroidal distances."""
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


        # Toroidal mean nearest-neighbor distance (wrap-aware)
        # Subsample if large to keep O(n²) tractable
        pts = embedded
        if len(pts) > 500:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), 500, replace=False)
            pts = pts[idx]
        if len(pts) > 1:
            diff = np.abs(pts[:, np.newaxis, :] - pts[np.newaxis, :, :])
            diff = np.minimum(diff, 1.0 - diff)
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            nn_distances = dist_matrix.min(axis=1)
            toroidal_mean_nn = float(np.mean(nn_distances))
        else:
            toroidal_mean_nn = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "coverage": coverage,
                "entropy": entropy,
                "toroidal_mean_nn_distance": toroidal_mean_nn,
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

    def description(self) -> str:

        return "Embeds data in the Poincare disk, where distances grow exponentially near the boundary. Hierarchical or tree-like data clusters near the edge."


    @property

    def view(self) -> str:

        return "topological"


    @property

    def detects(self) -> str:

        return "Hierarchy depth, branching, boundary clustering"

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
        """Compute hyperbolic curvature metrics: headline + decomposed components.

        Three-component evolved metric (ShinkaEvolve v5b gen 24, D1=0.839):
        1. Temporal variance: 1 + var(consecutive distances).
        2. Spatio-temporal correlation: exp(corr(|z_i|, d(z_i, z_{i+1}))).
        3. Spatial kNN ratio: global/local distance scale ratio.

        Plus: mean_hyperbolic_radius (pre-evolution metric, known workhorse).
        """
        embedded = self.embed(data)

        zero_metrics = {
            "curvature_structure": 0.0,
            "temporal_variance": 0.0,
            "spatio_temporal_corr": 0.0,
            "knn_scale_ratio": 0.0,
            "mean_hyperbolic_radius": 0.0,
        }
        if len(embedded) < 30:
            return GeometryResult(
                geometry_name=self.name,
                metrics=zero_metrics,
                raw_data={"embedded_points": embedded}
            )

        # Vectorized hyperbolic distance for complex arrays
        def _hyp_dist(z1, z2):
            diff = np.abs(z1 - z2)
            denom = np.abs(1.0 - np.conj(z1) * z2)
            ratio = np.clip(diff / (denom + 1e-15), 0, 0.9999)
            return 2.0 * np.arctanh(ratio)

        z_all = embedded[:, 0] + 1j * embedded[:, 1]

        # Mean hyperbolic radius (generic workhorse — pre-evolution metric)
        radii_all = np.abs(z_all)
        hyp_radii = 2.0 * np.arctanh(np.clip(radii_all, 0, 0.9999))
        mean_hyp_radius = float(np.mean(hyp_radii))

        # --- Component 1: Temporal variance ---
        dists_seq = _hyp_dist(z_all[:-1], z_all[1:])
        temporal_var = float(np.var(dists_seq)) if len(dists_seq) >= 2 else 0.0
        temporal_score = 1.0 + temporal_var

        # --- Component 2: Spatio-temporal correlation ---
        spatio_temporal_corr = 0.0
        spatio_temporal_score = 1.0
        if len(dists_seq) > 10:
            radii = np.abs(z_all[:-1])
            if np.std(radii) > 1e-9 and np.std(dists_seq) > 1e-9:
                spatio_temporal_corr = float(np.corrcoef(radii, dists_seq)[0, 1])
                spatio_temporal_score = float(np.exp(spatio_temporal_corr))

        # --- Component 3: Spatial kNN ratio ---
        z_unique = np.unique(z_all)
        n_u = len(z_unique)
        knn_ratio = 1.0
        if n_u >= 30:
            if n_u > 300:
                u_rng = np.random.default_rng(42)
                z_unique = z_unique[u_rng.choice(n_u, 300, replace=False)]
                n_u = 300
            dist_matrix = _hyp_dist(z_unique[:, np.newaxis], z_unique[np.newaxis, :])
            k = min(5, n_u - 2)
            if k > 0:
                sorted_dists = np.sort(dist_matrix, axis=1)
                local_scale = np.median(sorted_dists[:, k])
                i_idx, j_idx = np.triu_indices(n_u, k=1)
                global_scale = np.median(dist_matrix[i_idx, j_idx])
                if local_scale > 1e-9:
                    knn_ratio = float(global_scale / local_scale)

        curvature_structure = float(temporal_score * spatio_temporal_score * knn_ratio)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "curvature_structure": curvature_structure,
                "temporal_variance": temporal_var,
                "spatio_temporal_corr": spatio_temporal_corr,
                "knn_scale_ratio": knn_ratio,
                "mean_hyperbolic_radius": mean_hyp_radius,
            },
            raw_data={"embedded_points": embedded}
        )


# =============================================================================
# MOSTOW RIGIDITY GEOMETRY (3D Poincaré Ball)
# =============================================================================

class MostowRigidityGeometry(ExoticGeometry):
    """
    Mostow Rigidity Geometry — measures how determined the hyperbolic
    geometry is by the combinatorial/topological structure.

    Mostow's theorem (dim ≥ 3): the metric on a closed hyperbolic manifold
    is uniquely determined by its fundamental group. There are no continuous
    deformations — the volume is a topological invariant.

    We compute a discrete analog: embed byte triples into the 3D Poincaré
    ball, form tetrahedra, and measure (1) the hyperbolic volume distribution
    and (2) how much the geometry changes under rank-preserving perturbation.

    Rigid data: rank perturbation → small geometric change (crystal, self-similar).
    Flexible data: same perturbation → large change (noise, chaos).
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Mostow Rigidity"

    @property
    def description(self) -> str:
        return ("3D Poincaré ball embedding measuring geometric rigidity — "
                "how determined the hyperbolic metric is by combinatorial structure")

    @property
    def view(self) -> str:
        return "topological"

    @property
    def detects(self) -> str:
        return "Geometric rigidity, volume invariance, Margulis thickness"

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed into 3D Poincaré ball."""
        return self._to_ball(data)

    def _to_ball(self, data: np.ndarray) -> np.ndarray:
        """Map byte triples to points in the open unit ball in R^3."""
        data = self.validate_data(data)
        x = data.astype(np.float64) / self.input_scale
        n = len(x) // 3
        if n < 4:
            return np.zeros((0, 3))
        pts = x[:n * 3].reshape(n, 3)
        # Map [0,1]^3 → ball: shift to center, scale to use disk
        pts = (pts - 0.5) * 1.8
        # Project onto open ball (clamp radius < 1)
        r = np.linalg.norm(pts, axis=1, keepdims=True)
        mask = (r >= 1.0).ravel()
        if mask.any():
            pts[mask] = pts[mask] / (r[mask] + 0.01) * 0.99
        return pts

    def _hyp_dist(self, x: np.ndarray, y: np.ndarray) -> float:
        """Poincaré ball hyperbolic distance: d(x,y) = arccosh(1 + 2|x-y|²/((1-|x|²)(1-|y|²)))."""
        diff_sq = np.sum((x - y) ** 2)
        rx = np.sum(x ** 2)
        ry = np.sum(y ** 2)
        denom = (1 - rx) * (1 - ry)
        if denom < 1e-15:
            return 20.0  # near-boundary, very large
        arg = 1.0 + 2.0 * diff_sq / denom
        return float(np.arccosh(max(arg, 1.0)))

    def _hyp_dist_matrix(self, pts: np.ndarray) -> np.ndarray:
        """Pairwise hyperbolic distance matrix (vectorized)."""
        n = len(pts)
        r_sq = np.sum(pts ** 2, axis=1)
        # |x_i - x_j|^2
        diff_sq = np.sum(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2) ** 2
        # Wait, this is wrong. Let me compute it correctly.
        # diff_sq[i,j] = sum_k (pts[i,k] - pts[j,k])^2
        diff_sq = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2)
        denom = np.outer(1 - r_sq, 1 - r_sq)
        denom = np.clip(denom, 1e-15, None)
        arg = 1.0 + 2.0 * diff_sq / denom
        arg = np.clip(arg, 1.0, None)
        return np.arccosh(arg)

    def _tet_volume_hyp(self, pts4: np.ndarray) -> float:
        """Approximate hyperbolic volume of tetrahedron in Poincaré ball.

        Uses the ball-model volume element: dV_hyp = (2/(1-|x|²))³ dV_euc.
        Approximation: V_hyp ≈ V_euc × (2/(1-|c|²))³ where c = centroid.
        """
        # Euclidean volume via 1/6 |det([b-a, c-a, d-a])|
        a, b, c, d = pts4
        mat = np.array([b - a, c - a, d - a])
        v_euc = abs(np.linalg.det(mat)) / 6.0
        # Conformal factor at centroid
        centroid = pts4.mean(axis=0)
        r_sq = np.sum(centroid ** 2)
        if r_sq >= 1.0:
            r_sq = 0.99
        conformal = (2.0 / (1.0 - r_sq)) ** 3
        return v_euc * conformal

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        pts = self._to_ball(data)

        defaults = {
            "volume_entropy": 0.0,
            "volume_rigidity": 0.0,
            "distance_rigidity": 0.0,
            "margulis_ratio": 0.0,
            "spectral_rigidity": 0.0,
        }
        if len(pts) < 20:
            return GeometryResult(geometry_name=self.name, metrics=defaults)

        # Subsample for tractability
        rng = np.random.default_rng(0)
        n_pts = min(len(pts), 500)
        idx = rng.choice(len(pts), n_pts, replace=False)
        idx.sort()
        pts_sub = pts[idx]

        # ── Tetrahedron volumes ───────────────────────────────────────
        # Form tetrahedra from consecutive quadruples
        n_tets = len(pts_sub) // 4
        volumes = np.zeros(n_tets)
        for i in range(n_tets):
            tet = pts_sub[4 * i: 4 * i + 4]
            volumes[i] = self._tet_volume_hyp(tet)

        # Volume entropy: how spread is the volume distribution?
        volumes_pos = volumes[volumes > 1e-15]
        if len(volumes_pos) > 2:
            # Normalize to distribution
            v_norm = volumes_pos / volumes_pos.sum()
            volume_entropy = float(-np.sum(v_norm * np.log(v_norm + 1e-30)))
        else:
            volume_entropy = 0.0

        # ── Rigidity under rank-preserving perturbation ───────────────
        # Take a subset for distance matrix computation
        n_rig = min(n_pts, 150)
        rig_pts = pts_sub[:n_rig]

        # Original distance matrix
        D_orig = self._hyp_dist_matrix(rig_pts)

        # Rank-preserving perturbation: add small noise that preserves
        # the rank order of values within each coordinate
        data_sub = data[:len(data) // 3 * 3].reshape(-1, 3)[idx[:n_rig]].astype(np.float64)
        noise_scale = 0.05 * self.input_scale  # 5% of range
        perturbed = data_sub + rng.normal(0, noise_scale, data_sub.shape)
        # Rank-order preservation: replace values with rank-matched originals + noise
        # Actually, simpler: just use the noisy version directly
        perturbed = np.clip(perturbed, 0, self.input_scale)
        # Re-embed
        p_norm = perturbed / self.input_scale
        p_pts = (p_norm - 0.5) * 1.8
        r = np.linalg.norm(p_pts, axis=1, keepdims=True)
        mask = (r >= 1.0).ravel()
        if mask.any():
            p_pts[mask] = p_pts[mask] / (r[mask] + 0.01) * 0.99

        D_pert = self._hyp_dist_matrix(p_pts)

        # Distance rigidity: correlation between original and perturbed
        triu = np.triu_indices(n_rig, k=1)
        d_orig_flat = D_orig[triu]
        d_pert_flat = D_pert[triu]
        finite = np.isfinite(d_orig_flat) & np.isfinite(d_pert_flat)
        if finite.sum() > 10 and d_orig_flat[finite].std() > 0:
            distance_rigidity = float(np.corrcoef(
                d_orig_flat[finite], d_pert_flat[finite])[0, 1])
        else:
            distance_rigidity = 0.0

        # Volume rigidity: how much does total volume change?
        # Recompute volumes on perturbed points
        n_tets_rig = len(p_pts) // 4
        vols_pert = np.zeros(n_tets_rig)
        vols_orig_rig = np.zeros(n_tets_rig)
        for i in range(n_tets_rig):
            sl = slice(4 * i, 4 * i + 4)
            vols_pert[i] = self._tet_volume_hyp(p_pts[sl])
            vols_orig_rig[i] = self._tet_volume_hyp(rig_pts[sl])

        total_orig = vols_orig_rig.sum()
        total_pert = vols_pert.sum()
        if total_orig > 1e-15:
            volume_rigidity = 1.0 - abs(total_pert - total_orig) / total_orig
            volume_rigidity = max(0.0, volume_rigidity)
        else:
            volume_rigidity = 0.0

        # ── Margulis ratio: thin/thick decomposition ──────────────────
        # Ratio of minimum to maximum pairwise distance (sampled)
        d_sample = d_orig_flat[finite]
        if len(d_sample) > 10:
            d_nz = d_sample[d_sample > 1e-10]
            if len(d_nz) > 10:
                margulis_ratio = float(np.percentile(d_nz, 5) /
                                       np.percentile(d_nz, 95))
            else:
                margulis_ratio = 0.0
        else:
            margulis_ratio = 0.0

        # ── Spectral rigidity ─────────────────────────────────────────
        # Compare Laplacian eigenvalues before/after perturbation
        # Use graph Laplacian from distance matrix (Gaussian kernel)
        sigma = np.median(d_orig_flat[finite]) if finite.sum() > 0 else 1.0
        if sigma < 1e-10:
            sigma = 1.0

        W_orig = np.exp(-D_orig ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W_orig, 0)
        L_orig = np.diag(W_orig.sum(axis=1)) - W_orig

        W_pert = np.exp(-D_pert ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W_pert, 0)
        L_pert = np.diag(W_pert.sum(axis=1)) - W_pert

        try:
            eig_orig = np.sort(np.linalg.eigvalsh(L_orig))[:20]
            eig_pert = np.sort(np.linalg.eigvalsh(L_pert))[:20]
            # Correlation of first 20 eigenvalues
            if len(eig_orig) == len(eig_pert):
                spectral_rigidity = float(np.corrcoef(eig_orig, eig_pert)[0, 1])
                if not np.isfinite(spectral_rigidity):
                    spectral_rigidity = 0.0
            else:
                spectral_rigidity = 0.0
        except np.linalg.LinAlgError:
            spectral_rigidity = 0.0

        # --- Mean hyperbolic turn angle (evolved via ShinkaEvolve) ---
        # For consecutive triples p_{t-1}, p_t, p_{t+1}, compute the angle at p_t
        # using the hyperbolic law of cosines. Structured data traces straighter
        # paths (mean angle near pi); random data is erratic (mean angle near pi/3).
        if len(pts_sub) >= 3:
            angles = []
            for i in range(1, len(pts_sub) - 1):
                a = self._hyp_dist(pts_sub[i], pts_sub[i + 1])
                b = self._hyp_dist(pts_sub[i - 1], pts_sub[i])
                c = self._hyp_dist(pts_sub[i - 1], pts_sub[i + 1])
                denom = np.sinh(a) * np.sinh(b)
                if denom < 1e-12:
                    angles.append(np.pi)
                    continue
                cos_g = np.clip(
                    (np.cosh(a) * np.cosh(b) - np.cosh(c)) / denom, -1.0, 1.0)
                angles.append(np.arccos(cos_g))
            mean_turn_angle = float(np.mean(angles) / np.pi) if angles else 0.0
        else:
            mean_turn_angle = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "volume_entropy": volume_entropy,
                "volume_rigidity": volume_rigidity,
                "distance_rigidity": distance_rigidity,
                "margulis_ratio": margulis_ratio,
                "spectral_rigidity": spectral_rigidity,
                "mean_turn_angle": mean_turn_angle,
            },
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

    Metrics:
    - area_length_ratio: log(1 + |z_final|/n²). Captures nilpotent scaling
      (correlated data has z ~ n², noise ~ n^1.5). Replaces raw final_z.
    - z_rate_spectral_entropy: 1 - normalized spectral entropy of dz/dt.
      Structured data → peaked spectrum, noise → flat.
      (ShinkaEvolve heisenberg_v1 gen 10)
    - xy_spread: std dev of x,y path coordinates.
    """

    def __init__(self, input_scale: float = 255.0, center_data: bool = False):
        self.input_scale = input_scale
        self.center_data = center_data

    @property
    def name(self) -> str:
        suffix = " (centered)" if self.center_data else ""
        return f"Heisenberg (Nil){suffix}"


    @property

    def description(self) -> str:

        return "Lifts byte pairs to the 3D Heisenberg group, where the z-coordinate accumulates signed area (xy cross-products). Correlated data twists the path; uncorrelated data stays flat."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Correlation twist, phase coupling, area accumulation"

    @property
    def encoding_invariant(self) -> bool:
        return self.center_data

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
        """Compute Heisenberg metrics: area-length ratio, spectral entropy, spread."""
        path = self.embed(data)
        n_steps = len(path) - 1

        _nan = {k: float('nan') for k in [
            "area_length_ratio", "z_rate_spectral_entropy", "xy_spread"]}
        if n_steps < 20:
            return GeometryResult(self.name, _nan, {})

        # XY spread
        xy_std = np.std(path[:, :2])

        # Area-to-length ratio: log(1 + |z_final|/n²).
        # Better conditioned than raw final_z (kurtosis 39, range 33M).
        # Captures nilpotent scaling: correlated data has z ~ n², noise has z ~ n^1.5.
        z_final = path[-1, 2]
        area_length = float(np.log1p(abs(z_final) / (n_steps ** 2)))

        # Z-rate spectral entropy (ShinkaEvolve heisenberg_v1 gen 10).
        # dz/dt = z[i+1] - z[i] captures instantaneous non-commutativity.
        # Structured data → peaked spectrum (low entropy), noise → flat (high).
        z_increments = np.diff(path[:, 2])
        z_spectral_entropy = 0.0
        if np.std(z_increments) > 1e-10:
            psd = np.abs(np.fft.rfft(z_increments)) ** 2
            psd_sum = np.sum(psd)
            if psd_sum > 1e-10:
                psd_norm = psd / psd_sum
                entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
                n_freqs = len(psd_norm)
                if n_freqs > 1:
                    max_entropy = np.log2(n_freqs)
                    if max_entropy > 1e-10:
                        z_spectral_entropy = 1.0 - entropy / max_entropy

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "area_length_ratio": area_length,
                "z_rate_spectral_entropy": float(np.clip(z_spectral_entropy, 0, 1)),
                "xy_spread": float(xy_std),
            },
            raw_data={}
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

    def description(self) -> str:

        return "Maps byte triples to the 2-sphere via spherical coordinates. Measures directional concentration, hemisphere balance, and angular spread."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Directional clustering, angular uniformity"

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
        return "Cantor Set"


    @property

    def description(self) -> str:

        return "Interprets bits as a base-3 address into the Cantor set. Gap structure and dust fraction measure self-similarity in the ternary digit stream."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Gaps, dust fraction, ternary self-similarity"

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

    def _bit_plane_autocorrelation(self, data: np.ndarray) -> float:
        """Weighted lag-1 autocorrelation across 8 bit planes.

        Each byte has 8 bits; each bit plane is a binary time series.
        Lag-1 autocorrelation of each plane measures temporal structure
        at that bit position. Weights by ternary significance (MSB most
        important in the Cantor embedding).

        Structured data has correlated bit planes; shuffling destroys
        the temporal correlations while preserving bit distributions.
        """
        data_np = np.asarray(data, dtype=np.uint8)
        bit_planes = np.unpackbits(data_np[:, np.newaxis], axis=1)
        autocorrs = np.zeros(8, dtype=np.float64)
        for i in range(8):
            plane = bit_planes[:, i].astype(np.float64)
            if np.std(plane) < 1e-9:
                autocorrs[i] = 1.0
                continue
            c = np.corrcoef(plane[:-1], plane[1:])[0, 1]
            autocorrs[i] = np.abs(c) if np.isfinite(c) else 0.0
        weights = (1.0 / self.base) ** np.arange(1, 9)
        return float(np.sum(weights * autocorrs) / np.sum(weights))

    def _jump_entropy(self, coords: np.ndarray) -> float:
        """1 - normalized Shannon entropy of consecutive Cantor jump sizes.

        The sequence |coord[t+1] - coord[t]| has a characteristic distribution
        for structured data (clustered jumps) vs shuffled (more uniform).
        Low entropy = structured jumps = high metric value.
        """
        if len(coords) < 2:
            return 0.0
        jumps = np.abs(np.diff(coords))
        if np.std(jumps) < 1e-9:
            return 1.0
        hist, _ = np.histogram(jumps, bins=32, range=(0.0, 1.0))
        hist_sum = np.sum(hist)
        if hist_sum < 1:
            return 0.0
        p = hist / hist_sum
        entropy = -np.sum(p * np.log2(p + 1e-12))
        max_entropy = np.log2(32)
        return float(np.clip(1.0 - entropy / max_entropy, 0, 1))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Cantor set metrics: gaps, coverage, temporal structure."""
        data = self.validate_data(data)
        embedded = self.embed(data)
        _nan = {k: float('nan') for k in [
            "mean_gap", "max_gap", "coverage",
            "bit_plane_autocorrelation", "jump_entropy"]}
        if len(embedded) < 20:
            return GeometryResult(self.name, _nan, {})

        # Gap analysis
        sorted_coords = np.sort(embedded)
        gaps = np.diff(sorted_coords)
        mean_gap = np.mean(gaps)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0

        # Coverage (fraction of distinct embedded values)
        coverage = len(np.unique(embedded)) / len(embedded)

        # Temporal metrics (evolved via ShinkaEvolve)
        bit_autocorr = self._bit_plane_autocorrelation(data)
        jump_ent = self._jump_entropy(embedded)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "mean_gap": mean_gap,
                "max_gap": max_gap,
                "coverage": coverage,
                "bit_plane_autocorrelation": float(bit_autocorr),
                "jump_entropy": float(jump_ent),
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
    Ultrametric/p-adic Geometry — detects number-theoretic hierarchy.

    Computes p-adic distance between byte values, where d(a,b) = p^(-v_p(a-b))
    and v_p is the p-adic valuation. This measures divisibility structure:
    values whose difference is divisible by high powers of p are "close."

    Note: this detects hierarchy in the VALUE space (number-theoretic),
    not temporal hierarchy or structural nesting of the time series.
    """

    def __init__(self, p: int = 2):
        self.p = p

    @property
    def name(self) -> str:
        return f"{self.p}-adic"


    @property

    def description(self) -> str:

        return "Measures distances via 2-adic valuation: two bytes are close if they agree on many trailing bits. Ultrametric violations reveal non-hierarchical structure."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "p-adic clustering, divisibility depth, modular structure"

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

    def _consecutive_valuations(self, data: np.ndarray) -> np.ndarray:
        """Compute v_p(|data[t+1] - data[t]|) for consecutive pairs."""
        diffs = np.abs(np.diff(data.astype(np.int32)))
        vals = np.zeros(len(diffs), dtype=np.int32)
        for i, d in enumerate(diffs):
            if d == 0:
                vals[i] = 8
            else:
                v = 0
                dd = int(d)
                while dd % self.p == 0:
                    dd //= self.p
                    v += 1
                vals[i] = v
        return vals

    def _multiscale_markov_predictability(self, data: np.ndarray) -> float:
        """Conditional entropy at multiple 2-adic scales, averaged.

        At each scale k, reduces data to states mod 2^k, builds a Markov
        transition matrix, and computes normalized entropy rate. Averages
        across scales k=2..7. Returns 1 - mean_entropy: high = predictable.
        """
        n = len(data)
        entropies = []
        for k in range(2, 8):
            n_states = 1 << k
            mod_data = data % n_states
            counts = np.zeros((n_states, n_states), dtype=np.float64)
            for i in range(n - 1):
                counts[mod_data[i], mod_data[i + 1]] += 1
            row_sums = counts.sum(axis=1)
            total = row_sums.sum()
            if total < 1:
                continue
            pi = row_sums / total
            h_rate = 0.0
            for s in range(n_states):
                if pi[s] > 1e-12:
                    probs = counts[s] / row_sums[s]
                    nz = probs[probs > 1e-12]
                    h_rate += pi[s] * (-np.sum(nz * np.log2(nz)))
            max_ent = float(k)
            if max_ent > 1e-9:
                entropies.append(h_rate / max_ent)
        if not entropies:
            return 0.0
        return float(np.clip(1.0 - np.mean(entropies), 0, 1))

    def _valuation_spectral_concentration(self, val_seq: np.ndarray) -> float:
        """1 - normalized spectral entropy of the valuation sequence.

        Measures global periodicity in how 2-adic distances between
        consecutive values vary over time. High = structured/periodic.
        """
        n = len(val_seq)
        if n < 4:
            return 0.0
        ps = np.abs(np.fft.rfft(val_seq - np.mean(val_seq))) ** 2
        ps_sum = np.sum(ps)
        if ps_sum < 1e-9:
            return 1.0
        p_norm = ps / ps_sum
        nz = p_norm[p_norm > 1e-12]
        entropy = -np.sum(nz * np.log2(nz))
        max_ent = np.log2(len(ps))
        if max_ent < 1e-9:
            return 1.0
        return float(np.clip(1.0 - entropy / max_ent, 0, 1))

    def _valuation_transition_predictability(self, val_seq: np.ndarray) -> float:
        """1 - normalized conditional entropy of valuation-level transitions.

        Measures local Markov structure: given the current v_2 level,
        how predictable is the next level? High = deterministic transitions.
        """
        n_levels = 9  # v_2 ranges 0..8
        if len(val_seq) < 2:
            return 0.0
        counts = np.zeros((n_levels, n_levels), dtype=np.float64)
        for i in range(len(val_seq) - 1):
            counts[val_seq[i], val_seq[i + 1]] += 1
        row_sums = counts.sum(axis=1)
        total = row_sums.sum()
        if total < 1:
            return 0.0
        pi = row_sums / total
        h_rate = 0.0
        for s in range(n_levels):
            if pi[s] > 1e-12:
                probs = counts[s] / row_sums[s]
                nz = probs[probs > 1e-12]
                h_rate += pi[s] * (-np.sum(nz * np.log2(nz)))
        max_ent = np.log2(n_levels)
        if max_ent < 1e-9:
            return 1.0
        return float(np.clip(1.0 - h_rate / max_ent, 0, 1))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute ultrametric metrics: distributional + temporal."""
        embedded = self.embed(data)
        _nan = {k: float('nan') for k in [
            "mean_distance", "distance_entropy",
            "multiscale_markov_predictability",
            "valuation_spectral_concentration",
            "valuation_transition_predictability"]}
        if len(embedded) < 50:
            return GeometryResult(self.name, _nan, {})

        # --- Distributional (original) ---
        n = min(len(embedded), 500)
        rng = np.random.default_rng(0)
        indices = rng.choice(len(embedded), n, replace=False)
        sample = embedded[indices]

        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, min(i + 50, len(sample))):
                distances.append(self.p_adic_distance(sample[i], sample[j]))

        distances = np.array(distances)
        mean_dist = np.mean(distances)
        dist_entropy = -np.sum([
            (np.sum(distances == d) / len(distances)) *
            np.log2(np.sum(distances == d) / len(distances) + 1e-10)
            for d in set(distances)
        ])

        # --- Temporal (evolved via ShinkaEvolve) ---
        data_uint8 = data if data.dtype == np.uint8 else np.asarray(data, dtype=np.uint8)
        val_seq = self._consecutive_valuations(data_uint8)
        mmp = self._multiscale_markov_predictability(data_uint8)
        vsc = self._valuation_spectral_concentration(val_seq)
        vtp = self._valuation_transition_predictability(val_seq)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "mean_distance": mean_dist,
                "distance_entropy": dist_entropy,
                "multiscale_markov_predictability": mmp,
                "valuation_spectral_concentration": vsc,
                "valuation_transition_predictability": vtp,
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
    Piecewise-Linear Geometry — detects linear regimes and slope transitions.

    Analyzes the piecewise-linear structure of 1D signals: how many distinct
    linear segments exist, how they break, and the envelope structure.

    Note: Named "Tropical" for historical reasons. The metrics do not use
    tropical semiring (min-plus) operations. They detect piecewise-linear
    structure via second-derivative thresholding and running-minimum envelopes.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Tropical"


    @property

    def description(self) -> str:

        return "Tropical algebra replaces (add,multiply) with (min,add). The resulting piecewise-linear envelope reveals slope transitions and regime boundaries."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Piecewise-linear regimes, slope diversity, envelope structure"

    @property
    def dimension(self) -> str:
        return "piecewise-linear"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] for piecewise-linear analysis."""
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute piecewise-linear structure metrics."""
        embedded = self.embed(data)
        n = len(embedded)

        # Piecewise linearity: count slope changes via second-derivative threshold
        if n < 3:
            slope_changes = 0
            linearity = 1.0
        else:
            diffs = np.diff(embedded)
            slope_changes = np.sum(np.abs(np.diff(diffs)) > 0.01)
            # Linearity: fraction without slope change
            linearity = 1 - slope_changes / (n - 2)

        # Running-minimum envelope area
        # Measures how far the signal deviates above its cumulative minimum
        running_min = np.minimum.accumulate(embedded)
        envelope_area = np.sum(embedded - running_min)

        # Slope diversity: count distinct windowed slopes
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
                "slope_changes": slope_changes,
                "envelope_area": envelope_area,
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

    def description(self) -> str:

        return "Computes optimal transport (earth mover's) distance between windowed histograms. Self-similarity and concentration measure distributional stability."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Distribution shape, transport cost, self-similarity"

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

    def _windowed_histograms(self, data: np.ndarray,
                             n_windows: int = 16) -> list:
        """Split data into n_windows segments, return normalized histograms."""
        seg_len = len(data) // n_windows
        if seg_len < 100:
            return []
        hists = []
        for i in range(n_windows):
            seg = data[i * seg_len:(i + 1) * seg_len]
            x = seg.astype(float) / self.input_scale
            hist, _ = np.histogram(x, bins=self.n_bins, range=(0, 1), density=True)
            hists.append(hist / (np.sum(hist) + 1e-10))
        return hists

    def _transport_dynamics(self, data: np.ndarray) -> dict:
        """Windowed transport dynamics in Wasserstein space.

        transport_variability: std of W₁ between consecutive windowed
        distributions. Measures burstiness of distributional change.

        recurrence_distance: for each window, the minimum W₁ to any
        prior window. Low = trajectory revisits prior states (confined).
        High = trajectory wanders (random). (evolved via ShinkaEvolve)
        """
        hists = self._windowed_histograms(data, n_windows=16)
        if len(hists) < 4:
            return {'transport_variability': 0.0, 'recurrence_distance': 0.0}

        dists = np.array([self.wasserstein_1d(hists[i], hists[i + 1])
                          for i in range(len(hists) - 1)])
        transport_variability = float(np.std(dists))

        min_dists = []
        for i in range(1, len(hists)):
            d2past = [self.wasserstein_1d(hists[i], hists[j])
                      for j in range(i)]
            min_dists.append(np.min(d2past))
        recurrence_distance = float(np.mean(min_dists))

        return {'transport_variability': transport_variability,
                'recurrence_distance': recurrence_distance}

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Wasserstein metrics: distance from uniform, self-similarity."""
        data = self.validate_data(data)
        hist = self.embed(data)

        # Distance from uniform distribution
        uniform = np.ones(self.n_bins) / self.n_bins
        dist_from_uniform = self.wasserstein_1d(hist, uniform)

        # Self-similarity: compare first half to second half
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

        metrics = {
            "dist_from_uniform": dist_from_uniform,
            "self_similarity": self_similarity,
            "entropy": entropy,
            "concentration": concentration,
        }
        metrics.update(self._transport_dynamics(data))

        return GeometryResult(
            geometry_name=self.name,
            metrics=metrics,
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

    def __init__(self, n_points: int = 50, max_dim: int = 1):
        self.n_points = n_points
        self.max_dim = max_dim

    @property
    def name(self) -> str:
        return "Persistent Homology"


    @property

    def description(self) -> str:

        return "Builds a Vietoris-Rips filtration on delay-embedded points. Persistent features (connected components H0, loops H1) that survive across scales indicate robust topological structure."


    @property

    def view(self) -> str:

        return "topological"


    @property

    def detects(self) -> str:

        return "Holes, loops, connected components, topological persistence"

    @property
    def dimension(self) -> str:
        return "topological"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as 2D point cloud using delay embedding."""
        data = self.validate_data(data)
        data = data.astype(float)

        # Subsample: use contiguous block to preserve temporal adjacency.
        # np.linspace subsampling scatters points across the signal,
        # destroying the delay embedding's topology (e.g., a periodic
        # signal's ellipse becomes a disconnected scatter).
        if len(data) > self.n_points * 2:
            data = data[:self.n_points * 2]

        indices = np.arange(0, len(data) - 1)

        # Delay embedding: (x[i], x[i+1])
        points = np.array([[data[i], data[i + 1]] for i in indices])

        # Deduplicate: uint8 inputs create many exact-duplicate embedded
        # points (e.g., a sine with 89 points may have only 20 unique).
        # Duplicates create degenerate Rips complexes with thousands of
        # spurious zero-birth H1 features. Keep unique points only.
        points = np.unique(points, axis=0)

        # Cap at n_points to keep TDA tractable
        if len(points) > self.n_points:
            points = points[:self.n_points]

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

    def compute_rips_homology_h1(self, points: np.ndarray,
                                  dists: np.ndarray,
                                  max_filtration: float) -> List[Tuple[float, float]]:
        """
        Compute H1 (loops) persistent homology via boundary matrix reduction over Z_2.

        Algorithm:
        1. Build spanning tree (union-find) to identify non-tree edges = H1 births
        2. Enumerate triangles up to filtration threshold
        3. Column-reduce boundary matrix: each triangle may kill an H1 cycle
        """
        n = len(points)
        if n < 3:
            return []

        # Build sorted edges with index mapping
        edge_list = []    # [(filtration, i, j), ...]
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((dists[i, j], i, j))
        edge_list.sort()

        # Map (i,j) -> filtration-order index
        edge_order = {}
        for idx, (f, i, j) in enumerate(edge_list):
            edge_order[(i, j)] = idx

        # Union-find to identify tree vs non-tree edges
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        non_tree_edges = set()
        for f, i, j in edge_list:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
            else:
                non_tree_edges.add(edge_order[(i, j)])

        # Enumerate triangles up to max_filtration
        # Optimized: instead of O(n^3) combinations, only check neighbors in the threshold graph
        adj = [[] for _ in range(n)]
        for (f, i, j) in edge_list:
            if f > max_filtration:
                break
            adj[i].append(j)
            adj[j].append(i)

        triangles = []
        for i in range(n):
            # Only check neighbors with higher index to avoid double-counting
            neighbors = sorted([nj for nj in adj[i] if nj > i])
            for idx_j, j in enumerate(neighbors):
                for k in neighbors[idx_j + 1:]:
                    # Check if (j,k) exists and is within filtration
                    if dists[j, k] <= max_filtration:
                        tri_filt = max(dists[i, j], dists[j, k], dists[i, k])
                        triangles.append((tri_filt, i, j, k))
        triangles.sort()

        # Boundary matrix column reduction over Z_2
        pivot_to_col = {}
        h1_pairs = []

        for tri_filt, i, j, k in triangles:
            e1 = edge_order[(min(i, j), max(i, j))]
            e2 = edge_order[(min(j, k), max(j, k))]
            e3 = edge_order[(min(i, k), max(i, k))]
            boundary = {e1, e2, e3}

            while boundary:
                pivot = max(boundary)
                if pivot in pivot_to_col:
                    boundary ^= pivot_to_col[pivot]
                else:
                    pivot_to_col[pivot] = boundary
                    if pivot in non_tree_edges:
                        h1_pairs.append((edge_list[pivot][0], tri_filt))
                    break

        return h1_pairs

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute persistence metrics (H0 components + H1 loops)."""
        points = self.embed(data)

        if len(points) == 0:
            return GeometryResult(
                geometry_name=self.name,
                metrics={
                    "total_persistence": 0.0,
                    "n_significant_features": 0,
                    "persistence_entropy": 0.0,
                    "max_components": 0,
                    "max_lifetime": 0.0,
                    "n_h1_features": 0,
                    "max_h1_lifetime": 0.0,
                    "h1_total_persistence": 0.0,
                },
                raw_data={"points": points}
            )

        # Normalize points to [0,1]
        points = points - points.min()
        points = points / (points.max() + 1e-10)

        # Compute H0 persistence
        h0_pairs = self.compute_rips_homology_h0(points)

        # H0 statistics
        lifetimes = [death - birth for birth, death in h0_pairs]
        lifetimes = sorted(lifetimes, reverse=True)
        total_persistence = sum(lifetimes)
        threshold = 0.1
        n_significant = sum(1 for l in lifetimes if l > threshold)
        lifetimes_arr = np.array(lifetimes)
        probs = lifetimes_arr / (total_persistence + 1e-10)
        persistence_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_components = len(h0_pairs)

        # Compute H1 persistence (loops)
        h1_pairs = []
        h1_n_significant = 0
        h1_max_lifetime = 0.0
        h1_total_persistence = 0.0

        if self.max_dim >= 1 and len(points) >= 3:
            from scipy.spatial.distance import pdist, squareform
            dists = squareform(pdist(points))
            max_filt = max(d for _, d in h0_pairs) if h0_pairs else 1.0
            h1_pairs = self.compute_rips_homology_h1(points, dists, max_filt)

            if h1_pairs:
                h1_lifetimes = sorted([d - b for b, d in h1_pairs], reverse=True)
                h1_total_persistence = sum(h1_lifetimes)
                h1_n_significant = sum(1 for l in h1_lifetimes if l > threshold)
                h1_max_lifetime = h1_lifetimes[0]

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "total_persistence": total_persistence,
                "n_significant_features": n_significant,
                "persistence_entropy": persistence_entropy,
                "max_components": max_components,
                "max_lifetime": lifetimes[0] if lifetimes else 0,
                "n_h1_features": h1_n_significant,
                "max_h1_lifetime": h1_max_lifetime,
                "h1_total_persistence": h1_total_persistence,
            },
            raw_data={
                "h0_pairs": h0_pairs,
                "h1_pairs": h1_pairs,
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

    def description(self) -> str:

        return "Treats consecutive (time, value) pairs as events in 1+1 Minkowski spacetime. Classifies intervals as timelike, spacelike, or lightlike to detect causal structure."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Causal ordering, lightcone structure, timelike fraction"

    @property
    def atlas_exclude(self) -> set:
        return {"lightlike_fraction"}

    @property
    def dimension(self) -> str:
        return "1+1 spacetime"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as spacetime events (t, x) with resolution-matched scales.

        Scale time so that c=1 corresponds to sqrt(n_levels) byte values
        per step.  For uint8 (256 levels), the lightcone boundary is at
        16 byte levels per step — the geometric mean of the spatial
        resolution scales.  Without this, c=1 in the naive (i/N, v/255)
        embedding makes the minimum nonzero velocity ~N/255 ≈ 32 × c,
        collapsing causal_order_preserved to "fraction of identical bytes."
        """
        data = self.validate_data(data)
        data_norm = self._normalize_to_unit(data, self.input_scale)
        n = len(data_norm)

        n_levels = 256.0  # uint8 resolution
        t = np.arange(n) / np.sqrt(n_levels)  # Δt = 1/16 per step
        events = np.column_stack([t, data_norm])
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

        if n < 3:
            return GeometryResult(
                geometry_name=self.name,
                metrics={k: 0.0 for k in [
                    "spacelike_fraction", "lightlike_fraction",
                    "causal_order_preserved", "crossing_density",
                    "causal_persistence"]},
                raw_data={"events": events})

        # --- Sample intervals at log-spaced separations ---
        rng = np.random.default_rng(0)
        n_samples = min(500, n * (n - 1) // 2)
        log_seps = rng.uniform(0, np.log(max(n // 2, 2)), n_samples)
        seps = np.clip(np.exp(log_seps).astype(int), 1, n - 1)
        starts = np.array([rng.integers(0, n - s) for s in seps])
        ends = starts + seps

        dt = events[ends, 0] - events[starts, 0]
        dx = events[ends, 1] - events[starts, 1]
        intervals = -dt**2 + dx**2
        separations = dt**2 + dx**2

        rel_tol = 0.01
        near_lightcone = np.abs(intervals) < rel_tol * (separations + 1e-20)
        timelike_frac = float(np.mean((intervals < 0) & ~near_lightcone))
        spacelike_frac = float(np.mean((intervals > 0) & ~near_lightcone))
        lightlike_frac = float(np.mean(near_lightcone))

        # --- Causal ordering: consecutive events ---
        con_dt = events[1:, 0] - events[:-1, 0]
        con_dx = events[1:, 1] - events[:-1, 1]
        con_s2 = -con_dt**2 + con_dx**2
        is_timelike = con_s2 < 0
        causal_order_preserved = float(np.mean(is_timelike))

        # --- Lightcone crossing density (ShinkaEvolve atlas v1 gen 5) ---
        # How often consecutive steps switch between timelike and spacelike.
        # Measures burstiness relative to the lightcone boundary c=1.
        if len(is_timelike) >= 2:
            crossings = np.sum(is_timelike[:-1] != is_timelike[1:])
            crossing_density = float(crossings) / (len(is_timelike) - 1)
        else:
            crossing_density = 0.0

        # --- Causal persistence (ShinkaEvolve atlas v1 gen 43) ---
        # Lag-1 autocorrelation of the timelike/spacelike binary sequence,
        # scaled to [0, 1]. High = long runs of same causal character.
        p = causal_order_preserved
        if len(is_timelike) >= 2 and 1e-9 < p < 1.0 - 1e-9:
            is_tl = is_timelike.astype(np.float64)
            autocov = float(np.mean((is_tl[:-1] - p) * (is_tl[1:] - p)))
            causal_persistence = (autocov / (p * (1.0 - p)) + 1.0) / 2.0
        else:
            causal_persistence = 0.0

        # --- Light cone analysis: consecutive velocities ---
        good = con_dt > 1e-10
        velocities = np.abs(con_dx[good] / con_dt[good])
        mean_velocity = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
        superluminal_frac = float(np.mean(velocities > 1)) if len(velocities) > 0 else 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "spacelike_fraction": spacelike_frac,
                "lightlike_fraction": lightlike_frac,
                "causal_order_preserved": causal_order_preserved,
                "crossing_density": crossing_density,
                "causal_persistence": causal_persistence,
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
        return "Logarithmic Spiral"


    @property

    def description(self) -> str:

        return "Maps the time series to a logarithmic spiral in polar coordinates. Growth rate, winding number, and angular uniformity measure multiplicative structure."


    @property

    def view(self) -> str:

        return "scale"


    @property

    def detects(self) -> str:

        return "Growth rate, spiral tightness, radial regularity"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data on spiral. Angle advances by a data-dependent step each sample,
        so angular metrics (winding, uniformity) reflect the data's structure."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        points = []
        theta = 0.0
        for val in data:
            # Angular step: base rate + data modulation
            # val in [0,1]: 0 → slow rotation, 1 → fast rotation
            theta += 0.05 + val * 0.15  # step in [0.05, 0.20]

            if self.spiral_type == "logarithmic":
                r = (0.1 + val) * np.exp(0.02 * theta)
            elif self.spiral_type == "archimedean":
                r = (0.1 + val) + 0.05 * theta
            else:  # fermat
                r = np.sqrt((0.1 + val) * max(theta, 0.01))

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
        # Log-transform radius-based metrics: the logarithmic spiral
        # parameterization produces exp(0.02 * θ) growth, which overflows
        # for long data (N=16384 → θ ~ 2000 → exp(40) ~ 10^17).
        # Log/log preserves ordering while keeping values on comparable scale.
        if len(radii) > 1:
            radial_change = radii[-1] - radii[0]
            tightness = np.log1p(abs(radial_change)) / (np.log1p(abs(total_winding)) + 1e-10)
        else:
            tightness = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "angular_uniformity": angular_uniformity,
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

    def description(self) -> str:

        return "Lifts byte triples to projective 2-space (homogeneous coordinates). Cross-ratio variance measures departure from projective invariance."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Scale invariance, cross-ratio stability, projective curvature"

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

        # Cross-ratio invariant (projective invariant of 4 collinear points)
        # For 4 points in P^2, project p3,p4 onto span(p1,p2) and compute:
        #   p3 ≈ α·p1 + β·p2, p4 ≈ γ·p1 + δ·p2
        #   CR = (α·δ) / (β·γ)
        rng = np.random.default_rng(0)
        cross_ratios = []
        for _ in range(min(100, n // 4)):
            idx = rng.choice(n, 4, replace=False)
            p1, p2, p3, p4 = points[idx]
            # Project p3, p4 onto span(p1, p2) via least squares
            A = np.column_stack([p1, p2])
            coeff3, _, _, _ = np.linalg.lstsq(A, p3, rcond=None)
            coeff4, _, _, _ = np.linalg.lstsq(A, p4, rcond=None)
            alpha, beta = coeff3
            gamma, delta = coeff4
            denom = beta * gamma
            if abs(denom) > 1e-10:
                cr = (alpha * delta) / denom
                # Clip: near-coincident projected points produce divergent
                # cross-ratios that dominate the std. CR is a Möbius invariant
                # so ±100 is already far from the generic range [0,1].
                cr = np.clip(cr, -100, 100)
                cross_ratios.append(cr)

        cross_ratio_std = np.std(cross_ratios) if cross_ratios else 0

        # Collinearity: how often are 3 points nearly collinear?
        collinear_count = 0
        total_triples = 0
        for _ in range(min(100, n // 3)):
            idx = rng.choice(n, 3, replace=False)
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
# ZARISKI (NON-HAUSDORFF) GEOMETRY
# =============================================================================

class ZariskiGeometry(ExoticGeometry):
    """
    Zariski (Non-Hausdorff) Geometry — detects algebraic structure.

    The Zariski topology on R^d has closed sets = zero sets of polynomials.
    Unlike metric-space topologies used by all other geometries in this
    framework, the Zariski topology is non-Hausdorff: points on the same
    algebraic variety cannot be topologically separated.

    Method: delay-embed the byte stream into R^d, then test whether the
    point cloud lies on (or near) a low-degree algebraic variety via
    Vandermonde SVD. Also measures the Heyting algebra gap — deviation
    of the pattern lattice from Boolean (law of excluded middle).

    Detects:
    - Polynomial recurrence relations (linear, quadratic, cubic maps)
    - Algebraic dependencies between consecutive values
    - Non-Boolean pattern structure (substitution rules, QC, correlations)
    """

    def __init__(self, input_scale=255.0, embed_dim=2, max_degree=4):
        self.input_scale = input_scale
        self.embed_dim = embed_dim
        self.max_degree = max_degree

    @property
    def name(self) -> str:
        return "Zariski"

    @property
    def dimension(self) -> str:
        return "algebraic variety"

    @property
    def description(self) -> str:
        return ("Non-Hausdorff algebraic structure via Zariski topology. "
                "Detects polynomial recurrences via Vandermonde SVD and "
                "Heyting lattice gaps in pattern complement structure.")

    @property
    def view(self) -> str:
        return "topological"

    @property
    def detects(self) -> str:
        return "Polynomial recurrences, algebraic varieties, non-Boolean pattern lattice"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Delay embedding into R^d, normalized to [0,1]."""
        data = self.validate_data(data)
        x = self._normalize_to_unit(data, self.input_scale)
        n = len(x) - self.embed_dim + 1
        if n <= 0:
            return np.empty((0, self.embed_dim))
        X = np.zeros((n, self.embed_dim))
        for i in range(self.embed_dim):
            X[:, i] = x[i:i + n]
        return X

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        data = self.validate_data(data)
        metrics = {}

        # Delay embed and subsample for SVD
        X = self.embed(data)
        rng = np.random.default_rng(42)
        if len(X) > 2000:
            idx = rng.choice(len(X), 2000, replace=False)
            X = X[np.sort(idx)]

        if len(X) >= 50:
            metrics.update(self._vandermonde_residuals(X))
            metrics.update(self._zariski_separation(X, rng))
            metrics.update(self._residual_convexity(X))

        metrics.update(self._heyting_gap(data))
        metrics.update(self._heyting_stability(data))

        return GeometryResult(geometry_name=self.name, metrics=metrics)

    def _vandermonde_residuals(self, X: np.ndarray) -> dict:
        """SVD residuals of Vandermonde matrices at increasing polynomial degree.

        For data generated by a degree-k recurrence, the Vandermonde matrix
        at degree k develops a null space (near-zero singular values).
        """
        from itertools import combinations_with_replacement
        n, d = X.shape

        residuals = []
        for deg in range(1, self.max_degree + 1):
            columns = [np.ones(n)]
            for dd in range(1, deg + 1):
                for idx in combinations_with_replacement(range(d), dd):
                    col = np.ones(n)
                    for var in idx:
                        col = col * X[:, var]
                    columns.append(col)

            n_mono = len(columns)
            if n_mono >= n:
                break

            V = np.column_stack(columns)
            s = np.linalg.svd(V, compute_uv=False)
            residuals.append(s[-1] / np.sqrt(n) if s[0] > 1e-15 else 0.0)

        results = {}
        if len(residuals) >= 2:
            # Degree-2 residual: low for quadratic maps (logistic), high for noise
            results['algebraic_residual'] = float(residuals[1])
            # Slope of log-residual vs degree: more negative = more algebraic
            log_res = np.log10(np.array(residuals) + 1e-15)
            degs = np.arange(1, len(log_res) + 1)
            results['residual_slope'] = float(np.polyfit(degs, log_res, 1)[0])
        else:
            results['algebraic_residual'] = 1.0
            results['residual_slope'] = 0.0
        return results

    def _heyting_gap(self, data: np.ndarray) -> dict:
        """Deviation of pattern lattice from Boolean algebra.

        In Boolean (Hausdorff) topology: A ∨ ¬A = 1 (excluded middle).
        In Heyting (non-Hausdorff): A ∨ ¬A < 1 — some patterns lack complements.

        Binary quantize, then for each pattern length k, measure what fraction
        of observed k-grams are missing their bitwise complement.
        """
        unique_vals = np.unique(data)
        if len(unique_vals) <= 4:
            threshold = (float(unique_vals[0]) + float(unique_vals[-1])) / 2
        else:
            threshold = float(np.median(data))
        bits = (data > threshold).astype(np.uint8)

        gaps = []
        for k in range(2, 9):
            observed = set()
            for i in range(len(bits) - k + 1):
                observed.add(tuple(bits[i:i + k]))
            if not observed:
                continue
            n_with_complement = sum(
                1 for pat in observed
                if tuple(1 - b for b in pat) in observed
            )
            gaps.append(1.0 - n_with_complement / len(observed))

        return {'heyting_gap': float(np.mean(gaps)) if gaps else 0.0}

    def _zariski_separation(self, X: np.ndarray, rng) -> dict:
        """Fraction of point pairs algebraically non-separable at degree 2.

        Points p, q are Zariski-non-separable if every polynomial vanishing
        at p also vanishes at q. We approximate this by cosine similarity
        of their degree-2 monomial evaluation vectors.
        """
        from itertools import combinations_with_replacement
        n, d = X.shape
        if n < 10:
            return {'nonsep_fraction': 0.0}

        columns = [np.ones(n)]
        for dd in range(1, 3):
            for idx in combinations_with_replacement(range(d), dd):
                col = np.ones(n)
                for var in idx:
                    col = col * X[:, var]
                columns.append(col)
        V = np.column_stack(columns)

        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.maximum(norms, 1e-15)

        n_pairs = min(10000, n * (n - 1) // 2)
        i_idx = rng.integers(0, n, n_pairs)
        j_idx = rng.integers(0, n, n_pairs)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        if len(i_idx) == 0:
            return {'nonsep_fraction': 0.0}

        cos_sim = np.sum(V[i_idx] * V[j_idx], axis=1)
        return {'nonsep_fraction': float(np.mean(cos_sim > 0.999))}

    def _heyting_stability(self, data: np.ndarray, n_windows: int = 16) -> dict:
        """Stability of Heyting gap across data windows.

        Computes heyting_gap on non-overlapping windows and returns
        mean_gap / (std_gap + 0.01). Structured data has consistent
        non-Hausdorff gaps (high stability); noise has erratic gaps
        (low stability).

        Evolved via ShinkaEvolve zariski_discovery v1 gen 34+.
        """
        chunk_size = len(data) // n_windows
        if chunk_size < 100:
            gap = self._heyting_gap(data)['heyting_gap']
            return {'heyting_stability': gap / 0.01 if gap > 0 else 0.0}
        gaps = [self._heyting_gap(data[i * chunk_size:(i + 1) * chunk_size])['heyting_gap']
                for i in range(n_windows)]
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        return {'heyting_stability': float(mean_gap / (std_gap + 0.01))}

    def _residual_convexity(self, X: np.ndarray) -> dict:
        """Convexity (elbow) in the log-residual spectrum.

        Computes the discrete second derivative of log10(residuals) at
        degrees 1-3. A positive value means the residual drops sharply
        then plateaus — signature of data fitting a specific algebraic
        degree. Near zero for noise (smooth residual curve).

        Evolved via ShinkaEvolve zariski_discovery v1 gen 32+.
        """
        from itertools import combinations_with_replacement
        n, d = X.shape
        residuals = []
        for deg in range(1, self.max_degree + 1):
            columns = [np.ones(n)]
            for dd in range(1, deg + 1):
                for idx in combinations_with_replacement(range(d), dd):
                    col = np.ones(n)
                    for var in idx:
                        col = col * X[:, var]
                    columns.append(col)
            n_mono = len(columns)
            if n_mono >= n:
                break
            V = np.column_stack(columns)
            s = np.linalg.svd(V, compute_uv=False)
            residuals.append(s[-1] / np.sqrt(n) if s[0] > 1e-15 else 0.0)

        if len(residuals) >= 3:
            log_res = np.log10(np.array(residuals) + 1e-15)
            convexity = (log_res[0] - log_res[1]) - (log_res[1] - log_res[2])
            return {'residual_convexity': float(max(0, convexity))}
        return {'residual_convexity': 0.0}


# =============================================================================
# CAYLEY (GEOMETRIC GROUP THEORY) GEOMETRY
# =============================================================================

class CayleyGeometry(ExoticGeometry):
    """
    Cayley (Geometric Group Theory) Geometry — large-scale graph invariants.

    Builds a k-NN graph on the delay-embedded point cloud and computes
    invariants from geometric group theory:

    1. Gromov δ-hyperbolicity: model-free large-scale curvature via the
       4-point condition. δ≈0 = tree-like, δ/diam≈0.25 = flat/Euclidean.
    2. Growth exponent: polynomial exponent β of ball growth |B(r)| ~ r^β
       in the graph metric. β≈1 = 1D curve, β≈2 = area-filling, β≈3 = volume.
    3. Spectral gap: Fiedler eigenvalue of the graph Laplacian, measuring
       expansion/amenability (Cheeger inequality).

    These measure genuinely different properties from existing metrics:
    - δ-hyperbolicity is model-free (vs HyperbolicGeometry's Poincaré disk)
    - Growth exponent uses graph distance (vs correlation dimension's Euclidean)
    - Spectral gap measures global expansion (nothing comparable in framework)
    """

    def __init__(self, input_scale=255.0, embed_dim=3, k_nn=10, n_subsample=500):
        self.input_scale = input_scale
        self.embed_dim = embed_dim
        self.k_nn = k_nn
        self.n_subsample = n_subsample

    @property
    def name(self) -> str:
        return "Cayley"

    @property
    def dimension(self) -> str:
        return "graph"

    @property
    def description(self) -> str:
        return ("Geometric group theory invariants on k-NN graphs: "
                "Gromov hyperbolicity, polynomial growth exponent, "
                "and spectral gap (Cheeger constant).")

    @property
    def view(self) -> str:
        return "topological"

    @property
    def detects(self) -> str:
        return "Large-scale curvature, intrinsic dimension, graph expansion"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Delay embedding into R^d, normalized to [0,1]."""
        data = self.validate_data(data)
        x = self._normalize_to_unit(data, self.input_scale)
        n = len(x) - self.embed_dim + 1
        if n <= 0:
            return np.empty((0, self.embed_dim))
        X = np.zeros((n, self.embed_dim))
        for i in range(self.embed_dim):
            X[:, i] = x[i:i + n]
        return X

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        data = self.validate_data(data)
        X = self.embed(data)

        rng = np.random.default_rng(42)
        if len(X) > self.n_subsample:
            idx = rng.choice(len(X), self.n_subsample, replace=False)
            X = X[np.sort(idx)]

        # Tiny jitter to break ties from uint8 quantization
        X = X + rng.normal(0, 1e-6, X.shape)

        metrics = {}
        n = len(X)

        if n < 20:
            return GeometryResult(geometry_name=self.name, metrics={
                'delta_hyp_norm': 0.0,
                'growth_exponent': 0.0, 'spectral_gap': 0.0,
                'saturation_radius': 0.0, 'local_linearity': 1.0 / self.embed_dim,
            })

        # Pairwise distance matrix (needed for δ-hyperbolicity)
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(X))
        diameter = D.max()

        metrics.update(self._gromov_hyperbolicity(D, diameter, rng))

        # k-NN graph (needed for growth + spectral gap)
        adj = self._build_knn_graph(X)
        metrics.update(self._growth_exponent(adj, n, rng))
        metrics.update(self._spectral_gap(adj, n))
        metrics.update(self._local_linearity(X))

        return GeometryResult(geometry_name=self.name, metrics=metrics)

    def _build_knn_graph(self, X: np.ndarray):
        """Build symmetric k-NN graph as sparse adjacency matrix."""
        from scipy.spatial import KDTree
        from scipy.sparse import csr_matrix

        n = len(X)
        k = min(self.k_nn, n - 1)
        tree = KDTree(X)
        dists, indices = tree.query(X, k=k + 1)

        rows, cols, vals = [], [], []
        for i in range(n):
            for j_idx in range(1, k + 1):
                j = indices[i, j_idx]
                d = dists[i, j_idx]
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([d, d])

        adj = csr_matrix((vals, (rows, cols)), shape=(n, n))
        return adj.minimum(adj.T)

    def _gromov_hyperbolicity(self, D: np.ndarray, diameter: float,
                              rng, n_samples: int = 50000) -> dict:
        """Gromov δ via 4-point condition: sample random quadruples,
        compute δ = (S_max - S_mid) / 2 for the three pairwise-sum orderings."""
        n = len(D)
        if diameter < 1e-10 or n < 4:
            return {'delta_hyp_norm': 0.0}

        # Vectorized sampling
        quads = rng.choice(n, size=(n_samples, 4), replace=True)
        # Reject quads with repeated indices (rare for large n)
        x, y, z, w = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

        S1 = D[x, y] + D[z, w]
        S2 = D[x, z] + D[y, w]
        S3 = D[x, w] + D[y, z]

        sums = np.stack([S1, S2, S3], axis=1)
        sums.sort(axis=1)
        deltas = (sums[:, 2] - sums[:, 1]) / 2

        return {
            'delta_hyp_norm': float(np.max(deltas) / diameter),
        }

    def _growth_exponent(self, adj, n: int, rng, n_starts: int = 20,
                         max_r: int = 100) -> dict:
        """Polynomial growth exponent β from BFS ball counting on k-NN graph.

        |B(r)| ~ r^β in graph metric. β≈1 for 1D curves, β≈2 for surfaces.
        """
        from scipy.sparse import csr_matrix
        binary_adj = (adj > 0).astype(np.float32)
        starts = rng.choice(n, min(n_starts, n), replace=False)

        all_counts = []
        for start in starts:
            visited = {int(start)}
            frontier = {int(start)}
            counts = [1]

            for r in range(1, max_r + 1):
                new_frontier = set()
                for node in frontier:
                    neighbors = binary_adj.getrow(node).indices
                    for nb in neighbors:
                        nb = int(nb)
                        if nb not in visited:
                            visited.add(nb)
                            new_frontier.add(nb)
                frontier = new_frontier
                counts.append(len(visited))
                if len(visited) >= n or not frontier:
                    break
            all_counts.append(counts)

        # Average ball sizes at each radius
        max_len = max(len(c) for c in all_counts)
        avg_counts = np.zeros(max_len)
        count_at_r = np.zeros(max_len)
        for c in all_counts:
            for r, v in enumerate(c):
                avg_counts[r] += v
                count_at_r[r] += 1
        avg_counts /= np.maximum(count_at_r, 1)

        # Only use radii where ball hasn't saturated
        usable = (avg_counts < 0.9 * n) & (np.arange(max_len) > 0)

        results = {}
        if np.sum(usable) >= 3:
            r_vals = np.where(usable)[0].astype(float)
            b_vals = avg_counts[usable]
            log_r = np.log(r_vals + 1e-10)
            log_b = np.log(b_vals + 1e-10)
            poly_fit = np.polyfit(log_r, log_b, 1)
            results['growth_exponent'] = float(poly_fit[0])

            # Saturation: radius where B(r) reaches 50% of points
            half_idx = np.argmax(avg_counts >= 0.5 * n)
            # Normalize by n so it's scale-free
            results['saturation_radius'] = float(half_idx) / n if half_idx > 0 else 1.0
        else:
            results['growth_exponent'] = 0.0
            results['saturation_radius'] = 1.0 / n  # saturates immediately

        return results

    def _spectral_gap(self, adj, n: int) -> dict:
        """Normalized Fiedler eigenvalue of graph Laplacian.

        By Cheeger inequality: λ₁/2 ≤ h(G) ≤ √(2λ₁).
        λ₁≈0: bottleneck/amenable. λ₁ large: expander.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        A = (adj > 0).astype(np.float64)
        A = A.maximum(A.T)

        degrees = np.array(A.sum(axis=1)).flatten()
        D = csr_matrix((degrees, (range(n), range(n))), shape=(n, n))
        L = D - A

        # Normalized Laplacian
        deg_inv_sqrt = np.zeros(n)
        nonzero = degrees > 0
        deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = csr_matrix((deg_inv_sqrt, (range(n), range(n))), shape=(n, n))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        try:
            eigs = eigsh(L_norm, k=min(3, n - 1), which='SM',
                         return_eigenvectors=False)
            eigs = np.sort(np.real(eigs))
            fiedler = max(float(eigs[1]) if len(eigs) > 1 else 0.0, 0.0)

            max_eig = eigsh(L_norm, k=1, which='LM',
                            return_eigenvectors=False)[0]
            spectral_gap = fiedler / max(float(max_eig), 1e-10)
        except Exception:
            spectral_gap = 0.0

        return {'spectral_gap': spectral_gap}

    def _local_linearity(self, X: np.ndarray) -> dict:
        """Average local linearity via PCA of k-NN neighborhoods.

        For each point, compute PCA of its k nearest neighbors and measure
        the fraction of variance captured by the first principal component.
        Structured data on low-dimensional manifolds yields linearity ~ 1.0;
        isotropic noise yields ~ 1/embed_dim.
        """
        from scipy.spatial import KDTree

        n = len(X)
        if n <= self.k_nn:
            return {'local_linearity': 1.0 / self.embed_dim}

        tree = KDTree(X)
        _, indices = tree.query(X, k=self.k_nn + 1)

        scores = np.empty(n)
        for i in range(n):
            nbrs = X[indices[i, 1:]]
            centered = nbrs - nbrs.mean(axis=0)
            cov = centered.T @ centered
            eigs = np.linalg.eigh(cov)[0]
            total = eigs.sum()
            if total < 1e-9:
                scores[i] = 1.0 / self.embed_dim
            else:
                scores[i] = eigs[-1] / total

        return {'local_linearity': float(np.mean(scores))}


# =============================================================================
# SPECTRAL GRAPH GEOMETRY (HEAT KERNEL)
# =============================================================================

class SpectralGraphGeometry(ExoticGeometry):
    """
    Spectral Graph Geometry — heat kernel and Laplacian spectral invariants.

    Builds an epsilon-neighborhood graph on the delay-embedded point cloud
    and extracts the full shape of the Laplacian spectrum via:

    1. Spectral dimension (d_s): from the heat kernel trace Z(t) = Σ exp(-λ_i t),
       d_s = -2 d(log Z)/d(log t) at characteristic scale. Measures the effective
       dimensionality seen by diffusion on the data manifold.
       For regular lattices d_s = topological dimension.
       For fractals d_s < Hausdorff dimension (anomalous diffusion).

    2. Weyl exponent (α): fit λ_n ~ n^α. For d-dimensional Riemannian manifolds,
       Weyl's law gives α = 2/d. Captures dimensionality from the eigenvalue
       growth rate. Distinct from growth_exponent (BFS) and correlation dimension.

    3. Gap ratio (λ₃/λ₂): spectral gap above the Fiedler eigenvalue.
       Large → strong cluster structure. Near 1 → continuous spectrum.
       Different from Cayley's spectral_gap (λ₂/λ_max normalization).

    Uses epsilon-neighborhood graphs (different from CayleyGeometry's k-NN)
    with 5D delay embedding (vs Cayley's 3D). Non-redundant with Cayley:
    max |Spearman rho| = 0.77 across all metric pairs.
    """

    def __init__(self, input_scale=255.0, embed_dim=5, n_subsample=1500,
                 target_degree=8, n_eigs=60):
        self.input_scale = input_scale
        self.embed_dim = embed_dim
        self.n_subsample = n_subsample
        self.target_degree = target_degree
        self.n_eigs = n_eigs

    @property
    def name(self) -> str:
        return "Spectral Graph"

    @property
    def dimension(self) -> str:
        return "graph spectrum"

    @property
    def description(self) -> str:
        return ("Heat kernel spectral invariants on epsilon-neighborhood graphs: "
                "spectral dimension, Weyl exponent, eigenvalue gap ratio.")

    @property
    def view(self) -> str:
        return "topological"

    @property
    def detects(self) -> str:
        return "Diffusion dimensionality, spectral shape, cluster structure"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Delay embedding into R^d, normalized to [0,1]."""
        data = self.validate_data(data)
        x = self._normalize_to_unit(data, self.input_scale)
        n = len(x) - self.embed_dim + 1
        if n <= 0:
            return np.empty((0, self.embed_dim))
        X = np.zeros((n, self.embed_dim))
        for i in range(self.embed_dim):
            X[:, i] = x[i:i + n]
        return X

    def _build_epsilon_graph(self, X: np.ndarray):
        """Build epsilon-neighborhood graph with adaptive epsilon."""
        from scipy.spatial import KDTree
        from scipy.sparse import csr_matrix as sp_csr

        n = len(X)
        tree = KDTree(X)

        # Adaptive epsilon: use k-th neighbor distance as epsilon
        k = min(self.target_degree + 1, n - 1)
        dists, _ = tree.query(X[:min(200, n)], k=k)
        eps = np.median(dists[:, -1])

        # Floor for degenerate embeddings (binary data etc.)
        if eps < 1e-10:
            from scipy.spatial.distance import pdist
            pw = pdist(X[:min(50, n)])
            pw_pos = pw[pw > 1e-12]
            eps = np.percentile(pw_pos, 10) if len(pw_pos) > 0 else 0.01

        pairs = tree.query_pairs(r=eps, output_type='ndarray')
        if len(pairs) == 0:
            return None

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        vals = np.ones(len(rows))
        return sp_csr((vals, (rows, cols)), shape=(n, n))

    def _laplacian_spectrum(self, A):
        """Compute smallest eigenvalues of normalized Laplacian, largest component."""
        from scipy.sparse import csr_matrix as sp_csr
        from scipy.sparse.linalg import eigsh
        from scipy.sparse.csgraph import connected_components

        # Largest connected component
        n_comp, labels = connected_components(A, directed=False)
        if n_comp > 1:
            sizes = np.bincount(labels)
            largest = np.argmax(sizes)
            idx = np.where(labels == largest)[0]
            A = A[idx][:, idx]

        n = A.shape[0]
        if n < 20:
            return None

        n_eigs = min(self.n_eigs, n - 2)
        if n_eigs < 3:
            return None

        degrees = np.array(A.sum(axis=1)).ravel()
        mask = degrees > 0
        if mask.sum() < 20:
            return None

        A_sub = A[mask][:, mask]
        d_sub = degrees[mask]
        n_sub = len(d_sub)

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = 1.0 / np.sqrt(d_sub)
        D_inv_sqrt = sp_csr(
            (d_inv_sqrt, (np.arange(n_sub), np.arange(n_sub))),
            shape=(n_sub, n_sub)
        )
        L_norm = sp_csr(np.eye(n_sub)) - D_inv_sqrt @ A_sub @ D_inv_sqrt

        try:
            eigs = eigsh(L_norm, k=min(n_eigs, n_sub - 2),
                         which='SM', return_eigenvectors=False)
            eigs = np.sort(np.real(eigs))
            return np.maximum(eigs, 0)
        except Exception:
            return None

    def _spectral_dimension(self, eigs: np.ndarray) -> float:
        """d_s at characteristic scale from heat kernel trace Z(t) = Σ exp(-λ_i t)."""
        ts = np.logspace(-1, 1, 50)  # t in [0.1, 10]
        log_Z = np.array([
            np.log(np.sum(np.exp(-eigs * t)) + 1e-300) for t in ts
        ])
        log_t = np.log(ts)
        d_log_Z = np.gradient(log_Z, log_t)
        d_s = -2 * d_log_Z
        return float(d_s[len(ts) // 2])  # midpoint = characteristic scale

    def _weyl_exponent(self, eigs: np.ndarray) -> float:
        """Eigenvalue growth exponent: λ_n ~ n^α (Weyl's law)."""
        pos = eigs[eigs > 1e-10]
        if len(pos) < 5:
            return 0.0
        ns = np.arange(1, len(pos) + 1)
        slope, _ = np.polyfit(np.log(ns), np.log(pos), 1)
        return float(slope)

    def _gap_ratio(self, eigs: np.ndarray) -> float:
        """λ₃/λ₂ — gap above Fiedler eigenvalue."""
        pos = eigs[eigs > 1e-10]
        if len(pos) < 3:
            return 1.0
        return float(pos[2] / pos[1]) if pos[1] > 1e-10 else 1.0

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        data = self.validate_data(data)
        X = self.embed(data)

        fallback = GeometryResult(
            geometry_name=self.name,
            metrics={'spectral_dim': 0.0, 'weyl_exponent': 0.0},
        )

        if len(X) < 30:
            return fallback

        rng = np.random.default_rng(42)
        if len(X) > self.n_subsample:
            idx = rng.choice(len(X), self.n_subsample, replace=False)
            X = X[idx]

        A = self._build_epsilon_graph(X)
        if A is None:
            return fallback

        eigs = self._laplacian_spectrum(A)
        if eigs is None:
            return fallback

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'spectral_dim': self._spectral_dimension(eigs),
                'weyl_exponent': self._weyl_exponent(eigs),
            }
        )


# =============================================================================
# BOLTZMANN GEOMETRY (ENERGY LANDSCAPE)
# =============================================================================

class BoltzmannGeometry(ExoticGeometry):
    """
    Boltzmann (Energy Landscape) Geometry — pairwise Ising model analysis.

    Fits the maximum-entropy pairwise model (Ising model) to binary-quantized
    sliding windows:

        P(x) = (1/Z) exp(Σ J_ij x_i x_j + Σ h_i x_i)

    The coupling matrix J encodes which positions in a window interact and
    how strongly. Its structure reveals the energy landscape geometry:

    1. Coupling strength (mean |J_ij|): interaction energy scale. How much
       do positions constrain each other? Noise → 0, structured → high.

    2. Frustration: fraction of triangles (i,j,k) where the three pairwise
       couplings are mutually contradictory (sign(J_ij·J_jk·J_ik) < 0).
       Frustrated = can't satisfy all constraints simultaneously.
       Quasicrystals show maximal frustration (Fibonacci = 1.0) because
       incommensurable correlations create a spin-glass landscape.
       Monotonic signals have zero frustration. Noise ≈ 0.5.

    3. Spectral gap of J (λ₂/λ₁): how degenerate is the coupling spectrum.
       Near a phase transition the gap closes. Clear hierarchical structure
       has a dominant eigenvalue (small gap).

    Solves the "inverse Ising problem" (Schneidman et al. 2006, Weigt et al.
    2009) via J = -C_conn^{-1} where C_conn is the connected correlation matrix.
    """

    def __init__(self, input_scale=255.0, window=8):
        self.input_scale = input_scale
        self.window = window

    @property
    def name(self) -> str:
        return "Boltzmann"

    @property
    def dimension(self) -> str:
        return "energy landscape"

    @property
    def description(self) -> str:
        return ("Pairwise Ising model on binary windows: coupling strength, "
                "geometric frustration, and spectral gap of the interaction matrix.")

    @property
    def view(self) -> str:
        return "dynamical"

    @property
    def detects(self) -> str:
        return "Interaction strength, spin-glass frustration, criticality"

    @property
    def encoding_invariant(self) -> bool:
        return True

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Binary-quantize and create sliding windows in {-1, +1}^d."""
        data = self.validate_data(data)
        uniq = np.unique(data)
        if len(uniq) <= 4:
            threshold = (float(uniq.min()) + float(uniq.max())) / 2.0
        else:
            threshold = float(np.median(data))
        bits = np.where(data > threshold, 1.0, -1.0)

        n = len(bits) - self.window + 1
        if n < 100:
            return np.empty((0, self.window))
        windows = np.zeros((n, self.window))
        for i in range(self.window):
            windows[:, i] = bits[i:i + n]
        return windows

    def _fit_ising(self, windows: np.ndarray):
        """Fit pairwise Ising model via inverse correlation matrix."""
        n, d = windows.shape
        m = windows.mean(axis=0)
        C = (windows.T @ windows) / n
        C_conn = C - np.outer(m, m)

        # Regularize for invertibility
        C_reg = C_conn + 0.01 * np.eye(d)
        try:
            J = -np.linalg.inv(C_reg)
        except np.linalg.LinAlgError:
            J = -np.linalg.pinv(C_reg)

        np.fill_diagonal(J, 0)
        return J

    def _coupling_strength(self, J: np.ndarray) -> float:
        """Mean absolute coupling |J_ij|."""
        d = J.shape[0]
        triu = J[np.triu_indices(d, k=1)]
        return float(np.mean(np.abs(triu)))

    def _frustration(self, J: np.ndarray) -> float:
        """Fraction of frustrated triangles (sign product < 0).

        Samples 5000 triangles for d > 15, exhaustive otherwise.
        """
        d = J.shape[0]
        if d < 3:
            return 0.0

        rng = np.random.default_rng(42)

        if d > 15:
            n_samples = 5000
            n_frustrated = 0
            for _ in range(n_samples):
                i, j, k = rng.choice(d, 3, replace=False)
                if J[i, j] * J[j, k] * J[i, k] < 0:
                    n_frustrated += 1
            return float(n_frustrated / n_samples)
        else:
            n_triangles = 0
            n_frustrated = 0
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        n_triangles += 1
                        if J[i, j] * J[j, k] * J[i, k] < 0:
                            n_frustrated += 1
            return float(n_frustrated / n_triangles) if n_triangles > 0 else 0.0

    def _spectral_gap_J(self, J: np.ndarray) -> float:
        """λ₂/λ₁ of coupling matrix (by absolute value)."""
        eigs = np.sort(np.abs(np.linalg.eigvalsh(J)))[::-1]
        if len(eigs) < 2 or eigs[0] < 1e-10:
            return 1.0
        return float(eigs[1] / eigs[0])

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        data = self.validate_data(data)
        windows = self.embed(data)

        fallback = GeometryResult(
            geometry_name=self.name,
            metrics={'coupling_strength': 0.0, 'frustration': 0.5,
                     'spectral_gap_J': 1.0,
                     'nn_dominance': 0.0, 'coupling_temporal_variance': 0.0}
        )

        if len(windows) < 200:
            return fallback

        J = self._fit_ising(windows)
        cs = self._coupling_strength(J)

        # NN dominance: ratio of nearest-neighbor coupling strength to
        # overall coupling strength. Measures locality of interactions —
        # structured signals have stronger adjacent-position couplings.
        # (evolved via ShinkaEvolve)
        d = J.shape[0]
        nn_couplings = np.abs(np.diag(J, k=1))
        nn_dominance = float(np.mean(nn_couplings) / (cs + 1e-9)) if d > 1 else 1.0

        # Coupling temporal variance: relative Frobenius norm change of J
        # between first and second half of the signal. Measures how the
        # energy landscape evolves — structured signals have nonstationary
        # coupling patterns. (evolved via ShinkaEvolve)
        n_win = len(windows)
        mid = n_win // 2
        J1 = self._fit_ising(windows[:mid])
        J2 = self._fit_ising(windows[mid:])
        diff_norm = np.linalg.norm(J1 - J2, 'fro')
        avg_norm = (np.linalg.norm(J1, 'fro') + np.linalg.norm(J2, 'fro')) / 2.0
        coupling_temporal_variance = float(diff_norm / (avg_norm + 1e-9))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'coupling_strength': cs,
                'frustration': self._frustration(J),
                'spectral_gap_J': self._spectral_gap_J(J),
                'nn_dominance': nn_dominance,
                'coupling_temporal_variance': coupling_temporal_variance,
            }
        )


# =============================================================================
# GOTTWALD-MELBOURNE 0-1 TEST FOR CHAOS
# =============================================================================

class GottwaldMelbourneGeometry(ExoticGeometry):
    """
    Gottwald-Melbourne 0-1 test for chaos.

    A binary chaos detector that does not require phase-space reconstruction
    or embedding dimension estimation. Works directly on the 1D time series.

    Constructs 2D translation variables for random frequencies c:

        p_c(n) = Σ_{j=1}^{n} φ(j) cos(jc)
        q_c(n) = Σ_{j=1}^{n} φ(j) sin(jc)

    where φ(j) is the (normalized) time series value at step j.

    For chaotic signals, (p_c, q_c) performs a Brownian-like random walk
    (mean square displacement grows linearly → K ≈ 1).
    For regular signals, (p_c, q_c) is bounded (MSD bounded → K ≈ 0).

    Uses the correlation method (Gottwald & Melbourne 2009) for robustness:
    K_c = corr(Δ, ξ) where Δ(n) = MSD at lag n and ξ(n) = n, for
    n = 1..n_cut. This avoids the log-log regression instabilities of the
    original regression method.

    Median over 10 random c values (uniformly in (π/5, 4π/5), avoiding
    resonances) gives a robust estimate.

    Metrics:
    - k_statistic: median K across random frequencies. 0 = regular, 1 = chaotic.
    - k_variance: IQR of K values. Low = confident classification,
      high = ambiguous (e.g. edge-of-chaos, intermittency).

    Reference: Gottwald & Melbourne, "On the implementation of the 0-1 test
    for chaos", SIAM J. Appl. Dyn. Syst. 8(1), 2009.
    """

    def __init__(self, input_scale=255.0, n_freqs=10, n_cut_frac=0.1):
        self.input_scale = input_scale
        self.n_freqs = n_freqs
        self.n_cut_frac = n_cut_frac  # fraction of N used for correlation

    @property
    def name(self) -> str:
        return "Gottwald-Melbourne"

    @property
    def dimension(self) -> str:
        return "translation plane"

    @property
    def description(self) -> str:
        return ("0-1 test for chaos: K≈1 for chaotic dynamics, K≈0 for regular. "
                "No embedding required.")

    @property
    def view(self) -> str:
        return "dynamical"

    @property
    def detects(self) -> str:
        return "Chaos vs regular dynamics, classification confidence"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def _k_correlation(self, phi: np.ndarray, c: float) -> float:
        """Compute K_c via the correlation method.

        Build p(n), q(n), then M_c(n) = <(p(n+j)-p(j))^2 + (q(n+j)-q(j))^2>_j
        for n = 1..n_cut.  K_c = corr(M_c, ξ) where ξ(n) = n.
        """
        N = len(phi)
        n_cut = max(10, int(N * self.n_cut_frac))
        if n_cut >= N // 2:
            n_cut = N // 2 - 1
        if n_cut < 5:
            return 0.5

        # Build translation variables
        j = np.arange(1, N + 1, dtype=np.float64)
        p = np.cumsum(phi * np.cos(j * c))
        q = np.cumsum(phi * np.sin(j * c))

        # Mean square displacement for lags 1..n_cut
        # Subtract off the expected linear drift term for bounded signals
        # D_c(n) = M_c(n) - V_osc(n) where V_osc = E[φ]^2 * (1-cos(nc))/(1-cos(c))
        mean_phi = np.mean(phi)
        msd = np.empty(n_cut)
        for n in range(1, n_cut + 1):
            dp = p[n:] - p[:-n]
            dq = q[n:] - q[:-n]
            msd[n - 1] = np.mean(dp * dp + dq * dq)

        # Subtract oscillatory term to regularize
        cos_c = np.cos(c)
        if abs(1 - cos_c) > 1e-10:
            v_osc = mean_phi ** 2 * (1 - np.cos(np.arange(1, n_cut + 1) * c)) / (1 - cos_c)
            msd = msd - v_osc

        # Correlation of D_c with linear ramp
        xi = np.arange(1, n_cut + 1, dtype=np.float64)
        # K = correlation coefficient
        if np.std(msd) < 1e-15:
            return 0.0
        k = np.corrcoef(xi, msd)[0, 1]
        return float(np.clip(k, 0, 1))

    @staticmethod
    def _spectral_flatness(x):
        """Spectral flatness of a signal: geometric_mean(PSD) / arithmetic_mean(PSD).
        Returns 1 - flatness so that high = structured, low = noise-like."""
        if len(x) < 20 or np.std(x) < 1e-15:
            return 0.0
        ps = np.abs(np.fft.rfft(x)) ** 2 + 1e-16
        gmean = np.exp(np.mean(np.log(ps)))
        amean = np.mean(ps)
        flatness = gmean / amean if amean > 1e-16 else 1.0
        return float(1.0 - np.clip(flatness, 0, 1))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        phi = self.embed(data)
        N = len(phi)

        fallback = GeometryResult(
            geometry_name=self.name,
            metrics={'k_statistic': 0.5, 'k_variance': 0.5,
                     'angular_spectral_structure': 0.0,
                     'radial_spectral_structure': 0.0}
        )

        if N < 200:
            return fallback

        # Center the signal (remove mean to avoid drift artifacts)
        phi = phi - np.mean(phi)

        # Draw n_freqs random frequencies in (π/5, 4π/5), avoiding resonances
        rng = np.random.default_rng(42)
        c_vals = rng.uniform(np.pi / 5, 4 * np.pi / 5, self.n_freqs)

        k_vals = np.array([self._k_correlation(phi, c) for c in c_vals])

        k_median = float(np.median(k_vals))
        k_iqr = float(np.percentile(k_vals, 75) - np.percentile(k_vals, 25))

        # Radial-angular coherence (ShinkaEvolve gottwald_v1 gen 44).
        # Decompose (p,q) trajectory into polar coords, measure spectral
        # flatness of dθ/dt (angular velocity) and dr/dt (radial velocity).
        # Structured signals have non-flat spectra in at least one component.
        angular_scores = []
        radial_scores = []
        start_n = N // 10  # skip transient near origin

        for c in c_vals:
            j = np.arange(1, N + 1, dtype=np.float64)
            p = np.cumsum(phi * np.cos(j * c))
            q = np.cumsum(phi * np.sin(j * c))

            if N - start_n < 100:
                angular_scores.append(0.0)
                radial_scores.append(0.0)
                continue

            p_s, q_s = p[start_n:], q[start_n:]

            # Angular velocity spectral flatness
            theta = np.unwrap(np.arctan2(q_s, p_s))
            d_theta = np.diff(theta)
            angular_scores.append(self._spectral_flatness(d_theta))

            # Radial velocity spectral flatness
            r = np.sqrt(p_s ** 2 + q_s ** 2)
            dr = np.diff(r)
            radial_scores.append(self._spectral_flatness(dr))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'k_statistic': float(np.clip(k_median, 0, 1)),
                'k_variance': float(np.clip(k_iqr, 0, 1)),
                'angular_spectral_structure': float(np.clip(
                    np.median(angular_scores), 0, 1)),
                'radial_spectral_structure': float(np.clip(
                    np.median(radial_scores), 0, 1)),
            }
        )


# =============================================================================
# ORDINAL PARTITION GEOMETRY
# =============================================================================

class OrdinalPartitionGeometry(ExoticGeometry):
    """
    Ordinal Partition Geometry — transition dynamics of ordinal patterns.

    While HigherOrderGeometry counts ordinal patterns (Bandt-Pompe), this
    geometry analyzes how patterns *transition* to each other, capturing
    the dynamics of the symbolic sequence.

    Constructs the ordinal transition matrix T where T[π_i, π_j] counts
    how often pattern π_i at position t is followed by pattern π_j at
    position t+1 (overlapping windows of order d).

    Metrics:
    - transition_entropy: Conditional entropy H(π_{t+1} | π_t), normalized
      by log2(d) — the maximum for overlapping order-d patterns (since the
      next pattern is determined by where 1 new value ranks among d-1
      carry-over values, giving d possible outcomes).
      Periodic ≈ 0 (deterministic transitions), noise ≈ 1 (uniform over
      d positions), chaos ≈ 0.5-0.9 (structured but not deterministic).

    - time_irreversibility: Total variation distance between the ordinal
      pattern distribution P(π) and its time-reversed counterpart P(π̃),
      where π̃ is obtained by reversing the rank sequence.
      For time-reversible processes (IID noise, Hamiltonian systems):
      P(π) = P(π̃), so TV ≈ 0. For irreversible dynamics (dissipative
      chaos, driven oscillators): P(π) ≠ P(π̃), so TV > 0.

    - statistical_complexity: López-Ruiz-Mancini-Calbet complexity
      C_JS = Q_J × H_s on the ordinal pattern distribution, where Q_J
      is the normalized Jensen-Shannon divergence from uniform and H_s
      is the normalized Shannon entropy. Peaks at edge-of-chaos; zero
      for both perfect order and perfect disorder.

    - forbidden_transitions: Fraction of legal ordinal transitions that
      are absent. For overlapping order-d patterns, each pattern has
      exactly d possible successors (d × d! = 600 total for d=5).
      Deterministic 1D maps forbid many transitions (the map constrains
      which rank orderings can follow which). Noise and mixing maps
      allow all transitions. Orthogonal to perm_forbidden (which counts
      absent patterns, not absent transitions between patterns).

    - markov_mixing: Spectral gap (1 - |λ₂|) of the VALUE-transition
      matrix (16 equiprobable bins). Measures how quickly the process
      forgets its current value — complementary to the rank-based
      metrics above which ignore value magnitudes.
      White noise ≈ 0.9 (near-IID), sine ≈ 0.002 (maximally persistent),
      chaos ≈ 0.3-0.5 (deterministic but mixing).
      max |r| = 0.90 with Projective:collinearity (below 0.95 threshold).

    - memory_order: Log-likelihood improvement of order-2 over order-1
      Markov model on 8 equiprobable value bins. Measures whether the
      process has memory beyond one step — i.e., whether knowing TWO
      previous values helps predict the next, beyond what one value gives.
      Sine ≈ 0.001, Brownian ≈ 0.04, Hénon ≈ 0.45, Lorenz ≈ 0.50.
      High values indicate hidden state (higher-dimensional dynamics
      projected to 1D). max |r| = 0.57 — genuinely orthogonal to all
      192 existing metrics.

    Uses order d=5 (120 possible patterns).

    References:
    - Bandt & Pompe, Phys. Rev. Lett. 88, 2002 (ordinal patterns)
    - Rosso et al., Phys. Rev. Lett. 99, 2007 (complexity-entropy plane)
    - Zanin et al., Entropy 14, 2012 (ordinal transition graphs)
    """

    def __init__(self, input_scale=255.0, order=5):
        self.input_scale = input_scale
        self.order = order
        self._n_states = math.factorial(order)
        self._reverse_map = self._build_reverse_map()
        self._legal_mask = self._build_legal_transitions()

    def _build_reverse_map(self):
        """Lookup: Lehmer index -> Lehmer index of time-reversed pattern.

        Time reversal of rank pattern (r_0, ..., r_{d-1}) is
        (r_{d-1}, ..., r_0) — reverse the sequence, keep the ranks.
        """
        from itertools import permutations as iterperms
        d = self.order
        perm_to_idx = {}
        for idx, perm in enumerate(iterperms(range(d))):
            perm_to_idx[perm] = idx
        rev_map = np.zeros(self._n_states, dtype=np.int32)
        for idx, perm in enumerate(iterperms(range(d))):
            rev_perm = tuple(perm[d - 1 - k] for k in range(d))
            rev_map[idx] = perm_to_idx[rev_perm]
        return rev_map

    def _build_legal_transitions(self):
        """Precompute legal transition mask for overlapping order-d patterns.

        For overlapping windows, each pattern has exactly d possible successors:
        the new element can rank 0 through d-1 among the d elements of the
        next window. Returns a boolean mask (d!, d!) where True = legal.
        """
        from itertools import permutations as iterperms
        d = self.order
        all_perms = list(iterperms(range(d)))
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        n = self._n_states
        mask = np.zeros((n, n), dtype=bool)

        for idx, perm in enumerate(all_perms):
            # Drop first element, re-rank remaining d-1 elements
            sub = list(perm[1:])
            order = sorted(range(d - 1), key=lambda i: sub[i])
            re_ranked = [0] * (d - 1)
            for new_rank, old_pos in enumerate(order):
                re_ranked[old_pos] = new_rank

            # Insert new element at each possible rank k = 0..d-1
            for k in range(d):
                new_perm = tuple(
                    r + 1 if r >= k else r for r in re_ranked
                ) + (k,)
                mask[idx, perm_to_idx[new_perm]] = True

        return mask

    @property
    def name(self) -> str:
        return "Ordinal Partition"

    @property
    def dimension(self) -> str:
        return "symbolic"

    @property
    def description(self) -> str:
        return ("Ordinal pattern transition dynamics: conditional entropy, "
                "time irreversibility, and statistical complexity.")

    @property
    def view(self) -> str:
        return "dynamical"

    @property
    def detects(self) -> str:
        return "Transition predictability, time irreversibility, edge-of-chaos complexity"

    @property
    def encoding_invariant(self) -> bool:
        return True

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def _ordinal_patterns(self, x: np.ndarray) -> np.ndarray:
        """Convert time series to sequence of ordinal pattern indices.

        Each window of length `order` is mapped to the rank permutation,
        then to an integer index via the Lehmer code (factorial number system).
        """
        n = len(x) - self.order + 1
        if n < 2:
            return np.array([], dtype=np.int32)

        d = self.order
        # Factorials for Lehmer code
        factorials = np.array([math.factorial(d - 1 - k) for k in range(d - 1)])

        patterns = np.empty(n, dtype=np.int32)
        for i in range(n):
            window = x[i:i + d]
            # Rank: argsort of argsort gives rank
            rank = np.argsort(np.argsort(window))
            # Lehmer code: for each position, count how many remaining
            # elements are smaller
            code = 0
            for k in range(d - 1):
                count = 0
                for j in range(k + 1, d):
                    if rank[j] < rank[k]:
                        count += 1
                code += count * factorials[k]
            patterns[i] = code

        return patterns

    def _transition_matrix(self, patterns: np.ndarray) -> np.ndarray:
        """Build the transition count matrix from pattern sequence."""
        T = np.zeros((self._n_states, self._n_states), dtype=np.float64)
        for i in range(len(patterns) - 1):
            T[patterns[i], patterns[i + 1]] += 1
        return T

    def _transition_entropy(self, T: np.ndarray) -> float:
        """Conditional entropy H(π_{t+1} | π_t), normalized by log2(order).

        For overlapping order-d patterns, the next pattern is determined by
        where 1 new value ranks among d-1 carry-over values. There are d
        possible outcomes, so the maximum conditional entropy is log2(d).
        """
        row_sums = T.sum(axis=1)
        total = T.sum()
        if total < 1:
            return 0.0

        max_ent = np.log2(self.order)
        if max_ent < 1e-15:
            return 0.0

        h_cond = 0.0
        for i in range(self._n_states):
            if row_sums[i] == 0:
                continue
            p_row = T[i, :] / row_sums[i]
            p_row = p_row[p_row > 0]
            h_row = -np.sum(p_row * np.log2(p_row))
            h_cond += (row_sums[i] / total) * h_row

        return float(h_cond / max_ent)

    def _time_irreversibility(self, pattern_counts: np.ndarray) -> float:
        """Total variation distance between ordinal distribution and
        its time-reversed counterpart.

        Time reversal of rank pattern (r_0,...,r_{d-1}) is (r_{d-1},...,r_0).
        For time-reversible processes P(π) = P(π̃), so TV ≈ 0.
        """
        total = pattern_counts.sum()
        if total < 1:
            return 0.0

        p_fwd = pattern_counts / total

        # Build time-reversed distribution
        p_rev = np.zeros(self._n_states)
        for i in range(self._n_states):
            p_rev[self._reverse_map[i]] += p_fwd[i]

        # Total variation distance = ½ Σ |p - q|
        tv = 0.5 * np.sum(np.abs(p_fwd - p_rev))
        return float(tv)

    def _statistical_complexity(self, pattern_counts: np.ndarray) -> float:
        """Jensen-Shannon complexity C_JS = Q_J × H_s.

        Computed on the ordinal pattern distribution (d! bins).
        Q_J = JSD(p || uniform) / Q_0, H_s = H(p) / log2(d!).
        """
        total = pattern_counts.sum()
        if total < 1:
            return 0.0

        n_bins = self._n_states
        p = pattern_counts / total
        u = np.ones(n_bins) / n_bins

        # Normalized Shannon entropy
        p_pos = p[p > 0]
        h_max = np.log2(n_bins)
        if h_max < 1e-15:
            return 0.0
        h_s = -np.sum(p_pos * np.log2(p_pos)) / h_max

        # Jensen-Shannon divergence
        m = 0.5 * (p + u)
        kl_p_m = np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
        kl_u_m = np.sum(u * np.log2(u / m))
        jsd = 0.5 * (kl_p_m + kl_u_m)

        # Normalize Q_J to [0, 1] using the standard normalizer Q_0
        # (max JSD for n-bin distributions: delta vs uniform)
        n = n_bins
        Q_0 = -2.0 * ((n + 1.0) / n * np.log2(n + 1.0)
                       - 2.0 * np.log2(2.0 * n) + np.log2(n))
        q_j = jsd / Q_0 if Q_0 > 1e-15 else 0.0

        return float(np.clip(q_j * h_s, 0, 1))

    def _markov_mixing(self, data: np.ndarray, n_bins: int = 16) -> float:
        """Spectral gap of value-transition matrix: 1 - |λ₂(T)|.

        Quantizes data into equiprobable bins (normalizing the marginal
        so all structure is in transitions), builds the Markov transition
        matrix, and returns the spectral gap.

        Large gap = fast mixing (memoryless, IID-like).
        Small gap = slow mixing (persistent values, smooth trajectories).
        """
        n = len(data)
        if n < 50:
            return 0.5

        # Equiprobable binning
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(data, percentiles)
        edges[-1] += 1  # include max
        states = np.clip(np.digitize(data, edges[1:-1]), 0, n_bins - 1)

        # Build transition matrix
        T = np.zeros((n_bins, n_bins), dtype=np.float64)
        for i in range(n - 1):
            T[states[i], states[i + 1]] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T /= row_sums

        try:
            eigs = np.linalg.eigvals(T)
            mags = np.sort(np.abs(eigs))[::-1]
            if len(mags) >= 2:
                return float(np.clip(1.0 - mags[1], 0, 1))
            return 1.0
        except np.linalg.LinAlgError:
            return 0.5

    def _memory_order(self, data: np.ndarray, n_bins: int = 8) -> float:
        """Log-likelihood improvement of order-2 over order-1 Markov model.

        Quantizes data into n_bins equiprobable bins, fits both a first-order
        and second-order Markov model, and returns the per-step log-likelihood
        improvement. Positive values mean data has memory beyond one step.

        Uses n_bins=8 (not 16) to avoid sparse 3D contingency table with
        typical sample sizes (~2000).
        """
        n = len(data)
        if n < 100:
            return 0.0

        # Equiprobable binning
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(data, percentiles)
        edges[-1] += 1
        states = np.clip(np.digitize(data, edges[1:-1]), 0, n_bins - 1)

        # Order-1 transition matrix and log-likelihood
        T1 = np.zeros((n_bins, n_bins), dtype=np.float64)
        for i in range(n - 1):
            T1[states[i], states[i + 1]] += 1
        row_sums = T1.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T1 /= row_sums

        ll1 = 0.0
        for i in range(n - 1):
            p = T1[states[i], states[i + 1]]
            if p > 0:
                ll1 += np.log(p)
        ll1 /= max(n - 1, 1)

        # Order-2: 3D transition counts (s_{t-1}, s_t) -> s_{t+1}
        T2 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
        for i in range(n - 2):
            T2[states[i], states[i + 1], states[i + 2]] += 1

        ll2 = 0.0
        for i in range(n - 2):
            s0, s1, s2 = states[i], states[i + 1], states[i + 2]
            denom = T2[s0, s1].sum()
            if denom > 0:
                p = T2[s0, s1, s2] / denom
                if p > 0:
                    ll2 += np.log(p)
        ll2 /= max(n - 2, 1)

        return float(ll2 - ll1)

    def _forbidden_transitions(self, T: np.ndarray,
                                pattern_counts: np.ndarray) -> float:
        """Fraction of legal ordinal transitions that are absent.

        For overlapping order-d patterns, each pattern has exactly d legal
        successors (600 total for d=5). Deterministic 1D maps forbid many
        of these transitions; noise and mixing maps allow all of them.

        Only counts transitions from patterns observed at least 10 times,
        to avoid finite-sample artifacts.
        """
        min_count = 10
        observed = pattern_counts >= min_count
        n_observed = int(observed.sum())
        if n_observed < 2:
            return 0.0

        # For each well-observed pattern, count legal successors that occur
        legal_sub = self._legal_mask[observed]      # (n_observed, n_states)
        trans_sub = T[observed] > 0                  # (n_observed, n_states)

        total_legal = int(legal_sub.sum())           # n_observed * d
        total_present = int((legal_sub & trans_sub).sum())

        if total_legal == 0:
            return 0.0

        return float(1.0 - total_present / total_legal)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        N = len(x)

        fallback = GeometryResult(
            geometry_name=self.name,
            metrics={
                'transition_entropy': 0.0,
                'time_irreversibility': 0.0,
                'statistical_complexity': 0.0,
                'forbidden_transitions': 0.0,
                'markov_mixing': 0.5,
                'memory_order': 0.0,
            }
        )

        if N < 200:
            return fallback

        patterns = self._ordinal_patterns(x)
        if len(patterns) < 100:
            return fallback

        T = self._transition_matrix(patterns)
        pattern_counts = np.bincount(
            patterns, minlength=self._n_states
        ).astype(np.float64)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'transition_entropy': float(np.clip(
                    self._transition_entropy(T), 0, 1)),
                'time_irreversibility': float(np.clip(
                    self._time_irreversibility(pattern_counts), 0, 1)),
                'statistical_complexity': float(np.clip(
                    self._statistical_complexity(pattern_counts), 0, 1)),
                'forbidden_transitions': float(np.clip(
                    self._forbidden_transitions(T, pattern_counts), 0, 1)),
                'markov_mixing': self._markov_mixing(data),
                'memory_order': self._memory_order(data),
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

    def description(self) -> str:

        return "Treats windowed histograms as points on a statistical manifold. The Fisher metric measures how sharply the distribution changes — high curvature means the data is informationally rich."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Information gradient, statistical curvature, parameter sensitivity"

    @property
    def dimension(self) -> str:
        return "statistical manifold"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed as histogram with Laplace smoothing (empirical distribution)."""
        data = self.validate_data(data)
        data = data.astype(float)
        data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        hist, _ = np.histogram(data, bins=self.n_bins, range=(0, 1))
        # Laplace smoothing: add 1 pseudocount per bin to guarantee
        # all probabilities are positive and bounded away from zero.
        # This prevents catastrophic Fisher information values (1/p → ∞).
        return (hist + 1).astype(float) / (len(data) + self.n_bins)

    def fisher_information(self, p: np.ndarray) -> np.ndarray:
        """
        Compute Fisher information matrix for multinomial (Poisson approx).
        For multinomial with Laplace-smoothed probabilities:
        F_ij = δ_ij / p_i (diagonal, ignoring simplex constraint).
        All p_i > 0 is guaranteed by Laplace smoothing in embed().
        """
        return np.diag(1.0 / p)

    def _windowed_distributions(self, data: np.ndarray,
                                n_windows: int = 32) -> list:
        """Split data into n_windows segments, return Laplace-smoothed histograms."""
        seg_len = len(data) // n_windows
        if seg_len < 100:
            return []
        dists = []
        for i in range(n_windows):
            seg = data[i * seg_len:(i + 1) * seg_len].astype(float)
            seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-10)
            hist, _ = np.histogram(seg, bins=self.n_bins, range=(0, 1))
            dists.append((hist + 1).astype(float) / (seg_len + self.n_bins))
        return dists

    @staticmethod
    def _fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fisher-Rao geodesic distance: 2·arccos(Σ√(p_i·q_i))."""
        bc = np.sum(np.sqrt(p * q))
        return float(2.0 * np.arccos(np.clip(bc, -1.0, 1.0)))

    def _geodesic_dynamics(self, data: np.ndarray) -> dict:
        """Windowed distributional dynamics on the statistical manifold.

        geodesic_velocity: mean Fisher-Rao distance between consecutive
        windowed distributions. Measures how fast the distribution moves.

        velocity_spectral_gini: Gini coefficient of the power spectrum of
        the velocity series. Concentrated spectrum = regular dynamics.
        (evolved via ShinkaEvolve)
        """
        windows = self._windowed_distributions(data, n_windows=32)
        if len(windows) < 4:
            return {'geodesic_velocity': 0.0, 'velocity_spectral_gini': 0.0}

        vels = np.array([self._fisher_rao_distance(windows[i], windows[i + 1])
                         for i in range(len(windows) - 1)])
        geodesic_velocity = float(np.mean(vels))

        # Spectral Gini of velocity series
        v_det = vels - np.mean(vels)
        fft_c = np.fft.rfft(v_det)[1:]
        ps = np.abs(fft_c) ** 2
        if np.sum(ps) < 1e-12 or len(ps) < 2:
            gini = 0.0
        else:
            s = np.sort(ps)
            n = len(s)
            idx = np.arange(1, n + 1)
            gini = float(np.sum((2 * idx - n - 1) * s) / (n * np.sum(s) + 1e-12))

        return {'geodesic_velocity': geodesic_velocity,
                'velocity_spectral_gini': gini}

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Fisher geometry metrics."""
        data = self.validate_data(data)
        p = self.embed(data)

        # Fisher information matrix
        F = self.fisher_information(p)

        # Scalar curvature proxy: trace and log-determinant.
        # F = diag(1/p_i) so det(F) = prod(1/p_i) which spans 40+ orders of
        # magnitude (10^19 for uniform → 10^58 for concentrated data).
        # Log-scale is mandatory for PCA and distance-based analysis.
        trace_F = np.trace(F)
        log_det_F = float(np.sum(np.log(np.diag(F))))  # = -sum(log(p_i))

        # Effective dimension (number of significant parameters)
        eigenvalues = np.linalg.eigvalsh(F)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            # Participation ratio
            eff_dim = (np.sum(eigenvalues) ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)
        else:
            eff_dim = 0

        metrics = {
            "trace_fisher": trace_F,
            "log_det_fisher": log_det_F,
            "effective_dimension": eff_dim,
        }
        metrics.update(self._geodesic_dynamics(data))

        return GeometryResult(
            geometry_name=self.name,
            metrics=metrics,
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

    def description(self) -> str:

        return "Constructs a phase portrait (position vs. momentum) from the time series. The symplectic area form measures trajectory stationarity and return structure."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Phase space area, trajectory stationarity, recurrence"

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
        """Compute symplectic metrics: area form, stationarity, recurrence."""
        pts = self.embed(data)
        n = len(pts)

        metric_names = ["total_area", "windowed_area_cv", "q_spread",
                        "flux_sign_persistence", "recurrence_rate"]
        if n < 10:
            return GeometryResult(
                geometry_name=self.name,
                metrics={k: 0.0 for k in metric_names},
                raw_data={"points": pts})

        q, p = pts[:, 0], pts[:, 1]

        # Total symplectic area (shoelace on full trajectory)
        fluxes = q[:-1] * p[1:] - q[1:] * p[:-1]
        total_area = abs(np.sum(fluxes)) / 2.0

        # Windowed area CV: coefficient of variation of shoelace areas
        # across trajectory windows. Low CV = stationary dynamics,
        # high CV = transient or multi-regime.
        window_size = min(50, n // 4)
        win_areas = []
        for i in range(0, n - window_size, window_size):
            w = pts[i:i + window_size]
            win_areas.append(self.symplectic_area(w))
        if win_areas:
            wa = np.array(win_areas)
            windowed_area_cv = float(np.std(wa) / (np.mean(wa) + 1e-10))
        else:
            windowed_area_cv = 0.0

        # Phase space position spread
        q_spread = float(np.std(q))

        # Flux sign persistence: mean run length of same-sign consecutive
        # fluxes. High persistence = consistent circulation direction
        # (orbiting behavior). Low = random walk in phase space.
        signs = np.sign(fluxes)
        sign_changes = np.sum(signs[:-1] != signs[1:])
        flux_sign_persistence = float(len(fluxes) / (sign_changes + 1))

        # Recurrence rate: fraction of sampled phase-space point pairs
        # within ε-neighborhood. High = trajectory revisits regions.
        rng = np.random.default_rng(0)
        n_sample = min(200, n)
        idx = rng.choice(n, n_sample, replace=False)
        sub = pts[idx]
        dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        recurrence_rate = float(np.mean(dists < 0.1))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "total_area": total_area,
                "windowed_area_cv": windowed_area_cv,
                "q_spread": q_spread,
                "flux_sign_persistence": flux_sign_persistence,
                "recurrence_rate": recurrence_rate,
            },
            raw_data={
                "points": pts,
                "fluxes": fluxes,
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

    def description(self) -> str:

        return "One of Thurston's eight 3D geometries. The Sol metric stretches one direction exponentially while contracting the other, detecting anisotropic scaling."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Hyperbolic splitting, exponential divergence"

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
            step = np.array([data[3*i], data[3*i + 1], data[3*i + 2]])
            new_point = self.sol_multiply(path[-1], step)
            # Clamp z to [-5, 5] so exp(2z) stays in [~4e-5, ~22026],
            # safe for float64 and large enough for Sol structure to matter.
            new_point[2] = np.clip(new_point[2], -5, 5)
            new_point[:2] = np.clip(new_point[:2], -1000, 1000)
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

        # Path length in the Sol metric: ds² = e^(2z)dx² + e^(-2z)dy² + dz²
        diffs = np.diff(path, axis=0)
        z_vals = path[:-1, 2]  # z at start of each segment
        sol_ds = np.sqrt(
            np.exp(2 * z_vals) * diffs[:, 0]**2 +
            np.exp(-2 * z_vals) * diffs[:, 1]**2 +
            diffs[:, 2]**2
        )
        path_length = np.sum(sol_ds)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "z_drift": z_drift,
                "z_variance": z_variance,
                "path_length": path_length,
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

    def description(self) -> str:

        return "Thurston geometry: product of the 2-sphere and the real line. Detects data with layered spherical structure — directional concentration that drifts over time."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Spherical layering, radial drift"

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

    def description(self) -> str:

        return "Thurston geometry: product of the hyperbolic plane and the real line. Combines hierarchical depth (hyperbolic) with vertical drift (Euclidean)."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Hyperbolic layering, vertical drift"

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

    def _temporal_memory(self, series, max_lag=5):
        """Sum of absolute autocorrelation at lags 1..max_lag.

        Measures temporal persistence: high for structured data
        (correlated consecutive values), near zero for shuffled data.

        Evolved via ShinkaEvolve h2r_discovery v1.
        """
        n = len(series)
        if n < max_lag + 2 or np.var(series) < 1e-15:
            return 0.0
        total = 0.0
        for k in range(1, max_lag + 1):
            if n > k + 1:
                c = np.corrcoef(series[:-k], series[k:])[0, 1]
                total += abs(c) if np.isfinite(c) else 0.0
        return total

    def _boundary_dynamics(self, points, window_size=500, threshold=0.85):
        """Variance of windowed boundary fraction.

        Measures how the fraction of points near the Poincaré disk
        boundary varies over time. Structured data has regime changes
        (periods near/far from boundary); random data is uniform.

        Evolved via ShinkaEvolve h2r_discovery v1.
        """
        radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        n = len(radii)
        step = window_size // 2
        if n < window_size:
            return 0.0
        fracs = []
        for i in range(0, n - window_size + 1, step):
            fracs.append(np.mean(radii[i:i + window_size] > threshold))
        return float(np.var(fracs)) if len(fracs) > 1 else 0.0

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute H² × ℝ metrics."""
        points = self.embed(data)

        zero_metrics = {
            "hyperbolic_variance": 0.0,
            "depth_height_corr": 0.0,
            "radial_temporal_memory": 0.0,
            "boundary_dynamics": 0.0,
        }
        if len(points) < 10:
            return GeometryResult(
                geometry_name=self.name,
                metrics=zero_metrics,
                raw_data={"points": points}
            )

        # Hyperbolic distances from origin
        radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        hyp_dists = 2 * np.arctanh(np.clip(radii, 0, 0.9999))
        hyp_dist_variance = float(np.var(hyp_dists))

        # Heights
        heights = points[:, 2]

        # Correlation between hyperbolic depth and height
        if np.std(hyp_dists) > 1e-10 and np.std(heights) > 1e-10:
            depth_height_corr = float(np.corrcoef(hyp_dists, heights)[0, 1])
        else:
            depth_height_corr = 0.0

        # Temporal memory (sum of |ACF| at lags 1-5 for radial distances)
        radial_memory = self._temporal_memory(hyp_dists)

        # Boundary dynamics (variance of windowed boundary fraction)
        boundary_dyn = self._boundary_dynamics(points)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "hyperbolic_variance": hyp_dist_variance,
                "depth_height_corr": depth_height_corr,
                "radial_temporal_memory": float(np.clip(radial_memory, 0, 25)),
                "boundary_dynamics": float(np.clip(boundary_dyn, 0, 1)),
            },
            raw_data={
                "points": points,
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

    def description(self) -> str:

        return "Thurston geometry: the universal cover of the unit tangent bundle of H^2. Accumulates 2x2 matrices and classifies the path as elliptic, parabolic, or hyperbolic by trace."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Shear flow, geodesic divergence, rotation-shear coupling"

    @property
    def dimension(self) -> int:
        return 3  # SL(2,R) is 3-dimensional

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed data as elements of SL(2,ℝ) using the KAK decomposition.
        Each element = rotation(θ) @ diag(e^t, e^{-t}) @ rotation(φ).
        This naturally spans all three conjugacy classes:
          - small t → elliptic (rotation-dominated)
          - large t → hyperbolic (boost-dominated)
          - t ≈ 0 boundary → parabolic
        Three bytes → (θ, t, φ), covering the full group topology.
        """
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_matrices = len(data) // 3
        matrices = []

        for i in range(n_matrices):
            theta = data[3*i] * 2 * np.pi       # rotation angle [0, 2π)
            t = (data[3*i + 1] - 0.5) * 4.0     # boost parameter [-2, 2]
            phi = data[3*i + 2] * 2 * np.pi      # rotation angle [0, 2π)

            ct, st = np.cos(theta), np.sin(theta)
            cp, sp = np.cos(phi), np.sin(phi)
            et = np.exp(t)

            # R(θ) @ diag(e^t, e^{-t}) @ R(φ)
            M = np.array([[ct, -st], [st, ct]]) \
                @ np.array([[et, 0], [0, 1.0/et]]) \
                @ np.array([[cp, -sp], [sp, cp]])

            matrices.append(M)

        return matrices

    def sl2_distance(self, M1: np.ndarray, M2: np.ndarray) -> float:
        """
        Distance in SL(2,ℝ) based on Frobenius norm of difference.
        (Approximate - true metric involves geodesics)
        """
        return np.linalg.norm(M1 - M2, 'fro')

    def _lag1_autocorr(self, series):
        """Absolute lag-1 autocorrelation of a time series."""
        if len(series) < 3 or np.std(series) < 1e-9:
            return 0.0
        c = np.corrcoef(series[:-1], series[1:])[0, 1]
        return abs(c) if np.isfinite(c) else 0.0

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute SL(2,ℝ) metrics.

        Original metrics: trace/spectral classification + Lyapunov exponent.
        Evolved metrics (ShinkaEvolve sl2r_discovery v1):
        - boost_autocorrelation: temporal memory in KAK boost parameter
        - trace_autocorrelation: temporal memory in geometric trace sequence
        - lyapunov_std: burstiness of divergence across blocks
        """
        matrices = self.embed(data)

        zero_metrics = {
            "parabolic_fraction": 0.0,
            "hyperbolic_fraction": 0.0,
            "mean_trace": 0.0,
            "lyapunov_exponent": 0.0,
            "mean_spectral_radius": 0.0,
            "boost_autocorrelation": 0.0,
            "trace_autocorrelation": 0.0,
            "lyapunov_std": 0.0,
        }
        if len(matrices) < 2:
            return GeometryResult(
                geometry_name=self.name,
                metrics=zero_metrics,
                raw_data={}
            )

        # Trace classification for SL(2,R):
        #   Elliptic:   |trace| < 2  ⟺  trace² < 4
        #   Parabolic:  |trace| = 2  ⟺  trace² = 4
        #   Hyperbolic: |trace| > 2  ⟺  trace² > 4
        traces = np.array([np.trace(M) for M in matrices])
        trace_sq = traces**2
        eps = 0.2  # tolerance for parabolic boundary

        parabolic_frac = float(np.mean(np.abs(trace_sq - 4) <= eps))
        hyperbolic_frac = float(np.mean(trace_sq > 4 + eps))

        # Lyapunov exponent: average log of spectral radius over running product
        block_size = 50
        log_norms = []
        for start in range(0, len(matrices) - block_size + 1, block_size):
            product = np.eye(2)
            for M in matrices[start:start + block_size]:
                product = product @ M
            s = np.linalg.svd(product, compute_uv=False)
            log_norms.append(np.log(s[0] + 1e-300))
        lyap_arr = np.array(log_norms) / block_size if log_norms else np.array([0.0])
        lyapunov_exponent = float(np.mean(lyap_arr))

        # Spectral radius (largest eigenvalue magnitude)
        eigenvalues = [np.max(np.abs(np.linalg.eigvals(M))) for M in matrices]
        mean_spectral_radius = float(np.mean(eigenvalues))

        # Boost parameter autocorrelation (temporal memory in raw KAK params)
        norm_data = self._normalize_to_unit(
            self.validate_data(data), self.input_scale)
        boost_params = (norm_data[1::3] - 0.5) * 4.0
        boost_autocorr = self._lag1_autocorr(boost_params)

        # Trace autocorrelation (temporal memory in geometric representation)
        trace_autocorr = self._lag1_autocorr(traces)

        # Lyapunov std (burstiness of divergence)
        lyap_std = float(np.std(lyap_arr))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "parabolic_fraction": parabolic_frac,
                "hyperbolic_fraction": hyperbolic_frac,
                "mean_trace": float(np.mean(traces)),
                "lyapunov_exponent": lyapunov_exponent,
                "mean_spectral_radius": mean_spectral_radius,
                "boost_autocorrelation": float(np.clip(boost_autocorr, 0, 1)),
                "trace_autocorrelation": float(np.clip(trace_autocorr, 0, 1)),
                "lyapunov_std": float(np.clip(lyap_std, 0, 5)),
            },
            raw_data={
                "matrices": matrices,
                "traces": traces.tolist(),
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

    def description(self) -> str:

        return "Embeds the time series on the Clifford torus (flat torus inside S^3) via two angular coordinates. Phase coupling and winding rates reveal quasiperiodic structure."


    @property

    def view(self) -> str:

        return "symmetry"


    @property

    def detects(self) -> str:

        return "Phase coupling, torus flatness, S^3 embedding quality"

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
# QUASICRYSTAL DIFFRACTION ANALYSIS
# =============================================================================

def _quasicrystal_spectral_1d(data: np.ndarray, target_ratio: float) -> dict:
    """
    1D spectral metrics for quasicrystalline structure detection.

    Quasicrystals have three properties that separate them from both periodic
    and random signals:
      1. Discrete (Bragg-like) spectrum with peaks at ratio-harmonic positions
      2. Spectral self-similarity: the peak pattern is invariant under
         frequency scaling by the characteristic algebraic ratio
         (golden ratio φ for Penrose/Hat, silver ratio δ for Ammann-Beenker)
      3. Aperiodic complexity: subword complexity grows linearly
         (faster than periodic, slower than random)

    All four returned metrics use target_ratio, making them topology-specific:
    ratio_symmetry and acf_self_similarity measure correlation at ratio-scaled
    frequencies/lags; peak_sharpness measures energy in peaks forming geometric
    chains at the target ratio; subword_complexity is ratio-independent (binary
    symbolic dynamics).

    Args:
        data: uint8 array (raw signal)
        target_ratio: characteristic ratio (φ ≈ 1.618, δ ≈ 2.414)

    Returns:
        dict with: ratio_symmetry, peak_sharpness, subword_complexity,
                   acf_self_similarity
    """
    from scipy.signal import find_peaks

    x = data.astype(np.float64)
    N = len(x)
    x_centered = x - x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        return {"ratio_symmetry": 0.0, "peak_sharpness": 0.0,
                "subword_complexity": 0.0, "acf_self_similarity": 0.0}
    x_norm = x_centered / x_std

    # === Power spectrum (real FFT, skip DC) ===
    ps = np.abs(np.fft.rfft(x_norm))**2
    ps = ps[1:]
    n_freq = len(ps)
    log_ps = np.log(ps + 1e-30)

    # === Moving-average detrend ===
    # Remove the smooth spectral envelope while preserving discrete peaks.
    # Critical: a linear detrend leaves smooth residuals for 1/f noise,
    # which correlate at any scale factor. The moving average removes ALL
    # smooth structure, isolating only the peaks.
    if n_freq < 10:
        return {"ratio_symmetry": 0.0, "peak_sharpness": 0.0,
                "subword_complexity": 0.0, "acf_self_similarity": 0.0}
    idx = np.arange(n_freq, dtype=np.float64)
    win = min(max(n_freq // 10, 20), n_freq)
    ma_kernel = np.ones(win) / win
    smooth = np.convolve(log_ps, ma_kernel, mode='same')
    residual = log_ps - smooth

    # === Find prominent spectral peaks ===
    med_res = np.median(residual)
    mad = np.median(np.abs(residual - med_res))
    prominence_threshold = max(4.0 * mad, 1.0)
    peak_indices, peak_props = find_peaks(
        residual, prominence=prominence_threshold, distance=3)
    n_peaks = len(peak_indices)

    # === 1. Ratio symmetry: correlation of residuals at ratio-scaled frequencies ===
    # Pre-calculate mean/std of residual for faster correlation
    res_mean = residual.mean()
    res_centered = residual - res_mean
    res_var = np.sum(res_centered**2)

    def _corr_at_ratio(ratio):
        scaled = idx / ratio
        valid = (scaled >= 0) & (scaled < n_freq - 1)
        if valid.sum() < 50:
            return 0.0
        
        # Linear interpolation
        interp = np.interp(scaled[valid], idx, residual)
        
        # Faster correlation for centered data
        i_centered = interp - interp.mean()
        i_var = np.sum(i_centered**2)
        if i_var < 1e-10 or res_var < 1e-10:
            return 0.0
            
        # Pearson R = dot(a, b) / (||a|| * ||b||)
        # We must use the specific portion of 'residual' corresponding to 'valid'
        r_centered = residual[valid] - residual[valid].mean()
        r_var = np.sum(r_centered**2)
        if r_var < 1e-10: return 0.0
        
        return np.sum(r_centered * i_centered) / np.sqrt(r_var * i_var)

    target_cc = _corr_at_ratio(target_ratio)

    # Use fewer null ratios to speed up the common case while maintaining Z-score integrity
    null_ratios = [1.12, 1.31, 1.55, 1.78, 2.05, 2.33, 2.73, 3.17]
    null_ratios = [r for r in null_ratios
                   if abs(r - target_ratio) > 0.08
                   and abs(r - target_ratio**2) > 0.15
                   and abs(r - 1.0 / target_ratio) > 0.08]
    null_ccs = [_corr_at_ratio(r) for r in null_ratios]
    null_mean = np.mean(null_ccs)
    null_std = np.std(null_ccs)

    if null_std > 1e-10:
        z = (target_cc - null_mean) / null_std
    else:
        z = 10.0 if target_cc > null_mean + 0.01 else 0.0
    ratio_symmetry = float(np.clip(z / 5.0, 0, 1))

    # === 2. Peak sharpness: prominence in ratio-harmonic Bragg peaks ===
    # QC diffraction patterns have Bragg peaks at positions related by
    # powers of the inflation factor λ.  We find geometric chains of peaks
    # and measure their total prominence (height above envelope), then
    # normalize against null ratios so the metric is high only when the
    # target ratio gives significantly more chain prominence than others.
    # Using prominences (not raw spectral power) avoids envelope domination.
    prominences = peak_props.get('prominences', np.zeros(n_peaks))

    def _best_chain_prominence(ratio):
        """Best chain of 3+ peaks at geometric ratio, return total prominence."""
        best = 0.0
        for i in range(n_peaks):
            f0 = peak_pos[i]
            chain = set()
            for exp in range(10):
                expected = f0 * ratio**exp
                if expected >= n_freq:
                    break
                dists = np.abs(peak_pos - expected)
                j = int(np.argmin(dists))
                if dists[j] < max(2.0, 0.02 * expected):
                    chain.add(j)
            for exp in range(1, 5):
                expected = f0 / ratio**exp
                if expected < 1:
                    break
                dists = np.abs(peak_pos - expected)
                j = int(np.argmin(dists))
                if dists[j] < max(2.0, 0.02 * expected):
                    chain.add(j)
            if len(chain) >= 3:
                prom = sum(prominences[j] for j in chain)
                best = max(best, prom)
        return best

    if n_peaks >= 3:
        peak_pos = peak_indices.astype(np.float64) + 1  # +1: avoid log(0)

        target_prom = _best_chain_prominence(target_ratio)

        # Null ratios centered around target (multiplicative offsets).
        # Critical: must be same magnitude as target so chain-finding
        # difficulty is comparable. Ratios near 1.0 trivially find long
        # chains because expected positions are densely packed.
        null_ratios = []
        for factor in [0.7, 0.8, 0.85, 0.9, 1.1, 1.15, 1.2, 1.3]:
            r = target_ratio * factor
            if (r > 1.1
                    and abs(r - target_ratio) > 0.08
                    and abs(r - target_ratio**2) > 0.3):
                null_ratios.append(r)
        null_proms = [_best_chain_prominence(r) for r in null_ratios]
        null_mean = np.mean(null_proms)
        null_std = np.std(null_proms)

        if null_std > 1e-10:
            z = (target_prom - null_mean) / null_std
        else:
            z = 10.0 if target_prom > null_mean * 1.1 else 0.0
        peak_sharpness = float(np.clip(z / 5.0, 0, 1))
    else:
        peak_sharpness = 0.0

    # === 3. Subword complexity growth rate ===
    # Binary quantize, count distinct n-grams for several n values.
    # Quasicrystal (Fibonacci): p(n) = n+1 → constant differences.
    # Periodic: p(n) saturates → differences → 0.
    # Random: p(n) = 2^n → differences accelerate.
    # Metric: 1/(1 + CV of successive differences).  High = constant rate (QC).
    # Binarize: for binary-valued data (e.g. substitution words mapped to
    # {0, 255}), median thresholding fails when the majority symbol equals
    # the max value (median=255 → data>255 = all False). Fix: detect
    # few-valued data and threshold at the midpoint of unique values.
    unique_vals = np.unique(data)
    if len(unique_vals) <= 4:
        threshold = (float(unique_vals[0]) + float(unique_vals[-1])) / 2.0
    else:
        threshold = float(np.median(data))
    binary = (data > threshold).astype(np.uint8)

    complexities = []
    # Maximum window size to keep unique() tractable
    n_windows_max = 8192
    
    for n in [3, 5, 7, 9, 11]:
        if N - n + 1 < 100:
            break
        
        n_windows = min(N - n + 1, n_windows_max)
        # Optimized: use stride_tricks to create overlapping views without copying
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Binary windows of length n
        windows = sliding_window_view(binary[:n_windows + n - 1], n)
        
        # Pack bits into integers via dot product with powers of 2
        # powers = [1, 2, 4, ..., 2^(n-1)]
        powers = 2**np.arange(n, dtype=np.uint32)
        packed = windows @ powers
        
        complexities.append(len(np.unique(packed)))

    if len(complexities) >= 3:
        diffs = np.diff(complexities).astype(np.float64)
        mean_diff = np.mean(np.abs(diffs))
        if mean_diff < 0.5:
            # Complexity saturated (periodic) or degenerate (all same).
            # True QC has linear growth, so diffs must be positive.
            subword_complexity = 0.0
        else:
            std_diff = np.std(diffs)
            cv = std_diff / (mean_diff + 1e-10)
            subword_complexity = float(max(0.0, 1.0 / (1.0 + cv)))
    else:
        subword_complexity = 0.5

    # === 4. ACF peak self-similarity at ratio-scaled lags ===
    # Compute ACF via Wiener-Khinchin, then find local maxima.
    # For a quasicrystal, ACF has oscillatory peaks at lags related by
    # the target ratio. For 1/f noise, ACF decays monotonically — no peaks.
    n_fft = 2 * N
    x_padded = np.zeros(n_fft)
    x_padded[:N] = x_norm
    X = np.fft.rfft(x_padded)
    acf_raw = np.fft.irfft(np.abs(X)**2)[:N]
    acf = acf_raw / (acf_raw[0] + 1e-10)

    # Detrend ACF with moving average (removes monotone decay from 1/f noise)
    max_lag = min(N // 2, 2000)
    acf_seg = acf[1:max_lag]
    acf_idx = np.arange(len(acf_seg), dtype=np.float64)
    acf_win = max(len(acf_seg) // 10, 10)
    acf_smooth = np.convolve(acf_seg, np.ones(acf_win) / acf_win, mode='same')
    acf_residual = acf_seg - acf_smooth

    # Gate: if ACF residual is at the noise floor, correlation is meaningless.
    # White noise ACF residual std ≈ 0.88/√N; structured data >> 2/√N.
    if np.std(acf_residual) < 2.0 / np.sqrt(N):
        acf_self_similarity = 0.0
        return {
            "ratio_symmetry": ratio_symmetry,
            "peak_sharpness": peak_sharpness,
            "subword_complexity": subword_complexity,
            "acf_self_similarity": acf_self_similarity,
        }

    def _acf_corr_at_ratio(ratio):
        scaled = acf_idx / ratio
        valid = (scaled >= 0) & (scaled < len(acf_seg) - 1)
        if valid.sum() < 30:
            return 0.0
        interp = np.interp(scaled[valid], acf_idx, acf_residual)
        
        # Fast correlation
        a_centered = acf_residual[valid] - acf_residual[valid].mean()
        i_centered = interp - interp.mean()
        a_var = np.sum(a_centered**2)
        i_var = np.sum(i_centered**2)
        if a_var < 1e-10 or i_var < 1e-10:
            return 0.0
        return np.sum(a_centered * i_centered) / np.sqrt(a_var * i_var)

    # Scan |correlation| densely across [0.7·r, 1.3·r]. True QC has a sharp
    # peak at the target ratio (peak/mean >> 1); smooth ACFs (brownian, 1/f)
    # have flat profiles (peak/mean ≈ 1). The peak/mean ratio is extremely
    # seed-stable (CV < 0.07 on QC sources, CV ≈ 0 on non-QC).
    scan_ratios = np.linspace(target_ratio * 0.7, target_ratio * 1.3, 31)
    scan_ccs = np.array([abs(_acf_corr_at_ratio(r)) for r in scan_ratios])
    target_idx = np.argmin(np.abs(scan_ratios - target_ratio))
    target_cc = scan_ccs[target_idx]
    mean_cc = np.mean(scan_ccs)

    # Signal strength gate: if target correlation is below the Pearson R
    # noise floor (≈ 3/√n_pairs), the peak/mean ratio is meaningless —
    # random fluctuations among near-zero values create spurious ratios.
    # Fibonacci QC gives target_cc ≈ 0.107; Dragon Curve gives ≈ 0.008.
    n_valid = int(np.sum((acf_idx / target_ratio) < len(acf_seg) - 1))
    cc_noise_floor = 2.5 / np.sqrt(max(n_valid, 30))
    if target_cc < cc_noise_floor:
        acf_self_similarity = 0.0
    else:
        peak_ratio = target_cc / (mean_cc + 1e-10)
        # Gate: peak/mean < 2 is consistent with harmonic-series ACF
        # (oscillators like Rössler give ~1.6) or noise fluctuations.
        # True QC gives peak/mean > 20, so the gap is enormous.
        if peak_ratio < 2.0:
            acf_self_similarity = 0.0
        else:
            acf_self_similarity = float(np.clip((peak_ratio - 2.0) / 8.0, 0, 1))

    return {
        "ratio_symmetry": ratio_symmetry,
        "peak_sharpness": peak_sharpness,
        "subword_complexity": subword_complexity,
        "acf_self_similarity": acf_self_similarity,
    }


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

    def description(self) -> str:

        return "Projects the 1D signal into a 2D diffraction pattern and tests for fivefold Bragg peaks, the hallmark of Penrose quasicrystalline order."


    @property

    def view(self) -> str:

        return "quasicrystal"


    @property

    def detects(self) -> str:

        return "Fivefold diffraction symmetry, Bragg peak contrast, aperiodic order"

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
        """Compute Penrose quasicrystal metrics via 1D spectral analysis.

        Detects golden-ratio (φ) self-similarity in the signal's spectrum
        and autocorrelation — the defining signature of Penrose/Fibonacci
        quasicrystalline order.
        """
        data = self.validate_data(data)
        m = _quasicrystal_spectral_1d(data, self.PHI)

        # Evolved metric (ShinkaEvolve v2 gen 43)
        tower = self._algebraic_tower(data)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "long_range_order": m["acf_self_similarity"],
                "algebraic_tower": tower,
            },
            raw_data={}
        )

    def _detrended_residual(self, data: np.ndarray):
        """Shared spectral preprocessing for evolved phi metrics."""
        x = data.astype(np.float64)
        N = len(x)
        x = x - x.mean()
        s = x.std()
        if s < 1e-10 or N < 256:
            return None, None, N
        x = x / s
        ps = np.abs(np.fft.rfft(x))**2
        ps = ps[1:]
        n_freq = len(ps)
        if n_freq < 100:
            return None, None, N
        log_ps = np.log(ps + 1e-30)
        win = min(max(n_freq // 10, 20), n_freq)
        kernel = np.ones(win) / win
        smooth = np.convolve(log_ps, kernel, mode='same')
        return log_ps - smooth, np.arange(n_freq, dtype=np.float64), N

    def _algebraic_tower(self, data: np.ndarray) -> float:
        """Peak-weighted φ-coherence tower (ShinkaEvolve atlas v1 gen 27).

        Four-factor multiplicative product:
        1. Peak-weighted R² regression (exp(residual) weights focus on spectral peaks)
        2. Max-null φ-specificity (target R² - best null R², not mean)
        3. Cross-scale stability (penalizes R² variance across tower levels n=2..5)
        4. Residual fit quality (low lag-1 autocorrelation = good model fit)

        Produces continuous values in ~[-2, 1] with IQR≈1.2 across atlas sources.
        """
        residual, idx, _ = self._detrended_residual(data)
        if residual is None:
            return 0.0

        ratio = self.PHI
        n_freq = len(residual)
        freqs = np.arange(n_freq, dtype=np.float64)
        peakiness = np.std(residual)

        def _single_coherence(r, n):
            """Returns (R², residual_lag1_autocorrelation) or None."""
            if n < 2:
                return None
            r_p1, r_p2 = r**(n - 1), r**n
            start_idx = int(np.ceil(10 * r_p2))
            if start_idx >= n_freq - 50:
                return None
            y = residual[start_idx:]
            if y.std() < 1e-9:
                return 0.0, 0.0
            target_freqs = freqs[start_idx:]
            x1 = np.interp(target_freqs / r_p1, freqs, residual)
            x2 = np.interp(target_freqs / r_p2, freqs, residual)
            X = np.vstack([x1, x2, np.ones(len(x1))]).T
            # Peak-weighted regression: exp(residual) emphasizes spectral peaks
            weights = np.exp(np.clip(residual[start_idx:], -10, 10))
            if weights.sum() < 1e-12:
                return 0.0, 0.0
            sqrt_w = np.sqrt(weights)
            try:
                beta, res_ss_w, _, _ = np.linalg.lstsq(
                    X * sqrt_w[:, np.newaxis], y * sqrt_w, rcond=None)
                if len(res_ss_w) == 0:
                    return 0.0, 0.0
                weighted_mean_y = np.average(y, weights=weights)
                ss_tot_w = np.sum(weights * (y - weighted_mean_y)**2)
                if ss_tot_w < 1e-12:
                    return 0.0, 0.0
                r2 = float(np.clip(1.0 - res_ss_w[0] / ss_tot_w, 0.0, 1.0))
                # Lag-1 autocorrelation of unweighted residuals
                y_pred = X @ beta
                res_err = y - y_pred
                autocorr = 0.0
                if len(res_err) > 2:
                    c = np.corrcoef(res_err[:-1], res_err[1:])
                    if c.shape == (2, 2) and np.isfinite(c[0, 1]):
                        autocorr = c[0, 1]
                return r2, autocorr
            except np.linalg.LinAlgError:
                return 0.0, 0.0

        def _coherence_stats(r):
            r2s, autocorrs = [], []
            for n in range(2, 6):
                result = _single_coherence(r, n)
                if result is None:
                    break
                r2, ac = result
                r2s.append(r2)
                autocorrs.append(ac)
            if not r2s:
                return 0.0, 0.0, 0.0
            return np.mean(r2s), np.std(r2s), np.mean(autocorrs)

        target_r2, target_r2_std, target_autocorr = _coherence_stats(ratio)

        null_ratios = [1.21, 1.38, 1.55, 1.85, 2.15, 2.45, 2.8, 3.2]
        null_ratios = [r for r in null_ratios if abs(r - ratio) > 0.08]
        max_null_r2 = max((_coherence_stats(r)[0] for r in null_ratios), default=0.0)

        coherence_gap = target_r2 - max_null_r2
        stability = np.exp(-5.0 * target_r2_std)
        fit_quality = np.exp(-3.0 * abs(target_autocorr))

        return float(np.clip(
            10.0 * coherence_gap * peakiness * stability * fit_quality,
            -10.0, 20.0))


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

    def description(self) -> str:

        return "Tests for eightfold quasicrystalline order (the Ammann-Beenker tiling). Measures cardinal vs. diagonal anisotropy in the diffraction pattern."


    @property

    def view(self) -> str:

        return "quasicrystal"


    @property

    def detects(self) -> str:

        return "Eightfold diffraction symmetry, Bragg peak contrast, cardinal-diagonal anisotropy"

    @property
    def atlas_exclude(self) -> set:
        return {"pell_conformance", "peak_sharpness", "eightfold_symmetry"}

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
        """Compute octagonal quasicrystal metrics: headline + decomposed + workhorses.

        Headline: pell_conformance (Pell gate × convergent resonance, D1=59%).
        Decomposed: pell_gate, convergent_resonance (the two stages).
        Workhorses: eightfold_symmetry, peak_sharpness, convergent_profile.
        """
        data = self.validate_data(data)
        components = self._evolved_delta_components(data)

        # Generic QC spectral metrics
        m = _quasicrystal_spectral_1d(data, self.SILVER)

        # Convergent profile (ShinkaEvolve atlas v1 gen 48)
        conv_profile = self._convergent_profile(data)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "pell_conformance": components['pell_conformance'],
                "convergent_resonance": components['convergent_resonance'],
                "eightfold_symmetry": m["ratio_symmetry"],
                "peak_sharpness": m["peak_sharpness"],
                "convergent_profile": conv_profile,
            },
            raw_data={}
        )

    def _convergent_profile(self, data: np.ndarray) -> float:
        """Convergent-cascade coherence profile (ShinkaEvolve atlas v1 gen 48).

        Measures spectral self-similarity across continued-fraction convergents
        of the silver ratio, using quadratic detrending of the log power spectrum.
        Returns mean - std + 0.2*slope of coherences across convergent scales.
        """
        x = np.asarray(data, dtype=np.float64)
        N = len(x)
        x = x - x.mean()
        s = x.std()
        if s < 1e-10 or N < 512:
            return 0.0
        x = x / s

        ps = np.abs(np.fft.rfft(x))**2
        ps = ps[1:]
        freqs = np.fft.rfftfreq(N, d=1.0)[1:]
        if len(ps) < 20:
            return 0.0

        # Quadratic detrending in log-log space (key evolution discovery)
        log_ps = np.log(ps + 1e-30)
        log_f = np.log(freqs)
        try:
            poly = np.polyfit(log_f, log_ps, 2)
            residual = log_ps - np.polyval(poly, log_f)
        except (np.linalg.LinAlgError, ValueError):
            residual = log_ps

        # Pell convergents: 2/1, 5/2, 12/5, 29/12, 70/29
        convergents = [2.0, 2.5, 2.4, 29.0 / 12.0, 70.0 / 29.0]
        scales = [c for c in convergents if c > 1.05]
        if len(scales) < 2:
            return 0.0

        grid_freqs = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), num=256)
        spec_interp = np.interp(grid_freqs, freqs, residual)

        coherences = []
        for sc in scales:
            scaled_gf = grid_freqs / sc
            if scaled_gf[0] > freqs[-1] or scaled_gf[-1] < freqs[0]:
                continue
            spec_scaled = np.interp(scaled_gf, freqs, residual)
            if spec_interp.std() < 1e-10 or spec_scaled.std() < 1e-10:
                coherences.append(0.0)
                continue
            corr = np.corrcoef(spec_interp, spec_scaled)[0, 1]
            coherences.append(float(np.nan_to_num(corr)))

        if len(coherences) < 2:
            return 0.0

        c = np.array(coherences)
        slope = 0.0
        try:
            slope, _ = np.polyfit(np.arange(len(c)), c, 1)
        except (np.linalg.LinAlgError, ValueError):
            pass
        return float(np.mean(c) - np.std(c) + 0.2 * slope)

    def _evolved_delta_components(self, data: np.ndarray) -> dict:
        """Pell gate × convergent resonance (ShinkaEvolve v2 gen 45).

        Returns individual components AND the combined headline.
        """
        zero = {'pell_conformance': 0.0, 'pell_gate': 0.0, 'convergent_resonance': 0.0}

        x = data.astype(np.float64)
        N = len(x)
        x = x - x.mean()
        s = x.std()
        if s < 1e-10 or N < 512:
            return zero
        x = x / s

        ps = np.abs(np.fft.rfft(x))**2
        ps = ps[1:]
        n_freq = len(ps)
        if n_freq < 100:
            return zero

        log_ps = np.log(ps + 1e-30)
        win = min(max(n_freq // 10, 20), n_freq)
        kernel = np.ones(win) / win
        smooth = np.convolve(log_ps, kernel, mode='same')
        residual = log_ps - smooth

        freq_idx = np.arange(n_freq, dtype=np.float64)

        def _coherence(r):
            """Positive-part spectral self-coherence at scale r."""
            if r <= 1.0 or r > n_freq / 20.0:
                return 0.0
            mask = residual > 0.0
            if mask.sum() < 50:
                return 0.0
            vec_orig = residual[mask]
            scaled = freq_idx[mask] / r
            vec_interp = np.interp(scaled, freq_idx, residual,
                                   left=0.0, right=0.0)
            if vec_orig.std() < 1e-9 or vec_interp.std() < 1e-9:
                return 0.0
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.corrcoef(vec_orig, vec_interp)[0, 1]
            return 0.0 if np.isnan(corr) else corr

        ratio = self.SILVER

        # Stage 1: Pell gate (δ² = 2δ+1 identity check)
        pell_error = abs(_coherence(ratio**2) - _coherence(2 * ratio + 1))
        pell_gate = np.exp(-15.0 * pell_error)

        # Stage 2: Convergent resonance (Pell number ratios)
        # CF convergents of δ = 1+√2 = [2; 2,2,2,...]: 2/1, 5/2, 12/5, 29/12
        convergents = [2.0, 2.5, 2.4, 29.0 / 12.0]
        coherences = [np.clip(_coherence(c), 0, 1) for c in convergents]

        prod = np.prod(coherences)
        if prod < 1e-30:
            conv_res = 0.0
        else:
            conv_res = float(prod ** (1.0 / len(coherences)))

        return {
            'pell_conformance': float(pell_gate * conv_res),
            'pell_gate': float(pell_gate),
            'convergent_resonance': conv_res,
        }


# =============================================================================
# DODECAGONAL GEOMETRY (12-fold aperiodic)
# =============================================================================

class DodecagonalGeometry(ExoticGeometry):
    """
    Dodecagonal (Stampfli) Geometry - detects 12-fold quasicrystalline structure.

    Dodecagonal quasicrystals (like Ta-Te, V-Ni-Si) have 12-fold rotational
    symmetry and are connected to the ratio 2 + √3. They can be modeled by
    Square-Triangle tilings (Stampfli tiling).

    Complementary to Penrose (5-fold) and Ammann-Beenker (8-fold).
    """

    RATIO = 2 + np.sqrt(3)  # Dodecagonal ratio ~ 3.732

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Dodecagonal (Stampfli)"

    @property
    def description(self) -> str:
        return "Tests for 12-fold quasicrystalline order (Stampfli square-triangle tiling, ratio 2+sqrt(3)). Found in Ta-Te and V-Ni-Si alloys."

    @property
    def view(self) -> str:
        return "quasicrystal"

    @property
    def detects(self) -> str:
        return "Twelvefold diffraction symmetry, square-triangle tiling order"

    @property
    def dimension(self) -> str:
        return "2D with 12-fold"

    def dodecagonal_projection(self, x: float, y: float) -> Tuple[float, ...]:
        """Project 2D point onto 12 directions (dodecagonal symmetry)."""
        angles = [2 * np.pi * k / 12 for k in range(12)]
        return tuple(x * np.cos(a) + y * np.sin(a) for a in angles)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data pairs in 12-direction dodecagonal space."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        embedded = []

        for i in range(n_points):
            x = data[2*i] * 10 - 5
            y = data[2*i + 1] * 10 - 5
            proj = self.dodecagonal_projection(x, y)
            embedded.append(list(proj) + [x, y])

        return np.array(embedded)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute dodecagonal QC metrics: combined + decomposed + workhorse.

        Combined:    dodec_phase_coherence (3-test complex FFT ensemble, ShinkaEvolve v2)
        Decomposed:  z_poly_identity, z_sqrt3_coherence, z_self_similarity
        Workhorse:   ratio_symmetry (generic QC spectral)
        """
        data = self.validate_data(data)
        z1, z2, z3 = self._phase_coherence_components(data)
        combined = float(np.clip((z1 + z2 + z3) / 9.0, -1.0, 1.0))

        m = _quasicrystal_spectral_1d(data, self.RATIO)
        pisot = self._pisot_triplet_coherence(data)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "dodec_phase_coherence": combined,
                "z_sqrt3_coherence": float(np.clip(z2 / 4.0, -1.0, 1.0)),
                "pisot_triplet_coherence": pisot,
            },
            raw_data={}
        )

    @staticmethod
    def _interp_complex(new_x, xp, fp):
        """Interpolate complex-valued array."""
        return (np.interp(new_x, xp, fp.real, left=0, right=0) +
                1j * np.interp(new_x, xp, fp.imag, left=0, right=0))

    @staticmethod
    def _smooth_complex(vec, window_size=11):
        """Smooth a complex vector with moving average."""
        if len(vec) < window_size:
            return vec
        if window_size % 2 == 0:
            window_size += 1
        kernel = np.ones(window_size) / window_size
        return (np.convolve(vec.real, kernel, mode='same') +
                1j * np.convolve(vec.imag, kernel, mode='same'))

    def _phase_coherence_components(self, data: np.ndarray):
        """Complex FFT phase-coherence ensemble (ShinkaEvolve v2 gen 28, D1=80.5%).

        Uses complex FFT (not just power spectrum) to probe phase structure.
        Three tests:
          1. Polynomial identity: δ²−4δ+1=0 on complex coefficients
          2. √3 coherence: spectral coherence between S(f/(r−2)²) and S(f/3)
          3. Self-similarity: coherence between S(f) and S(f/r)

        Returns (z_poly, z_sqrt3, z_self_sim) — robust MAD-based z-scores.
        """
        x = data.astype(np.float64)
        N = len(x)
        x = x - x.mean()
        s = x.std()
        if s < 1e-10 or N < 800:
            return 0.0, 0.0, 0.0
        x = x / s

        X = np.fft.rfft(x)
        n_freq = len(X)
        if n_freq < 200:
            return 0.0, 0.0, 0.0

        freqs = np.arange(n_freq, dtype=np.float64)
        r = self.RATIO
        ic = self._interp_complex
        sc = self._smooth_complex

        # --- Test functions ---
        def poly_coherence(test_r):
            """δ²−4δ+1=0 on complex FFT coefficients."""
            if not np.isfinite(test_r) or test_r <= 1.1:
                return 0.0
            r2 = test_r ** 2
            end = int(n_freq / r2)
            if end < 50:
                return 0.0
            vf = freqs[:end]
            X0 = X[:end]
            X1 = ic(vf / test_r, freqs, X)
            X2 = ic(vf / r2, freqs, X)
            res = X2 - 4.0 * X1 + X0
            pwr_res = np.sum(np.abs(res) ** 2)
            pwr_comp = (np.sum(np.abs(X2) ** 2) +
                        16.0 * np.sum(np.abs(X1) ** 2) +
                        np.sum(np.abs(X0) ** 2))
            if pwr_comp < 1e-12:
                return 0.0
            return 1.0 - pwr_res / pwr_comp

        def sqrt3_coherence(test_r):
            """Coherence between S(f/(r−2)²) and S(f/3)."""
            y = test_r - 2.0
            if not np.isfinite(y) or y <= 1.1:
                return 0.0
            y2 = y ** 2
            end = int(n_freq / max(y2, 3.0))
            if end < 50:
                return 0.0
            vf = freqs[:end]
            Xy2 = ic(vf / y2, freqs, X)
            X3 = ic(vf / 3.0, freqs, X)
            cross = sc(Xy2 * np.conj(X3))
            a1 = sc(np.abs(Xy2) ** 2).real
            a2 = sc(np.abs(X3) ** 2).real
            denom = np.sqrt(a1 * a2) + 1e-12
            return float(np.mean(np.abs(cross) / denom))

        def self_sim_coherence(test_r):
            """Coherence between S(f) and S(f/r)."""
            if not np.isfinite(test_r) or test_r <= 1.1:
                return 0.0
            end = int(n_freq / test_r)
            if end < 50:
                return 0.0
            Xb = X[:end]
            Xs = ic(freqs[:end] / test_r, freqs, X)
            cross = sc(Xb * np.conj(Xs))
            a1 = sc(np.abs(Xb) ** 2).real
            a2 = sc(np.abs(Xs) ** 2).real
            denom = np.sqrt(a1 * a2) + 1e-12
            return float(np.mean(np.abs(cross) / denom))

        # --- Dense local null scan ---
        hw, np_ = 0.6, 40
        null_ratios = np.concatenate([
            np.linspace(r - hw, r - 0.05, np_ // 2),
            np.linspace(r + 0.05, r + hw, np_ // 2),
        ])
        null_ratios = null_ratios[null_ratios > 1.2]
        if len(null_ratios) < 10:
            return 0.0, 0.0, 0.0

        def robust_z(target_val, null_vals):
            arr = np.asarray(null_vals)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            scale = 1.4826 * mad + 1e-9
            return float((target_val - med) / scale)

        # z1: polynomial identity
        t1 = poly_coherence(r)
        n1 = [poly_coherence(nr) for nr in null_ratios]
        z1 = robust_z(t1, n1)

        # z2: √3 coherence
        t2 = sqrt3_coherence(r)
        n2 = [sqrt3_coherence(nr) for nr in null_ratios]
        z2 = robust_z(t2, n2)

        # z3: self-similarity coherence
        t3 = self_sim_coherence(r)
        n3 = [self_sim_coherence(nr) for nr in null_ratios]
        z3 = robust_z(t3, n3)

        return z1, z2, z3

    def _pisot_triplet_coherence(self, data: np.ndarray) -> float:
        """Pisot triplet verification (ShinkaEvolve dodecagonal_v1 gen 9, D1=0.343).

        Finds spectral peak triplets at (f, f·δ, f·δ²) and verifies the
        algebraic identity X(f·δ²) + X(f) ≈ 4·X(f·δ) via cosine similarity
        on complex FFT coefficients. Only the correct ratio satisfies this
        at detected peaks.
        """
        x = data.astype(np.float64)
        N = len(x)
        x = x - x.mean()
        s = x.std()
        if s < 1e-10 or N < 2000:
            return 0.0
        x = x / s

        r = self.RATIO
        X = np.fft.rfft(x)[1:]  # skip DC
        ps = np.abs(X) ** 2
        n_freq = len(X)
        freqs = np.arange(n_freq, dtype=np.float64)

        # Detrended residual for peak detection
        log_ps = np.log(ps + 1e-30)
        win = min(max(n_freq // 10, 20), n_freq)
        kernel = np.ones(win) / win
        smooth = np.convolve(log_ps, kernel, mode='same')
        residual = log_ps - smooth

        # Simple peak finder
        threshold = 1.5 * np.std(residual)
        peaks = []
        for i in range(1, len(residual) - 1):
            if residual[i] > threshold and residual[i] > residual[i-1] and residual[i] > residual[i+1]:
                peaks.append(i)
        if len(peaks) < 5:
            return 0.0

        peak_set = set(peaks)
        tol = 3
        ic = self._interp_complex
        scores = []
        weights = []

        max_base = int(n_freq / (r ** 2))
        for f0 in peaks:
            if f0 > max_base:
                break
            f1_exp = f0 * r
            f2_exp = f0 * r ** 2
            f1_found = any(p in peak_set for p in range(int(f1_exp) - tol, int(f1_exp) + tol + 1))
            if not f1_found:
                continue
            f2_found = any(p in peak_set for p in range(int(f2_exp) - tol, int(f2_exp) + tol + 1))
            if not f2_found:
                continue

            X0 = X[f0]
            X1 = ic([f1_exp], freqs, X)[0]
            X2 = ic([f2_exp], freqs, X)[0]

            va = 4.0 * X1
            vb = X0 + X2
            abs_va, abs_vb = np.abs(va), np.abs(vb)
            if abs_va < 1e-9 or abs_vb < 1e-9:
                continue

            cos_sim = float(np.real(va * np.conj(vb)) / (abs_va * abs_vb))
            scores.append(cos_sim)
            weights.append(np.log1p(ps[f0]))

        if not scores:
            return 0.0
        return float(np.clip(np.average(scores, weights=weights), 0, 1))


# =============================================================================
# DECAGONAL GEOMETRY (10-fold aperiodic)
# =============================================================================

# =============================================================================
# SEPTAGONAL GEOMETRY (7-fold aperiodic)
# =============================================================================

class SeptagonalGeometry(ExoticGeometry):
    """
    Septagonal Geometry - detects 7-fold quasicrystalline structure.

    Sevenfold symmetry is impossible in periodic crystals. It is associated
    with the ratio ρ ≈ 2.247 (root of x³ - x² - 2x + 1 = 0), related to
    2*cos(π/7).
    """

    RATIO = 1 + 2 * np.cos(2 * np.pi / 7)  # approx 2.24698

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Septagonal (Danzer)"

    @property
    def description(self) -> str:
        return "Tests for 7-fold quasicrystalline order (Danzer tiling). Sevenfold symmetry is crystallographically forbidden, arising only in aperiodic structures."

    @property
    def view(self) -> str:
        return "quasicrystal"

    @property
    def detects(self) -> str:
        return "Sevenfold diffraction symmetry, Danzer tiling order"

    @property
    def dimension(self) -> str:
        return "2D with 7-fold"

    def septagonal_projection(self, x: float, y: float) -> Tuple[float, ...]:
        """Project 2D point onto 7 directions."""
        angles = [2 * np.pi * k / 7 for k in range(7)]
        return tuple(x * np.cos(a) + y * np.sin(a) for a in angles)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data pairs in 7-direction septagonal space."""
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        embedded = []

        for i in range(n_points):
            x = data[2*i] * 10 - 5
            y = data[2*i + 1] * 10 - 5
            proj = self.septagonal_projection(x, y)
            embedded.append(list(proj) + [x, y])

        return np.array(embedded)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute septagonal metrics: combined + decomposed + workhorses.

        Combined:    cubic_coherence (triple-conjugate ensemble, ShinkaEvolve v2)
        Decomposed:  z_primary, z_conjugate, z_reciprocal (individual polynomial tests)
        Workhorses:  ratio_symmetry, peak_sharpness (generic QC spectral)
        """
        data = self.validate_data(data)
        z1, z2, z3 = self._conjugate_ensemble_components(data)
        combined = float(np.clip(
            (z1 + 0.5 * max(0, z2) + 0.5 * max(0, z3)) / 7.0, -1.0, 1.0))

        # Generic QC spectral metrics (workhorses)
        m = _quasicrystal_spectral_1d(data, self.RATIO)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "cubic_coherence": combined,
                "z_primary": float(np.clip(z1 / 4.0, -1.0, 1.0)),
                "z_conjugate": float(np.clip(z2 / 4.0, -1.0, 1.0)),
                "z_reciprocal": float(np.clip(z3 / 4.0, -1.0, 1.0)),
                "ratio_symmetry": m["ratio_symmetry"],
            },
            raw_data={}
        )

    def _detrended_residual(self, data: np.ndarray):
        """Shared spectral preprocessing for evolved septagonal metrics."""
        x = data.astype(np.float64)
        N = len(x)
        x = x - x.mean()
        s = x.std()
        if s < 1e-10 or N < 800:
            return None, None, N
        x = x / s
        ps = np.abs(np.fft.rfft(x))**2
        ps = ps[1:]
        n_freq = len(ps)
        if n_freq < 200:
            return None, None, N
        log_ps = np.log(ps + 1e-30)
        win = min(max(n_freq // 10, 20), n_freq)
        kernel = np.ones(win) / win
        smooth = np.convolve(log_ps, kernel, mode='same')
        return log_ps - smooth, np.arange(n_freq, dtype=np.float64), N

    @staticmethod
    def _poly_relative_error(r, spectrum, idx, coeffs):
        """Relative error for a cubic identity applied to a detrended spectrum.

        coeffs are for [x³, x², x¹, x⁰].
        """
        n_freq = len(spectrum)
        if not np.isfinite(r) or r <= 1.1:
            return 1.0
        r2, r3 = r * r, r * r * r
        if n_freq / r3 < 20:
            return 1.0
        s0 = spectrum
        s1 = np.interp(idx / r, idx, spectrum)
        s2 = np.interp(idx / r2, idx, spectrum)
        s3 = np.interp(idx / r3, idx, spectrum)
        error_vec = coeffs[0] * s3 + coeffs[1] * s2 + coeffs[2] * s1 + coeffs[3] * s0
        ac = np.abs(coeffs)
        denom = (ac[0] * np.linalg.norm(s3) + ac[1] * np.linalg.norm(s2) +
                 ac[2] * np.linalg.norm(s1) + ac[3] * np.linalg.norm(s0))
        if denom < 1e-9:
            return 1.0
        return np.linalg.norm(error_vec) / denom

    def _robust_dip_z(self, target_r, spectrum, idx, coeffs):
        """Robust z-score of the error dip at target_r using MAD.

        Compares the polynomial relative error at target_r against a dense
        local null scan (±0.5 around target, 40 points, excluding ±0.05).
        Uses median + MAD for outlier resistance.
        """
        if not np.isfinite(target_r) or target_r <= 1.1:
            return 0.0
        target_err = self._poly_relative_error(target_r, spectrum, idx, coeffs)
        hw, np_ = 0.5, 40
        test_ratios = np.concatenate([
            np.linspace(target_r - hw, target_r - 0.05, np_ // 2),
            np.linspace(target_r + 0.05, target_r + hw, np_ // 2),
        ])
        test_ratios = test_ratios[test_ratios > 1.1]
        if len(test_ratios) < 10:
            return 0.0
        null_errs = np.array([
            self._poly_relative_error(r, spectrum, idx, coeffs)
            for r in test_ratios])
        med = np.median(null_errs)
        mad = np.median(np.abs(null_errs - med))
        scale = 1.4826 * mad + 1e-9
        return float((med - target_err) / scale)

    def _conjugate_ensemble_components(self, data: np.ndarray):
        """Triple-conjugate algebraic test (ShinkaEvolve v2 gen 33, D1=87.6%).

        Tests three interconnected cubic identities derived from the minimal
        polynomial of ρ = 1 + 2cos(2π/7):

          P1: x³ − 2x² − x + 1 = 0  →  test at r = ρ ≈ 2.247
          P2: x³ + x² − 2x − 1 = 0  →  test at r = ρ−1 ≈ 1.247
          P3: x³ − x² − 2x + 1 = 0  →  test at r = 1/(ρ²−2ρ) ≈ 1.802

        Returns (z_p1, z_p2, z_p3) — robust MAD-based z-scores.
        """
        residual, idx, _ = self._detrended_residual(data)
        if residual is None:
            return 0.0, 0.0, 0.0

        r = self.RATIO

        # P1: primary polynomial, root ρ
        P1 = np.array([1.0, -2.0, -1.0, 1.0])
        z1 = self._robust_dip_z(r, residual, idx, P1)

        # P2: conjugate polynomial, root ρ−1
        P2 = np.array([1.0, 1.0, -2.0, -1.0])
        z2 = self._robust_dip_z(r - 1.0, residual, idx, P2)

        # P3: reciprocal polynomial, root 1/(ρ²−2ρ)
        d = r * r - 2.0 * r
        z3 = 0.0
        if d > 1e-6:
            P3 = np.array([1.0, -1.0, -2.0, 1.0])
            z3 = self._robust_dip_z(1.0 / d, residual, idx, P3)

        return z1, z2, z3


# =============================================================================
# FRACTAL MANDELBROT GEOMETRY
# =============================================================================

class FractalMandelbrotGeometry(ExoticGeometry):
    """
    Mandelbrot Fractal Geometry - detects sensitivity to initial conditions.

    Maps data pairs to the complex plane and iterates z = z² + c.
    Measures escape velocity, orbit traps, and connectivity.
    Good for distinguishing different types of deterministic chaos.
    """

    def __init__(self, input_scale: float = 255.0, max_iter: int = 64):
        self.input_scale = input_scale
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "Fractal (Mandelbrot)"


    @property

    def description(self) -> str:

        return "Uses byte pairs as starting points z0 for Mandelbrot iteration z -> z^2 + c. Escape times and orbit statistics measure fractal boundary structure."


    @property

    def view(self) -> str:

        return "topological"


    @property

    def detects(self) -> str:

        return "Escape rate, fractal dimension, boundary complexity"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed data pairs as parameters c in the complex plane.
        Maps [0, 1] to [-1.5, 1.5].
        """
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        pairs = data[:2*n_points].reshape(n_points, 2)
        c_vals = (pairs[:, 0] * 3.0 - 1.5) + 1j * (pairs[:, 1] * 3.0 - 1.5)

        return c_vals

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Mandelbrot escape metrics."""
        c_vals = self.embed(data)

        if len(c_vals) == 0:
            return GeometryResult(self.name, {}, {})

        escapes = []
        final_moduli = []

        for c in c_vals:
            z = 0j
            it = 0
            for _ in range(self.max_iter):
                z = z*z + c
                if abs(z) > 2.0:
                    break
                it += 1
            escapes.append(it)
            final_moduli.append(abs(z))

        escapes = np.array(escapes)
        final_moduli = np.array(final_moduli)

        # Metrics
        mean_escape = np.mean(escapes)
        escape_variance = np.var(escapes)

        # "Interior" fraction (points that didn't escape)
        interior_fraction = np.mean(escapes == self.max_iter)

        # Entropy of escape times
        counts = np.bincount(escapes)
        probs = counts[counts > 0] / len(escapes)
        escape_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Orbit magnitude (for escaped points, this grows huge; for interior, bounded)
        # We use log magnitude to handle the explosion
        log_moduli = np.log(final_moduli + 1e-10)
        mean_log_modulus = np.mean(log_moduli)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "escape_time_variance": escape_variance,
                "interior_fraction": interior_fraction,
                "escape_entropy": escape_entropy,
            },
            raw_data={
                "c_vals": c_vals,
                "escapes": escapes
            }
        )


class FractalJuliaGeometry(ExoticGeometry):
    """
    Julia Fractal Geometry - a tunable dynamical sensor.

    Data define the starting position z0. 
    A fixed parameter c defines the landscape (physics).
    Changing c acts like tuning a sensor to different "resonance" patterns.
    """

    def __init__(self, c_real: float = -0.8, c_imag: float = 0.156, 
                 input_scale: float = 255.0, max_iter: int = 64):
        self.c = complex(c_real, c_imag)
        self.input_scale = input_scale
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "Julia Set"


    @property

    def description(self) -> str:

        return "Fixed Julia set parameter c acts as a tunable sensor. Data values become starting points z0; escape time and orbit stability detect dynamical trapping."


    @property

    def view(self) -> str:

        return "topological"


    @property

    def detects(self) -> str:

        return "Dynamical trapping, basin structure, Julia dimension"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed data pairs as starting positions z0.
        Maps [0, 1] to [-1.5, 1.5].
        """
        data = self.validate_data(data)
        data = self._normalize_to_unit(data, self.input_scale)

        n_points = len(data) // 2
        pairs = data[:2*n_points].reshape(n_points, 2)
        z0_vals = (pairs[:, 0] * 3.0 - 1.5) + 1j * (pairs[:, 1] * 3.0 - 1.5)

        return z0_vals

    def _continuous_potential(self, escapes: np.ndarray,
                              final_mod: np.ndarray) -> np.ndarray:
        """Normalized iteration count: smooth escape time via log-log.

        For escaping points: n + 1 - log(log|z_final|)/log(2).
        For trapped points: max_iter. This removes the discrete steps
        in the integer escape landscape.
        """
        pot = escapes.astype(np.float64)
        esc_idx = np.where(escapes < self.max_iter)[0]
        if len(esc_idx) > 0:
            fm = final_mod[esc_idx]
            pot[esc_idx] += 1.0 - np.log(np.log(fm)) / np.log(2.0)
        return pot

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Julia escape metrics."""
        z0_vals = self.embed(data)

        if len(z0_vals) == 0:
            return GeometryResult(self.name, {
                "escape_entropy": 0.0, "stability": 0.0,
                "potential_smoothness": 0.0, "potential_variance": 0.0,
            }, {})

        escapes = []
        final_moduli = []

        for z0 in z0_vals:
            z = z0
            it = 0
            for _ in range(self.max_iter):
                z = z*z + self.c
                if abs(z) > 2.0:
                    break
                it += 1
            escapes.append(it)
            final_moduli.append(abs(z))

        escapes = np.array(escapes)
        final_moduli = np.array(final_moduli)

        # Metrics (similar to Mandelbrot but for fixed c)
        counts = np.bincount(escapes)
        probs = counts[counts > 0] / len(escapes)
        escape_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Stability: variance of escape times
        stability = np.var(escapes)

        # Continuous potential and derived metrics (evolved via ShinkaEvolve)
        pot = self._continuous_potential(escapes, final_moduli)

        # Potential smoothness: mean |diff| of continuous potential.
        # Structured data has correlated z0 path → smooth potential landscape.
        # Shuffled data has random z0 pairs → jumpy potential.
        potential_smoothness = float(np.mean(np.abs(np.diff(pot)))) if len(pot) > 1 else 0.0

        # Potential variance: diversity of dynamical behavior.
        # Structured data explores specific fractal regions → high variance.
        potential_variance = float(np.var(pot))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "escape_entropy": escape_entropy,
                "stability": stability,
                "potential_smoothness": potential_smoothness,
                "potential_variance": potential_variance,
            },
            raw_data={
                "z0_vals": z0_vals,
                "escapes": escapes,
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

    def description(self) -> str:

        return "Computes windowed skewness, kurtosis, permutation entropy, and bispectral coherence — the statistical fingerprint beyond mean and variance."


    @property

    def view(self) -> str:

        return "dynamical"


    @property

    def detects(self) -> str:

        return "Skewness, kurtosis, tail asymmetry, non-Gaussianity"

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

        # Zero-variance data: higher moments are undefined
        if np.ptp(fdata) == 0:
            return GeometryResult(self.name, {k: float('nan') for k in [
                'skew_mean', 'kurt_max', 'perm_entropy', 'perm_forbidden',
                'bicoherence_max']}, {})

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

            metrics['skew_mean'] = float(np.mean(np.abs(skewnesses)))
            metrics['kurt_max'] = float(np.max(np.abs(kurtoses)))
        else:
            metrics.update({'skew_mean': 0, 'kurt_max': 0})

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

        # --- Bicoherence (segment-averaged normalized bispectrum) ---
        # Single-FFT bicoherence is trivially 1. Must average over segments.
        n = len(fdata)
        seg_len = 256
        n_segs = n // seg_len
        max_freq = min(seg_len // 2, self.bispec_max_freq)

        if n_segs >= 4 and max_freq > 2:
            # Accumulate bispectrum and normalization across segments
            # Kim & Powers (1979) bicoherence:
            #   b²(f1,f2) = |<X(f1)X(f2)X*(f3)>|² /
            #               (<|X(f1)X(f2)|²> · <|X(f3)|²>)
            bispec_sum = {}      # (f1,f2) → sum of X[f1]*X[f2]*conj(X[f3])
            pair_power_sum = {}  # (f1,f2) → sum of |X[f1]*X[f2]|²
            single_power_sum = np.zeros(seg_len // 2)  # sum of |X[f]|²
            for s in range(n_segs):
                seg = centered[s*seg_len:(s+1)*seg_len]
                X = np.fft.fft(seg)
                X_half = X[:seg_len//2]
                single_power_sum += np.abs(X_half)**2
                for f1 in range(1, max_freq):
                    for f2 in range(f1, max_freq):
                        f3 = f1 + f2
                        if f3 < seg_len // 2:
                            B = X[f1] * X[f2] * np.conj(X[f3])
                            key = (f1, f2)
                            if key not in bispec_sum:
                                bispec_sum[key] = 0.0
                                pair_power_sum[key] = 0.0
                            bispec_sum[key] += B
                            pair_power_sum[key] += float(
                                np.abs(X[f1] * X[f2])**2)

            bicoherence_vals = []
            for (f1, f2), B_sum in bispec_sum.items():
                f3 = f1 + f2
                denom = np.sqrt(pair_power_sum[(f1, f2)] *
                                single_power_sum[f3] + 1e-30)
                bicoherence_vals.append(np.abs(B_sum) / denom)

            bic = np.array(bicoherence_vals)
            # Use top-1% mean rather than single max for stability
            top_k = max(1, len(bic) // 100)
            top_vals = np.partition(bic, -top_k)[-top_k:]
            metrics['bicoherence_max'] = float(np.mean(top_vals))
        else:
            metrics['bicoherence_max'] = 0

        return GeometryResult(
            geometry_name=self.name,
            metrics=metrics,
            raw_data={}
        )


# =============================================================================
# HÖLDER REGULARITY GEOMETRY
# =============================================================================

class HolderRegularityGeometry(ExoticGeometry):
    """
    Hölder Regularity Geometry — local regularity and multifractal analysis.

    Measures how "rough" or "smooth" the signal is at each point and how
    that roughness varies. Uses structure functions S(q, ℓ) to compute
    scaling exponents ζ(q) and the multifractal spectrum f(α).

    This geometry targets the blind spot in the 7D signature space:
    signals with pathological local regularity (Lévy flights, Devil's
    staircases, space-filling curves) that global summary statistics miss.

    Metrics:
    - hurst_exponent: ζ(2)/2, generalized Hurst exponent
    - holder_mean: mean local Hölder exponent
    - holder_std: std of local exponents (0 = monofractal)
    - holder_min: minimum exponent (roughest point)
    - holder_max: maximum exponent (smoothest point)
    - multifractal_width: range of the f(α) spectrum
    - structure_curvature: ζ(q) nonlinearity (0 = monofractal)
    """

    def __init__(self, input_scale: float = 255.0,
                 q_values=(-2, -1, -0.5, 0.5, 1, 2, 3, 4),
                 n_scales: int = 8):
        self.input_scale = input_scale
        self.q_values = np.array(q_values, dtype=float)
        self.n_scales = n_scales

    @property
    def name(self) -> str:
        return "Hölder Regularity"


    @property

    def description(self) -> str:

        return "Estimates pointwise Holder exponents via wavelet leaders. The regularity spectrum (multifractal formalism) measures local smoothness variation."


    @property

    def view(self) -> str:

        return "scale"


    @property

    def detects(self) -> str:

        return "Local roughness, regularity spectrum, singularity strength"

    @property
    def encoding_invariant(self) -> bool:
        return True

    @property
    def dimension(self) -> str:
        return "function space"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Normalize signal to [0, 1]."""
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute Hölder regularity and multifractal metrics."""
        x = self.embed(data)
        n = len(x)
        _nan_result = lambda: GeometryResult(self.name, {k: float('nan') for k in [
            "hurst_exponent", "holder_mean", "holder_std",
            "holder_min", "holder_max"]}, {})

        if n < 32:
            return _nan_result()

        # Constant data has zero increments → structure functions degenerate
        if np.ptp(x) == 0:
            return _nan_result()

        # Scales: powers of 2 up to n/4
        max_scale = min(n // 4, 2 ** self.n_scales)
        scales = []
        s = 1
        while s <= max_scale:
            scales.append(s)
            s *= 2
        scales = np.array(scales)

        # Structure functions: S(q, ℓ) = mean(|X(t+ℓ) - X(t)|^q)
        # For q < 0, use |increment|^q only where increment != 0
        zeta = np.zeros(len(self.q_values))  # scaling exponents
        log_scales = np.log2(scales)
        S_matrix = np.zeros((len(self.q_values), len(scales)))

        for j, ell in enumerate(scales):
            increments = np.abs(x[ell:] - x[:-ell])
            for i, q in enumerate(self.q_values):
                if q < 0:
                    # Avoid zero increments for negative q
                    inc = increments[increments > 1e-15]
                    if len(inc) > 10:
                        S_matrix[i, j] = np.mean(inc ** q)
                    else:
                        S_matrix[i, j] = np.nan
                else:
                    S_matrix[i, j] = np.mean(increments ** q) if len(increments) > 0 else np.nan

        # Fit ζ(q) via log-log regression of S(q, ℓ) vs ℓ
        for i, q in enumerate(self.q_values):
            log_S = np.log2(np.maximum(S_matrix[i, :], 1e-30))
            valid = np.isfinite(log_S)
            if np.sum(valid) >= 3:
                coeffs = np.polyfit(log_scales[valid], log_S[valid], 1)
                zeta[i] = coeffs[0]
            else:
                zeta[i] = np.nan

        # Hurst exponent from ζ(2)/2
        q2_idx = np.argmin(np.abs(self.q_values - 2.0))
        hurst = zeta[q2_idx] / 2.0 if np.isfinite(zeta[q2_idx]) else 0.5

        # Structure curvature: deviation of ζ(q) from linear (monofractal)
        valid_zeta = np.isfinite(zeta)
        if np.sum(valid_zeta) >= 3:
            q_valid = self.q_values[valid_zeta]
            z_valid = zeta[valid_zeta]
            linear_fit = np.polyfit(q_valid, z_valid, 1)
            linear_pred = np.polyval(linear_fit, q_valid)
            curvature = np.sqrt(np.mean((z_valid - linear_pred) ** 2))
        else:
            curvature = 0.0

        # Local Hölder exponents via finest scale increments
        # α(t) ≈ log|X(t+1) - X(t)| / log(1) — use ratio of scales 1 and 2
        inc1 = np.abs(x[1:] - x[:-1])
        inc2 = np.abs(x[2:] - x[:-2])
        # α(t) ≈ log2(inc2/inc1) where both are nonzero
        n_loc = min(len(inc1), len(inc2))
        local_alpha = np.full(n_loc, np.nan)
        for t in range(n_loc):
            if inc1[t] > 1e-15 and inc2[t] > 1e-15:
                local_alpha[t] = np.log2(inc2[t] / inc1[t])

        valid_alpha = local_alpha[np.isfinite(local_alpha)]
        if len(valid_alpha) > 10:
            # Clip extreme values for robustness
            valid_alpha = np.clip(valid_alpha, -2, 4)
            holder_mean = float(np.mean(valid_alpha))
            holder_std = float(np.std(valid_alpha))
            holder_min = float(np.percentile(valid_alpha, 2))
            holder_max = float(np.percentile(valid_alpha, 98))
            multifractal_width = holder_max - holder_min
        else:
            holder_mean = hurst
            holder_std = 0.0
            holder_min = hurst
            holder_max = hurst
            multifractal_width = 0.0

        # Increment autocorrelation: lag-1 autocorr of |increments| averaged
        # across scales 1,2,4,8,16,32. Measures volatility persistence —
        # structured signals have clustered large/small changes.
        # (evolved via ShinkaEvolve)
        inc_ac_sum, inc_ac_count = 0.0, 0
        for ell in [1, 2, 4, 8, 16, 32]:
            if n < ell + 2:
                break
            inc_ell = np.abs(x[ell:] - x[:-ell])
            if len(inc_ell) >= 30 and np.std(inc_ell) > 1e-9:
                ac = float(np.corrcoef(inc_ell[1:], inc_ell[:-1])[0, 1])
                if np.isfinite(ac):
                    inc_ac_sum += max(0.0, ac)
                    inc_ac_count += 1
        increment_autocorrelation = inc_ac_sum / inc_ac_count if inc_ac_count > 0 else 0.0

        # Alpha autocorrelation: lag-1 autocorr of local Hölder exponents.
        # Measures spatial clustering of smooth/rough regions — structured
        # signals have coherent regularity zones, shuffled signals don't.
        # (evolved via ShinkaEvolve)
        if len(valid_alpha) >= 30 and np.std(valid_alpha) > 1e-9:
            ac_alpha = float(np.corrcoef(valid_alpha[1:], valid_alpha[:-1])[0, 1])
            alpha_autocorrelation = max(0.0, ac_alpha) if np.isfinite(ac_alpha) else 0.0
        else:
            alpha_autocorrelation = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "hurst_exponent": float(np.clip(hurst, -2, 4)),
                "holder_mean": holder_mean,
                "holder_std": holder_std,
                "holder_min": holder_min,
                "holder_max": holder_max,
                "increment_autocorrelation": increment_autocorrelation,
                "alpha_autocorrelation": alpha_autocorrelation,
            },
            raw_data={
                "zeta": zeta,
                "q_values": self.q_values,
                "S_matrix": S_matrix,
            }
        )


# =============================================================================
# p-VARIATION GEOMETRY
# =============================================================================

class PVariationGeometry(ExoticGeometry):
    """
    p-Variation Geometry — path roughness characterization.

    Computes V_p = Σ|X(t+1) - X(t)|^p for multiple p values and at
    multiple partition scales. The variation index γ (critical p where
    V_p transitions from divergent to convergent with refinement) is a
    fundamental path invariant:

      γ = 2 for Brownian motion
      γ = α for Lévy-α stable processes
      γ = 1/H for fractional Brownian motion with Hurst H
      γ = ∞ for smooth (C¹) functions

    Also computes how V_p scales with partition refinement, which reveals
    the Devil's Staircase (anomalous scaling) and space-filling curves
    (dimension visible in V_1 scaling).

    Metrics:
    - var_p05, var_p1, var_p2: p-variation at each p
    - variation_index: estimated critical p (roughness invariant)
    - var_scaling_ratio: V_p(fine) / V_p(coarse) characterizes self-similarity
    """

    def __init__(self, input_scale: float = 255.0,
                 p_values=(0.5, 1.0, 2.0, 3.0)):
        self.input_scale = input_scale
        self.p_values = list(p_values)

    @property
    def name(self) -> str:
        return "p-Variation"


    @property

    def description(self) -> str:

        return "Computes the p-th variation (sum of |increments|^p) for multiple p. The critical p where variation transitions from finite to infinite characterizes path roughness."


    @property

    def view(self) -> str:

        return "scale"


    @property

    def detects(self) -> str:

        return "Path roughness, variation index, regularity"

    @property
    def dimension(self) -> str:
        return "path space"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Normalize signal to [0, 1]."""
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def _compute_pvar(self, x, p, partition_step=1):
        """Compute p-variation over a given partition."""
        sub = x[::partition_step]
        if len(sub) < 2:
            return 0.0
        increments = np.abs(np.diff(sub))
        return float(np.sum(increments ** p))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute p-variation metrics at multiple scales."""
        x = self.embed(data)
        n = len(x)

        if n < 16:
            return GeometryResult(self.name, {k: 0.0 for k in [
                "var_p2",
                "variation_index"]}, {})

        # p-variation at finest scale
        vp = {}
        for p in self.p_values:
            vp[p] = self._compute_pvar(x, p, partition_step=1)

        # Normalize by length for comparability
        metrics = {
            "var_p2": vp[2.0] / n,
        }

        # Variation index: use scaling across partition refinements
        # V_p(step=k) vs k for each p. If V_p grows with refinement (decreasing k),
        # the path has infinite p-variation. The critical p is where it stabilizes.
        steps = [1, 2, 4, 8, 16]
        steps = [s for s in steps if s < n // 4]

        if len(steps) >= 3:
            log_steps = np.log2(np.array(steps, dtype=float))

            # For each p, compute slope of log(V_p) vs log(1/step)
            slopes = {}
            for p in self.p_values:
                vp_at_steps = [self._compute_pvar(x, p, s) for s in steps]
                vp_at_steps = np.array(vp_at_steps)
                vp_at_steps[vp_at_steps < 1e-30] = 1e-30
                log_vp = np.log2(vp_at_steps)
                # Finer partitions = smaller steps = more points
                # log(V_p) vs log(n_points) where n_points ~ 1/step
                log_npts = np.log2(n / np.array(steps, dtype=float))
                coeffs = np.polyfit(log_npts, log_vp, 1)
                slopes[p] = coeffs[0]

            # Variation index: find p where slope transitions from positive
            # (V_p grows with refinement = infinite variation) to negative/zero
            # (V_p converges = finite variation)
            slope_vals = np.array([slopes[p] for p in self.p_values])
            p_arr = np.array(self.p_values)

            # Interpolate to find zero crossing
            if slope_vals[0] > 0 and slope_vals[-1] <= 0:
                # Find crossing point
                for i in range(len(slope_vals) - 1):
                    if slope_vals[i] > 0 and slope_vals[i + 1] <= 0:
                        # Linear interpolation
                        frac = slope_vals[i] / (slope_vals[i] - slope_vals[i + 1])
                        variation_index = p_arr[i] + frac * (p_arr[i + 1] - p_arr[i])
                        break
                else:
                    variation_index = p_arr[-1]
            elif slope_vals[0] <= 0:
                variation_index = p_arr[0]  # Already convergent at lowest p
            else:
                variation_index = p_arr[-1] + 1.0  # Still divergent at highest p

            metrics["variation_index"] = float(variation_index)
        else:
            metrics["variation_index"] = 2.0  # default

        # --- Temporal increment metrics (evolved via ShinkaEvolve) ---
        dx = np.diff(x)
        if len(dx) >= 2:
            # Volatility clustering: lag-1 autocorrelation of squared increments.
            # Measures whether large changes predict large changes.
            dx_sq = dx ** 2
            if np.std(dx_sq) > 1e-12:
                vc = float(np.corrcoef(dx_sq[:-1], dx_sq[1:])[0, 1])
                metrics["volatility_clustering"] = vc if np.isfinite(vc) else 0.0
            else:
                metrics["volatility_clustering"] = 0.0

            # Increment persistence: lag-1 autocorrelation of signed increments.
            # Positive = trending (persistent), ~ -0.5 = iid, negative = mean-reverting.
            if np.std(dx) > 1e-12:
                ip = float(np.corrcoef(dx[:-1], dx[1:])[0, 1])
                metrics["increment_persistence"] = ip if np.isfinite(ip) else 0.0
            else:
                metrics["increment_persistence"] = 0.0
        else:
            metrics["volatility_clustering"] = 0.0
            metrics["increment_persistence"] = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics=metrics,
            raw_data={"p_values": self.p_values, "vp": vp}
        )


# =============================================================================
# MULTI-SCALE WASSERSTEIN GEOMETRY
# =============================================================================

class MultiScaleWassersteinGeometry(ExoticGeometry):
    """
    Multi-Scale Wasserstein Geometry — distributional self-similarity.

    Extends the single-scale Wasserstein comparison (half vs half) to a
    full cascade: compares block distributions at scales 2^k for k=1..K.

    For self-similar signals (Devil's Staircase), the Wasserstein distance
    between adjacent blocks is constant across scales. For non-stationary
    signals, it drifts. For scale-free processes (Lévy), it follows a
    power law. The scaling exponent and spectral shape characterize the
    signal's multi-scale distributional structure.

    Metrics:
    - w_mean: mean Wasserstein distance across all scales
    - w_slope: log-log slope of W(scale) vs scale (self-similarity exponent)
    - w_std: variability of W across scales (flatness indicator)
    - w_max_ratio: W_max / W_min (dynamic range)
    - w_fine: average W at finest scale (local distributional variation)
    - w_coarse: average W at coarsest scale (global distributional shift)
    """

    def __init__(self, input_scale: float = 255.0, n_bins: int = 32):
        self.input_scale = input_scale
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "Multi-Scale Wasserstein"


    @property

    def description(self) -> str:

        return "Computes Wasserstein distances between histograms at multiple coarsening scales. Scale coherence reveals distributional self-similarity."


    @property

    def view(self) -> str:

        return "scale"


    @property

    def detects(self) -> str:

        return "Scale coherence, distributional drift, self-similarity"

    @property
    def dimension(self) -> str:
        return "distribution cascade"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Normalize signal to [0, 1]."""
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    @staticmethod
    def _wasserstein_1d(p, q):
        """1D Wasserstein distance via CDF difference."""
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)) / len(p))

    def _block_histogram(self, block):
        """Compute normalized histogram for a block."""
        hist, _ = np.histogram(block, bins=self.n_bins, range=(0, 1),
                               density=False)
        total = hist.sum()
        if total > 0:
            return hist.astype(float) / total
        return np.ones(self.n_bins) / self.n_bins

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute multi-scale Wasserstein metrics."""
        x = self.embed(data)
        n = len(x)

        if n < 32:
            return GeometryResult(self.name, {k: 0.0 for k in [
                "w_mean", "w_slope", "w_std", "w_max_ratio",
                "w_fine"]}, {})

        # Scales: number of blocks = 2, 4, 8, 16, ...
        max_blocks = n // 8  # each block needs >= 8 samples
        n_levels = int(np.floor(np.log2(max_blocks))) if max_blocks >= 2 else 0

        if n_levels < 2:
            return GeometryResult(self.name, {k: 0.0 for k in [
                "w_mean", "w_slope", "w_std", "w_max_ratio",
                "w_fine"]}, {})

        scale_distances = []  # (n_blocks, mean_W)

        for level in range(1, n_levels + 1):
            n_blocks = 2 ** level
            block_size = n // n_blocks

            # Compute histogram for each block
            hists = []
            for b in range(n_blocks):
                block = x[b * block_size:(b + 1) * block_size]
                hists.append(self._block_histogram(block))

            # Compute W between adjacent blocks
            distances = []
            for b in range(n_blocks - 1):
                w = self._wasserstein_1d(hists[b], hists[b + 1])
                distances.append(w)

            mean_w = np.mean(distances)
            scale_distances.append((n_blocks, mean_w))

        n_blocks_arr = np.array([sd[0] for sd in scale_distances], dtype=float)
        w_arr = np.array([sd[1] for sd in scale_distances])

        # Guard against zero/negative values in log
        w_arr_safe = np.maximum(w_arr, 1e-15)

        # Metrics
        w_mean = float(np.mean(w_arr))
        w_std = float(np.std(w_arr))

        # Log-log slope: how does W scale with number of blocks?
        log_nb = np.log2(n_blocks_arr)
        log_w = np.log2(w_arr_safe)
        if len(log_nb) >= 2 and np.std(log_w) > 1e-15:
            coeffs = np.polyfit(log_nb, log_w, 1)
            w_slope = float(coeffs[0])
        else:
            w_slope = 0.0

        w_max = np.max(w_arr_safe)
        w_min = np.min(w_arr_safe)
        w_max_ratio = float(w_max / w_min) if w_min > 1e-15 else 0.0

        # Fine vs coarse scale
        w_fine = float(w_arr[-1])   # most blocks = finest scale
        w_coarse = float(w_arr[0])  # fewest blocks = coarsest scale

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "w_mean": w_mean,
                "w_slope": w_slope,
                "w_std": w_std,
                "w_max_ratio": w_max_ratio,
                "w_fine": w_fine,
            },
            raw_data={
                "scale_distances": scale_distances,
            }
        )


# =============================================================================
# SPECTRAL GEOMETRY (1D)
# =============================================================================

class SpectralGeometry(ExoticGeometry):
    """
    Spectral Geometry — frequency-domain analysis of 1D signals.

    Computes the power spectral density via FFT and extracts statistics
    characterizing the spectral structure. This fills the biggest gap in
    the 1D geometry suite: no existing geometry operates in the frequency
    domain.

    Metrics:
    - spectral_slope: log-log slope of PSD (β in P(f) ∝ f^β).
      White noise ≈ 0, pink ≈ -1, brown ≈ -2, periodic → steep.
    - spectral_r2: goodness of power-law fit (1 = perfect power law)
    - spectral_entropy: Shannon entropy of normalized PSD (high = flat
      spectrum, low = peaked/narrowband)
    - spectral_flatness: Wiener entropy = exp(mean(log P)) / mean(P).
      1 = white noise, 0 = pure tone.
    - spectral_centroid: first moment of PSD, <f> (low = bass-heavy)
    - spectral_bandwidth: std of PSD about centroid (narrow = tonal)
    - peak_frequency: frequency bin with maximum power (normalized 0–1)
    - high_freq_ratio: fraction of power above Nyquist/2
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Spectral Analysis"


    @property

    def description(self) -> str:

        return "FFT power spectrum analysis: spectral slope (1/f^beta exponent), entropy, flatness, centroid, and bandwidth characterize the frequency content."


    @property

    def view(self) -> str:

        return "dynamical"


    @property

    def detects(self) -> str:

        return "Dominant frequency, spectral slope, bandwidth, periodicity"

    @property
    def encoding_invariant(self) -> bool:
        return True

    @property
    def dimension(self) -> str:
        return "frequency"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        n = len(x)

        if n < 16:
            return GeometryResult(self.name, {k: 0.0 for k in [
                "spectral_slope", "spectral_r2", "spectral_entropy",
                "spectral_flatness", "spectral_bandwidth",
                "peak_frequency", "high_freq_ratio"]}, {})

        # Power spectral density via FFT (one-sided)
        x_centered = x - np.mean(x)
        F = np.fft.rfft(x_centered)
        psd = np.abs(F) ** 2 / n
        # Double one-sided (except DC and Nyquist)
        psd[1:-1] *= 2
        freqs = np.fft.rfftfreq(n)  # normalized [0, 0.5]

        # Skip DC component
        psd = psd[1:]
        freqs = freqs[1:]

        if len(psd) < 3:
            return GeometryResult(self.name, {k: 0.0 for k in [
                "spectral_slope", "spectral_r2", "spectral_entropy",
                "spectral_flatness", "spectral_bandwidth",
                "peak_frequency", "high_freq_ratio"]}, {})

        # Spectral slope: log-log regression
        valid = psd > 0
        if np.sum(valid) >= 3:
            log_f = np.log(freqs[valid])
            log_p = np.log(psd[valid])
            coeffs = np.polyfit(log_f, log_p, 1)
            slope = float(coeffs[0])
            predicted = np.polyval(coeffs, log_f)
            ss_res = np.sum((log_p - predicted) ** 2)
            ss_tot = np.sum((log_p - np.mean(log_p)) ** 2) + 1e-12
            r2 = float(1.0 - ss_res / ss_tot)
        else:
            slope = 0.0
            r2 = 0.0

        # Normalize PSD as probability distribution
        total = np.sum(psd) + 1e-30
        p_norm = psd / total

        # Spectral entropy (normalized by max possible)
        p_pos = p_norm[p_norm > 0]
        max_ent = np.log(len(p_norm)) if len(p_norm) > 1 else 1.0
        spectral_entropy = float(-np.sum(p_pos * np.log(p_pos)) / max_ent) if max_ent > 0 else 0.0

        # Spectral flatness (Wiener entropy): geometric mean / arithmetic mean
        psd_pos = psd[psd > 1e-30]
        if len(psd_pos) >= 2:
            log_psd = np.log(psd_pos)
            geo_mean = np.exp(np.mean(log_psd))
            arith_mean = np.mean(psd_pos)
            spectral_flatness = float(geo_mean / arith_mean)
        else:
            spectral_flatness = 0.0

        # Spectral centroid and bandwidth
        centroid = float(np.sum(freqs * p_norm))
        bandwidth = float(np.sqrt(np.sum((freqs - centroid) ** 2 * p_norm)))

        # Peak frequency
        peak_idx = np.argmax(psd)
        peak_frequency = float(freqs[peak_idx])

        # High-frequency ratio (above half-Nyquist = 0.25)
        hf_mask = freqs > 0.25
        high_freq_ratio = float(np.sum(psd[hf_mask]) / total)

        # Phase coherence: entropy of adjacent-bin phase differences.
        # PSD discards phase; this recovers it. Structured signals have
        # locked phase relationships (low entropy → high coherence);
        # shuffling randomizes phases. (evolved via ShinkaEvolve)
        phases = np.angle(F[1:])  # skip DC (already removed from psd/freqs)
        phase_diffs = np.diff(phases)
        phase_diffs_wrapped = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
        ph_hist, _ = np.histogram(phase_diffs_wrapped, bins=32,
                                  range=(-np.pi, np.pi))
        ph_prob = ph_hist / np.sum(ph_hist)
        ph_pos = ph_prob[ph_prob > 0]
        ph_ent = float(-np.sum(ph_pos * np.log2(ph_pos)))
        ph_max_ent = np.log2(32)
        phase_coherence = float(1.0 - ph_ent / ph_max_ent) if ph_max_ent > 0 else 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "spectral_slope": np.clip(slope, -10, 2),
                "spectral_r2": np.clip(r2, 0, 1),
                "spectral_entropy": spectral_entropy,
                "spectral_flatness": np.clip(spectral_flatness, 0, 1),
                "spectral_bandwidth": bandwidth,
                "peak_frequency": peak_frequency,
                "high_freq_ratio": high_freq_ratio,
                "phase_coherence": phase_coherence,
            },
            raw_data={"psd": psd, "freqs": freqs}
        )


# =============================================================================
# INFORMATION / COMPLEXITY GEOMETRY (1D)
# =============================================================================

class InformationGeometry(ExoticGeometry):
    """
    Information Geometry — complexity and predictability measures.

    Computes entropy at multiple block sizes (entropy rate), compression-
    based complexity (Lempel-Ziv), and temporal dependence (mutual
    information at multiple lags). These measure *how predictable* and
    *how compressible* the signal is — fundamentally different from any
    geometric embedding.

    Metrics:
    - block_entropy_1: Shannon entropy of single-byte distribution
    - block_entropy_2: Shannon entropy of byte-pair distribution
    - block_entropy_4: Shannon entropy of 4-grams (normalized)
    - entropy_rate: estimated entropy per symbol (h = H_k - H_{k-1})
    - compression_ratio: zlib compression ratio (0 = trivial, 1 = incompressible)
    - mutual_info_1: mutual information at lag 1 (sequential dependence)
    - mutual_info_8: mutual information at lag 8 (longer-range dependence)
    - excess_entropy: total predictable information (H_1 - h) × N
    """

    def __init__(self, input_scale: float = 255.0, n_bins: int = 16):
        self.input_scale = input_scale
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "Information Theory"


    @property

    def description(self) -> str:

        return "Block entropies at multiple depths (1, 2, 4 bytes), entropy rate, compression ratio, and mutual information at lag 1 and 8. Measures sequential redundancy."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Shannon entropy, complexity, redundancy"

    @property
    def dimension(self) -> str:
        return "information"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    @staticmethod
    def _shannon_entropy(counts):
        """Shannon entropy in nats from count array."""
        total = np.sum(counts)
        if total == 0:
            return 0.0
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def _compression_complexity(data_bytes):
        """Compression ratio via zlib as a proxy for Kolmogorov complexity.

        Returns value in [0, 1] where 0 = perfectly compressible,
        1 = incompressible (random).
        """
        import zlib
        raw = bytes(data_bytes)
        n = len(raw)
        if n == 0:
            return 0.0
        compressed = zlib.compress(raw, level=9)
        # Subtract zlib header (~11 bytes) for fairer ratio
        compressed_size = max(len(compressed) - 11, 1)
        return float(min(compressed_size / n, 1.0))

    def _mutual_information(self, x, lag, n_bins):
        """Mutual information I(X_t; X_{t+lag}) via histogram estimator."""
        n = len(x) - lag
        if n < 10:
            return 0.0
        a = x[:n]
        b = x[lag:lag + n]

        # Joint and marginal histograms
        joint, _, _ = np.histogram2d(a, b, bins=n_bins, range=[[0, 1], [0, 1]])
        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)

        total = np.sum(joint)
        if total == 0:
            return 0.0

        # MI = H(X) + H(Y) - H(X,Y)
        hx = self._shannon_entropy(px)
        hy = self._shannon_entropy(py)
        hxy = self._shannon_entropy(joint.ravel())
        return float(max(0.0, hx + hy - hxy))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        n = len(x)

        if n < 32:
            return GeometryResult(self.name, {k: 0.0 for k in [
                "block_entropy_2", "block_entropy_4",
                "compression_ratio", "mutual_info_1",
                "mutual_info_8", "excess_entropy"]}, {})

        # Quantize to n_bins levels for entropy estimation
        x_q = np.clip((x * self.n_bins).astype(int), 0, self.n_bins - 1)

        # Block entropy at different scales
        # H_1: single symbol
        counts_1 = np.bincount(x_q, minlength=self.n_bins)
        h1 = self._shannon_entropy(counts_1) / np.log(self.n_bins)  # normalized

        # H_2: pairs
        pairs = x_q[:-1] * self.n_bins + x_q[1:]
        counts_2 = np.bincount(pairs, minlength=self.n_bins ** 2)
        h2_raw = self._shannon_entropy(counts_2)
        h2 = h2_raw / (2 * np.log(self.n_bins)) if self.n_bins > 1 else 0.0

        # H_4: 4-grams (if enough data)
        if n >= 100:
            quads = (x_q[:-3] * self.n_bins ** 3 + x_q[1:-2] * self.n_bins ** 2 +
                     x_q[2:-1] * self.n_bins + x_q[3:])
            # Use unique counts instead of full bincount (n_bins^4 can be large)
            _, quad_counts = np.unique(quads, return_counts=True)
            h4_raw = self._shannon_entropy(quad_counts)
            h4 = h4_raw / (4 * np.log(self.n_bins)) if self.n_bins > 1 else 0.0
        else:
            h4 = h2

        # Entropy rate estimate: h ≈ H_2 - H_1 (conditional entropy of 2nd given 1st)
        h1_raw = self._shannon_entropy(counts_1)
        entropy_rate = (h2_raw - h1_raw) / np.log(self.n_bins) if self.n_bins > 1 else 0.0

        # Excess entropy: total predictable information
        excess_entropy = float(max(0.0, h1 - entropy_rate))

        # Lempel-Ziv complexity (on quantized sequence)
        lz = self._compression_complexity(self.validate_data(data))

        # Mutual information at lags 1 and 8
        mi_1 = self._mutual_information(x, 1, self.n_bins)
        mi_8 = self._mutual_information(x, 8, self.n_bins)
        # Normalize by H_1
        if h1_raw > 1e-10:
            mi_1 /= h1_raw
            mi_8 /= h1_raw

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "block_entropy_2": h2,
                "block_entropy_4": h4,
                "compression_ratio": float(np.clip(lz, 0, 1)),
                "mutual_info_1": float(np.clip(mi_1, 0, 1)),
                "mutual_info_8": float(np.clip(mi_8, 0, 1)),
                "excess_entropy": float(np.clip(excess_entropy, 0, 2)),
            },
            raw_data={}
        )


# =============================================================================
# RECURRENCE GEOMETRY (1D)
# =============================================================================

class RecurrenceGeometry(ExoticGeometry):
    """
    Recurrence Geometry — recurrence quantification analysis.

    Constructs the recurrence matrix R(i,j) = Θ(ε - ||x_i - x_j||) from
    a delay-embedded signal and extracts statistics characterizing the
    diagonal and vertical line structures.  These directly measure
    dynamical properties: determinism, laminarity, and trapping.

    Uses 3D delay embedding with lag = median of first autocorrelation zero
    (capped at 50) and a recurrence threshold at the 10th percentile of
    pairwise distances. Subsampled to keep the matrix tractable.

    Metrics:
    - recurrence_rate: fraction of recurrent points (density of R)
    - determinism: fraction of recurrent points forming diagonal lines (≥2)
      High = deterministic dynamics, low = stochastic.
    - avg_diagonal: mean length of diagonal lines (predictability horizon)
    - max_diagonal: longest diagonal (Lyapunov time proxy)
    - laminarity: fraction of recurrent points forming vertical lines (≥2)
      High = intermittent/laminar dynamics.
    - trapping_time: mean length of vertical lines (stickiness)
    - entropy_diagonal: Shannon entropy of diagonal line length distribution
    - transitivity: global clustering coefficient of the recurrence network.
      Treats R as adjacency matrix; measures triangle density.
      High = geometrically coherent attractor (ShinkaEvolve rqa_v1 gen 6).
    """

    def __init__(self, input_scale: float = 255.0, embed_dim: int = 3,
                 max_points: int = 500):
        self.input_scale = input_scale
        self.embed_dim = embed_dim
        self.max_points = max_points  # subsample for tractability

    @property
    def name(self) -> str:
        return "Recurrence Quantification"


    @property

    def description(self) -> str:

        return "Recurrence quantification analysis (RQA) on a delay-embedded phase portrait. Determinism, laminarity, and trapping time distinguish chaos from noise."


    @property

    def view(self) -> str:

        return "dynamical"


    @property

    def detects(self) -> str:

        return "Recurrence rate, determinism, laminarity, trapping time"

    @property
    def encoding_invariant(self) -> bool:
        return True

    @property
    def dimension(self) -> str:
        return "phase space"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    @staticmethod
    def _find_lag(x, max_lag=50):
        """Find delay lag as first zero-crossing of autocorrelation."""
        n = len(x)
        x_centered = x - np.mean(x)
        var = np.sum(x_centered ** 2)
        if var < 1e-15:
            return 1
        for lag in range(1, min(max_lag, n // 3)):
            acf = np.sum(x_centered[:n - lag] * x_centered[lag:]) / var
            if acf <= 0:
                return lag
        return max_lag

    @staticmethod
    def _line_stats(lengths):
        """Compute statistics from a list of line lengths (≥2)."""
        lines = [l for l in lengths if l >= 2]
        if not lines:
            return 0.0, 0.0, 0, 0.0
        total = sum(lines)
        avg = total / len(lines)
        mx = max(lines)
        # Shannon entropy of line length distribution
        counts = {}
        for l in lines:
            counts[l] = counts.get(l, 0) + 1
        n_lines = len(lines)
        ent = 0.0
        for c in counts.values():
            p = c / n_lines
            if p > 0:
                ent -= p * np.log(p)
        return float(total), float(avg), int(mx), float(ent)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        n = len(x)

        _nan = {k: float('nan') for k in [
            "recurrence_rate", "determinism", "avg_diagonal",
            "max_diagonal", "laminarity", "trapping_time",
            "entropy_diagonal", "transitivity"]}

        if n < 30:
            return GeometryResult(self.name, _nan, {})

        # Find delay and build embedding
        lag = self._find_lag(x)
        dim = self.embed_dim
        N = n - (dim - 1) * lag
        if N < 20:
            return GeometryResult(self.name, _nan, {})

        # Delay embedding matrix: (N, dim)
        embedded = np.zeros((N, dim))
        for d in range(dim):
            embedded[:, d] = x[d * lag:d * lag + N]

        # Subsample if too large — use contiguous block to preserve
        # temporal adjacency (linspace subsampling destroys diagonal
        # line structure for signals with short Lyapunov time)
        if N > self.max_points:
            embedded = embedded[:self.max_points]
            N = self.max_points

        # Distance matrix
        # Use Chebyshev (max-norm) for efficiency — standard in RQA
        diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
        dist = np.max(np.abs(diff), axis=2)

        # Threshold: 10th percentile of distances (excluding self-pairs)
        upper_tri = dist[np.triu_indices(N, k=1)]
        if len(upper_tri) == 0:
            return GeometryResult(self.name, _nan, {})
        epsilon = np.percentile(upper_tri, 10)
        if epsilon < 1e-15:
            epsilon = np.percentile(upper_tri, 20)

        # Recurrence matrix with Theiler window — exclude diagonals
        # |k| < w to avoid trivially short lines from embedding overlap.
        # Standard: w = lag * (dim - 1) + 1
        R = (dist <= epsilon).astype(np.int8)
        theiler_w = lag * (dim - 1) + 1
        for tw in range(-theiler_w + 1, theiler_w):
            np.fill_diagonal(R[max(0, tw):, max(0, -tw):], 0)

        total_recurrent = int(np.sum(R))
        total_possible = N * (N - 1) - 2 * sum(N - abs(tw) for tw in range(1, theiler_w))
        total_possible = max(total_possible, 1)
        recurrence_rate = total_recurrent / total_possible if total_possible > 0 else 0.0

        # Diagonal line statistics (lines parallel to main diagonal)
        diag_lengths = []
        for k in range(-N + 1, N):
            if abs(k) < theiler_w:
                continue
            diag = np.diag(R, k)
            # Count consecutive 1s
            length = 0
            for val in diag:
                if val:
                    length += 1
                else:
                    if length > 0:
                        diag_lengths.append(length)
                    length = 0
            if length > 0:
                diag_lengths.append(length)

        diag_total, avg_diag, max_diag, ent_diag = self._line_stats(diag_lengths)
        determinism = diag_total / total_recurrent if total_recurrent > 0 else 0.0

        # Vertical line statistics (columns of R)
        vert_lengths = []
        for j in range(N):
            col = R[:, j]
            length = 0
            for val in col:
                if val:
                    length += 1
                else:
                    if length > 0:
                        vert_lengths.append(length)
                    length = 0
            if length > 0:
                vert_lengths.append(length)

        vert_total, avg_vert, _, _ = self._line_stats(vert_lengths)
        laminarity = vert_total / total_recurrent if total_recurrent > 0 else 0.0

        # Network transitivity (global clustering coefficient).
        # Treats R as adjacency matrix; transitivity = trace(R³) / Σ d_i(d_i-1).
        # ShinkaEvolve rqa_v1 gen 6: 2× baseline, huge on deterministic systems.
        R_f = R.astype(np.float32)
        degrees = R_f.sum(axis=1)
        num_wedges = float(np.sum(degrees * (degrees - 1)))
        if num_wedges > 0:
            num_triangles = float(np.einsum('ij,jk,ki->', R_f, R_f, R_f,
                                            optimize=True))
            transitivity = num_triangles / num_wedges
        else:
            transitivity = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "recurrence_rate": float(np.clip(recurrence_rate, 0, 1)),
                "determinism": float(np.clip(determinism, 0, 1)),
                "avg_diagonal": float(np.clip(avg_diag, 0, N)),
                "max_diagonal": float(min(max_diag, N)),
                "laminarity": float(np.clip(laminarity, 0, 1)),
                "trapping_time": float(np.clip(avg_vert, 0, N)),
                "entropy_diagonal": float(np.clip(ent_diag, 0, 10)),
                "transitivity": float(np.clip(transitivity, 0, 1)),
            },
            raw_data={}
        )


# =============================================================================
# SPATIAL FIELD GEOMETRY (2D only)
# =============================================================================

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
        return "Spatial Field"


    @property

    def description(self) -> str:

        return "Analyzes 2D fields via gradient magnitude, Laplacian, and critical point classification. Detects anisotropy, smoothness, and topographic structure."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "Gradient field, curvature, critical points"

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

    def description(self) -> str:

        return "Treats 2D fields as height maps and computes Gaussian and mean curvature via the first and second fundamental forms. Positive Gaussian curvature = dome, negative = saddle."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "Gaussian curvature, mean curvature, surface area"

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
        
        if field.shape[0] < 2 or field.shape[1] < 2:
            metrics = {
                'gaussian_curvature_mean': 0.0,
                'gaussian_curvature_std': 0.0,
                'mean_curvature_mean': 0.0,
                'mean_curvature_std': 0.0,
                'shape_index_mean': 0.0,
                'shape_index_std': 0.0,
                'curvedness_mean': 0.0,
                'gauss_bonnet_integral': 0.0,
                'total_metric_area': 0.0
            }
            return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})

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
        return "Persistent Homology 2D"


    @property

    def description(self) -> str:

        return "Sublevel and superlevel set persistence on 2D grids. Counts topological births and deaths (components, loops) as a threshold sweeps through the field."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "2D topological features, level-set connectivity"

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
        return "Conformal 2D"


    @property

    def description(self) -> str:

        return "Measures conformal distortion and angle preservation in 2D fields. The Cauchy-Riemann residual quantifies departure from holomorphicity."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "Conformal distortion, angle preservation"

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

        if H < 2 or W < 2:
            return GeometryResult(
                geometry_name=self.name,
                metrics={
                    'mean_distortion': 0.0,
                    'max_distortion': 0.0,
                    'conformal_energy': 0.0,
                    'riesz_energy': 0.0,
                    'isotropy_index': 0.0,
                    'liouville_curvature': 0.0
                },
                raw_data={}
            )

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
        return "Minkowski Functionals"


    @property

    def description(self) -> str:

        return "Computes Minkowski functionals (area, perimeter, Euler characteristic) of level sets at multiple thresholds. These three numbers completely characterize the morphology of 2D binary images."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "Area, perimeter, Euler characteristic of level sets"

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
        return "Multiscale Fractal 2D"


    @property

    def description(self) -> str:

        return "Box-counting fractal dimension and lacunarity of 2D fields at multiple thresholds. Lacunarity measures the gappiness of the fractal — two sets can share the same dimension but differ in lacunarity."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "2D fractal dimension, lacunarity"

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
        return "Hodge–Laplacian"


    @property

    def description(self) -> str:

        return "Decomposes a 2D gradient field into exact, co-exact, and harmonic components via the Hodge-Helmholtz decomposition. The harmonic fraction measures topological obstruction to gradient flow."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "Harmonic content, co-exact forms, Betti numbers"

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
        return "Spectral Power 2D"


    @property

    def description(self) -> str:

        return "2D FFT power spectrum analysis. Radial spectral slope (1/f^beta), isotropy ratio, and dominant spatial frequency characterize the field's scale structure."


    @property

    def view(self) -> str:

        return "other"


    @property

    def detects(self) -> str:

        return "2D spectral slope, isotropy, dominant frequency"

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
# POWER LAW & TOPOLOGICAL GEOMETRIES
# =============================================================================

class VisibilityGraphGeometry(ExoticGeometry):
    """
    Visibility Graph Geometry - transforms time series into a topological network.

    A Natural Visibility Graph (NVG) connects two points (ti, yi) and (tj, yj) 
    if any point (tk, yk) between them (ti < tk < tj) satisfies:
        yk < yj + (yi - yj) * (tj - tk) / (tj - ti)

    This mapping is invariant under affine transformations of the data and 
    captures structural properties (periodicity, chaos, fractality) in the 
    graph's degree distribution and modularity.

    Validated for:
    - Distinguishing fGn (fractional Gaussian noise) from fBm (fractional Brownian motion).
    - Detecting transitions to chaos in dynamical systems (logistic map).
    """

    def __init__(self, max_points: int = 1024):
        self.max_points = max_points

    @property
    def name(self) -> str:
        return "Visibility Graph"


    @property

    def description(self) -> str:

        return "Converts the time series to a graph: two points are connected if no intermediate value blocks the line of sight. Degree distribution and clustering reveal dynamical class."


    @property

    def view(self) -> str:

        return "topological"


    @property

    def detects(self) -> str:

        return "Graph degree distribution, clustering, small-worldness"

    @property
    def dimension(self) -> str:
        return "Network"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """The embedding is the graph adjacency matrix itself."""
        return self.validate_data(data)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        y = self.validate_data(data).astype(float)

        # Constant data: equal heights block all NVG visibility → empty graph
        if np.ptp(y) == 0:
            return GeometryResult(self.name, {k: float('nan') for k in [
                'max_degree', 'degree_exponent_gamma', 'degree_r_squared',
                'graph_density', 'avg_clustering_coeff', 'assortativity',
                'nvg_hvg_edge_divergence', 'nvg_hvg_reach_divergence']}, {})

        if len(y) > self.max_points:
            # Take a contiguous block from the center.
            # np.linspace subsampling would destroy temporal adjacency,
            # creating spurious visibility edges between non-adjacent points.
            start = (len(y) - self.max_points) // 2
            y = y[start:start + self.max_points]

        N = len(y)
        t = np.arange(N, dtype=float)

        # O(N^2) visibility graph construction
        # Optimization: points only see neighbors that aren't blocked
        adj = [[] for _ in range(N)]
        for i in range(N):
            max_slope = -np.inf
            for j in range(i + 1, N):
                slope = (y[j] - y[i]) / (t[j] - t[i])
                if slope > max_slope:
                    adj[i].append(j)
                    adj[j].append(i)
                    max_slope = slope

        degrees = np.array([len(neighbors) for neighbors in adj], dtype=float)
        
        metrics: Dict[str, float] = {}
        metrics['max_degree'] = float(np.max(degrees))
        
        # Power Law Fitting P(k) ~ k^-gamma
        # Use log-log regression on the cumulative distribution
        k_vals, counts = np.unique(degrees, return_counts=True)
        if len(k_vals) > 3:
            # P(K >= k) distribution
            pk = np.cumsum(counts[::-1])[::-1] / N
            log_k = np.log(k_vals)
            log_pk = np.log(pk)
            coeffs = np.polyfit(log_k, log_pk, 1)
            metrics['degree_exponent_gamma'] = float(-coeffs[0])
            
            # Goodness of fit (R^2)
            predicted = np.polyval(coeffs, log_k)
            ss_res = np.sum((log_pk - predicted)**2)
            ss_tot = np.sum((log_pk - np.mean(log_pk))**2) + 1e-12
            metrics['degree_r_squared'] = float(1.0 - ss_res / ss_tot)
        else:
            metrics['degree_exponent_gamma'] = 0.0
            metrics['degree_r_squared'] = 0.0

        # Graph Density: actual edges / potential edges
        n_edges = np.sum(degrees) / 2
        metrics['graph_density'] = float(n_edges / (N * (N - 1) / 2))

        # Average Clustering Coefficient (Local density)
        # Optimized via set intersections
        cluster_coeffs = []
        # Pre-convert all adjacency lists to sets for O(1) lookup
        adj_sets = [set(neighbors) for neighbors in adj]
        
        for i in range(N):
            ki = degrees[i]
            if ki < 2:
                cluster_coeffs.append(0.0)
                continue
            
            # Count edges between neighbors of i
            neighbors = adj[i]
            edges_between = 0
            for idx_n1, n1 in enumerate(neighbors):
                s1 = adj_sets[n1]
                for n2 in neighbors[idx_n1 + 1:]:
                    if n2 in s1:
                        edges_between += 1
            cluster_coeffs.append(edges_between / (ki * (ki - 1) / 2))
        
        metrics['avg_clustering_coeff'] = float(np.mean(cluster_coeffs))

        # Degree Assortativity (Pearson correlation of degrees at ends of edges)
        # Vectorized edge degree extraction
        edge_list = []
        for i in range(N):
            for j in adj[i]:
                if j > i:
                    edge_list.append((i, j))
        
        if len(edge_list) > 1:
            edge_arr = np.array(edge_list)
            degs_i = degrees[edge_arr[:, 0]]
            degs_j = degrees[edge_arr[:, 1]]
            r = np.corrcoef(degs_i, degs_j)[0, 1]
            metrics['assortativity'] = float(r) if np.isfinite(r) else 0.0
        else:
            metrics['assortativity'] = 0.0

        # NVG vs HVG divergence (ShinkaEvolve visibility_v1, gen 4).
        # The Horizontal Visibility Graph connects i and j iff all
        # intermediate y[k] < min(y[i], y[j]).  It is always a subgraph
        # of the NVG.  The fraction of NVG edges/reach NOT in the HVG
        # measures how much visibility depends on slope structure vs
        # local height barriers — a signal property no existing atlas
        # metric captures.
        n_edges_nvg = int(np.sum(degrees) // 2)
        if n_edges_nvg > 0:
            # Build HVG
            hvg_adj = [[] for _ in range(N)]
            for i in range(N):
                for j in range(i + 1, N):
                    threshold = min(y[i], y[j])
                    blocked = False
                    for k in range(i + 1, j):
                        if y[k] >= threshold:
                            blocked = True
                            break
                    if not blocked:
                        hvg_adj[i].append(j)
                        hvg_adj[j].append(i)

            n_edges_hvg = sum(len(nb) for nb in hvg_adj) // 2
            metrics['nvg_hvg_edge_divergence'] = float(
                (n_edges_nvg - n_edges_hvg) / n_edges_nvg)

            # Reach-weighted: sum edge lengths |j - i|
            w_nvg = sum(j - i for i in range(N) for j in adj[i] if j > i)
            w_hvg = sum(j - i for i in range(N) for j in hvg_adj[i] if j > i)
            metrics['nvg_hvg_reach_divergence'] = float(
                (w_nvg - w_hvg) / w_nvg) if w_nvg > 0 else 0.0
        else:
            metrics['nvg_hvg_edge_divergence'] = 0.0
            metrics['nvg_hvg_reach_divergence'] = 0.0

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


class ZipfMandelbrotGeometry(ExoticGeometry):
    """
    Zipf-Mandelbrot Geometry - analyzes the linguistic/symbolic "vocabulary" of data.

    Zipf's Law: Frequency f(r) of the r-th most common "word" follows f(r) ∝ r^-alpha.
    Mandelbrot's refinement: f(r) ∝ (r + q)^-alpha, where q accounts for low-rank 
    vocabulary structure.

    This geometry treats N-bit sequences as symbols and measures the richness, 
    diversity, and decay of the resulting "dictionary."
    """

    def __init__(self, n_bits: int = 8, sliding: bool = True):
        self.n_bits = n_bits
        self.sliding = sliding

    @property
    def name(self) -> str:
        return f"Zipf–Mandelbrot ({self.n_bits}-bit)"


    @property

    def description(self) -> str:

        return "Treats N-bit sequences as symbols and fits Zipf-Mandelbrot frequency decay f(r) ~ (r+q)^-alpha. Vocabulary richness, hapax ratio, and Gini coefficient measure symbolic diversity."


    @property

    def view(self) -> str:

        return "distributional"


    @property

    def detects(self) -> str:

        return "Zipf exponent, vocabulary richness, frequency decay"

    @property
    def dimension(self) -> str:
        return "Linguistic"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Converts data into a frequency-ranked vector of word counts."""
        return self.validate_data(data)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        y = self.validate_data(data)
        
        # Extract "words" (N-bit atoms)
        if self.n_bits == 8:
            words = y
        else:
            # For 16-bit, etc., pack bits or use sliding window
            if self.sliding:
                # 16-bit sliding: pair bytes i and i+1
                words = (y[:-1].astype(np.uint16) << 8) | y[1:].astype(np.uint16)
            else:
                # 16-bit non-overlapping
                n = (len(y) // 2) * 2
                words = (y[:n:2].astype(np.uint16) << 8) | y[1:n:2].astype(np.uint16)

        counts = Counter(words)
        frequencies = sorted(counts.values(), reverse=True)
        N_unique = len(frequencies)
        N_total = sum(frequencies)
        
        metrics: Dict[str, float] = {}
        
        # Fit Zipf-Mandelbrot f(r) = C * (r + q)^-alpha
        # via log-log: log(f) = log(C) - alpha * log(r + q)
        # We'll estimate alpha using the Zipf limit (q=0) first
        if N_unique >= 5:
            ranks = np.arange(1, N_unique + 1, dtype=float)
            log_r = np.log(ranks)
            log_f = np.log(frequencies)
            
            coeffs = np.polyfit(log_r, log_f, 1)
            metrics['zipf_alpha'] = float(-coeffs[0])
            
            # Mandelbrot q estimation: 
            # We look for q that maximizes linearity of log(f) vs log(r+q)
            best_q = 0.0
            max_r2 = -1.0
            for q_test in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
                log_rq = np.log(ranks + q_test)
                c_test = np.polyfit(log_rq, log_f, 1)
                pred = np.polyval(c_test, log_rq)
                r2 = 1.0 - np.sum((log_f - pred)**2) / (np.sum((log_f - np.mean(log_f))**2) + 1e-12)
                if r2 > max_r2:
                    max_r2 = r2
                    best_q = q_test
            
            metrics['mandelbrot_q'] = float(best_q)
            metrics['zipf_r_squared'] = float(max_r2)
        else:
            # Too few unique values to fit a rank-frequency distribution
            metrics['zipf_alpha'] = float('nan')
            metrics['mandelbrot_q'] = float('nan')
            metrics['zipf_r_squared'] = float('nan')

        # Hapax Legomena Ratio (words appearing exactly once)
        hapax = sum(1 for f in frequencies if f == 1)
        metrics['hapax_ratio'] = float(hapax / N_unique) if N_unique > 0 else float('nan')

        # Gini Coefficient of frequencies (concentration of vocabulary)
        # G = (2 * Σ i*f_i) / (n * Σ f_i) - (n + 1) / n
        if N_unique > 1:
            f_arr = np.array(frequencies[::-1]) # Sort ascending for standard Gini formula
            index = np.arange(1, N_unique + 1)
            gini = (np.sum((2 * index - N_unique - 1) * f_arr)) / (N_unique * np.sum(f_arr))
            metrics['gini_coefficient'] = float(gini)
        else:
            metrics['gini_coefficient'] = float('nan')

        return GeometryResult(geometry_name=self.name, metrics=metrics, raw_data={})


# =============================================================================
# DYNAMICAL GEOMETRIES — Multifractal, Predictability, Attractor
# =============================================================================


class MultifractalGeometry(ExoticGeometry):
    """
    Multifractal Geometry — singularity spectrum f(α) via structure functions.

    Computes the multifractal spectrum of a 1D signal by estimating the
    scaling exponents τ(q) of the partition function Z(q, ε) = Σ|μ_i|^q
    at multiple moment orders q, then Legendre-transforming to get f(α).

    This directly measures whether a signal has uniform scaling (monofractal,
    like fBm) or variable local regularity (multifractal, like turbulence
    or financial returns).

    Metrics:
    - spectrum_width: width of the f(α) curve (α_max - α_min). Zero for
      monofractal, large for strongly multifractal.
    - alpha_peak: position of the maximum of f(α) — the most common
      singularity strength. Low = rough, high = smooth.
    - asymmetry: left/right asymmetry of f(α). Positive = left-skewed
      (more smooth than rough events), negative = right-skewed.
    - hurst_estimate: H = τ(2)/2, the self-affinity exponent.
      H < 0.5 = anti-persistent, H > 0.5 = persistent.
    - tau_curvature: curvature of τ(q) at q=2. Zero for monofractal,
      negative for multifractal. Measures intermittency.
    """

    def __init__(self, input_scale: float = 255.0, n_scales: int = 8):
        self.input_scale = input_scale
        self.n_scales = n_scales

    @property
    def name(self) -> str:
        return "Multifractal Spectrum"


    @property

    def description(self) -> str:

        return "Estimates the multifractal singularity spectrum f(alpha) via structure functions. Spectrum width measures scaling heterogeneity — monofractal vs. rich multiscale structure."


    @property

    def view(self) -> str:

        return "dynamical"


    @property

    def detects(self) -> str:

        return "Spectrum width, dominant singularity, scaling nonlinearity"

    @property
    def dimension(self) -> str:
        return "variable"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        n = len(x)
        _nan = {k: float('nan') for k in ["spectrum_width", "hurst_estimate"]}

        if n < 64:
            return GeometryResult(self.name, _nan, {})

        # Constant data: zero increments → structure functions degenerate
        if np.ptp(x) == 0:
            return GeometryResult(self.name, _nan, {})

        # Structure function approach: for scale ε, compute moments of |Δx|
        # Use dyadic scales
        qs = np.array([-3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5])
        max_scale_exp = min(self.n_scales, int(np.log2(n // 4)))
        if max_scale_exp < 3:
            return GeometryResult(self.name, _nan, {})
        scales = 2 ** np.arange(1, max_scale_exp + 1)

        # Compute τ(q) via linear regression of log(Z(q,ε)) vs log(ε)
        log_scales = np.log(scales.astype(float))
        tau_q = np.full(len(qs), np.nan)

        for iq, q in enumerate(qs):
            log_Z = np.zeros(len(scales))
            for isc, sc in enumerate(scales):
                # Increments at scale sc
                increments = np.abs(x[sc:] - x[:-sc])
                increments = increments[increments > 1e-15]
                if len(increments) < 10:
                    log_Z[isc] = np.nan
                    continue
                if q > 0:
                    log_Z[isc] = np.log(np.mean(increments ** q))
                else:
                    # For negative q, avoid amplifying near-zero increments
                    clipped = np.maximum(increments, 1e-10)
                    log_Z[isc] = np.log(np.mean(clipped ** q))

            valid = np.isfinite(log_Z)
            if valid.sum() < 3:
                continue
            # Fit slope: τ(q) = d log Z(q,ε) / d log ε
            coeffs = np.polyfit(log_scales[valid], log_Z[valid], 1)
            tau_q[iq] = coeffs[0]

        valid_tau = np.isfinite(tau_q)
        if valid_tau.sum() < 5:
            return GeometryResult(self.name, _nan, {})

        # Hurst exponent from τ(2)
        q2_idx = np.argmin(np.abs(qs - 2))
        hurst = tau_q[q2_idx] / 2.0 if np.isfinite(tau_q[q2_idx]) else 0.5

        # Legendre transform via quadratic fit of τ(q).
        # Only use positive q for spectrum estimation: negative-q structure
        # function moments are dominated by the uint8 quantization floor
        # (smallest nonzero increment = 1/255) and produce systematically
        # biased τ(q), creating false-positive spectrum width for
        # monofractal signals like fBm.
        pos_mask = valid_tau & (qs >= 0.5)
        pos_qs = qs[pos_mask]
        pos_taus = tau_q[pos_mask]

        if len(pos_qs) < 3:
            return GeometryResult(self.name, _nan | {"hurst_estimate": float(np.clip(hurst, 0, 1.5))}, {})

        # Quadratic: τ(q) = a*q² + b*q + c
        # α(q) = 2aq + b, spectrum_width = |2a| * Δq
        # Monofractal: a ≈ 0, multifractal: a < 0 (concave)
        tau_poly = np.polyfit(pos_qs, pos_taus, min(2, len(pos_qs) - 1))
        if len(tau_poly) == 3:
            a_coeff = tau_poly[0]
        else:
            a_coeff = 0.0

        tau_deriv = np.polyder(tau_poly)
        alphas = np.polyval(tau_deriv, pos_qs)
        tau_smooth = np.polyval(tau_poly, pos_qs)
        f_alpha = pos_qs * alphas - tau_smooth

        # Spectrum properties
        alpha_range = abs(alphas.max() - alphas.min())
        f_peak_idx = np.argmax(f_alpha)
        alpha_peak = alphas[f_peak_idx]

        # Asymmetry: compare left and right widths of f(α) relative to peak
        left_width = alpha_peak - alphas.min()
        right_width = alphas.max() - alpha_peak
        if left_width + right_width > 1e-10:
            asymmetry = (left_width - right_width) / (left_width + right_width)
        else:
            asymmetry = 0.0

        # τ curvature = 2a (the quadratic coefficient)
        curvature = 2.0 * a_coeff

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "spectrum_width": float(np.clip(alpha_range, 0, 5)),
                "hurst_estimate": float(np.clip(hurst, 0, 1.5)),
            },
            raw_data={}
        )


class PredictabilityGeometry(ExoticGeometry):
    """
    Predictability Geometry — how quickly information decays with lag.

    Measures the conditional entropy H(X_t | X_{t-1}, ..., X_{t-k}) at
    multiple embedding depths k using coarse-grained symbolization, plus
    sample entropy (SampEn) which quantifies self-similarity of patterns
    at different tolerance levels.

    These metrics directly distinguish:
    - Periodic signals: conditional entropy → 0 at sufficient depth
    - Deterministic chaos: moderate conditional entropy, slow decay
    - Stochastic processes: high conditional entropy, fast saturation
    - IID noise: conditional entropy = marginal entropy at all depths

    Metrics:
    - cond_entropy_k1/k2/k4/k8: conditional entropy at depths 1,2,4,8
    - entropy_decay_rate: how fast conditional entropy drops with depth
      (slope of H(X|past_k) vs k). Near 0 = unpredictable, large negative = predictable.
    - sample_entropy: SampEn(m=2, r=0.2σ) — regularity measure.
      Low = regular/predictable, high = complex/unpredictable.
    - excess_predictability: H(X) - H(X|past_8) — total information
      gained from knowing the past 8 steps.
    - transition_entropy_variance: variance of per-symbol next-symbol entropy.
      High = heterogeneous predictability (some contexts predictive, others not).
    """

    def __init__(self, input_scale: float = 255.0, n_symbols: int = 8):
        self.input_scale = input_scale
        self.n_symbols = n_symbols

    @property
    def name(self) -> str:
        return "Predictability"


    @property

    def description(self) -> str:

        return "Measures conditional entropy at increasing history depths (1, 2, 4, 8 bytes). The decay rate quantifies memory — fast decay means unpredictable, slow decay means structured."


    @property

    def view(self) -> str:

        return "dynamical"


    @property

    def detects(self) -> str:

        return "Conditional entropy, memory depth, sample entropy"

    @property
    def dimension(self) -> str:
        return "information"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def _symbolize(self, x):
        """Coarse-grain to n_symbols levels for entropy estimation."""
        # Floor-based binning: map [0,1] → {0, 1, ..., n_symbols-1}
        s = np.floor(x * self.n_symbols).astype(int)
        return np.clip(s, 0, self.n_symbols - 1)

    def _conditional_entropy(self, symbols, k):
        """H(X_t | X_{t-1}, ..., X_{t-k}) via counting."""
        n = len(symbols)
        if n < k + 100:
            return np.nan

        # Optimized: vectorize context extraction using sliding windows
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Windows of length k+1: first k are context, last is outcome
        all_windows = sliding_window_view(symbols, k + 1)
        total = len(all_windows)
        
        # Use unique() to count occurrences of each context-outcome pair
        # Map rows to tuples for hashing or use unique rows
        # For small n_symbols and k, packing into an integer is faster
        if k < 10 and self.n_symbols <= 10:
            # Packed: c0*S^k + c1*S^(k-1) + ... + out
            powers = (self.n_symbols**np.arange(k + 1, dtype=np.uint64))[::-1]
            packed = all_windows.astype(np.uint64) @ powers
            unique_packed, counts = np.unique(packed, return_counts=True)
            
            # Pack contexts too for fast matching
            contexts_packed = unique_packed // self.n_symbols
            # Sum up joint counts per context
            context_unique, context_counts = np.unique(all_windows[:, :-1].astype(np.uint64) @ powers[1:], return_counts=True)
            
            # Data sufficiency check: avoid bias toward zero if samples per context are too few.
            n_contexts = len(context_unique)
            avg_per_context = total / max(n_contexts, 1)
            if avg_per_context < 5:
                return np.nan

            # Map context_unique -> count
            context_map = dict(zip(context_unique, context_counts))
            
            probs_joint = counts / total
            # Conditional probability: p(joint) / p(context)
            # Find context count for each joint pair
            h_cond = 0.0
            for i, p_joint in enumerate(probs_joint):
                ctx = contexts_packed[i]
                p_cond = counts[i] / context_map[ctx]
                h_cond -= p_joint * np.log2(p_cond)
            return h_cond
        else:
            # Fallback for large k (too large for packing)
            from collections import Counter
            joint_counts = Counter(map(tuple, all_windows))
            context_counts = Counter(map(tuple, all_windows[:, :-1]))
            
            # Data sufficiency check
            n_contexts = len(context_counts)
            avg_per_context = total / max(n_contexts, 1)
            if avg_per_context < 5:
                return np.nan

            h_cond = 0.0
            for pair, count in joint_counts.items():
                p_joint = count / total
                p_cond = count / context_counts[pair[:-1]]
                h_cond -= p_joint * np.log2(p_cond)
            return h_cond

    def _sample_entropy(self, x, m=2, r_frac=0.2):
        """Sample entropy (SampEn): count template matches at length m and m+1."""
        n = len(x)
        if n < 100:
            return np.nan

        # Subsample for speed
        if n > 2000:
            idx = np.linspace(0, n - 1, 2000, dtype=int)
            x = x[idx]
            n = len(x)

        r = r_frac * np.std(x)
        if r < 1e-15:
            return 0.0

        def count_matches(length):
            """Count pairs of matching templates of given length."""
            from numpy.lib.stride_tricks import sliding_window_view
            templates = sliding_window_view(x, length)
            count = 0
            # We still need to avoid self-matches and double-counting pairs.
            # Vectorized approach for all pairs is memory-intensive (N^2).
            # But we can vectorize the INNER loop completely.
            for i in range(len(templates) - 1):
                # Chebyshev distance: max(|a - b|) <= r is same as all(|a - b| <= r)
                matches = np.all(np.abs(templates[i+1:] - templates[i]) <= r, axis=1)
                count += np.sum(matches)
            return count

        A = count_matches(m + 1)
        B = count_matches(m)

        if B == 0:
            return np.nan
        if A == 0:
            return np.log(B)  # conservative estimate

        return -np.log(A / B)

    def _transition_entropy_variance(self, symbols):
        """Variance of per-symbol next-symbol entropy (normalized).

        Builds the k=1 transition matrix P(x_t | x_{t-1}), computes the
        entropy of each row, and returns the variance across rows.

        High variance = some contexts are highly predictive while others
        are not (heterogeneous predictability). Low variance = all contexts
        are equally (un)predictable.

        Evolved via ShinkaEvolve predictability_discovery v1 gen 45.
        """
        n_sym = self.n_symbols
        counts = np.zeros((n_sym, n_sym), dtype=np.int32)
        np.add.at(counts, (symbols[:-1], symbols[1:]), 1)
        row_sums = counts.sum(axis=1)
        valid = row_sums > 5
        if np.sum(valid) <= 1:
            return 0.0
        p_mat = counts[valid] / row_sums[valid, np.newaxis]
        log_p = np.log2(p_mat, where=p_mat > 0,
                        out=np.zeros_like(p_mat, dtype=float))
        row_entropies = -np.sum(p_mat * log_p, axis=1)
        max_ent = np.log2(n_sym)
        if max_ent < 1e-9:
            return 0.0
        return float(np.var(row_entropies / max_ent))

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        n = len(x)
        _nan = {k: float('nan') for k in [
            "cond_entropy_k1",
            "cond_entropy_k8",
            "entropy_decay_rate", "sample_entropy",
            "excess_predictability",
            "transition_entropy_variance"]}

        if n < 200:
            return GeometryResult(self.name, _nan, {})

        symbols = self._symbolize(x)

        # Marginal entropy
        from collections import Counter
        counts = Counter(symbols)
        total = sum(counts.values())
        h_marginal = -sum((c/total) * np.log2(c/total) for c in counts.values())

        # Conditional entropies at different depths
        depths = [1, 2, 4, 8]
        h_cond = {}
        for k in depths:
            h = self._conditional_entropy(symbols, k)
            h_cond[k] = h if np.isfinite(h) else h_marginal

        # Entropy decay rate: slope of H(X|past_k) vs k
        valid_ks = [k for k in depths if np.isfinite(h_cond[k])]
        if len(valid_ks) >= 2:
            ks_arr = np.array(valid_ks, dtype=float)
            hs_arr = np.array([h_cond[k] for k in valid_ks])
            decay_rate = np.polyfit(ks_arr, hs_arr, 1)[0]
        else:
            decay_rate = 0.0

        # Sample entropy
        sampen = self._sample_entropy(x)
        if not np.isfinite(sampen):
            sampen = 5.0  # max complexity

        # Excess predictability
        excess = h_marginal - h_cond.get(8, h_marginal)

        # Transition entropy variance
        trans_ent_var = self._transition_entropy_variance(symbols)

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "cond_entropy_k1": float(np.clip(h_cond.get(1, 0), 0, 10)),
                "cond_entropy_k8": float(np.clip(h_cond.get(8, 0), 0, 10)),
                "entropy_decay_rate": float(np.clip(decay_rate, -2, 0.5)),
                "sample_entropy": float(np.clip(sampen, 0, 10)),
                "excess_predictability": float(np.clip(excess, 0, 10)),
                "transition_entropy_variance": float(np.clip(trans_ent_var, 0, 1)),
            },
            raw_data={}
        )


class AttractorGeometry(ExoticGeometry):
    """
    Attractor Geometry — dimensionality and divergence of embedded dynamics.

    Estimates the correlation dimension D2 via Grassberger-Procaccia and
    the maximum Lyapunov exponent λ1 via the Rosenstein (1993) method,
    both from a Takens delay-embedded signal.

    These directly separate:
    - Low-dimensional chaos: D2 ≈ 2-5, λ1 > 0 (Lorenz, Rössler)
    - High-dimensional noise: D2 → ∞ (saturates at embedding dim), λ1 undefined
    - Periodic signals: D2 ≈ 1, λ1 = 0
    - Quasiperiodic: D2 ≈ 2, λ1 = 0

    Limitation: Grassberger-Procaccia requires N ≳ 10^(D2/2) points for a
    reliable scaling region. With max_points=1500, D2 estimates are reliable
    for D2 ≲ 6 and unreliable above. High-dimensional attractors will show
    unsaturated D2 (d2_saturation near 0).

    Metrics:
    - correlation_dimension: D2 estimate from log C(r) vs log r slope.
      Low = low-dimensional attractor, high = high-dimensional/stochastic.
    - d2_saturation: whether D2 saturates with embedding dimension.
      Near 1.0 = saturated (true dimension found), near 0 = not converged.
    - lyapunov_max: largest Lyapunov exponent (Rosenstein method).
      Positive = chaotic divergence, near 0 = periodic/quasiperiodic,
      negative = damped/contracting.
    - prediction_horizon: estimated steps before exponential divergence
      makes prediction impossible. -1/λ1 in practice.
    - filling_ratio: fraction of embedding space actually occupied.
      Low = signal lives on a low-dimensional manifold.
    """

    def __init__(self, input_scale: float = 255.0, max_points: int = 1500):
        self.input_scale = input_scale
        self.max_points = max_points

    @property
    def name(self) -> str:
        return "Attractor Reconstruction"


    @property

    def description(self) -> str:

        return "Delay-embedding reconstruction of the phase space attractor. Grassberger-Procaccia correlation dimension and maximum Lyapunov exponent distinguish chaos from noise."


    @property

    def view(self) -> str:

        return "dynamical"


    @property

    def detects(self) -> str:

        return "Correlation dimension, Lyapunov exponent, attractor filling"

    @property
    def dimension(self) -> str:
        return "phase space"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    @staticmethod
    def _find_lag(x, max_lag=50):
        """First zero-crossing of autocorrelation."""
        n = len(x)
        xc = x - np.mean(x)
        var = np.sum(xc ** 2)
        if var < 1e-15:
            return 1
        for lag in range(1, min(max_lag, n // 3)):
            acf = np.sum(xc[:n-lag] * xc[lag:]) / var
            if acf <= 0:
                return lag
        return max_lag

    def _delay_embed_matrix(self, x, dim, lag):
        """Build delay embedding matrix."""
        N = len(x) - (dim - 1) * lag
        if N < 10:
            return None
        embedded = np.zeros((N, dim))
        for d in range(dim):
            embedded[:, d] = x[d * lag:d * lag + N]
        return embedded

    def _correlation_dimension(self, x, lag):
        """Grassberger-Procaccia D2 at embedding dims 2-8."""
        dims = [2, 3, 4, 5, 6, 8]
        d2_values = []

        for dim in dims:
            emb = self._delay_embed_matrix(x, dim, lag)
            if emb is None or len(emb) < 50:
                continue

            # Subsample
            N = len(emb)
            if N > self.max_points:
                idx = np.random.default_rng(0).choice(N, self.max_points, replace=False)
                idx.sort()
                emb = emb[idx]
                N = len(emb)

            # Pairwise distances (Chebyshev for speed)
            # Use vectorized block computation
            n_samples = min(N, 500)
            dists = []
            for i in range(n_samples):
                # Compute distance from point i to all points after it
                # Using max(abs(diff)) is Chebyshev distance
                diffs = np.abs(emb[i+1:] - emb[i])
                if diffs.size > 0:
                    dists.append(np.max(diffs, axis=1))
            
            if not dists:
                continue
            dists = np.concatenate(dists)
            dists = dists[dists > 1e-15]
            if len(dists) < 100:
                continue

            # C(r) at multiple radii
            # Sorting dists makes C(r) counting O(log N) per radius
            dists.sort()
            log_dists = np.log(dists)
            r_min = np.percentile(log_dists, 5)
            r_max = np.percentile(log_dists, 50)
            if r_max - r_min < 0.5:
                continue

            n_radii = 15
            log_rs = np.linspace(r_min, r_max, n_radii)
            n_pairs = len(dists)
            # Use searchsorted for extremely fast counting on sorted distances
            counts = np.searchsorted(log_dists, log_rs, side='right')
            log_Cr = np.log(np.maximum(counts / n_pairs, 1e-30))

            # Linear fit in scaling region
            valid = np.isfinite(log_Cr) & (log_Cr > np.log(1e-10))
            if valid.sum() < 5:
                continue
            slope = np.polyfit(log_rs[valid], log_Cr[valid], 1)[0]
            d2_values.append((dim, max(slope, 0.0)))

        if not d2_values:
            return 0.0, 0.0

        # Find first plateau in D2 vs embedding dim where D2 is well
        # below the embedding dimension. GP overestimates D2 when the
        # scaling region degrades in high-D, so early convergence at
        # D2 << dim is the most reliable signal of a true attractor.
        # A "plateau" at D2 ≈ dim is just noise saturating at
        # embedding capacity and should be rejected.
        dims_used = [d for d, _ in d2_values]
        d2s = [d2 for _, d2 in d2_values]

        d2_final = d2s[-1]
        saturation = 0.0
        for i in range(len(d2s) - 1):
            if d2s[i] < 0.1:
                continue
            ratio = min(d2s[i] / d2s[i+1], d2s[i+1] / d2s[i])
            avg_d2 = (d2s[i] + d2s[i+1]) / 2.0
            avg_dim = (dims_used[i] + dims_used[i+1]) / 2.0
            # Accept plateau only if D2 is below 60% of embedding dim
            if ratio > 0.75 and avg_d2 < avg_dim * 0.6:
                d2_final = avg_d2
                saturation = ratio
                break
        else:
            # No low-D plateau found — report highest-dim estimate
            d2_final = d2s[-1]
            if len(d2s) >= 2 and d2s[-2] > 0.1:
                saturation = min(d2s[-2] / d2s[-1], d2s[-1] / d2s[-2])
                saturation = min(saturation, 1.0)

        return d2_final, saturation

    def _lyapunov_rosenstein(self, x, lag, dim=5):
        """Maximum Lyapunov exponent via Rosenstein (1993) method."""
        emb = self._delay_embed_matrix(x, dim, lag)
        if emb is None or len(emb) < 100:
            return 0.0

        N = len(emb)
        if N > self.max_points:
            emb = emb[:self.max_points]
            N = self.max_points

        # For each point, find nearest neighbor (excluding temporal neighbors)
        min_temporal_sep = max(lag * dim, 10)
        max_iter = min(N // 4, 200)
        
        n_query = min(N - max_iter, 500)
        divergences = np.full((n_query, max_iter), np.nan)
        
        # Pre-build distance exclusion mask for the whole matrix to speed up searches
        # But for N=1500, a simple loop over queries is often fine if the distance part is vectorized.
        for i in range(n_query):
            # Find nearest neighbor
            # Chebyshev distance is faster to compute than Euclidean
            diffs = np.abs(emb - emb[i])
            dists = np.max(diffs, axis=1)
            
            # Mask temporal neighbors
            low = max(0, i - min_temporal_sep)
            high = min(N, i + min_temporal_sep + 1)
            dists[low:high] = np.inf
            
            j = np.argmin(dists)
            d0 = dists[j]
            if d0 < 1e-15:
                continue

            # Track divergence over time: vectorized across k for this i
            # dk[k] = ||emb[i+k] - emb[j+k]||
            # Ensure we don't exceed N
            actual_k = min(max_iter, N - i, N - j)
            if actual_k < 5:
                continue
            
            # Vectorized divergence for all k steps simultaneously
            traj_i = emb[i:i+actual_k]
            traj_j = emb[j:j+actual_k]
            dk = np.max(np.abs(traj_i - traj_j), axis=1)
            
            valid_dk = dk > 1e-15
            divergences[i, :actual_k][valid_dk] = np.log(dk[valid_dk])

        # Average divergence curve (suppress empty-slice warning)
        with np.errstate(all='ignore'):
            mean_div = np.nanmean(divergences, axis=0)
        valid = np.isfinite(mean_div)
        if valid.sum() < 10:
            return 0.0

        # Fit slope to initial linear region (first 20% of valid points)
        n_valid = valid.sum()
        n_fit = max(5, n_valid // 5)
        valid_idx = np.where(valid)[0][:n_fit]
        if len(valid_idx) < 3:
            return 0.0

        t = valid_idx.astype(float)
        y = mean_div[valid_idx]
        slope = np.polyfit(t, y, 1)[0]
        return slope

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        x = self.embed(data)
        n = len(x)
        _nan = {k: float('nan') for k in [
            "correlation_dimension", "d2_saturation",
            "lyapunov_max", "filling_ratio"]}

        if n < 200:
            return GeometryResult(self.name, _nan, {})

        lag = self._find_lag(x)

        # Correlation dimension
        d2, saturation = self._correlation_dimension(x, lag)

        # Lyapunov exponent — embedding dim should exceed D2 by ~2
        # (Takens theorem: d >= 2*D2 + 1 suffices, but over-embedding is safe
        # while under-embedding corrupts the estimate)
        lya_dim = max(5, int(np.ceil(d2)) + 2) if d2 > 0 else 5
        lya_dim = min(lya_dim, 10)  # cap to avoid sparse embeddings
        lam = self._lyapunov_rosenstein(x, lag, dim=lya_dim)

        # Filling ratio: how much of the bounding box does the signal fill?
        dim_embed = lya_dim
        emb = self._delay_embed_matrix(x, dim_embed, lag)
        if emb is not None and len(emb) > 50:
            # Fraction of occupied cells in a coarse grid
            n_bins = 4  # 4^5 = 1024 cells
            mins = emb.min(axis=0)
            maxs = emb.max(axis=0)
            ranges = maxs - mins
            ranges[ranges < 1e-15] = 1.0
            binned = ((emb - mins) / ranges * (n_bins - 0.001)).astype(int)
            binned = np.clip(binned, 0, n_bins - 1)
            # Count unique cells
            cells = set(map(tuple, binned))
            total_cells = n_bins ** dim_embed
            filling = len(cells) / total_cells
        else:
            filling = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "correlation_dimension": float(np.clip(d2, 0, 20)),
                "d2_saturation": float(np.clip(saturation, 0, 1)),
                "lyapunov_max": float(np.clip(lam, -2, 5)),
                "filling_ratio": float(np.clip(filling, 0, 1)),
            },
            raw_data={}
        )


class InflationGeometry(ExoticGeometry):
    """
    Inflation (Substitution Rule) Geometry — detects hierarchical self-similar
    structure characteristic of substitution dynamical systems.

    A substitution rule σ maps each symbol to a word (e.g. Fibonacci: a→ab, b→a).
    Iterating σ generates an infinite aperiodic sequence whose statistical
    structure is invariant under block-renormalization at scale λ, where λ is
    the Perron-Frobenius eigenvalue of the substitution matrix. For Fibonacci
    λ = φ ≈ 1.618; for Thue-Morse λ = 2; for period-doubling λ = 2.

    All metrics operate on a binary-quantized symbolic sequence to match the
    theoretical framework (symbolic dynamics over finite alphabets).

    Metrics:
    1. complexity_linearity — R² of p(n) vs n on binary symbols. Substitution
       sequences have p(n) = Θ(n). Periodic: bounded. Random: 2^n.
    2. entropy_rate — topological entropy h = lim log(p(n))/n. Zero for
       substitution and periodic, positive for random and chaos.
    3. discrepancy — max|D(n)|/√N where D(n) = cumulative symbol count minus
       expected. O(1/√N) for balanced substitutions, O(1) for random.
    4. return_concentration — 1 - mean(n_distinct_gaps / n_gaps) for top
       subwords. Substitutions have few distinct return times (hierarchical
       tiling constrains positions). Random: many distinct gaps.
    5. acf_geometric — prominence of ACF peaks at geometric spacings (λ^k)
       vs arithmetic spacings. High for substitution (peaks at powers of λ),
       low for periodic (peaks at multiples of T) and random (no peaks).

    References:
    - Queffelec, M. (2010). Substitution Dynamical Systems — Spectral Analysis.
      Lecture Notes in Mathematics 1294. Springer.
    - Baake, M. & Grimm, U. (2013). Aperiodic Order, Vol. 1. Cambridge.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Inflation (Substitution)"

    @property
    def description(self) -> str:
        return (
            "Detects substitution-rule structure via linear subword complexity, "
            "zero topological entropy, bounded discrepancy, concentrated return "
            "times, and geometric ACF peaks. Lights up on Fibonacci, Thue-Morse, "
            "and L-system sequences."
        )

    @property
    def view(self) -> str:
        return "quasicrystal"

    @property
    def detects(self) -> str:
        return "Substitution inflation symmetry, linear complexity, bounded discrepancy"

    @property
    def encoding_invariant(self) -> bool:
        return True

    @property
    def dimension(self) -> str:
        return "1D multi-scale"

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Multi-scale block representation: original + coarse-grained versions."""
        data = self.validate_data(data)
        x = data.astype(np.float64) / max(self.input_scale, 1.0)
        cols = [x]
        for bs in [2, 3]:
            n_blocks = len(x) // bs
            if n_blocks > 0:
                blocked = x[:n_blocks * bs].reshape(n_blocks, bs).mean(axis=1)
                padded = np.full(len(x), np.nan)
                padded[:n_blocks] = blocked
                cols.append(padded)
        return np.column_stack(cols)

    def _binarize(self, data: np.ndarray) -> np.ndarray:
        """Robust binary quantization.

        For data with ≤4 unique values: threshold at midpoint of extremes
        (avoids degenerate case where majority == max).
        Otherwise: threshold at median.
        """
        uniq = np.unique(data)
        if len(uniq) <= 1:
            return np.zeros(len(data), dtype=np.int8)
        if len(uniq) <= 4:
            threshold = (float(uniq[0]) + float(uniq[-1])) / 2.0
        else:
            threshold = float(np.median(data))
        return (data.astype(np.float64) > threshold).astype(np.int8)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        """Compute inflation/substitution metrics on binary-quantized symbols."""
        data = self.validate_data(data)
        x = data.astype(np.float64)
        N = len(x)

        # === Binary quantization ===
        binary = self._binarize(data)
        bin_tuple = tuple(binary.tolist())

        # === 1. Subword complexity linearity ===
        # p(n) = number of distinct n-grams in the binary sequence.
        # Substitution: p(n) = c·n + d (strictly linear).
        # Periodic with period T: p(n) = T for n > some threshold.
        # Random binary: p(n) = min(2^n, N-n+1).
        # Measure up to the largest n where 2^n stays below sample saturation.
        max_n = min(11, N // 10)
        while max_n > 2 and 2 ** max_n > 0.8 * (N - max_n):
            max_n -= 1

        if max_n < 3:
            complexity_linearity = 0.0
            pn_array = np.array([])
            ns_array = np.array([])
        else:
            ns_array = np.arange(2, max_n + 1)
            pn_array = np.empty(len(ns_array))
            for i, n in enumerate(ns_array):
                subwords = set()
                for j in range(N - n + 1):
                    subwords.add(bin_tuple[j:j + n])
                pn_array[i] = len(subwords)

            coeffs = np.polyfit(ns_array, pn_array, 1)
            pn_pred = np.polyval(coeffs, ns_array)
            ss_res = np.sum((pn_array - pn_pred) ** 2)
            ss_tot = np.sum((pn_array - pn_array.mean()) ** 2)
            if ss_tot < 1e-10:
                complexity_linearity = 1.0
            else:
                complexity_linearity = float(np.clip(1.0 - ss_res / ss_tot, 0, 1))

        # === 2. Topological entropy rate (growth ratio) ===
        # Instead of log(p(n))/n (converges too slowly), use the mean
        # growth ratio p(n+1)/p(n). For substitution (linear p(n)):
        # ratio → 1. For random binary: ratio → 2. Normalize so that
        # ratio=1 → 0.0, ratio=2 → 1.0.
        if len(pn_array) >= 3:
            ratios = pn_array[1:] / np.maximum(pn_array[:-1], 1)
            mean_ratio = float(np.mean(ratios))
            # Clamp: ratio < 1 shouldn't happen (p(n) is non-decreasing)
            entropy_rate = float(np.clip((mean_ratio - 1.0), 0, 1))
        else:
            entropy_rate = 0.0

        # === 3. Discrepancy ===
        # D(n) = |#{1's in first n symbols} - n·freq(1)|.
        # Balanced substitutions: max D(n) = O(1), so D*/√N → 0.
        # Random binary: max D(n) = O(√N), so D*/√N → const ≈ 0.5.
        # Periodic (e.g. binarized sine): D*/√N can be large (long runs).
        freq_1 = binary.mean()
        if freq_1 < 0.01 or freq_1 > 0.99:
            discrepancy = 0.0
        else:
            cumsum = np.cumsum(binary)
            expected = np.arange(1, N + 1) * freq_1
            D = np.abs(cumsum - expected)
            discrepancy = float(D.max() / np.sqrt(N))

        # === 4. Return-time concentration (CV-based) ===
        # For each common subword, compute the coefficient of variation
        # (std/mean) of gaps between occurrences. Substitution systems have
        # low CV (few distinct, regular gap values). Random sequences have
        # geometric return times → CV ≈ 1. Periodic sequences have CV ≈ 0.
        # Metric: 1 / (1 + mean_CV) — high for substitution+periodic, low
        # for random. Combined with discrepancy, separates substitution from
        # both periodic and random.
        from collections import Counter
        subword_len = 3
        if N < subword_len + 20:
            return_concentration = 0.0
        else:
            sw_counts = Counter()
            for j in range(N - subword_len + 1):
                sw_counts[bin_tuple[j:j + subword_len]] += 1

            top_k = min(5, len(sw_counts))
            top_sws = [sw for sw, _ in sw_counts.most_common(top_k)]

            cvs = []
            for sw in top_sws:
                positions = []
                for j in range(N - subword_len + 1):
                    if bin_tuple[j:j + subword_len] == sw:
                        positions.append(j)
                if len(positions) < 5:
                    continue
                gaps = np.diff(positions).astype(np.float64)
                gap_mean = gaps.mean()
                if gap_mean < 1e-10:
                    continue
                cvs.append(float(gaps.std() / gap_mean))

            if cvs:
                return_concentration = float(1.0 / (1.0 + np.mean(cvs)))
            else:
                return_concentration = 0.0

        # === 5. ACF peak-ratio constancy ===
        # Substitution sequences (inflation factor λ) have ACF peaks at
        # positions that follow a geometric progression: Fibonacci numbers
        # for λ=φ, powers of 2 for λ=2, etc. The ratio between consecutive
        # peak positions converges to λ.
        # Periodic sequences have ACF peaks at arithmetic spacings (T, 2T, ...).
        # Random/chaotic sequences have no genuine ACF peaks.
        #
        # Metric: 1/(1 + CV(consecutive_peak_ratios)).
        # High → constant ratios → geometric (substitution).
        # Low → varying ratios → arithmetic (periodic) or no structure.
        # Zero → fewer than 4 genuine peaks detected.
        from scipy.signal import find_peaks as _find_peaks
        x_centered = x - x.mean()
        x_std = x.std()
        if x_std < 1e-10 or N < 100:
            acf_geometric = 0.0
        else:
            x_norm = x_centered / x_std
            n_fft = 2 * N
            xf = np.fft.rfft(x_norm, n=n_fft)
            acf_full = np.fft.irfft(xf * np.conj(xf), n=n_fft)[:N] / N
            acf = acf_full[1:]  # Skip lag 0

            max_lag = min(N // 2, len(acf))
            acf = acf[:max_lag]
            abs_acf = np.abs(acf)

            # Require high prominence to reject noise fluctuations.
            # White noise ACF has std ≈ 1/√N, so 4/√N ≈ 4σ.
            min_prom = 4.0 / np.sqrt(N)
            peak_idx, peak_props = _find_peaks(
                abs_acf, distance=3, prominence=min_prom)

            if len(peak_idx) < 4:
                acf_geometric = 0.0
            else:
                # Sort by prominence, take the most prominent peaks
                proms = peak_props['prominences']
                top_order = np.argsort(proms)[::-1]
                n_use = min(15, len(peak_idx))
                top_peaks = np.sort(peak_idx[top_order[:n_use]]) + 1  # 1-indexed

                if len(top_peaks) < 4:
                    acf_geometric = 0.0
                else:
                    consec_ratios = (top_peaks[1:].astype(np.float64) /
                                     top_peaks[:-1].astype(np.float64))
                    ratio_mean = consec_ratios.mean()
                    if ratio_mean < 0.01:
                        acf_geometric = 0.0
                    else:
                        ratio_cv = consec_ratios.std() / ratio_mean
                        acf_geometric = float(1.0 / (1.0 + ratio_cv))

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                "complexity_linearity": complexity_linearity,
                "entropy_rate": entropy_rate,
                "discrepancy": discrepancy,
                "return_concentration": return_concentration,
                "acf_geometric": acf_geometric,
            },
            raw_data={}
        )


class KleinBottleGeometry(ExoticGeometry):
    """
    Klein Bottle Geometry — detects GF(2) linear structure via the topology
    of the Klein bottle.

    The Klein bottle K has homology H_1(K) = Z + Z/2Z. The Z/2Z torsion is
    algebraically GF(2), making K the natural topological space for detecting
    linear recurrences over GF(2) — XorShift generators, LFSRs, and related
    constructions.

    The non-orientability of K corresponds to the "twist" in GF(2) algebra:
    XOR operations flip bits without maintaining a consistent orientation,
    mirroring how paths on K can return orientation-reversed.

    Metrics:

    1. linear_complexity — Per-bit-channel Berlekamp-Massey LFSR length,
       averaged across 8 bit positions, normalized by n/2. For true random
       bits, the shortest LFSR has length ~n/2 (value ~1.0). For GF(2)-linear
       generators, the LFSR length equals the state size (XorShift32 = 0.0625).
       Structured signals (Brownian, financial) show intermediate values.

    2. rank_deficit — Form 16x16 binary matrices from consecutive bits and
       compute rank over GF(2) via Gaussian elimination. The normalized rank
       deficit (16 - rank)/16 reflects how much linear dependence exists in the
       bit stream. Interestingly, GF(2)-linear generators produce *zero*
       deficit (always full rank) — their algebraic structure guarantees
       independence, while true random data occasionally produces rank-deficient
       matrices (~0.92 expected deficit for 16x16).

    3. orientation_coherence — Maps consecutive byte pairs onto the Klein
       bottle (unit square with twisted identification (0,y)~(1,1-y)) and
       measures how consistently the trajectory respects local orientation.
       Computed via signed areas of consecutive triangles in the delay
       embedding, with GF(2) twist correction at boundary crossings.
    """

    def __init__(self, input_scale: float = 255.0):
        self.input_scale = input_scale

    @property
    def name(self) -> str:
        return "Klein Bottle"

    @property
    def dimension(self) -> str:
        return "algebraic-topological"

    @property
    def description(self) -> str:
        return ("Detects GF(2) linear structure via Klein bottle topology: "
                "LFSR complexity, binary rank over GF(2), and non-orientable "
                "trajectory coherence. The Klein bottle's Z/2Z torsion is the "
                "topological manifestation of GF(2) algebra.")

    @property
    def view(self) -> str:
        return "topological"

    @property
    def detects(self) -> str:
        return "GF(2) linear recurrences, LFSRs, XorShift generators, bit-level linear dependencies"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def _berlekamp_massey(self, bits: np.ndarray, max_len: int) -> int:
        """Berlekamp-Massey algorithm: shortest LFSR length for a bit sequence."""
        n = min(len(bits), max_len)
        s = bits[:n].astype(np.int8)
        c = np.zeros(n + 1, dtype=np.int8)
        b = np.zeros(n + 1, dtype=np.int8)
        c[0] = 1
        b[0] = 1
        L = 0
        m = 1
        for i in range(n):
            d = s[i]
            for j in range(1, L + 1):
                d ^= c[j] & s[i - j]
            d &= 1
            if d == 0:
                m += 1
            else:
                t = c.copy()
                for j in range(n + 1 - m):
                    c[j + m] ^= b[j]
                if 2 * L <= i:
                    L = i + 1 - L
                    b = t.copy()
                    m = 1
                else:
                    m += 1
        return L

    def _gf2_rank(self, matrix: np.ndarray) -> int:
        """Rank of a binary matrix over GF(2) via Gaussian elimination."""
        m = matrix.copy().astype(np.uint8)
        rows, cols = m.shape
        rank = 0
        for col in range(cols):
            pivot = None
            for row in range(rank, rows):
                if m[row, col] == 1:
                    pivot = row
                    break
            if pivot is None:
                continue
            m[[rank, pivot]] = m[[pivot, rank]]
            for row in range(rows):
                if row != rank and m[row, col] == 1:
                    m[row] = m[row] ^ m[rank]
            rank += 1
        return rank

    def compute_metrics(self, data: np.ndarray) -> 'GeometryResult':
        n = len(data)
        fallback = {
            'linear_complexity': 1.0,
            'rank_deficit': 0.0,
            'orientation_coherence': 0.5,
        }

        if n < 128:
            return GeometryResult(
                geometry_name=self.name,
                metrics=dict(fallback),
                raw_data={}
            )

        bits = np.unpackbits(data.astype(np.uint8))

        # --- 1. Linear complexity: per-bit-channel BM ---
        bm_len = min(512, n)  # BM on up to 512 bits per channel
        channel_lcs = []
        for bit_pos in range(8):
            channel = ((data >> bit_pos) & 1).astype(np.int8)
            lc = self._berlekamp_massey(channel, max_len=bm_len)
            channel_lcs.append(lc / (bm_len / 2.0))
        linear_complexity = float(np.mean(channel_lcs))

        # --- 2. Rank deficit: 16x16 GF(2) matrices ---
        mat_size = 16
        bits_per_mat = mat_size * mat_size
        n_matrices = len(bits) // bits_per_mat
        if n_matrices >= 3:
            ranks = []
            for i in range(min(n_matrices, 30)):
                mat = bits[i * bits_per_mat:(i + 1) * bits_per_mat].reshape(
                    mat_size, mat_size)
                ranks.append(self._gf2_rank(mat))
            rank_deficit = float((mat_size - np.mean(ranks)) / mat_size)
        else:
            rank_deficit = 0.0

        # --- 3. Orientation coherence on the Klein bottle ---
        # Map consecutive byte pairs onto [0,1]^2, apply Klein bottle
        # identification (0,y) ~ (1, 1-y).
        #
        # For each consecutive triple (p_t, p_{t+1}, p_{t+2}), form the
        # triangle in the Klein bottle's fundamental domain and compute its
        # signed area. On an orientable surface, signed areas are consistent
        # locally. On the Klein bottle, triangles near the twist boundary
        # (x near 0 or 1) can flip sign.
        #
        # The orientation coherence is the autocorrelation of signed-area
        # signs: high coherence means the trajectory respects a local
        # orientation (like random data on a torus), low coherence means
        # frequent sign flips (like GF(2)-linear data threading through
        # the twist).
        if n >= 4:
            x = data[:-1].astype(np.float64) / 255.0
            y = data[1:].astype(np.float64) / 255.0

            # Signed areas of consecutive triangles in (x, y) space
            n_tri = len(x) - 1
            # Triangle vertices: (x[i], y[i]), (x[i+1], y[i+1])
            # but we use the trajectory: p_t = (x_t, y_t), p_{t+1}, p_{t+2}
            # For Klein bottle: apply twist when x wraps
            # Detect twist crossings: when |dx| > 0.5, the path wraps
            dx = np.diff(x)
            wrap_mask = np.abs(dx) > 0.5

            # Signed area = 0.5 * (x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1))
            if n_tri >= 2:
                x0, y0 = x[:-1], y[:-1]
                x1, y1 = x[1:], y[1:]
                # Apply Klein twist: when wrapping, reflect y
                y1_twisted = y1.copy()
                y1_twisted[wrap_mask[:len(y1_twisted)]] = 1.0 - y1_twisted[
                    wrap_mask[:len(y1_twisted)]]

                # Use twisted coordinates for signed area of segments
                areas = dx[:len(x0)] * (y0 + y1_twisted) / 2.0
                signs = np.sign(areas)
                signs = signs[signs != 0]

                if len(signs) >= 4:
                    # Autocorrelation of signs at lag 1
                    sc = signs - signs.mean()
                    c0 = np.dot(sc, sc)
                    if c0 > 1e-10:
                        c1 = np.dot(sc[:-1], sc[1:])
                        orientation_coherence = float(
                            np.clip((c1 / c0 + 1) / 2, 0, 1))
                    else:
                        orientation_coherence = 0.5
                else:
                    orientation_coherence = 0.5
            else:
                orientation_coherence = 0.5
        else:
            orientation_coherence = 0.5

        # --- 4. WHT spectral kurtosis (evolved via ShinkaEvolve) ---
        # Walsh-Hadamard Transform is the natural Fourier transform over GF(2).
        # Spectral kurtosis measures energy concentration: high = structured
        # (few dominant coefficients), low = random (flat spectrum).
        target_len = 1 << (n.bit_length() - 1)
        if target_len >= 128:
            data_trunc = data[:target_len]
            channel_kurtosis = []
            for bit_pos in range(8):
                ch = ((data_trunc >> bit_pos) & 1).astype(np.float64)
                signal = 1.0 - 2.0 * ch  # {0,1} → {+1,-1}
                # In-place Fast Walsh-Hadamard Transform
                h = 1
                nn = len(signal)
                while h < nn:
                    for i in range(0, nn, h * 2):
                        for j in range(i, i + h):
                            x_val = signal[j]
                            y_val = signal[j + h]
                            signal[j] = x_val + y_val
                            signal[j + h] = x_val - y_val
                    h *= 2
                ac_power = signal[1:] ** 2
                sum_sq = np.sum(ac_power)
                if sum_sq < 1e-9:
                    channel_kurtosis.append(float(nn - 1))
                else:
                    channel_kurtosis.append(
                        float((nn - 1) * np.sum(ac_power ** 2) / (sum_sq ** 2)))
            wht_spectral_kurtosis = float(np.mean(np.log1p(channel_kurtosis)))
        else:
            wht_spectral_kurtosis = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'linear_complexity': linear_complexity,
                'rank_deficit': rank_deficit,
                'orientation_coherence': orientation_coherence,
                'wht_spectral_kurtosis': wht_spectral_kurtosis,
            },
            raw_data={}
        )


class NonstationarityGeometry(ExoticGeometry):
    """
    Nonstationarity / Volatility-of-Volatility Geometry — measures how the
    local geometric character of a sequence changes over time.

    The framework's other geometries compute static, full-sequence metrics.
    This geometry tracks the trajectory through a space of local descriptors
    on sliding windows, characterizing the *derivative* of the metric space.

    A 5D local descriptor g(t) = [entropy, acf1, variance, kurtosis, perm_entropy]
    is computed on overlapping windows. The sequence of g(t) traces a curve in
    descriptor space. Metrics characterize that curve:

    1. metric_volatility  — mean speed ||dg/dt|| in z-scored descriptor space.
       How fast is the local geometry changing? High for nonstationary processes
       (chirps, regime switches), low for stationary ones (noise, periodic).
    2. vol_of_vol — CV of ||dg/dt||. Is the rate of geometric change itself
       stable or bursty? High for regime-switching and concatenated processes.
       This is the actual "volatility of volatility."
    3. regime_persistence — ACF decay time of g(t), normalized to [0, 1].
       How long do geometric regimes last? High for piecewise-stationary
       data, low for rapidly fluctuating processes.
    4. trajectory_dim — PCA participation ratio of the g(t) cloud, normalized
       by descriptor count. Does the geometry walk through a low-D subspace
       (constrained evolution, e.g. a chirp sweeps one axis) or fill a high-D
       space (unconstrained, e.g. noise)?
    """

    def __init__(self, input_scale: float = 255.0,
                 window_size: int = 128, step: int = 32):
        self.input_scale = input_scale
        self.window_size = window_size
        self.step = step

    @property
    def name(self) -> str:
        return "Nonstationarity"

    @property
    def dimension(self) -> str:
        return "trajectory"

    @property
    def description(self) -> str:
        return ("Measures how local geometric character changes over time: "
                "speed of change, burstiness, regime duration, and trajectory "
                "dimensionality in a 5D local descriptor space.")

    @property
    def view(self) -> str:
        return "dynamical"

    @property
    def detects(self) -> str:
        return "heteroskedasticity, regime switches, geometric non-stationarity"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def _perm_entropy(self, x: np.ndarray, order: int = 4) -> float:
        """Permutation entropy of a short segment, normalized to [0, 1]."""
        n = len(x)
        if n < order + 1:
            return 0.0
        counts: Dict[tuple, int] = {}
        for i in range(n - order + 1):
            pattern = tuple(int(j) for j in np.argsort(x[i:i + order]))
            counts[pattern] = counts.get(pattern, 0) + 1
        total = sum(counts.values())
        probs = np.array(list(counts.values())) / total
        h = -np.sum(probs * np.log2(probs + 1e-15))
        import math
        max_h = np.log2(min(math.factorial(order), total))
        return float(h / max_h) if max_h > 0 else 0.0

    def _local_descriptor(self, window: np.ndarray) -> np.ndarray:
        """Compute 5D local descriptor vector for a window of uint8 data."""
        x = window.astype(np.float64)
        n = len(x)

        # 1. Entropy (Shannon on histogram), normalized by log2(256)=8
        hist = np.bincount(window.astype(int), minlength=256)
        p = hist[hist > 0] / n
        entropy = float(-np.sum(p * np.log2(p)) / 8.0)

        # 2. Lag-1 autocorrelation
        xc = x - x.mean()
        c0 = np.dot(xc, xc)
        acf1 = float(np.dot(xc[:-1], xc[1:]) / (c0 + 1e-15)) if c0 > 1e-10 else 0.0

        # 3. Variance (normalized by 255^2)
        variance = float(np.var(x) / (255.0 ** 2))

        # 4. Excess kurtosis (0.0 for constant data — undefined moment)
        from scipy.stats import kurtosis as _kurt
        kurt = float(_kurt(x, fisher=True))
        if not np.isfinite(kurt):
            kurt = 0.0

        # 5. Permutation entropy (order 4)
        pe = self._perm_entropy(window, order=4)

        return np.array([entropy, acf1, variance, kurt, pe])

    def compute_metrics(self, data: np.ndarray) -> 'GeometryResult':
        n = len(data)
        ws = self.window_size
        st = self.step

        fallback = {
            'metric_volatility': 0.0,
            'vol_of_vol': 0.0,
            'regime_persistence': 0.0,
            'trajectory_dim': 0.0,
        }

        if n < ws + st:
            return GeometryResult(
                geometry_name=self.name,
                metrics=dict(fallback),
                raw_data={}
            )

        # Compute trajectory g(t)
        positions = list(range(0, n - ws + 1, st))
        if len(positions) < 5:
            return GeometryResult(
                geometry_name=self.name,
                metrics=dict(fallback),
                raw_data={}
            )

        G = np.array([self._local_descriptor(data[i:i + ws]) for i in positions])
        n_pts = len(G)

        # Z-score each descriptor dimension
        stds = G.std(axis=0)
        stds[stds < 1e-10] = 1.0
        Gz = (G - G.mean(axis=0)) / stds

        # Velocities
        dG = np.diff(Gz, axis=0)
        speeds = np.linalg.norm(dG, axis=1)

        # 5. dynamic_coupling: cross-dimension velocity correlation (evolved via ShinkaEvolve)
        #    Sum of |off-diagonal| correlations of descriptor velocity vectors.
        #    High when geometric properties change together (regime switches);
        #    low when changes are independent (noise).
        if dG.shape[0] >= 2:
            dG_stds = dG.std(axis=0)
            dG_stds[dG_stds < 1e-10] = 1.0
            dGz = (dG - dG.mean(axis=0)) / dG_stds
            vel_corr = np.cov(dGz, rowvar=False)
            dc = np.sum(np.abs(np.triu(vel_corr, k=1)))
            dynamic_coupling = float(dc) if np.isfinite(dc) else 0.0
        else:
            dynamic_coupling = 0.0

        # 1. metric_volatility: mean speed
        metric_volatility = float(speeds.mean())

        # 2. vol_of_vol: CV of speed
        vol_of_vol = float(speeds.std() / (speeds.mean() + 1e-15)) if speeds.mean() > 1e-10 else 0.0

        # 3. regime_persistence: ACF decay time of g(t)
        max_lag = min(n_pts // 3, 30)
        if max_lag < 2:
            regime_persistence = 1.0
        else:
            acfs = []
            for d in range(Gz.shape[1]):
                col = Gz[:, d]
                c0 = np.var(col)
                if c0 < 1e-15:
                    acfs.append(np.zeros(max_lag))
                    continue
                col_c = col - col.mean()
                acf = np.array([
                    np.dot(col_c[:n_pts - lag], col_c[lag:]) / (c0 * n_pts)
                    for lag in range(1, max_lag + 1)
                ])
                acfs.append(acf)
            mean_acf = np.mean(acfs, axis=0)
            threshold = 1.0 / np.e
            decay_lag = max_lag
            for lag_idx, a in enumerate(mean_acf):
                if a < threshold:
                    decay_lag = lag_idx + 1
                    break
            regime_persistence = float(decay_lag / max_lag)

        # 4. trajectory_dim: PCA participation ratio
        if n_pts < 3:
            trajectory_dim = 1.0
        else:
            try:
                cov = np.cov(Gz.T)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) == 0:
                    trajectory_dim = 0.0
                else:
                    trajectory_dim = float(
                        (eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-15)
                        / Gz.shape[1]
                    )
            except np.linalg.LinAlgError:
                trajectory_dim = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'metric_volatility': metric_volatility,
                'vol_of_vol': vol_of_vol,
                'regime_persistence': regime_persistence,
                'trajectory_dim': trajectory_dim,
                'dynamic_coupling': dynamic_coupling,
            },
            raw_data={}
        )


class NavierStokesGeometry(ExoticGeometry):
    """
    Navier-Stokes turbulence geometry.

    Tests whether 1D signal statistics match predictions from turbulence theory:
    She-Leveque structure function exponents, energy cascade direction,
    intermittency flatness growth, and the exact Kolmogorov 4/5 law.

    What's genuinely non-redundant vs existing geometries:
    - MultifractalGeometry measures generic multifractal spectrum width;
      this tests fit to a *specific physical model* (She-Leveque).
    - SpectralGeometry fits generic power-law slopes;
      this tests cascade *direction* (forward vs inverse).
    - Flatness slope and 4/5 residual have no existing analogues.
    """

    def __init__(self, input_scale: float = 255.0, n_scales: int = 8):
        self.input_scale = input_scale
        self.n_scales = n_scales

    @property
    def name(self) -> str:
        return "Navier-Stokes"

    @property
    def dimension(self) -> str:
        return "inertial range"

    @property
    def description(self) -> str:
        return (
            "Tests turbulence scaling predictions: She-Leveque vs K41 structure "
            "function exponents (ESS), signed energy cascade direction, "
            "intermittency flatness growth rate, and the exact 4/5 third-order law."
        )

    @property
    def view(self) -> str:
        return "dynamical"

    @property
    def detects(self) -> str:
        return "Turbulence cascade, intermittency, K41/She-Leveque scaling"

    def embed(self, data: np.ndarray) -> np.ndarray:
        data = self.validate_data(data)
        return self._normalize_to_unit(data, self.input_scale)

    def compute_metrics(self, data: np.ndarray) -> GeometryResult:
        u = self.embed(data)
        n = len(u)

        fallback_metrics = {
            'sl_fit_quality': 0.0,
            'cascade_asymmetry': 0.0,
            'flatness_slope': 0.0,
            'ess_quality': 0.0,
        }
        if n < 128:
            return GeometryResult(geometry_name=self.name, metrics=fallback_metrics)

        # Dyadic scales for structure functions
        max_exp = min(self.n_scales, int(np.log2(n // 4)))
        if max_exp < 3:
            return GeometryResult(geometry_name=self.name, metrics=fallback_metrics)
        scales = 2 ** np.arange(1, max_exp + 1)
        log_scales = np.log(scales.astype(float))

        # Structure functions S_q(r) = <|δu|^q> for orders 1-6
        orders = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        S = np.zeros((len(orders), len(scales)))
        S3_signed = np.zeros(len(scales))

        for j, r in enumerate(scales):
            increments = u[r:] - u[:-r]
            abs_inc = np.abs(increments)
            for i, q in enumerate(orders):
                S[i, j] = np.mean(abs_inc ** q)
            S3_signed[j] = np.mean(increments ** 3)

        # Fit exponents ζ_q via log-log regression: S_q(r) ~ r^ζ_q
        zeta = np.zeros(len(orders))
        for i in range(len(orders)):
            log_S = np.log(np.maximum(S[i], 1e-30))
            coeffs = np.polyfit(log_scales, log_S, 1)
            zeta[i] = coeffs[0]

        # --- sl_fit_quality: She-Leveque vs K41 in ESS form ---
        # K41: ζ_n = n/3.  SL: ζ_n = n/9 + 2(1 - (2/3)^(n/3))
        # ESS normalizes by ζ_3 to remove inertial-range dependence.
        zeta_sl = orders / 9.0 + 2.0 * (1.0 - (2.0 / 3.0) ** (orders / 3.0))
        zeta_k41 = orders / 3.0
        zeta3 = zeta[2]  # observed ζ_3
        if abs(zeta3) > 1e-10:
            ess_obs = zeta / zeta3
            ess_sl = zeta_sl / zeta_sl[2]
            ess_k41 = zeta_k41 / zeta_k41[2]
            res_k41 = np.sum((ess_obs - ess_k41) ** 2)
            res_sl = np.sum((ess_obs - ess_sl) ** 2)
            sl_fit_quality = (res_k41 - res_sl) / (res_k41 + res_sl + 1e-10)
        else:
            sl_fit_quality = 0.0

        # --- cascade_asymmetry: signed third-order skewness ---
        # <(δu)³> / S_2^{3/2} — negative means forward cascade (large→small)
        # Use median across scales (robust to outlier at extreme scale),
        # then tanh compression (raw skewness can be arbitrarily large for trending data).
        S2_32 = S[1] ** 1.5
        cascade_raw = S3_signed / np.maximum(S2_32, 1e-30)
        cascade_asymmetry = float(np.tanh(np.median(cascade_raw)))

        # --- flatness_slope: intermittency growth rate ---
        # F(r) = S_4(r) / S_2(r)^2.  Gaussian: F=3 (flat). Turbulent: F grows as r→0.
        # Guard against degenerate flatness (constant F → meaningless slope).
        F = S[3] / np.maximum(S[1] ** 2, 1e-30)
        F_range = np.max(F) - np.min(F)
        if F_range < 1e-10 * (np.mean(F) + 1e-30):
            flatness_slope = 0.0
        else:
            log_F = np.log(np.maximum(F, 1e-30))
            coeffs_f = np.polyfit(log_scales, log_F, 1)
            flatness_slope = float(np.tanh(coeffs_f[0]))

        # --- ess_quality: Extended Self-Similarity improvement ---
        # ESS: plot log S_q vs log S_3 instead of vs log r.
        # For turbulent cascades, ESS dramatically improves linearity.
        # Metric = mean R² improvement across orders q != 3.
        log_S3 = np.log(np.maximum(S[2], 1e-30))  # S[2] = S_3
        r2_raw_sum = 0.0
        r2_ess_sum = 0.0
        n_orders = 0
        for i, q in enumerate(orders):
            if q == 3:
                continue
            log_Sq = np.log(np.maximum(S[i], 1e-30))
            # R² of log S_q vs log r
            coeffs_raw = np.polyfit(log_scales, log_Sq, 1)
            pred_raw = np.polyval(coeffs_raw, log_scales)
            ss_res_raw = np.sum((log_Sq - pred_raw) ** 2)
            ss_tot_raw = np.sum((log_Sq - np.mean(log_Sq)) ** 2)
            r2_raw = 1.0 - ss_res_raw / (ss_tot_raw + 1e-30) if ss_tot_raw > 1e-30 else 1.0
            # R² of log S_q vs log S_3 (ESS)
            # Guard: if S_3 is nearly constant across scales, ESS is undefined.
            if np.ptp(log_S3) < 1e-10:
                n_orders += 1
                r2_raw_sum += r2_raw
                r2_ess_sum += r2_raw  # no improvement possible
                continue
            coeffs_ess = np.polyfit(log_S3, log_Sq, 1)
            pred_ess = np.polyval(coeffs_ess, log_S3)
            ss_res_ess = np.sum((log_Sq - pred_ess) ** 2)
            r2_ess = 1.0 - ss_res_ess / (ss_tot_raw + 1e-30) if ss_tot_raw > 1e-30 else 1.0
            r2_raw_sum += r2_raw
            r2_ess_sum += r2_ess
            n_orders += 1
        if n_orders > 0:
            ess_quality = (r2_ess_sum - r2_raw_sum) / n_orders
        else:
            ess_quality = 0.0

        return GeometryResult(
            geometry_name=self.name,
            metrics={
                'sl_fit_quality': float(np.clip(sl_fit_quality, -1, 1)),
                'cascade_asymmetry': cascade_asymmetry,
                'flatness_slope': flatness_slope,
                'ess_quality': float(np.clip(ess_quality, -1, 1)),
            },
            raw_data={
                'zeta_observed': zeta.tolist(),
                'zeta_sl': zeta_sl.tolist(),
                'zeta_k41': zeta_k41.tolist(),
                'scales': scales.tolist(),
            }
        )


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

    # Module-level hash: computed once, invalidates all per-data caches
    # when any part of the framework changes.  (Previously per-class, but
    # that missed changes to module-level helper functions.)
    _module_hash: Optional[str] = None

    def __init__(self, cache_dir=None):
        self.geometries: List[ExoticGeometry] = []
        self.cache_dir = cache_dir
        self._metric_keys: Dict[str, List[str]] = {}  # geometry name → metric keys
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

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

    # Geometry tiers based on greedy forward selection over 93 atlas sources.
    # Each tier adds geometries that contribute the most unique variance to the
    # rank-normalized PCA space (cumulative R² on top-10 PCs).
    TIERS = {
        'quick': [
            # 24 metrics, ~8.6x speedup, 90% R²
            'Information',
            'H² × ℝ (Thurston)',
            'Higher-Order Statistics',
        ],
        'standard': [
            # 32 metrics, ~8.3x speedup, 95% R²
            'Information',
            'H² × ℝ (Thurston)',
            'Higher-Order Statistics',
            'Spectral',
        ],
        'full': [
            # ~47 metrics, ~3.5x speedup, 99% R²
            'Information',
            'H² × ℝ (Thurston)',
            'Higher-Order Statistics',
            'Spectral',
            'VisibilityGraph',
            'Sol (Thurston)',
        ],
        # 'complete' = add_all_geometries() — all 36 geometries, 205 metrics
    }

    def add_tier_geometries(self, tier: str = 'complete',
                            data_mode: str = 'bytes') -> 'GeometryAnalyzer':
        """Add geometries for a specific analysis tier.

        Tiers are ordered by coverage of the atlas's principal structure space:
          'quick'    — 3 geometries, ~24 metrics, 90% variance, ~8.6x speedup
          'standard' — 4 geometries, ~32 metrics, 95% variance, ~8.3x speedup
          'full'     — 7 geometries, ~54 metrics, 99% variance, ~3.5x speedup
          'complete' — all geometries, ~205 metrics, 100% variance (default)
        """
        if tier == 'complete':
            return self.add_all_geometries(data_mode=data_mode)

        tier_names = self.TIERS.get(tier)
        if tier_names is None:
            valid = list(self.TIERS.keys()) + ['complete']
            raise ValueError(f"Unknown tier '{tier}'. Use one of: {valid}")

        # Build the full set, then filter to tier
        self.add_all_geometries(data_mode=data_mode)
        self.geometries = [g for g in self.geometries if g.name in tier_names]
        return self

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
            # Lie / Coxeter root systems
            G2Geometry(),
            D4Geometry(),
            H3CoxeterGeometry(),
            H4CoxeterGeometry(),
            TorusGeometry(bins=16),
            HyperbolicGeometry(input_scale=s),
            MostowRigidityGeometry(input_scale=s),
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
            ZariskiGeometry(input_scale=s),
            CayleyGeometry(input_scale=s),
            SpectralGraphGeometry(input_scale=s),
            # Statistical
            WassersteinGeometry(input_scale=s),
            FisherGeometry(),
            PersistentHomologyGeometry(),
            # Physical
            LorentzianGeometry(input_scale=s),
            SymplecticGeometry(input_scale=s),
            SpiralGeometry(input_scale=s),
            # Aperiodic / quasicrystal
            PenroseGeometry(input_scale=s),
            AmmannBeenkerGeometry(input_scale=s),
            DodecagonalGeometry(input_scale=s),
            SeptagonalGeometry(input_scale=s),
            InflationGeometry(input_scale=s),
            # Fractal
            FractalMandelbrotGeometry(input_scale=s),
            FractalJuliaGeometry(input_scale=s),
            # Higher-order (3rd/4th order, independent from all above)
            HigherOrderGeometry(),
            # Local regularity (targets 7D blind spot)
            HolderRegularityGeometry(input_scale=s),
            PVariationGeometry(input_scale=s),
            MultiScaleWassersteinGeometry(input_scale=s),
            # Spectral / information / recurrence
            SpectralGeometry(input_scale=s),
            InformationGeometry(input_scale=s),
            RecurrenceGeometry(input_scale=s),
            # Topological / Power Law
            VisibilityGraphGeometry(),
            ZipfMandelbrotGeometry(n_bits=8),
            ZipfMandelbrotGeometry(n_bits=16, sliding=True),
            # Dynamical / Complexity
            MultifractalGeometry(input_scale=s),
            PredictabilityGeometry(input_scale=s),
            AttractorGeometry(input_scale=s),
            BoltzmannGeometry(input_scale=s),
            GottwaldMelbourneGeometry(input_scale=s),
            OrdinalPartitionGeometry(input_scale=s),
            NonstationarityGeometry(input_scale=s),
            NavierStokesGeometry(input_scale=s),
            KleinBottleGeometry(input_scale=s),
        ]
        # Cantor uses bit-extraction: only meaningful for integer/byte data
        if data_mode == 'bytes':
            self.geometries.insert(11, CantorGeometry())
        return self

    def geometry_catalog(self) -> list:
        """Export metadata for all loaded geometries."""
        return [g.metadata() for g in self.geometries]

    def view_lenses(self) -> dict:
        """Build VIEW_LENSES from geometry metadata. Returns {view_name: {geometries, ...}}."""
        from collections import defaultdict
        by_view = defaultdict(list)
        for g in self.geometries:
            by_view[g.view].append(g.name)
        # Standard questions/detects per view
        VIEW_META = {
            "distributional": ("What does the value distribution look like?",
                               "Alphabet size, uniformity, skew, tail weight"),
            "topological": ("What is the shape of the data's phase space?",
                            "Holes, fractal dimension, self-similarity, curvature"),
            "dynamical": ("How does the signal evolve in time?",
                          "Chaos, predictability, Lyapunov divergence, memory"),
            "symmetry": ("What algebraic structure underlies the data?",
                         "Lattice alignment, phase coupling, hyperbolic splitting"),
            "scale": ("How does structure change across scales?",
                      "Roughness, regularity, visibility connectivity"),
            "quasicrystal": ("Does the data have aperiodic order?",
                             "Aperiodic tiling alignment, forbidden symmetries"),
        }
        result = {}
        for view_name, geo_names in sorted(by_view.items()):
            if view_name == "other":
                continue  # 2D-only geometries, not relevant for 1D atlas
            q, d = VIEW_META.get(view_name, ("", ""))
            result[view_name.title()] = {
                "question": q,
                "detects": d,
                "geometries": geo_names,
            }
        # Cross-cutting ordinal view: geometries whose metrics survive re-encoding
        ordinal_geos = [g.name for g in self.geometries if g.encoding_invariant]
        if ordinal_geos:
            result["Ordinal"] = {
                "question": "What structure survives re-encoding?",
                "detects": "Rank-order patterns, symbolic dynamics, local roughness",
                "geometries": ordinal_geos,
            }
        return result

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

        # Pre-compute data hash once if caching is enabled
        if self.cache_dir is not None:
            data_hash = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()

        for geom in self.geometries:
            try:
                # Check cache — key includes module-level hash so that
                # changes to helper functions (not just class methods)
                # automatically invalidate entries.
                if self.cache_dir is not None:
                    if GeometryAnalyzer._module_hash is None:
                        _mod_file = os.path.abspath(inspect.getfile(type(geom)))
                        try:
                            with open(_mod_file, 'rb') as _f:
                                GeometryAnalyzer._module_hash = \
                                    hashlib.sha256(_f.read()).hexdigest()[:16]
                        except OSError:
                            GeometryAnalyzer._module_hash = ""
                    code_hash = GeometryAnalyzer._module_hash
                    cache_key = hashlib.sha256(
                        (geom.name + "|" + data_hash + "|" + code_hash).encode()
                    ).hexdigest()
                    cache_path = os.path.join(self.cache_dir, cache_key + ".pkl")
                    if os.path.exists(cache_path):
                        try:
                            with open(cache_path, "rb") as f:
                                geom_result = pickle.load(f)
                            result.results.append(geom_result)
                            continue
                        except Exception:
                            # Corrupted cache file — delete and recompute
                            os.remove(cache_path)

                geom_result = geom.compute_metrics(data)

                # Store in cache (atomic write via temp + rename)
                if self.cache_dir is not None:
                    import tempfile
                    fd, tmp_path = tempfile.mkstemp(
                        dir=self.cache_dir, suffix=".pkl.tmp"
                    )
                    try:
                        with os.fdopen(fd, "wb") as f:
                            pickle.dump(geom_result, f, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(tmp_path, cache_path)
                    except Exception:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                result.results.append(geom_result)
            except Exception as e:
                import traceback
                warnings.warn(
                    f"Geometry {geom.name} failed: {e}\n"
                    f"{traceback.format_exc()}",
                    stacklevel=2,
                )

        return result

    def clear_cache(self, geometry_name=None):
        """Clear cached results.

        Parameters
        ----------
        geometry_name : str, optional
            If given, only clear entries for this geometry.
            If None, wipe the entire cache directory.
        """
        if self.cache_dir is None or not os.path.exists(self.cache_dir):
            return
        if geometry_name is None:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            # Must recompute to find matching keys — just remove all
            # (geometry name is hashed into the key, no way to selectively
            # identify without metadata). Pragmatic: wipe all.
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)

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
                warnings.warn(
                    f"Metric count mismatch for signature '{sig['name']}': "
                    f"got {len(input_vec)} metrics, expected {len(means)}. "
                    f"Truncating to {min(len(input_vec), len(means))}. "
                    f"Rebuild signatures after metric changes.",
                    stacklevel=2,
                )
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

    e8_diversity = e8_result.metrics["diversity_ratio"]
    coverage = torus_result.metrics["coverage"]

    # Thresholds from validation
    # diversity_ratio is fraction of effective directions used (0-1 range)
    issues = []
    if e8_diversity < 0.15:
        issues.append(f"Low E8 diversity ({e8_diversity:.2f} < 0.15): algebraic structure likely")
    if e8_diversity < 0.10:
        issues.append(f"Very low E8 diversity ({e8_diversity:.2f} < 0.10): definitely constrained")

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
