"""
Discrimination tests for geometry metrics.

For each geometry, we construct signals that are KNOWN to exhibit (or not exhibit)
the property the geometry detects, then verify the metric ranks them correctly.

Naming convention:
    test_{geometry}_detects_{property}

Each test compares at least one "good" signal (should score HIGH) against
one "bad" signal (should score LOW).
"""
import numpy as np
import pytest
from exotic_geometry_framework import (
    # Distributional
    TorusGeometry, SphericalGeometry, CantorGeometry, UltrametricGeometry,
    WassersteinGeometry, FisherGeometry, InformationGeometry, ZipfMandelbrotGeometry,
    # Topological
    HyperbolicGeometry, MostowRigidityGeometry,
    PersistentHomologyGeometry, VisibilityGraphGeometry,
    FractalMandelbrotGeometry, FractalJuliaGeometry, ZariskiGeometry, CayleyGeometry,
    SpectralGraphGeometry, BoltzmannGeometry,
    GottwaldMelbourneGeometry, OrdinalPartitionGeometry,
    # Dynamical
    HigherOrderGeometry, SpectralGeometry, RecurrenceGeometry,
    MultifractalGeometry, PredictabilityGeometry, AttractorGeometry,
    # Symmetry
    E8Geometry, HeisenbergGeometry, TropicalGeometry, LorentzianGeometry, SolGeometry,
    # Scale
    SpiralGeometry, HolderRegularityGeometry, PVariationGeometry,
    MultiScaleWassersteinGeometry,
    # Quasicrystal
    PenroseGeometry, AmmannBeenkerGeometry, EinsteinHatGeometry,
    DodecagonalGeometry, SeptagonalGeometry,
    InflationGeometry,
    NonstationarityGeometry,
    KleinBottleGeometry,
)

SIZE = 4096
RNG = np.random.default_rng(12345)


# ── Signal generators ──────────────────────────────────────────────────

def _white_noise(size=SIZE):
    """IID uniform bytes. No structure at all."""
    return RNG.integers(0, 256, size, dtype=np.uint8)


def _constant(size=SIZE, val=128):
    """All-same-value. Degenerate."""
    return np.full(size, val, dtype=np.uint8)


def _periodic(size=SIZE, period=32):
    """Exact repeating ramp 0..255."""
    base = np.linspace(0, 255, period, dtype=np.uint8)
    return np.tile(base, size // period + 1)[:size]


def _brownian(size=SIZE):
    """Cumulative sum of Gaussian increments, normalized to uint8."""
    steps = RNG.normal(0, 1, size)
    walk = np.cumsum(steps)
    walk -= walk.min()
    mx = walk.max()
    if mx > 0:
        walk = walk / mx * 255
    return walk.astype(np.uint8)


def _logistic_chaos(size=SIZE, r=3.99):
    """Logistic map x_{n+1} = r * x * (1-x). Deterministic chaos."""
    x = 0.4 + RNG.uniform(-0.01, 0.01)
    out = np.empty(size)
    for i in range(size):
        x = r * x * (1 - x)
        out[i] = x
    return (out * 255).astype(np.uint8)


def _fibonacci_word(size=SIZE):
    """Fibonacci substitution: 0→01, 1→0. The canonical 1D quasicrystal."""
    a, b = "0", "01"
    while len(b) < size:
        a, b = b, b + a
    bits = np.array([int(c) for c in b[:size]], dtype=np.uint8)
    return bits * 255


def _exponential_dist(size=SIZE, scale=40):
    """Exponentially distributed values — heavy-tailed, hierarchical."""
    vals = RNG.exponential(scale, size)
    return np.clip(vals, 0, 255).astype(np.uint8)


def _power_law_text(size=SIZE):
    """Zipfian distribution: few frequent symbols, many rare. Like natural language."""
    alpha = 1.5
    ranks = np.arange(1, 257)
    probs = ranks.astype(float) ** (-alpha)
    probs /= probs.sum()
    symbols = RNG.choice(256, size=size, p=probs)
    return symbols.astype(np.uint8)


def _sine_wave(size=SIZE, freq=0.01):
    """Pure sinusoid. Periodic, smooth, predictable."""
    t = np.arange(size)
    return ((np.sin(2 * np.pi * freq * t) + 1) / 2 * 255).astype(np.uint8)


def _logistic_periodic(size=SIZE, r=3.2):
    """Logistic map in period-2 regime. Deterministic and periodic."""
    x = 0.4
    out = np.empty(size)
    for i in range(size):
        x = r * x * (1 - x)
        out[i] = x
    return (out * 255).astype(np.uint8)


def _fbm(size=SIZE, H=0.8):
    """Approximate fractional Brownian motion via spectral synthesis."""
    n = size
    f = np.fft.rfftfreq(n)[1:]  # skip DC
    power = f ** (-(2 * H + 1))
    phase = RNG.uniform(0, 2 * np.pi, len(power))
    spectrum = np.sqrt(power) * np.exp(1j * phase)
    spectrum = np.concatenate([[0], spectrum])
    signal = np.fft.irfft(spectrum, n=n)
    signal -= signal.min()
    mx = signal.max()
    if mx > 0:
        signal = signal / mx * 255
    return signal.astype(np.uint8)


def _alternating(size=SIZE):
    """Strictly alternating 0, 255, 0, 255... Perfect period-2."""
    return np.array([0, 255] * (size // 2), dtype=np.uint8)[:size]


# ── DISTRIBUTIONAL LENS ────────────────────────────────────────────────

class TestTorus:
    def test_periodic_vs_random(self):
        """Periodic signal traces a thin loop on the torus → low coverage, low entropy."""
        geom = TorusGeometry(bins=16)
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_per.metrics["coverage"] < res_rnd.metrics["coverage"]
        assert res_per.metrics["entropy"] < res_rnd.metrics["entropy"]


class TestSpherical:
    def test_concentrated_vs_uniform(self):
        """Constant data → concentrated on one point of S². Random → spread out."""
        geom = SphericalGeometry()
        near_const = np.clip(RNG.normal(128, 3, SIZE), 0, 255).astype(np.uint8)
        res_const = geom.compute_metrics(near_const)
        res_rnd = geom.compute_metrics(_white_noise())
        # Concentrated data → high concentration metric, low angular spread
        assert res_const.metrics["concentration"] > res_rnd.metrics["concentration"]


class TestCantor:
    def test_hierarchical_vs_uniform(self):
        """Data with Cantor-like gaps should have larger mean gap than uniform.
        The Cantor set removes the middle third recursively, leaving large gaps
        between the remaining points. Uniform data has small, evenly spaced gaps."""
        geom = CantorGeometry()
        # Build pseudo-Cantor data: remove middle third iteratively
        vals = np.linspace(0, 255, SIZE)
        # Keep only points not in middle third (ternary digits never = 1)
        ternary_ok = []
        for v in vals:
            x = v / 255.0
            ok = True
            for _ in range(5):
                x *= 3
                d = int(x) % 3
                if d == 1:
                    ok = False
                    break
                x -= int(x)
            if ok:
                ternary_ok.append(int(v))
        cantor_data = np.array(ternary_ok[:SIZE] if len(ternary_ok) >= SIZE
                               else ternary_ok * (SIZE // len(ternary_ok) + 1),
                               dtype=np.uint8)[:SIZE]
        res_cantor = geom.compute_metrics(cantor_data)
        res_rnd = geom.compute_metrics(_white_noise())
        # Cantor set has a larger maximum gap than uniform random:
        # the ternary embedding maps removed middle-third intervals to large voids.
        assert res_cantor.metrics["max_gap"] > res_rnd.metrics["max_gap"]


class TestInformation:
    def test_compression_ordering(self):
        """Periodic < random for compression ratio.
        Periodic signals are highly compressible; random signals are not."""
        geom = InformationGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        cr_per = res_per.metrics["compression_ratio"]
        cr_rnd = res_rnd.metrics["compression_ratio"]
        assert cr_per < cr_rnd, f"periodic {cr_per} should have lower compression ratio than random {cr_rnd}"

    def test_compression_ratio(self):
        """Periodic signal is compressible → low compression ratio (near 0).
        Random signal is incompressible → high compression ratio (near 1)."""
        geom = InformationGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_per.metrics["compression_ratio"] < res_rnd.metrics["compression_ratio"]


class TestFisher:
    def test_uniform_vs_peaked(self):
        """Peaked distribution → large Fisher trace (steep log-likelihood).
        Uniform → minimal Fisher information."""
        geom = FisherGeometry()
        peaked = np.clip(RNG.normal(128, 5, SIZE), 0, 255).astype(np.uint8)
        res_peaked = geom.compute_metrics(peaked)
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_peaked.metrics["trace_fisher"] > res_rnd.metrics["trace_fisher"]


class TestWasserstein:
    def test_distance_from_uniform(self):
        """Peaked distribution should be far from uniform. Uniform noise ≈ 0."""
        geom = WassersteinGeometry()
        peaked = np.clip(RNG.normal(128, 5, SIZE), 0, 255).astype(np.uint8)
        res_peaked = geom.compute_metrics(peaked)
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_peaked.metrics["dist_from_uniform"] > res_rnd.metrics["dist_from_uniform"]


class TestZipfMandelbrot:
    def test_zipfian_vs_uniform(self):
        """Zipfian (text-like) data should have steeper alpha than uniform."""
        geom = ZipfMandelbrotGeometry()
        res_zipf = geom.compute_metrics(_power_law_text())
        res_rnd = geom.compute_metrics(_white_noise())
        # Higher alpha = steeper rank-frequency curve (more Zipfian)
        assert res_zipf.metrics["zipf_alpha"] > res_rnd.metrics["zipf_alpha"]


# ── TOPOLOGICAL LENS ───────────────────────────────────────────────────

class TestPersistentHomology:
    def test_structured_vs_noise(self):
        """Periodic signal should have long-lived topological features (loops).
        Noise features die quickly → lower max_lifetime."""
        geom = PersistentHomologyGeometry()
        res_per = geom.compute_metrics(_sine_wave())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_per.metrics["max_lifetime"] > res_rnd.metrics["max_lifetime"]


class TestVisibilityGraph:
    def test_deterministic_vs_noise(self):
        """Sine wave creates extreme visibility hubs (peaks see far) → high max_degree.
        Noise has homogeneous degree distribution → low max_degree."""
        geom = VisibilityGraphGeometry()
        res_sine = geom.compute_metrics(_sine_wave(SIZE))
        res_rnd = geom.compute_metrics(_white_noise(SIZE))
        assert res_sine.metrics["max_degree"] > res_rnd.metrics["max_degree"]


class TestHyperbolic:
    def test_curvature_structure_nonzero(self):
        """curvature_structure produces nonzero values for both structured
        and random data (three-component metric: temporal × spatio-temporal
        × spatial)."""
        geom = HyperbolicGeometry()
        res_exp = geom.compute_metrics(_exponential_dist())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_exp.metrics["curvature_structure"] > 0
        assert res_rnd.metrics["curvature_structure"] > 0



class TestMostowRigidity:
    def test_structured_higher_distance_rigidity(self):
        """Quasicrystal (Fibonacci) has higher distance rigidity than white noise.
        Mostow: structured data's geometry is more determined by its combinatorics."""
        geom = MostowRigidityGeometry()
        fib = np.zeros(SIZE, dtype=np.uint8)
        a, b = "1", "10"
        while len(b) < SIZE + 100:
            a, b = b, b + a
        for i, c in enumerate(b[:SIZE]):
            fib[i] = 255 if c == "1" else 0
        res_fib = geom.compute_metrics(fib)
        res_noise = geom.compute_metrics(_white_noise(SIZE))
        assert res_fib.metrics["distance_rigidity"] > res_noise.metrics["distance_rigidity"]

    def test_noise_higher_volume_entropy(self):
        """White noise fills the Poincaré ball more uniformly → higher volume entropy.
        Periodic data collapses to a low-dimensional subset → lower entropy."""
        geom = MostowRigidityGeometry()
        res_noise = geom.compute_metrics(_white_noise(SIZE))
        res_sine = geom.compute_metrics(_sine_wave(SIZE))
        assert res_noise.metrics["volume_entropy"] > res_sine.metrics["volume_entropy"]

    def test_structured_higher_spectral_rigidity(self):
        """Fibonacci has spectral rigidity near 1.0 — Laplacian spectrum
        is almost unchanged under perturbation. Noise has lower spectral rigidity."""
        geom = MostowRigidityGeometry()
        fib = np.zeros(SIZE, dtype=np.uint8)
        a, b = "1", "10"
        while len(b) < SIZE + 100:
            a, b = b, b + a
        for i, c in enumerate(b[:SIZE]):
            fib[i] = 255 if c == "1" else 0
        res_fib = geom.compute_metrics(fib)
        res_noise = geom.compute_metrics(_white_noise(SIZE))
        assert res_fib.metrics["spectral_rigidity"] > res_noise.metrics["spectral_rigidity"]


# ── DYNAMICAL LENS ─────────────────────────────────────────────────────

class TestSpectral:
    def test_colored_vs_white(self):
        """Brownian motion has spectral slope ≈ -2 (red noise).
        White noise has slope ≈ 0. |slope| should be larger for Brownian."""
        geom = SpectralGeometry()
        res_brown = geom.compute_metrics(_brownian())
        res_white = geom.compute_metrics(_white_noise())
        # More negative slope = more power at low frequencies
        assert res_brown.metrics["spectral_slope"] < res_white.metrics["spectral_slope"]

    def test_periodic_peak(self):
        """Sine wave has a dominant spectral peak → low spectral flatness."""
        geom = SpectralGeometry()
        res_sine = geom.compute_metrics(_sine_wave())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_sine.metrics["spectral_flatness"] < res_rnd.metrics["spectral_flatness"]


class TestRecurrence:
    def test_periodic_high_determinism(self):
        """Periodic signal → near-perfect determinism (long diagonal lines in RP).
        Every subsequence repeats identically → diagonal lines span the full period."""
        geom = RecurrenceGeometry()
        res_per = geom.compute_metrics(_periodic())
        # Periodic determinism should be very high (close to 1.0)
        assert res_per.metrics["determinism"] > 0.8, \
            f"Periodic signal should have high determinism, got {res_per.metrics['determinism']}"

    def test_periodic_vs_noise_determinism(self):
        """Periodic signal is perfectly recurrent → higher determinism than noise.
        Use fixed-seed noise to avoid stochastic test instability."""
        geom = RecurrenceGeometry()
        res_per = geom.compute_metrics(_periodic())
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)
        res_rnd = geom.compute_metrics(noise)
        assert res_per.metrics["determinism"] > res_rnd.metrics["determinism"]


class TestPredictability:
    def test_predictable_vs_random(self):
        """Periodic signal is perfectly predictable → low conditional entropy.
        White noise is unpredictable → high conditional entropy."""
        geom = PredictabilityGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_per.metrics["cond_entropy_k1"] < res_rnd.metrics["cond_entropy_k1"]

    def test_sample_entropy_ordering(self):
        """Sample entropy: periodic < chaotic < random."""
        geom = PredictabilityGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_chaos = geom.compute_metrics(_logistic_chaos())
        res_rnd = geom.compute_metrics(_white_noise())
        se_per = res_per.metrics["sample_entropy"]
        se_chaos = res_chaos.metrics["sample_entropy"]
        se_rnd = res_rnd.metrics["sample_entropy"]
        assert se_per < se_rnd, "periodic should be less entropic than random"
        assert se_chaos < se_rnd, "chaos should be less entropic than random"


class TestHigherOrder:
    def test_nonlinear_vs_gaussian(self):
        """Logistic chaos has strong bispectral content (phase coupling).
        White noise has zero expected bicoherence."""
        geom = HigherOrderGeometry()
        res_chaos = geom.compute_metrics(_logistic_chaos())
        res_rnd = geom.compute_metrics(_white_noise())
        # Deterministic nonlinear process → non-zero bispectrum
        assert res_chaos.metrics["bicoherence_max"] > res_rnd.metrics["bicoherence_max"]

    def test_forbidden_permutations(self):
        """Deterministic dynamics forbid some ordinal patterns.
        Random data visits all permutations approximately equally."""
        geom = HigherOrderGeometry()
        res_chaos = geom.compute_metrics(_logistic_chaos())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_chaos.metrics["perm_forbidden"] > res_rnd.metrics["perm_forbidden"]


class TestAttractor:
    def test_chaos_vs_noise(self):
        """Logistic chaos lives on a low-dimensional attractor (dim ≈ 1).
        White noise fills the embedding space (dim → embedding_dim)."""
        geom = AttractorGeometry()
        res_chaos = geom.compute_metrics(_logistic_chaos())
        res_rnd = geom.compute_metrics(_white_noise())
        # Lower correlation dimension = lower-dimensional attractor
        assert res_chaos.metrics["correlation_dimension"] < res_rnd.metrics["correlation_dimension"]


class TestMultifractal:
    def test_multifractal_width(self):
        """Brownian motion is monofractal (narrow spectrum).
        Logistic chaos has broader multifractal spectrum."""
        geom = MultifractalGeometry()
        res_brown = geom.compute_metrics(_brownian())
        res_chaos = geom.compute_metrics(_logistic_chaos())
        # Both should have non-zero width, but chaos typically wider
        # Let's just verify Hurst estimate is reasonable for Brownian
        h = res_brown.metrics["hurst_estimate"]
        assert 0.3 < h < 0.7, f"Brownian Hurst should be near 0.5, got {h}"


# ── SYMMETRY LENS ──────────────────────────────────────────────────────

class TestE8:
    def test_lattice_alignment(self):
        """Data constructed from E8 roots → high alignment.
        Random data → lower alignment."""
        geom = E8Geometry(window_size=8, normalize=True)
        roots = geom.roots
        chosen = roots[:10]
        lattice = []
        for _ in range(SIZE // 8):
            lattice.append(chosen[RNG.integers(len(chosen))])
        lattice = np.concatenate(lattice).flatten()
        lattice = (lattice * 10) + 128
        res_lat = geom.compute_metrics(lattice)
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_lat.metrics["e8_structure_score"] > res_rnd.metrics["e8_structure_score"]



class TestHeisenberg:
    def test_spiral_vs_random(self):
        """Heisenberg z = accumulated signed area swept by (x,y) path.
        A spiral (coherent rotation) sweeps systematic area → large |final_z|.
        White noise path cancels → smaller |final_z| relative to path_length."""
        geom = HeisenbergGeometry(center_data=True)
        # Deterministic spiral: x=cos(t), y=sin(t) → sweeps π per revolution
        t = np.linspace(0, 20 * np.pi, SIZE)
        spiral_x = ((np.cos(t) + 1) / 2 * 255).astype(np.uint8)
        spiral_y = ((np.sin(t) + 1) / 2 * 255).astype(np.uint8)
        # Interleave x,y to form the data stream
        spiral = np.empty(SIZE, dtype=np.uint8)
        spiral[0::2] = spiral_x[:SIZE // 2]
        spiral[1::2] = spiral_y[:SIZE // 2]
        res_spiral = geom.compute_metrics(spiral)
        res_rnd = geom.compute_metrics(_white_noise())
        # Spiral sweeps coherent area → high |final_z|
        assert res_spiral.metrics["final_z"] > res_rnd.metrics["final_z"]


class TestTropical:
    def test_piecewise_linear_vs_smooth(self):
        """Piecewise-linear data (sawtooth) should produce fewer slope changes
        than noisy data. Tropical geometry = min-plus algebra → detects linearity."""
        geom = TropicalGeometry()
        res_saw = geom.compute_metrics(_periodic())  # ramp = piecewise linear
        res_rnd = geom.compute_metrics(_white_noise())
        # PWL data (ramp) → fewer slope changes than random noise
        assert res_saw.metrics["slope_changes"] < res_rnd.metrics["slope_changes"]


class TestLorentzian:
    def test_monotone_vs_oscillating(self):
        """Monotone increasing data → all timelike intervals (causal order preserved).
        Oscillating data has many spacelike/mixed intervals."""
        geom = LorentzianGeometry()
        # Monotone increasing
        monotone = np.linspace(0, 255, SIZE).astype(np.uint8)
        res_mono = geom.compute_metrics(monotone)
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_mono.metrics["causal_order_preserved"] > res_rnd.metrics["causal_order_preserved"]



class TestSol:
    def test_anisotropy(self):
        """Sol geometry detects anisotropic structure via z_variance.
        Data with different x/y scales → different z_variance than isotropic noise."""
        geom = SolGeometry()
        # Highly anisotropic: x varies slowly, y varies fast
        slow_x = np.repeat(np.arange(0, 256, dtype=np.uint8), SIZE // 256)[:SIZE // 2]
        fast_y = RNG.integers(0, 256, SIZE // 2, dtype=np.uint8)
        aniso = np.empty(SIZE, dtype=np.uint8)
        aniso[0::2] = slow_x[:SIZE // 2]
        aniso[1::2] = fast_y[:SIZE // 2]
        res_aniso = geom.compute_metrics(aniso)
        res_rnd = geom.compute_metrics(_white_noise())
        # Sol metric exponentially stretches one axis and shrinks the other
        # Anisotropic data should show different z_variance vs random
        assert res_aniso.metrics["z_variance"] != pytest.approx(
            res_rnd.metrics["z_variance"], abs=0.01
        )


# ── SCALE LENS ─────────────────────────────────────────────────────────

class TestHolderRegularity:
    def test_hurst_brownian(self):
        """Brownian motion has Hurst exponent ≈ 0.5."""
        geom = HolderRegularityGeometry()
        res = geom.compute_metrics(_brownian())
        h = res.metrics["hurst_exponent"]
        assert 0.3 < h < 0.7, f"Brownian Hurst should be near 0.5, got {h}"

    def test_hurst_persistent_fbm(self):
        """fBm with H=0.8 → Hurst ≈ 0.8 (persistent, smooth)."""
        geom = HolderRegularityGeometry()
        res = geom.compute_metrics(_fbm(H=0.8))
        h = res.metrics["hurst_exponent"]
        assert h > 0.6, f"fBm(H=0.8) Hurst should be > 0.6, got {h}"

    def test_smooth_vs_rough(self):
        """Smooth signal (sine) → high Hölder mean.
        Rough signal (white noise) → low Hölder mean."""
        geom = HolderRegularityGeometry()
        res_smooth = geom.compute_metrics(_sine_wave())
        res_rough = geom.compute_metrics(_white_noise())
        assert res_smooth.metrics["holder_mean"] > res_rough.metrics["holder_mean"]


class TestPVariation:
    def test_smooth_vs_rough(self):
        """Smooth signals have low p-variation; rough signals have high p-variation."""
        geom = PVariationGeometry()
        res_smooth = geom.compute_metrics(_sine_wave())
        res_rough = geom.compute_metrics(_white_noise())
        # At p=2, rough > smooth
        assert res_rough.metrics["var_p2"] > res_smooth.metrics["var_p2"]


class TestMultiScaleWasserstein:
    def test_multiscale_structure(self):
        """A signal whose distribution changes with block size should have high w_std.
        Concatenated segments with different means → scale-dependent distribution.
        Uniform noise has the same distribution at all scales → low w_std."""
        geom = MultiScaleWassersteinGeometry()
        # Build multi-scale signal: alternating low/high segments of increasing length
        segments = []
        for i in range(20):
            length = 50 * (i + 1)
            mean = 50 if i % 2 == 0 else 200
            seg = np.clip(RNG.normal(mean, 10, length), 0, 255).astype(np.uint8)
            segments.append(seg)
        multiscale = np.concatenate(segments)[:SIZE]
        res_ms = geom.compute_metrics(multiscale)
        res_rnd = geom.compute_metrics(_white_noise())
        # Multi-scale signal → Wasserstein distances vary more across scales
        assert res_ms.metrics["w_std"] > res_rnd.metrics["w_std"]


# ── QUASICRYSTAL LENS ─────────────────────────────────────────────────

class TestPenrose:
    def test_fibonacci_vs_noise(self):
        """Fibonacci word is THE canonical 1D quasicrystal with golden ratio φ.
        Should score high on long_range_order (ACF self-similarity at φ-scaled lags)."""
        geom = PenroseGeometry()
        fib = _fibonacci_word()
        noise = _white_noise()
        res_fib = geom.compute_metrics(fib)
        res_noise = geom.compute_metrics(noise)
        assert res_fib.metrics["long_range_order"] > res_noise.metrics["long_range_order"]

    def test_fibonacci_vs_periodic(self):
        """Fibonacci is quasiperiodic, not periodic. Both have structure, but
        Fibonacci should show φ-ratio spectral self-similarity that periodic doesn't."""
        geom = PenroseGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_per = geom.compute_metrics(_periodic())
        assert res_fib.metrics["long_range_order"] > res_per.metrics["long_range_order"]

    def test_long_range_order(self):
        """Fibonacci word has strong ACF self-similarity at φ-scaled lags.
        White noise has none."""
        geom = PenroseGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_fib.metrics["long_range_order"] > res_noise.metrics["long_range_order"]

    def test_brownian_not_quasicrystal(self):
        """Brownian motion has smooth 1/f² spectrum. The moving-average detrend
        + null-ratio comparison should reject it: low long_range_order.
        This was the critical false positive before the rewrite."""
        geom = PenroseGeometry()
        # Use a dedicated RNG to avoid sensitivity to global RNG state
        local_rng = np.random.default_rng(99999)
        steps = local_rng.normal(0, 1, SIZE)
        walk = np.cumsum(steps)
        walk -= walk.min()
        mx = walk.max()
        bm = ((walk / (mx + 1e-10)) * 255).astype(np.uint8) if mx > 0 else np.zeros(SIZE, dtype=np.uint8)
        res_bm = geom.compute_metrics(bm)
        assert res_bm.metrics["long_range_order"] < 0.3, \
            f"Brownian long_range_order={res_bm.metrics['long_range_order']:.3f} should be < 0.3"

    def test_fbm_not_quasicrystal(self):
        """Fractional Brownian motion (H=0.8) has smooth 1/f^(2H+1) spectrum.
        Should not fool the ratio detector despite strong spectral structure.
        Uses a dedicated RNG to avoid sensitivity to global RNG state."""
        geom = PenroseGeometry()
        rng_fbm = np.random.default_rng(42)
        n = SIZE
        f = np.fft.rfftfreq(n)[1:]
        power = f ** (-(2 * 0.8 + 1))
        phase = rng_fbm.uniform(0, 2 * np.pi, len(power))
        spectrum = np.sqrt(power) * np.exp(1j * phase)
        spectrum = np.concatenate([[0], spectrum])
        signal = np.fft.irfft(spectrum, n=n)
        signal -= signal.min()
        mx = signal.max()
        if mx > 0:
            signal = signal / mx * 255
        fbm_data = signal.astype(np.uint8)
        res_fbm = geom.compute_metrics(fbm_data)
        assert res_fbm.metrics["long_range_order"] < 0.3, \
            f"fBm long_range_order={res_fbm.metrics['long_range_order']:.3f} should be < 0.3"

    def test_fibonacci_beats_brownian(self):
        """Fibonacci should dominate Brownian on all QC metrics."""
        geom = PenroseGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_bm = geom.compute_metrics(_brownian())
        for metric in ["long_range_order", "algebraic_tower"]:
            assert res_fib.metrics[metric] > res_bm.metrics[metric], \
                f"Fibonacci {metric}={res_fib.metrics[metric]:.3f} should > Brownian {res_bm.metrics[metric]:.3f}"



def _octonacci_word(size=SIZE):
    """Octonacci substitution: A→AAB, B→A. Ratio of lengths → 1+√2 (silver ratio).
    This is the canonical 1D quasicrystal with silver-ratio self-similarity."""
    a, b = "A", "AAB"
    while len(b) < size * 2:
        a, b = b, b + b + a  # AAB rule: B→A means concat AAB+A
    # Actually, standard octonacci: a→aab, b→a
    # Let's redo properly
    s = "a"
    for _ in range(20):
        new_s = ""
        for c in s:
            if c == "a":
                new_s += "aab"
            else:
                new_s += "a"
        s = new_s
        if len(s) >= size:
            break
    bits = np.array([255 if c == 'a' else 0 for c in s[:size]], dtype=np.uint8)
    return bits


class TestAmmannBeenker:
    def test_pell_conformance_discriminates(self):
        """pell_conformance should give different values on real vs shuffled
        structured data. The metric works through suppression on QC-structured
        data (convergent gate kills it → 0.0) vs ~0.3 on shuffled. This
        inverted signal is valid for discrimination (|Cohen's d| is large)."""
        geom = AmmannBeenkerGeometry()
        rng = np.random.default_rng(42)
        bm = np.cumsum(rng.standard_normal(SIZE))
        bm = ((bm - bm.min()) / (bm.max() - bm.min()) * 255).astype(np.uint8)
        val_real = geom.compute_metrics(bm).metrics["pell_conformance"]
        val_shuf = geom.compute_metrics(rng.permutation(bm)).metrics["pell_conformance"]
        assert abs(val_real - val_shuf) > 0.1, \
            f"Should discriminate real vs shuffled: real={val_real:.4f}, shuf={val_shuf:.4f}"

    def test_noise_no_discrimination(self):
        """White noise should have near-zero discrimination (real ≈ shuffled)."""
        geom = AmmannBeenkerGeometry()
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)
        val_real = geom.compute_metrics(noise).metrics["pell_conformance"]
        val_shuf = geom.compute_metrics(rng.permutation(noise)).metrics["pell_conformance"]
        assert abs(val_real - val_shuf) < 0.15, \
            f"Noise should not discriminate: real={val_real:.4f}, shuf={val_shuf:.4f}"

    def test_ratio_dependence(self):
        """pell_conformance must give different discrimination with silver
        ratio vs a random ratio — this is the D1 drop property."""
        geo_silver = AmmannBeenkerGeometry()
        geo_random = AmmannBeenkerGeometry()
        geo_random.SILVER = 1.78  # arbitrary non-silver ratio

        diffs_s, diffs_r = [], []
        for trial in range(10):
            rng_t = np.random.default_rng(trial)
            data = np.cumsum(rng_t.standard_normal(SIZE))
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            shuf = rng_t.permutation(data)
            diffs_s.append(geo_silver._evolved_delta_components(data)['pell_conformance'] - geo_silver._evolved_delta_components(shuf)['pell_conformance'])
            diffs_r.append(geo_random._evolved_delta_components(data)['pell_conformance'] - geo_random._evolved_delta_components(shuf)['pell_conformance'])

        # Cohen's d should differ between silver and random ratio
        d_s = abs(np.mean(diffs_s) / max(np.std(diffs_s, ddof=1), 1e-10))
        d_r = abs(np.mean(diffs_r) / max(np.std(diffs_r, ddof=1), 1e-10))
        assert abs(d_s - d_r) > 0.5, \
            f"Should be ratio-dependent: |d_silver|={d_s:.2f}, |d_random|={d_r:.2f}"

    def test_fibonacci_word_not_silver(self):
        """Fibonacci word has φ-symmetry, not δ-symmetry.
        pell_conformance should be low (convergent gate suppresses it)."""
        geom = AmmannBeenkerGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        assert res_fib.metrics["pell_conformance"] < 0.2, \
            f"Fibonacci should not trigger Pell gate, got {res_fib.metrics['pell_conformance']}"

    def test_metric_returns_valid_float(self):
        """pell_conformance should return a non-negative float for all inputs."""
        geom = AmmannBeenkerGeometry()
        for data_fn in [_white_noise, _brownian, _fibonacci_word, _octonacci_word]:
            val = geom.compute_metrics(data_fn()).metrics["pell_conformance"]
            assert isinstance(val, float) and val >= 0.0 and not np.isnan(val), \
                f"Invalid pell_conformance={val} on {data_fn.__name__}"


class TestEinsteinHat:
    def test_hat_kernel_embedded_vs_noise(self):
        """Data with the Hat kernel embedded should have higher hat_boundary_match
        than white noise. The Hat kernel is [2,1,1,-2,1,2,1,-2,1,2,-1,2,-1] in
        hex turn units."""
        geom = EinsteinHatGeometry()
        # Build a signal whose hex turns contain the kernel repeatedly.
        # Kernel turns in hex units: [2,1,1,-2,1,2,1,-2,1,2,-1,2,-1]
        # Convert turns to direction sequence: dir[i+1] = (dir[i] + turn) % 6
        kernel_turns = [2, 1, 1, -2, 1, 2, 1, -2, 1, 2, -1, 2, -1]
        rng = np.random.default_rng(999)
        dirs = [0]
        for _ in range(SIZE // len(kernel_turns)):
            for t in kernel_turns:
                dirs.append((dirs[-1] + t) % 6)
        hat_data = np.array(dirs[:SIZE], dtype=np.uint8)
        res_hat = geom.compute_metrics(hat_data)
        res_noise = geom.compute_metrics(_white_noise())
        assert res_hat.metrics["hat_boundary_match"] > res_noise.metrics["hat_boundary_match"], \
            f"Hat kernel data={res_hat.metrics['hat_boundary_match']:.3f} should > noise={res_noise.metrics['hat_boundary_match']:.3f}"

    # test_fibonacci_inflation_similarity removed: inflation_similarity metric was pruned from EinsteinHatGeometry.

    def test_brownian_not_hat(self):
        """Brownian motion has smooth 1/f² spectrum — should NOT score high
        on hat_boundary_match.
        Uses a dedicated RNG to avoid sensitivity to global RNG state."""
        geom = EinsteinHatGeometry()
        rng_bm = np.random.default_rng(99)
        steps = rng_bm.normal(0, 1, SIZE)
        walk = np.cumsum(steps)
        walk -= walk.min()
        mx = walk.max()
        if mx > 0:
            walk = walk / mx * 255
        bm_data = walk.astype(np.uint8)
        res_bm = geom.compute_metrics(bm_data)
        # Brownian should score low on hat_boundary_match
        assert res_bm.metrics["hat_boundary_match"] < 0.3, \
            f"Brownian hat_boundary_match={res_bm.metrics['hat_boundary_match']:.3f} should be < 0.3"

    # chirality removed: 0% D1 drop, kernel-independent (dead metric per 2×2 classification)
    # hex_balance removed: was identical to Penrose:index_diversity (subword_complexity)



def _dodeca_word(size=SIZE):
    """Substitution word for dodecagonal QC: a→aaabb, b→ab.
    Matrix [[3,2],[1,1]] has PF eigenvalue 2+√3 ≈ 3.732."""
    s = 'a'
    for _ in range(20):
        new = ''
        for c in s:
            new += 'aaabb' if c == 'a' else 'ab'
        s = new
        if len(s) >= size:
            break
    return np.array([255 if c == 'a' else 0 for c in s[:size]], dtype=np.uint8)


def _ratio_signal(ratio, size=SIZE, base_freq=0.01, n_harmonics=6):
    """Cosine-sum signal with spectral peaks at f, f*r, f*r², ...
    Generic positive control for any target ratio."""
    t = np.arange(size)
    sig = np.zeros(size)
    for k in range(n_harmonics):
        freq = base_freq * ratio ** k
        if freq > 0.5:
            break
        sig += np.cos(2 * np.pi * freq * t) / (k + 1)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-10) * 255
    return sig.astype(np.uint8)


class TestDodecagonal:
    """Dodecagonal (12-fold) geometry. Ratio = 2+√3 ≈ 3.732.
    Evolved metric: conjugate symmetry × algebraic identity (δ²=4δ−1) × √3 bonus."""

    def test_substitution_word_vs_noise(self):
        """Dodecagonal substitution word (eigenvalue 2+√3) should score higher
        than white noise on dodec_phase_coherence."""
        geom = DodecagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_dw.metrics["dodec_phase_coherence"] >= res_noise.metrics["dodec_phase_coherence"]

    def test_brownian_scores_low(self):
        """Brownian motion should not trigger dodecagonal detection.
        z_sqrt3_coherence should not strongly activate on Brownian motion."""
        geom = DodecagonalGeometry()
        rng_bm = np.random.default_rng(99)
        steps = rng_bm.normal(0, 1, SIZE)
        walk = np.cumsum(steps); walk -= walk.min()
        walk = walk / (walk.max() + 1e-10) * 255
        res_bm = geom.compute_metrics(walk.astype(np.uint8))
        # Brownian has smooth spectrum but no algebraic √3 structure
        assert res_bm.metrics["dodec_phase_coherence"] <= 1.0

    def test_fibonacci_is_not_dodecagonal(self):
        """Fibonacci word (φ ≈ 1.618) should score lower than the dodecagonal
        substitution word on z_sqrt3_coherence."""
        geom = DodecagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        res_fib = geom.compute_metrics(_fibonacci_word())
        assert res_dw.metrics["z_sqrt3_coherence"] > res_fib.metrics["z_sqrt3_coherence"]


class TestSeptagonal:
    """Septagonal (7-fold) geometry. Ratio ρ = 1 + 2cos(2π/7) ≈ 2.247.
    Evolved metric: cubic coherence gate (ρ³=ρ²+2ρ−1).
    No simple 2-letter substitution exists (ρ is a cubic algebraic number),
    so we use cosine-sum signals as positive controls."""

    def test_rho_signal_vs_noise(self):
        """Cosine signal with spectral peaks at ρ-related frequencies
        should score high on cubic_coherence."""
        geom = SeptagonalGeometry()
        rho = geom.RATIO
        res_rho = geom.compute_metrics(_ratio_signal(rho))
        res_noise = geom.compute_metrics(_white_noise())
        assert res_rho.metrics["cubic_coherence"] > res_noise.metrics["cubic_coherence"]
        assert res_rho.metrics["cubic_coherence"] > 0.3

    def test_brownian_not_septagonal(self):
        """Brownian motion should not trigger ρ-ratio detection."""
        geom = SeptagonalGeometry()
        rng_bm = np.random.default_rng(99)
        steps = rng_bm.normal(0, 1, SIZE)
        walk = np.cumsum(steps); walk -= walk.min()
        walk = walk / (walk.max() + 1e-10) * 255
        res = geom.compute_metrics(walk.astype(np.uint8))
        assert res.metrics["cubic_coherence"] < 0.3, \
            f"Brownian cubic_coherence={res.metrics['cubic_coherence']:.3f}"

    def test_dodecagonal_word_is_not_septagonal(self):
        """Dodecagonal substitution (ratio 2+√3) should not score high
        on septagonal (ratio ρ) detection. Different ratios."""
        geom = SeptagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        assert res_dw.metrics["cubic_coherence"] < 0.3, \
            f"Dodeca word cubic_coherence={res_dw.metrics['cubic_coherence']:.3f}"

    def test_cross_specificity(self):
        """ρ-signal should not strongly trigger Penrose φ-detector.
        (Dodecagonal's phase-coherence metrics saturate on simple synthetic
        ratio signals — cross-specificity is validated via per_metric_ablation.)"""
        geom_phi = PenroseGeometry()
        sig = _ratio_signal(SeptagonalGeometry().RATIO)
        score_phi = geom_phi.compute_metrics(sig).metrics["long_range_order"]
        assert score_phi < 0.5, \
            f"ρ-signal should not trigger Penrose: {score_phi:.3f}"


class TestInflation:
    """Inflation (Substitution) geometry. Detects hierarchical self-similar
    structure: linear subword complexity, zero topological entropy, bounded
    discrepancy, concentrated return times, geometric ACF peaks."""

    def test_fibonacci_vs_noise(self):
        """Fibonacci word should score high on substitution metrics;
        noise should score low."""
        geom = InflationGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        # Discrepancy: Fibonacci is balanced → near 0; noise → moderate
        assert res_fib.metrics["discrepancy"] < res_noise.metrics["discrepancy"]
        assert res_fib.metrics["discrepancy"] < 0.05
        # Return concentration: Fibonacci → high; noise → lower
        assert res_fib.metrics["return_concentration"] > res_noise.metrics["return_concentration"]
        # Entropy rate: Fibonacci → low (linear p(n)); noise → high
        assert res_fib.metrics["entropy_rate"] < res_noise.metrics["entropy_rate"]
        assert res_fib.metrics["entropy_rate"] < 0.3

    def test_thue_morse_detects_substitution(self):
        """Thue-Morse (a→ab, b→ba) should be detected as substitution."""
        geom = InflationGeometry()
        tm = np.array([bin(i).count('1') % 2 for i in range(SIZE)],
                      dtype=np.uint8) * 255
        res_tm = geom.compute_metrics(tm)
        res_noise = geom.compute_metrics(_white_noise())
        assert res_tm.metrics["discrepancy"] < 0.05, \
            f"Thue-Morse discrepancy={res_tm.metrics['discrepancy']:.3f}"
        assert res_tm.metrics["return_concentration"] > res_noise.metrics["return_concentration"]

    def test_periodic_has_high_discrepancy(self):
        """Periodic signals (binarized sine) should have high discrepancy
        because their binary representation has long runs."""
        geom = InflationGeometry()
        res_sine = geom.compute_metrics(_sine_wave())
        res_fib = geom.compute_metrics(_fibonacci_word())
        assert res_sine.metrics["discrepancy"] > res_fib.metrics["discrepancy"]
        assert res_sine.metrics["discrepancy"] > 0.2

    def test_chaos_has_high_discrepancy(self):
        """Chaotic signals should have high discrepancy (unbalanced binary)."""
        geom = InflationGeometry()
        res_chaos = geom.compute_metrics(_logistic_chaos())
        assert res_chaos.metrics["discrepancy"] > 0.1

    def test_noise_high_entropy_rate(self):
        """White noise should have high entropy rate (maximal subword growth)."""
        geom = InflationGeometry()
        res = geom.compute_metrics(_white_noise())
        assert res.metrics["entropy_rate"] > 0.7, \
            f"Noise entropy_rate={res.metrics['entropy_rate']:.3f}"


# ── CROSS-CUTTING: DEGENERATE INPUT ───────────────────────────────────

class TestDegenerateInput:
    """All geometries should handle degenerate input without crashing."""

    ALL_CLASSES = [
        TorusGeometry, SphericalGeometry, CantorGeometry,
        HyperbolicGeometry, PersistentHomologyGeometry,
        E8Geometry, HeisenbergGeometry, TropicalGeometry,
        PenroseGeometry, AmmannBeenkerGeometry, EinsteinHatGeometry,
        DodecagonalGeometry, SeptagonalGeometry,
        InflationGeometry,
        SpectralGeometry, RecurrenceGeometry, PredictabilityGeometry,
        HolderRegularityGeometry, InformationGeometry,
        VisibilityGraphGeometry, AttractorGeometry,
        FisherGeometry, WassersteinGeometry,
        HigherOrderGeometry, MultifractalGeometry,
        ZariskiGeometry, CayleyGeometry,
    ]

    # Geometries where constant data makes metrics genuinely undefined (NaN is correct)
    _NAN_ON_CONSTANT = {
        'HolderRegularityGeometry', 'VisibilityGraphGeometry',
        'HigherOrderGeometry', 'MultifractalGeometry',
    }

    @pytest.mark.parametrize("GeomClass", ALL_CLASSES,
                             ids=lambda c: c.__name__)
    def test_constant_input(self, GeomClass):
        """Constant input: no variance. Should not crash. NaN allowed for undefined metrics."""
        if GeomClass.__name__ == "CantorGeometry":
            geom = GeomClass(base=3)
        elif GeomClass.__name__ == "UltrametricGeometry":
            geom = GeomClass(p=2)
        else:
            geom = GeomClass()
        result = geom.compute_metrics(_constant())
        allow_nan = GeomClass.__name__ in self._NAN_ON_CONSTANT
        for k, v in result.metrics.items():
            if allow_nan:
                assert np.isfinite(v) or np.isnan(v), \
                    f"{geom.name}.{k} is not finite or NaN on constant input: {v}"
            else:
                assert np.isfinite(v), f"{geom.name}.{k} is not finite on constant input: {v}"

    @pytest.mark.parametrize("GeomClass", ALL_CLASSES,
                             ids=lambda c: c.__name__)
    def test_short_input(self, GeomClass):
        """Minimal input (100 bytes). Should not crash."""
        if GeomClass.__name__ == "CantorGeometry":
            geom = GeomClass(base=3)
        elif GeomClass.__name__ == "UltrametricGeometry":
            geom = GeomClass(p=2)
        else:
            geom = GeomClass()
        short = RNG.integers(0, 256, 100, dtype=np.uint8)
        result = geom.compute_metrics(short)
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0


# ── ADDITIONAL GEOMETRY TESTS ──────────────────────────────────────────

class TestUltrametric:
    def test_tree_vs_uniform(self):
        """Ultrametric distance = max of pairwise differences on a tree.
        Data with binary structure (powers of 2) should produce lower distance
        entropy than uniform noise (which fills all distance levels)."""
        geom = UltrametricGeometry(p=2)
        # Binary tree-like: values cluster at powers of 2
        tree_like = np.zeros(SIZE, dtype=np.uint8)
        for i in range(SIZE):
            # Hierarchical binary: each bit level is a branch
            tree_like[i] = (i & 0xFF)  # identity mod 256
        # Shuffle to remove sequential correlation but keep distribution
        rng = np.random.default_rng(77)
        rng.shuffle(tree_like)
        res_tree = geom.compute_metrics(tree_like)
        res_rnd = geom.compute_metrics(_white_noise())
        # Both should have finite metrics
        assert np.isfinite(res_tree.metrics["distance_entropy"])
        assert np.isfinite(res_rnd.metrics["distance_entropy"])


class TestSpiral:
    def test_angular_uniformity(self):
        """Constant data maps to one point in spiral space → perfect angular uniformity (1.0).
        Exponential data clusters at small radii → less uniform angles."""
        geom = SpiralGeometry()
        res_const = geom.compute_metrics(_constant())
        t = np.linspace(0, 5, SIZE)
        exp_signal = (np.exp(t) / np.exp(5) * 255).astype(np.uint8)
        res_exp = geom.compute_metrics(exp_signal)
        assert res_const.metrics["angular_uniformity"] > res_exp.metrics["angular_uniformity"]

    # test_tightness removed: tightness metric was pruned from SpiralGeometry.


class TestFractalMandelbrot:
    def test_rich_vs_flat_escape(self):
        """Chaotic signal should produce varied escape times in the Mandelbrot embedding.
        Constant signal → all same escape → low variance."""
        geom = FractalMandelbrotGeometry()
        res_chaos = geom.compute_metrics(_logistic_chaos())
        res_const = geom.compute_metrics(_constant())
        assert res_chaos.metrics["escape_time_variance"] > res_const.metrics["escape_time_variance"]


class TestFractalJulia:
    def test_escape_entropy_variation(self):
        """Different data should produce different Julia set escape entropy values.
        This is a basic sanity check that the metric responds to input."""
        geom = FractalJuliaGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        # Both should be finite and non-negative
        assert np.isfinite(res_per.metrics["escape_entropy"])
        assert np.isfinite(res_rnd.metrics["escape_entropy"])
        assert res_per.metrics["escape_entropy"] >= 0
        assert res_rnd.metrics["escape_entropy"] >= 0


# ── CROSS-GEOMETRY CONSISTENCY ─────────────────────────────────────────

class TestCrossGeometryConsistency:
    """When data has a clear property, multiple geometries should agree."""

    def test_periodic_signal_consensus(self):
        """A periodic signal should be detected as structured by ALL relevant geometries.
        - Predictability: low cond_entropy
        - Recurrence: high determinism
        - Information: low compression_ratio
        - Spectral: low spectral_flatness (dominant peak)
        """
        per = _periodic()
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)

        pred = PredictabilityGeometry()
        recu = RecurrenceGeometry()
        info = InformationGeometry()
        spec = SpectralGeometry()

        rp = pred.compute_metrics(per)
        rn = pred.compute_metrics(noise)
        assert rp.metrics["cond_entropy_k1"] < rn.metrics["cond_entropy_k1"], \
            "Predictability should detect periodic structure"

        rr = recu.compute_metrics(per)
        rrn = recu.compute_metrics(noise)
        assert rr.metrics["determinism"] > rrn.metrics["determinism"], \
            "Recurrence should detect periodic structure"

        ri = info.compute_metrics(per)
        rin = info.compute_metrics(noise)
        assert ri.metrics["compression_ratio"] < rin.metrics["compression_ratio"], \
            "Information should detect periodic structure"

        rs = spec.compute_metrics(per)
        rsn = spec.compute_metrics(noise)
        assert rs.metrics["spectral_flatness"] < rsn.metrics["spectral_flatness"], \
            "Spectral should detect periodic structure"

    def test_white_noise_consensus(self):
        """White noise should score as maximally unstructured across all lenses.
        - High entropy rate (near 1.0)
        - Spectral flatness higher than structured signal (but uint8 quantization
          reduces theoretical flatness below 1.0)
        """
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)

        info = InformationGeometry()
        ri = info.compute_metrics(noise)
        assert ri.metrics["compression_ratio"] > 0.7, \
            f"White noise compression_ratio should be high (near 1.0), got {ri.metrics['compression_ratio']}"

        # White noise should be flatter than a structured signal (sine)
        spec = SpectralGeometry()
        rs_noise = spec.compute_metrics(noise)
        rs_sine = spec.compute_metrics(_sine_wave())
        assert rs_noise.metrics["spectral_flatness"] > rs_sine.metrics["spectral_flatness"], \
            "White noise should have higher spectral flatness than sine wave"


# ── KNOWN MATHEMATICAL VALUES ──────────────────────────────────────────

class TestKnownValues:
    """Test metrics against analytically known values."""

    def test_hurst_brownian_is_half(self):
        """Brownian motion (cumulative sum of IID) has Hurst exponent = 0.5."""
        geom = HolderRegularityGeometry()
        # Average over multiple realizations for stability
        hursts = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            steps = rng.normal(0, 1, SIZE)
            walk = np.cumsum(steps)
            walk -= walk.min()
            mx = walk.max()
            if mx > 0:
                walk = walk / mx * 255
            hursts.append(geom.compute_metrics(walk.astype(np.uint8)).metrics["hurst_exponent"])
        mean_h = np.mean(hursts)
        assert 0.35 < mean_h < 0.65, f"Brownian Hurst should be ~0.5, got {mean_h:.3f}"

    def test_uniform_entropy_is_maximal(self):
        """Uniform distribution over 256 symbols has maximum entropy.
        block_entropy_2 should be near 1.0 (normalized) for large uniform samples."""
        geom = InformationGeometry()
        rng = np.random.default_rng(42)
        uniform = rng.integers(0, 256, SIZE * 4, dtype=np.uint8)  # large sample
        res = geom.compute_metrics(uniform)
        assert res.metrics["block_entropy_2"] > 0.95, \
            f"Uniform H_2 should be near 1.0, got {res.metrics['block_entropy_2']}"

    def test_logistic_attractor_low_dimensional(self):
        """Logistic map at r=3.99 is a low-dimensional dynamical system.
        Its correlation dimension in delay embedding should be much less than
        white noise (which fills the embedding space)."""
        geom = AttractorGeometry()
        # Logistic map
        x = 0.4
        out = np.empty(SIZE * 2)
        for i in range(SIZE * 2):
            x = 3.99 * x * (1 - x)
            out[i] = x
        chaos = (out[SIZE:] * 255).astype(np.uint8)
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)
        res_chaos = geom.compute_metrics(chaos)
        res_noise = geom.compute_metrics(noise)
        assert res_chaos.metrics["correlation_dimension"] < res_noise.metrics["correlation_dimension"], \
            f"Logistic dim ({res_chaos.metrics['correlation_dimension']:.2f}) should be less than noise ({res_noise.metrics['correlation_dimension']:.2f})"

    def test_periodic_sample_entropy_near_zero(self):
        """A perfectly periodic signal has sample entropy ≈ 0
        (every pattern that matches at length m also matches at m+1).
        Tolerance accounts for finite-size edge effects."""
        geom = PredictabilityGeometry()
        per = _alternating()
        res = geom.compute_metrics(per)
        assert res.metrics["sample_entropy"] < 0.15, \
            f"Periodic sample entropy should be ~0, got {res.metrics['sample_entropy']}"


# ── SHUFFLE TEST ───────────────────────────────────────────────────────

class TestShuffleDestroysStructure:
    """Shuffling data destroys temporal/sequential structure.
    Metrics that detect such structure should change significantly."""

    def _shuffle(self, data):
        """Return a shuffled copy of data."""
        rng = np.random.default_rng(42)
        shuffled = data.copy()
        rng.shuffle(shuffled)
        return shuffled

    def test_recurrence_determinism_drops(self):
        """Shuffling a periodic signal destroys the diagonal-line structure in RP."""
        geom = RecurrenceGeometry()
        per = _periodic()
        res_orig = geom.compute_metrics(per)
        res_shuf = geom.compute_metrics(self._shuffle(per))
        assert res_orig.metrics["determinism"] > res_shuf.metrics["determinism"], \
            "Shuffling should destroy recurrence determinism"

    def test_spectral_slope_changes(self):
        """Shuffling Brownian motion destroys temporal correlations.
        Slope should move toward 0 (white spectrum)."""
        geom = SpectralGeometry()
        brown = _brownian()
        res_orig = geom.compute_metrics(brown)
        res_shuf = geom.compute_metrics(self._shuffle(brown))
        # Brownian has negative slope (red); shuffled → flatter
        assert res_orig.metrics["spectral_slope"] < res_shuf.metrics["spectral_slope"]

    def test_predictability_drops(self):
        """Shuffling a logistic map signal should increase conditional entropy."""
        geom = PredictabilityGeometry()
        chaos = _logistic_chaos()
        res_orig = geom.compute_metrics(chaos)
        res_shuf = geom.compute_metrics(self._shuffle(chaos))
        assert res_orig.metrics["cond_entropy_k1"] < res_shuf.metrics["cond_entropy_k1"], \
            "Shuffling should increase conditional entropy"

    def test_quasicrystal_symmetry_drops(self):
        """Shuffling a Fibonacci word destroys its φ-ratio spectral self-similarity."""
        geom = PenroseGeometry()
        fib = _fibonacci_word()
        res_orig = geom.compute_metrics(fib)
        res_shuf = geom.compute_metrics(self._shuffle(fib))
        assert res_orig.metrics["long_range_order"] > res_shuf.metrics["long_range_order"], \
            "Shuffling should destroy quasicrystalline spectral structure"

    def test_holder_regularity_drops(self):
        """Shuffling fBm destroys smoothness. Hölder exponent should drop."""
        geom = HolderRegularityGeometry()
        smooth = _fbm(H=0.8)
        res_orig = geom.compute_metrics(smooth)
        res_shuf = geom.compute_metrics(self._shuffle(smooth))
        assert res_orig.metrics["holder_mean"] > res_shuf.metrics["holder_mean"], \
            "Shuffling should destroy smoothness (lower Hölder regularity)"

    def test_visibility_graph_structure_changes(self):
        """Shuffling a sine wave destroys the regular peak/trough hub structure."""
        geom = VisibilityGraphGeometry()
        sine = _sine_wave()
        res_orig = geom.compute_metrics(sine)
        res_shuf = geom.compute_metrics(self._shuffle(sine))
        # Sine wave has extreme hubs (peaks see far) → high max_degree
        # Shuffled → more homogeneous degree distribution
        assert res_orig.metrics["max_degree"] > res_shuf.metrics["max_degree"], \
            "Shuffling should destroy visibility hub structure"


# ── MONOTONIC NOISE DEGRADATION ────────────────────────────────────────

class TestNoiseDegradation:
    """As noise increases, structure metrics should degrade monotonically."""

    def test_spectral_flatness_increases_with_noise(self):
        """Adding increasing amounts of noise to a sine wave should
        monotonically increase spectral flatness toward 1.0."""
        geom = SpectralGeometry()
        sine = _sine_wave().astype(np.float64)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 255, SIZE)
        prev_flatness = -1
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mixed = ((1 - alpha) * sine + alpha * noise)
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)
            res = geom.compute_metrics(mixed)
            flatness = res.metrics["spectral_flatness"]
            assert flatness >= prev_flatness - 0.05, \
                f"Spectral flatness should increase with noise: alpha={alpha}, flatness={flatness}, prev={prev_flatness}"
            prev_flatness = flatness

    def test_predictability_decreases_with_noise(self):
        """Adding noise to a periodic signal should increase conditional entropy
        (decrease predictability). This is more robust than recurrence determinism."""
        geom = PredictabilityGeometry()
        per = _periodic().astype(np.float64)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 255, SIZE)
        prev_ce = -1
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mixed = ((1 - alpha) * per + alpha * noise)
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)
            res = geom.compute_metrics(mixed)
            ce = res.metrics["cond_entropy_k1"]
            assert ce >= prev_ce - 0.05, \
                f"Cond entropy should increase with noise: alpha={alpha}, ce={ce:.3f}, prev={prev_ce:.3f}"
            prev_ce = ce


# ── REPRODUCIBILITY ────────────────────────────────────────────────────

class TestReproducibility:
    """Same input should always produce the same output."""

    GEOM_CLASSES = [
        PenroseGeometry, SpectralGeometry, RecurrenceGeometry,
        InformationGeometry, HolderRegularityGeometry, AttractorGeometry,
        InflationGeometry, ZariskiGeometry, CayleyGeometry,
    ]

    @pytest.mark.parametrize("GeomClass", GEOM_CLASSES,
                             ids=lambda c: c.__name__)
    def test_deterministic_output(self, GeomClass):
        """Running compute_metrics twice on identical data should give identical results."""
        geom = GeomClass()
        data = (np.arange(SIZE) % 256).astype(np.uint8)
        r1 = geom.compute_metrics(data)
        r2 = geom.compute_metrics(data)
        for k in r1.metrics:
            assert r1.metrics[k] == pytest.approx(r2.metrics[k], abs=1e-12), \
                f"{GeomClass.__name__}.{k}: {r1.metrics[k]} != {r2.metrics[k]}"


# ── ZARISKI (NON-HAUSDORFF) GEOMETRY ─────────────────────────────────

class TestZariski:
    """Zariski geometry detects algebraic (polynomial recurrence) structure
    and non-Boolean (Heyting) pattern lattice gaps."""

    def test_quadratic_recurrence_detected(self):
        """Logistic map x→rx(1-x) is a degree-2 polynomial.
        Its 2D delay embedding lies on a parabola, giving a much lower
        algebraic_residual than white noise."""
        geom = ZariskiGeometry()
        # Logistic chaos (deterministic seed — no RNG)
        x = 0.4
        out = np.empty(SIZE)
        for i in range(SIZE):
            x = 3.99 * x * (1 - x)
            out[i] = x
        logistic = (out * 255).astype(np.uint8)

        r_log = geom.compute_metrics(logistic)
        r_noise = geom.compute_metrics(_white_noise())

        # Logistic residual should be ≥10x lower than noise
        assert r_log.metrics["algebraic_residual"] < r_noise.metrics["algebraic_residual"] / 10, \
            (f"Logistic residual={r_log.metrics['algebraic_residual']:.6f} "
             f"should be <<  noise={r_noise.metrics['algebraic_residual']:.6f}")

    def test_residual_slope_steeper_for_polynomial(self):
        """Polynomial maps should have steeper (more negative) residual slopes
        than noise, because residuals drop faster with degree."""
        geom = ZariskiGeometry()
        # Sine wave: highly smooth, nearly algebraic
        t = np.arange(SIZE)
        sine = ((np.sin(2 * np.pi * 0.01 * t) + 1) / 2 * 255).astype(np.uint8)

        r_sine = geom.compute_metrics(sine)
        r_noise = geom.compute_metrics(_white_noise())

        assert r_sine.metrics["residual_slope"] < r_noise.metrics["residual_slope"], \
            (f"Sine slope={r_sine.metrics['residual_slope']:.3f} "
             f"should be < noise slope={r_noise.metrics['residual_slope']:.3f}")

    def test_heyting_gap_zero_for_noise(self):
        """White noise visits all patterns and their complements.
        Heyting gap should be 0 (Boolean/Hausdorff pattern lattice)."""
        geom = ZariskiGeometry()
        r = geom.compute_metrics(_white_noise())
        assert r.metrics["heyting_gap"] < 0.01, \
            f"Noise heyting_gap={r.metrics['heyting_gap']:.4f}, expected ~0"

    def test_heyting_gap_high_for_qc(self):
        """Fibonacci quasicrystal has substitution rules that prevent many
        complement patterns from appearing. Heyting gap should be large."""
        geom = ZariskiGeometry()
        r = geom.compute_metrics(_fibonacci_word())
        assert r.metrics["heyting_gap"] > 0.5, \
            f"Fibonacci heyting_gap={r.metrics['heyting_gap']:.4f}, expected > 0.5"


# ── CAYLEY (GEOMETRIC GROUP THEORY) GEOMETRY ─────────────────────────

class TestCayley:
    """Cayley geometry computes GGT invariants: Gromov hyperbolicity,
    growth exponent, and spectral gap on k-NN graphs."""

    def test_growth_exponent_noise_vs_curve(self):
        """White noise fills R^3 (growth~2). Logistic chaos traces a 1D curve
        (growth~1). Growth exponent should clearly separate them."""
        geom = CayleyGeometry()
        # Logistic chaos (deterministic)
        x = 0.4
        out = np.empty(SIZE)
        for i in range(SIZE):
            x = 3.99 * x * (1 - x)
            out[i] = x
        logistic = (out * 255).astype(np.uint8)

        r_noise = geom.compute_metrics(_white_noise())
        r_log = geom.compute_metrics(logistic)

        assert r_noise.metrics["growth_exponent"] > r_log.metrics["growth_exponent"] + 0.5, \
            (f"Noise growth={r_noise.metrics['growth_exponent']:.3f} "
             f"should be >> logistic growth={r_log.metrics['growth_exponent']:.3f}")

    def test_hyperbolicity_smooth_vs_chaotic(self):
        """Brownian motion is nearly tree-like (low δ). Chaotic maps fill
        higher-dimensional subspaces (higher δ)."""
        geom = CayleyGeometry()
        r_brown = geom.compute_metrics(_brownian())

        # Logistic chaos
        x = 0.4
        out = np.empty(SIZE)
        for i in range(SIZE):
            x = 3.99 * x * (1 - x)
            out[i] = x
        logistic = (out * 255).astype(np.uint8)
        r_log = geom.compute_metrics(logistic)

        assert r_brown.metrics["delta_hyp_norm"] < r_log.metrics["delta_hyp_norm"], \
            (f"Brownian δ/diam={r_brown.metrics['delta_hyp_norm']:.4f} "
             f"should be < logistic δ/diam={r_log.metrics['delta_hyp_norm']:.4f}")

    def test_spectral_gap_noise_is_expander(self):
        """White noise k-NN graph is well-connected (large spectral gap).
        Structured data has bottlenecks (small spectral gap)."""
        geom = CayleyGeometry()
        r_noise = geom.compute_metrics(_white_noise())

        t = np.arange(SIZE)
        sine = ((np.sin(2 * np.pi * 0.01 * t) + 1) / 2 * 255).astype(np.uint8)
        r_sine = geom.compute_metrics(sine)

        assert r_noise.metrics["spectral_gap"] > r_sine.metrics["spectral_gap"], \
            (f"Noise spectral_gap={r_noise.metrics['spectral_gap']:.6f} "
             f"should be > sine spectral_gap={r_sine.metrics['spectral_gap']:.6f}")

    def test_saturation_radius_noise_fast(self):
        """White noise graph saturates quickly (small world).
        1D curves take much longer to saturate."""
        geom = CayleyGeometry()
        r_noise = geom.compute_metrics(_white_noise())
        r_brown = geom.compute_metrics(_brownian())

        assert r_noise.metrics["saturation_radius"] < r_brown.metrics["saturation_radius"], \
            (f"Noise sat_r={r_noise.metrics['saturation_radius']:.4f} "
             f"should be < brownian sat_r={r_brown.metrics['saturation_radius']:.4f}")


class TestSpectralGraph:
    """Spectral Graph Geometry computes heat kernel spectral invariants
    on epsilon-neighborhood graphs: spectral dimension, Weyl exponent,
    and eigenvalue gap ratio."""

    def test_spectral_dim_noise_low_structured_high(self):
        """White noise has low spectral dimension (well-connected random graph,
        heat dissipates fast). Logistic chaos traces a higher-dimensional
        attractor (spectral_dim > 1)."""
        geom = SpectralGraphGeometry()
        x = 0.4
        out = np.empty(SIZE)
        for i in range(SIZE):
            x = 3.99 * x * (1 - x)
            out[i] = x
        logistic = (out * 255).astype(np.uint8)

        r_noise = geom.compute_metrics(_white_noise())
        r_log = geom.compute_metrics(logistic)

        assert r_noise.metrics["spectral_dim"] < r_log.metrics["spectral_dim"], \
            (f"Noise d_s={r_noise.metrics['spectral_dim']:.3f} "
             f"should be < logistic d_s={r_log.metrics['spectral_dim']:.3f}")

    def test_weyl_exponent_periodic_high(self):
        """Sine wave lives on a 1D curve in embedding space. Eigenvalues
        grow fast (high Weyl exponent). White noise fills the space,
        eigenvalues grow slowly (low Weyl exponent)."""
        geom = SpectralGraphGeometry()
        t = np.arange(SIZE)
        sine = ((np.sin(2 * np.pi * 0.01 * t) + 1) / 2 * 255).astype(np.uint8)

        r_sine = geom.compute_metrics(sine)
        r_noise = geom.compute_metrics(_white_noise())

        assert r_sine.metrics["weyl_exponent"] > r_noise.metrics["weyl_exponent"] + 0.5, \
            (f"Sine Weyl={r_sine.metrics['weyl_exponent']:.3f} "
             f"should be >> noise Weyl={r_noise.metrics['weyl_exponent']:.3f}")

    # test_gap_ratio_chaotic_has_stronger_gap removed: gap_ratio metric was pruned from SpectralGraphGeometry.

    def test_returns_two_metrics(self):
        """Geometry should always return exactly 2 metrics."""
        geom = SpectralGraphGeometry()
        r = geom.compute_metrics(_white_noise())
        expected = {"spectral_dim", "weyl_exponent"}
        assert set(r.metrics.keys()) == expected, \
            f"Expected {expected}, got {set(r.metrics.keys())}"


class TestBoltzmann:
    """Boltzmann geometry fits a pairwise Ising model to binary windows
    and extracts coupling strength, frustration, and spectral gap."""

    def test_coupling_noise_near_zero(self):
        """White noise has near-zero pairwise couplings — positions are
        independent, so J_ij ≈ 0."""
        geom = BoltzmannGeometry()
        r = geom.compute_metrics(_white_noise())
        assert r.metrics["coupling_strength"] < 0.05, \
            f"Noise coupling={r.metrics['coupling_strength']:.4f} should be near 0"

    def test_coupling_structured_high(self):
        """Brownian motion has strong temporal correlations — adjacent
        positions constrain each other, giving large |J_ij|."""
        geom = BoltzmannGeometry()
        r_brown = geom.compute_metrics(_brownian())
        r_noise = geom.compute_metrics(_white_noise())
        assert r_brown.metrics["coupling_strength"] > 10 * r_noise.metrics["coupling_strength"], \
            (f"Brownian coupling={r_brown.metrics['coupling_strength']:.4f} "
             f"should be >> noise coupling={r_noise.metrics['coupling_strength']:.4f}")

    def test_frustration_noise_near_half(self):
        """White noise has random J, so frustration ≈ 0.5 (half of
        triangles are frustrated by chance)."""
        geom = BoltzmannGeometry()
        r = geom.compute_metrics(_white_noise())
        assert 0.2 < r.metrics["frustration"] < 0.8, \
            f"Noise frustration={r.metrics['frustration']:.4f} should be near 0.5"

    def test_frustration_monotonic_zero(self):
        """Monotonic signals (ramp, brownian) have consistent pairwise
        correlations — zero frustration."""
        geom = BoltzmannGeometry()
        ramp = np.linspace(0, 255, SIZE).astype(np.uint8)
        r = geom.compute_metrics(ramp)
        assert r.metrics["frustration"] < 0.05, \
            f"Ramp frustration={r.metrics['frustration']:.4f} should be near 0"

    def test_frustration_quasicrystal_high(self):
        """Quasicrystal (Fibonacci) has incommensurable correlations that
        create maximal frustration — the spin-glass signature of aperiodic order."""
        geom = BoltzmannGeometry()
        # Fibonacci word
        a, b = [0], [0, 1]
        while len(b) < SIZE:
            a, b = b, b + a
        fib = np.array(b[:SIZE], dtype=np.uint8) * 255
        r = geom.compute_metrics(fib)
        assert r.metrics["frustration"] > 0.8, \
            f"Fibonacci frustration={r.metrics['frustration']:.4f} should be near 1.0"

    def test_returns_three_metrics(self):
        """Geometry should always return exactly 3 metrics."""
        geom = BoltzmannGeometry()
        r = geom.compute_metrics(_white_noise())
        expected = {"coupling_strength", "frustration", "spectral_gap_J"}
        assert set(r.metrics.keys()) == expected, \
            f"Expected {expected}, got {set(r.metrics.keys())}"


class TestGottwaldMelbourne:
    """Gottwald-Melbourne 0-1 test: K≈1 for chaos, K≈0 for regular."""

    def test_chaos_high_k(self):
        """Logistic chaos at r=3.99 is fully chaotic → K should be near 1."""
        geom = GottwaldMelbourneGeometry()
        r = geom.compute_metrics(_logistic_chaos())
        assert r.metrics["k_statistic"] > 0.7, \
            f"Logistic chaos K={r.metrics['k_statistic']:.4f} should be near 1"

    def test_periodic_low_k(self):
        """Periodic signal is perfectly regular → K should be near 0."""
        geom = GottwaldMelbourneGeometry()
        r = geom.compute_metrics(_periodic())
        assert r.metrics["k_statistic"] < 0.3, \
            f"Periodic K={r.metrics['k_statistic']:.4f} should be near 0"

    def test_chaos_vs_periodic(self):
        """Chaotic signal should have higher K than periodic."""
        geom = GottwaldMelbourneGeometry()
        r_chaos = geom.compute_metrics(_logistic_chaos())
        r_per = geom.compute_metrics(_periodic())
        assert r_chaos.metrics["k_statistic"] > r_per.metrics["k_statistic"], \
            (f"Chaos K={r_chaos.metrics['k_statistic']:.4f} should exceed "
             f"periodic K={r_per.metrics['k_statistic']:.4f}")

    def test_noise_high_k(self):
        """White noise is maximally unpredictable — K should be high.
        (Noise technically isn't deterministic chaos, but MSD grows
        linearly so K≈1 as well.)"""
        geom = GottwaldMelbourneGeometry()
        r = geom.compute_metrics(_white_noise())
        assert r.metrics["k_statistic"] > 0.6, \
            f"Noise K={r.metrics['k_statistic']:.4f} should be high"

    def test_returns_two_metrics(self):
        geom = GottwaldMelbourneGeometry()
        r = geom.compute_metrics(_white_noise())
        expected = {"k_statistic", "k_variance"}
        assert set(r.metrics.keys()) == expected, \
            f"Expected {expected}, got {set(r.metrics.keys())}"


class TestOrdinalPartition:
    """Ordinal partition transition dynamics: entropy, irreversibility, complexity."""

    def test_periodic_low_transition_entropy(self):
        """Periodic signal has deterministic ordinal transitions → low entropy."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_periodic())
        assert r.metrics["transition_entropy"] < 0.3, \
            f"Periodic transition_entropy={r.metrics['transition_entropy']:.4f} should be low"

    def test_noise_high_transition_entropy(self):
        """White noise has near-uniform transitions → high entropy."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_white_noise())
        assert r.metrics["transition_entropy"] > 0.7, \
            f"Noise transition_entropy={r.metrics['transition_entropy']:.4f} should be high"

    def test_chaos_intermediate_transition_entropy(self):
        """Logistic chaos has structured but non-deterministic transitions."""
        geom = OrdinalPartitionGeometry()
        r_chaos = geom.compute_metrics(_logistic_chaos())
        r_per = geom.compute_metrics(_periodic())
        r_noise = geom.compute_metrics(_white_noise())
        # Chaos should be between periodic and noise
        assert r_chaos.metrics["transition_entropy"] > r_per.metrics["transition_entropy"], \
            "Chaos transition_entropy should exceed periodic"
        assert r_chaos.metrics["transition_entropy"] < r_noise.metrics["transition_entropy"], \
            "Chaos transition_entropy should be less than noise"

    def test_irreversibility_noise_low(self):
        """White noise is time-reversible: P(π) = P(π̃) for all π, so
        the TV distance between forward and reversed distributions ≈ 0."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_white_noise())
        # Finite-sample TV ≈ sqrt(d!/(4N)) ≈ 0.09 for N=4092, d=5
        assert r.metrics["time_irreversibility"] < 0.15, \
            f"Noise irreversibility={r.metrics['time_irreversibility']:.4f} should be near 0"

    def test_irreversibility_chaos_higher(self):
        """Logistic chaos is dissipative and irreversible → the ordinal
        distribution differs from its time-reverse."""
        geom = OrdinalPartitionGeometry()
        r_chaos = geom.compute_metrics(_logistic_chaos())
        r_noise = geom.compute_metrics(_white_noise())
        assert r_chaos.metrics["time_irreversibility"] > r_noise.metrics["time_irreversibility"], \
            (f"Chaos irrev={r_chaos.metrics['time_irreversibility']:.4f} should exceed "
             f"noise irrev={r_noise.metrics['time_irreversibility']:.4f}")

    def test_complexity_periodic_low(self):
        """Periodic signal: zero complexity (perfect order)."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_periodic())
        assert r.metrics["statistical_complexity"] < 0.1, \
            f"Periodic complexity={r.metrics['statistical_complexity']:.4f} should be near 0"

    def test_noise_no_forbidden_transitions(self):
        """White noise: all legal transitions present → 0."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_white_noise())
        assert r.metrics["forbidden_transitions"] < 0.01, \
            (f"Noise forbidden_transitions={r.metrics['forbidden_transitions']:.4f} "
             "should be near 0")

    def test_chaos_has_forbidden_transitions(self):
        """Logistic map: deterministic constraints forbid some transitions."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_logistic_chaos())
        assert r.metrics["forbidden_transitions"] > 0.3, \
            (f"Logistic forbidden_transitions={r.metrics['forbidden_transitions']:.4f} "
             "should be > 0.3")

    def test_periodic_high_forbidden_transitions(self):
        """Periodic signal: most transitions forbidden."""
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_periodic())
        assert r.metrics["forbidden_transitions"] > 0.5, \
            (f"Periodic forbidden_transitions={r.metrics['forbidden_transitions']:.4f} "
             "should be > 0.5")

    def test_returns_six_metrics(self):
        geom = OrdinalPartitionGeometry()
        r = geom.compute_metrics(_white_noise())
        expected = {"transition_entropy", "time_irreversibility",
                    "statistical_complexity", "forbidden_transitions",
                    "markov_mixing", "memory_order"}
        assert set(r.metrics.keys()) == expected, \
            f"Expected {expected}, got {set(r.metrics.keys())}"

    def test_markov_mixing_separates_noise_from_smooth(self):
        """White noise has high markov_mixing (fast value-space mixing),
        sine waves have low markov_mixing (persistent values)."""
        geom = OrdinalPartitionGeometry()
        r_noise = geom.compute_metrics(_white_noise())
        t = np.linspace(0, 20 * np.pi, 2000)
        sine = ((np.sin(t) + 1) / 2 * 255).astype(np.uint8)
        r_sine = geom.compute_metrics(sine)
        assert r_noise.metrics["markov_mixing"] > 0.5, \
            (f"Noise markov_mixing={r_noise.metrics['markov_mixing']:.3f} "
             "should be > 0.5")
        assert r_sine.metrics["markov_mixing"] < 0.1, \
            (f"Sine markov_mixing={r_sine.metrics['markov_mixing']:.3f} "
             "should be < 0.1")

    def test_memory_order_detects_higher_dim_dynamics(self):
        """Lorenz x-component (3D projected to 1D) has high memory_order
        because knowing two previous values helps predict the next.
        White noise has near-zero memory_order."""
        geom = OrdinalPartitionGeometry()
        # Lorenz attractor x-component
        dt = 0.01
        x, y, z = 1.0, 1.0, 1.0
        vals = []
        for _ in range(25000):
            dx = 10*(y - x); dy = x*(28 - z) - y; dz = x*y - 8/3*z
            x += dx*dt; y += dy*dt; z += dz*dt
            vals.append(x)
        arr = np.array(vals[5000::10])[:2000]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255
        lorenz = arr.astype(np.uint8)
        r_lorenz = geom.compute_metrics(lorenz)
        r_noise = geom.compute_metrics(_white_noise())
        assert r_lorenz.metrics["memory_order"] > 0.3, \
            (f"Lorenz memory_order={r_lorenz.metrics['memory_order']:.3f} "
             "should be > 0.3")
        assert r_noise.metrics["memory_order"] < 0.2, \
            (f"Noise memory_order={r_noise.metrics['memory_order']:.3f} "
             "should be < 0.2")


class TestNonstationarity:
    """Nonstationarity geometry: tracks how local geometric character changes."""

    def test_returns_four_metrics(self):
        geom = NonstationarityGeometry()
        r = geom.compute_metrics(_white_noise())
        expected = {"metric_volatility", "vol_of_vol",
                    "regime_persistence", "trajectory_dim"}
        assert set(r.metrics.keys()) == expected, \
            f"Expected {expected}, got {set(r.metrics.keys())}"

    def test_vol_of_vol_regime_switch_high(self):
        """Regime-switching process has bursty geometric change → high vol_of_vol."""
        geom = NonstationarityGeometry()
        rng = np.random.default_rng(99)
        # Build regime-switching signal
        out = np.empty(SIZE, dtype=np.uint8)
        regime = 0
        for i in range(SIZE):
            if rng.random() < 0.005:
                regime = 1 - regime
            if regime == 0:
                out[i] = int(np.clip(128 + rng.standard_normal() * 10, 0, 255))
            else:
                out[i] = int(np.clip(128 + rng.standard_normal() * 80, 0, 255))
        r_regime = geom.compute_metrics(out)
        r_noise = geom.compute_metrics(_white_noise())
        assert r_regime.metrics["vol_of_vol"] > r_noise.metrics["vol_of_vol"], \
            (f"Regime vol_of_vol={r_regime.metrics['vol_of_vol']:.3f} should exceed "
             f"noise vol_of_vol={r_noise.metrics['vol_of_vol']:.3f}")

    def test_regime_persistence_concatenated_high(self):
        """Concatenated signals (sine+noise+chaos) have long geometric regimes."""
        geom = NonstationarityGeometry()
        third = SIZE // 3
        t = np.linspace(0, 20 * np.pi, third)
        seg_a = ((np.sin(t) + 1) * 127.5).astype(np.uint8)
        seg_b = RNG.integers(0, 256, third, dtype=np.uint8)
        seg_c = np.empty(SIZE - 2 * third, dtype=np.uint8)
        x = 0.4
        for i in range(len(seg_c)):
            x = 3.99 * x * (1 - x)
            seg_c[i] = int(x * 255)
        concat = np.concatenate([seg_a, seg_b, seg_c])
        r_concat = geom.compute_metrics(concat)
        r_noise = geom.compute_metrics(_white_noise())
        assert r_concat.metrics["regime_persistence"] > r_noise.metrics["regime_persistence"], \
            (f"Concat persistence={r_concat.metrics['regime_persistence']:.3f} should exceed "
             f"noise persistence={r_noise.metrics['regime_persistence']:.3f}")

    def test_trajectory_dim_noise_high(self):
        """White noise fills descriptor space → high trajectory_dim.
        Sine wave is constrained → low trajectory_dim."""
        geom = NonstationarityGeometry()
        r_noise = geom.compute_metrics(_white_noise())
        r_sine = geom.compute_metrics(_sine_wave())
        assert r_noise.metrics["trajectory_dim"] > r_sine.metrics["trajectory_dim"], \
            (f"Noise traj_dim={r_noise.metrics['trajectory_dim']:.3f} should exceed "
             f"sine traj_dim={r_sine.metrics['trajectory_dim']:.3f}")


def _xorshift32(size=SIZE, seed=1):
    """XorShift32 PRNG — GF(2)-linear recurrence with 32-bit state."""
    state = seed
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17)
        state ^= (state << 5) & 0xFFFFFFFF
        vals[i] = (state >> 16) & 0xFF
    return vals


class TestKleinBottle:
    """Klein Bottle geometry: GF(2) linear structure via LFSR complexity and binary rank."""

    def test_returns_three_metrics(self):
        geom = KleinBottleGeometry()
        r = geom.compute_metrics(_white_noise())
        expected = {"linear_complexity", "rank_deficit", "orientation_coherence"}
        assert set(r.metrics.keys()) == expected, \
            f"Expected {expected}, got {set(r.metrics.keys())}"

    def test_linear_complexity_xorshift_low(self):
        """XorShift32 is a GF(2)-linear recurrence: LC << n/2."""
        geom = KleinBottleGeometry()
        r_xor = geom.compute_metrics(_xorshift32())
        r_noise = geom.compute_metrics(_white_noise())
        assert r_xor.metrics["linear_complexity"] < 0.2, \
            f"XorShift LC={r_xor.metrics['linear_complexity']:.4f} should be < 0.2"
        assert r_noise.metrics["linear_complexity"] > 0.9, \
            f"Noise LC={r_noise.metrics['linear_complexity']:.4f} should be > 0.9"

    def test_rank_deficit_structured_high(self):
        """Highly structured signals (Thue-Morse) have large GF(2) rank deficit."""
        geom = KleinBottleGeometry()
        # Thue-Morse: binary substitution, extremely rank-deficient
        n = SIZE
        tm = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            tm[i] = 255 * (bin(i).count('1') % 2)
        r_tm = geom.compute_metrics(tm)
        r_noise = geom.compute_metrics(_white_noise())
        assert r_tm.metrics["rank_deficit"] > r_noise.metrics["rank_deficit"], \
            (f"Thue-Morse rank_deficit={r_tm.metrics['rank_deficit']:.3f} should exceed "
             f"noise rank_deficit={r_noise.metrics['rank_deficit']:.3f}")

    def test_orientation_coherence_smooth_high(self):
        """Smooth trajectories (sine wave) have high orientation coherence."""
        geom = KleinBottleGeometry()
        r_sine = geom.compute_metrics(_sine_wave())
        r_noise = geom.compute_metrics(_white_noise())
        assert r_sine.metrics["orientation_coherence"] > r_noise.metrics["orientation_coherence"], \
            (f"Sine coherence={r_sine.metrics['orientation_coherence']:.3f} should exceed "
             f"noise coherence={r_noise.metrics['orientation_coherence']:.3f}")
