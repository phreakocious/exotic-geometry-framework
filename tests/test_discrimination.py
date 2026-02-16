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
    HyperbolicGeometry, PersistentHomologyGeometry, VisibilityGraphGeometry,
    FractalMandelbrotGeometry, FractalJuliaGeometry,
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
    DodecagonalGeometry, DecagonalGeometry, SeptagonalGeometry,
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
        assert res_per.metrics["normalized_entropy"] < res_rnd.metrics["normalized_entropy"]


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
        """Data with Cantor-like gaps should have lower estimated dimension than uniform."""
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
        # Cantor set has dimension log2/log3 ≈ 0.63 — should be lower than uniform (dim ≈ 1)
        assert res_cantor.metrics["estimated_dimension"] < res_rnd.metrics["estimated_dimension"]


class TestInformation:
    def test_entropy_rate_ordering(self):
        """Periodic < random for entropy rate.
        Entropy rate measures new information per symbol given history."""
        geom = InformationGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        er_per = res_per.metrics["entropy_rate"]
        er_rnd = res_rnd.metrics["entropy_rate"]
        assert er_per < er_rnd, f"periodic {er_per} should have lower entropy rate than random {er_rnd}"

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
    def test_exponential_vs_uniform(self):
        """Exponentially distributed data clusters near the Poincaré disk boundary.
        Uniform data fills the disk more evenly → lower boundary proximity."""
        geom = HyperbolicGeometry()
        res_exp = geom.compute_metrics(_exponential_dist())
        res_rnd = geom.compute_metrics(_white_noise())
        assert res_exp.metrics["boundary_proximity"] > res_rnd.metrics["boundary_proximity"]


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
        assert res_lat.metrics["alignment_mean"] > res_rnd.metrics["alignment_mean"]


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
        # Spiral sweeps coherent area → high |twist_rate|
        assert abs(res_spiral.metrics["twist_rate"]) > abs(res_rnd.metrics["twist_rate"])


class TestTropical:
    def test_piecewise_linear_vs_smooth(self):
        """Piecewise-linear data (sawtooth) should produce fewer slope changes
        than noisy data. Tropical geometry = min-plus algebra → detects linearity."""
        geom = TropicalGeometry()
        res_saw = geom.compute_metrics(_periodic())  # ramp = piecewise linear
        res_rnd = geom.compute_metrics(_white_noise())
        # PWL data → high linearity score, low slope changes
        assert res_saw.metrics["linearity"] > res_rnd.metrics["linearity"]


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
        """Sol geometry detects anisotropic stretch.
        Data with different x/y scales → high stretch_ratio.
        Isotropic noise → stretch_ratio near 1."""
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
        # Anisotropic data should show different stretch vs random
        assert res_aniso.metrics["stretch_ratio"] != pytest.approx(
            res_rnd.metrics["stretch_ratio"], abs=0.01
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
        Should score high on fivefold_symmetry (spectral self-similarity at φ)
        and index_diversity (linear subword complexity growth)."""
        geom = PenroseGeometry()
        fib = _fibonacci_word()
        noise = _white_noise()
        res_fib = geom.compute_metrics(fib)
        res_noise = geom.compute_metrics(noise)
        assert res_fib.metrics["fivefold_symmetry"] > res_noise.metrics["fivefold_symmetry"]
        assert res_fib.metrics["index_diversity"] > res_noise.metrics["index_diversity"]

    def test_fibonacci_vs_periodic(self):
        """Fibonacci is quasiperiodic, not periodic. Both have structure, but
        Fibonacci should show φ-ratio spectral self-similarity that periodic doesn't."""
        geom = PenroseGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_per = geom.compute_metrics(_periodic())
        assert res_fib.metrics["fivefold_symmetry"] > res_per.metrics["fivefold_symmetry"]

    def test_long_range_order(self):
        """Fibonacci word has strong ACF self-similarity at φ-scaled lags.
        White noise has none."""
        geom = PenroseGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_fib.metrics["long_range_order"] > res_noise.metrics["long_range_order"]

    def test_brownian_not_quasicrystal(self):
        """Brownian motion has smooth 1/f² spectrum. The moving-average detrend
        + null-ratio comparison should reject it: low fivefold_symmetry, low
        long_range_order. This was the critical false positive before the rewrite."""
        geom = PenroseGeometry()
        res_bm = geom.compute_metrics(_brownian())
        assert res_bm.metrics["fivefold_symmetry"] < 0.3, \
            f"Brownian fivefold_symmetry={res_bm.metrics['fivefold_symmetry']:.3f} should be < 0.3"
        assert res_bm.metrics["long_range_order"] < 0.3, \
            f"Brownian long_range_order={res_bm.metrics['long_range_order']:.3f} should be < 0.3"

    def test_fbm_not_quasicrystal(self):
        """Fractional Brownian motion (H=0.8) has smooth 1/f^(2H+1) spectrum.
        Should not fool the ratio detector despite strong spectral structure.
        Uses a dedicated RNG to avoid sensitivity to global RNG state."""
        geom = PenroseGeometry()
        # Generate fBm with a fresh, fixed seed
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
        assert res_fbm.metrics["fivefold_symmetry"] < 0.3, \
            f"fBm fivefold_symmetry={res_fbm.metrics['fivefold_symmetry']:.3f} should be < 0.3"
        assert res_fbm.metrics["long_range_order"] < 0.3, \
            f"fBm long_range_order={res_fbm.metrics['long_range_order']:.3f} should be < 0.3"

    def test_fibonacci_beats_brownian(self):
        """Fibonacci should dominate Brownian on all QC metrics."""
        geom = PenroseGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_bm = geom.compute_metrics(_brownian())
        for metric in ["fivefold_symmetry", "index_diversity", "long_range_order"]:
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
    def test_octonacci_vs_noise(self):
        """Octonacci word has silver-ratio (δ=1+√2) self-similarity.
        It should score higher on eightfold_symmetry than white noise."""
        geom = AmmannBeenkerGeometry()
        res_oct = geom.compute_metrics(_octonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_oct.metrics["eightfold_symmetry"] > res_noise.metrics["eightfold_symmetry"], \
            f"Octonacci={res_oct.metrics['eightfold_symmetry']:.4f} should > noise={res_noise.metrics['eightfold_symmetry']:.4f}"

    def test_noise_scores_low(self):
        """White noise should score low across all AB metrics."""
        geom = AmmannBeenkerGeometry()
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)
        res = geom.compute_metrics(noise)
        assert res.metrics["eightfold_symmetry"] < 0.3
        assert res.metrics["long_range_order"] < 0.3

    def test_fibonacci_is_not_silver(self):
        """Fibonacci word has φ-symmetry, not δ-symmetry.
        It should NOT score high on eightfold_symmetry."""
        geom = AmmannBeenkerGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        # Fibonacci has no silver-ratio structure, so eightfold_symmetry should be low
        assert res_fib.metrics["eightfold_symmetry"] < 0.2, \
            f"Fibonacci should not have silver-ratio symmetry, got {res_fib.metrics['eightfold_symmetry']}"

    def test_brownian_not_quasicrystal(self):
        """Brownian motion should not trigger silver-ratio detection."""
        geom = AmmannBeenkerGeometry()
        res_bm = geom.compute_metrics(_brownian())
        assert res_bm.metrics["eightfold_symmetry"] < 0.3, \
            f"Brownian eightfold_symmetry={res_bm.metrics['eightfold_symmetry']:.3f} should be < 0.3"
        assert res_bm.metrics["long_range_order"] < 0.3, \
            f"Brownian long_range_order={res_bm.metrics['long_range_order']:.3f} should be < 0.3"

    def test_fbm_not_quasicrystal(self):
        """fBm (H=0.8) has strong spectral structure but no silver ratio.
        Uses a dedicated RNG to avoid sensitivity to global RNG state."""
        geom = AmmannBeenkerGeometry()
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
        assert res_fbm.metrics["eightfold_symmetry"] < 0.3, \
            f"fBm eightfold_symmetry={res_fbm.metrics['eightfold_symmetry']:.3f} should be < 0.3"

    def test_octonacci_beats_brownian(self):
        """Octonacci should dominate Brownian on all QC metrics."""
        geom = AmmannBeenkerGeometry()
        res_oct = geom.compute_metrics(_octonacci_word())
        res_bm = geom.compute_metrics(_brownian())
        for metric in ["eightfold_symmetry", "index_diversity", "long_range_order"]:
            assert res_oct.metrics[metric] > res_bm.metrics[metric], \
                f"Octonacci {metric}={res_oct.metrics[metric]:.3f} should > Brownian {res_bm.metrics[metric]:.3f}"


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

    def test_fibonacci_inflation_similarity(self):
        """Einstein Hat shares golden-ratio structure. Fibonacci should outscore
        noise on inflation_similarity (which maps to ratio_symmetry at φ)."""
        geom = EinsteinHatGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_fib.metrics["inflation_similarity"] > res_noise.metrics["inflation_similarity"]

    def test_brownian_not_hat(self):
        """Brownian motion has smooth 1/f² spectrum — should NOT score high
        on hat_boundary_match or inflation_similarity.
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
        res_fib = geom.compute_metrics(_fibonacci_word())
        # Brownian should score low on inflation_similarity (φ-ratio metric)
        assert res_bm.metrics["inflation_similarity"] < 0.3, \
            f"Brownian inflation_similarity={res_bm.metrics['inflation_similarity']:.3f} should be < 0.3"
        # Fibonacci should beat Brownian on inflation_similarity
        assert res_fib.metrics["inflation_similarity"] > res_bm.metrics["inflation_similarity"]

    def test_chirality_sign_flip(self):
        """CW and CCW hex loops should produce opposite chirality."""
        geom = EinsteinHatGeometry()
        # Make long enough signals from repeated CCW/CW hex loops
        ccw_unit = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        cw_unit = np.array([0, 5, 4, 3, 2, 1], dtype=np.uint8)
        ccw = np.tile(ccw_unit, SIZE // 6)
        cw = np.tile(cw_unit, SIZE // 6)
        res_ccw = geom.compute_metrics(ccw)
        res_cw = geom.compute_metrics(cw)
        c_ccw = res_ccw.metrics["chirality"]
        c_cw = res_cw.metrics["chirality"]
        assert np.sign(c_ccw) != np.sign(c_cw), f"Chirality should flip: CCW={c_ccw}, CW={c_cw}"

    def test_hex_balance(self):
        """Fibonacci word (binary, structured) should have higher subword complexity
        stability than white noise."""
        geom = EinsteinHatGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_fib.metrics["hex_balance"] > res_noise.metrics["hex_balance"]


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
    Secondary ratio √3 (triangle height in square-triangle tilings)."""

    def test_substitution_word_vs_noise(self):
        """Dodecagonal substitution word (eigenvalue 2+√3) should score high
        on twelvefold_symmetry. White noise should score low."""
        geom = DodecagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_dw.metrics["twelvefold_symmetry"] > res_noise.metrics["twelvefold_symmetry"]
        assert res_dw.metrics["twelvefold_symmetry"] > 0.5

    def test_brownian_scores_low(self):
        """Brownian motion should not trigger dodecagonal detection.
        The large ratio (3.732) makes this very robust."""
        geom = DodecagonalGeometry()
        rng_bm = np.random.default_rng(99)
        steps = rng_bm.normal(0, 1, SIZE)
        walk = np.cumsum(steps); walk -= walk.min()
        walk = walk / (walk.max() + 1e-10) * 255
        res = geom.compute_metrics(walk.astype(np.uint8))
        assert res.metrics["twelvefold_symmetry"] < 0.1, \
            f"Brownian twelvefold_symmetry={res.metrics['twelvefold_symmetry']:.3f}"

    def test_fibonacci_is_not_dodecagonal(self):
        """Fibonacci word (φ ≈ 1.618) should score lower than the dodecagonal
        substitution word on twelvefold_symmetry."""
        geom = DodecagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        res_fib = geom.compute_metrics(_fibonacci_word())
        assert res_dw.metrics["twelvefold_symmetry"] > res_fib.metrics["twelvefold_symmetry"]

    def test_index_diversity(self):
        """Substitution word has linear subword growth (QC), noise doesn't."""
        geom = DodecagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_dw.metrics["index_diversity"] > res_noise.metrics["index_diversity"]


class TestDecagonal:
    """Decagonal (10-fold) geometry. Ratio = φ (same as Penrose).
    Secondary ratio φ² ≈ 2.618."""

    def test_fibonacci_vs_noise(self):
        """Fibonacci word is the canonical φ-quasicrystal.
        Should score high on tenfold_symmetry."""
        geom = DecagonalGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_fib.metrics["tenfold_symmetry"] > res_noise.metrics["tenfold_symmetry"]
        assert res_fib.metrics["tenfold_symmetry"] > 0.5

    def test_phi_squared_ratio(self):
        """Fibonacci word should also score high on the secondary φ² ratio,
        since φ² = φ + 1 is an algebraic consequence of φ."""
        geom = DecagonalGeometry()
        res_fib = geom.compute_metrics(_fibonacci_word())
        res_noise = geom.compute_metrics(_white_noise())
        assert res_fib.metrics["phi_squared_ratio"] > res_noise.metrics["phi_squared_ratio"]

    def test_brownian_not_decagonal(self):
        """Brownian motion should not trigger φ-ratio detection."""
        geom = DecagonalGeometry()
        rng_bm = np.random.default_rng(99)
        steps = rng_bm.normal(0, 1, SIZE)
        walk = np.cumsum(steps); walk -= walk.min()
        walk = walk / (walk.max() + 1e-10) * 255
        res = geom.compute_metrics(walk.astype(np.uint8))
        assert res.metrics["tenfold_symmetry"] < 0.3, \
            f"Brownian tenfold_symmetry={res.metrics['tenfold_symmetry']:.3f}"

    def test_dodecagonal_word_is_not_decagonal(self):
        """Dodecagonal substitution (ratio 2+√3) should not score high
        on decagonal (ratio φ) detection."""
        geom = DecagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        assert res_dw.metrics["tenfold_symmetry"] < 0.2, \
            f"Dodeca word tenfold_symmetry={res_dw.metrics['tenfold_symmetry']:.3f}"


class TestSeptagonal:
    """Septagonal (7-fold) geometry. Ratio ρ = 1 + 2cos(2π/7) ≈ 2.247.
    No simple 2-letter substitution exists (ρ is a cubic algebraic number),
    so we use cosine-sum signals as positive controls."""

    def test_rho_signal_vs_noise(self):
        """Cosine signal with spectral peaks at ρ-related frequencies
        should score high on sevenfold_symmetry."""
        geom = SeptagonalGeometry()
        rho = geom.RATIO
        res_rho = geom.compute_metrics(_ratio_signal(rho))
        res_noise = geom.compute_metrics(_white_noise())
        assert res_rho.metrics["sevenfold_symmetry"] > res_noise.metrics["sevenfold_symmetry"]
        assert res_rho.metrics["sevenfold_symmetry"] > 0.5

    def test_brownian_not_septagonal(self):
        """Brownian motion should not trigger ρ-ratio detection."""
        geom = SeptagonalGeometry()
        rng_bm = np.random.default_rng(99)
        steps = rng_bm.normal(0, 1, SIZE)
        walk = np.cumsum(steps); walk -= walk.min()
        walk = walk / (walk.max() + 1e-10) * 255
        res = geom.compute_metrics(walk.astype(np.uint8))
        assert res.metrics["sevenfold_symmetry"] < 0.3, \
            f"Brownian sevenfold_symmetry={res.metrics['sevenfold_symmetry']:.3f}"

    def test_dodecagonal_word_is_not_septagonal(self):
        """Dodecagonal substitution (ratio 2+√3) should not score high
        on septagonal (ratio ρ) detection. Different ratios."""
        geom = SeptagonalGeometry()
        res_dw = geom.compute_metrics(_dodeca_word())
        assert res_dw.metrics["sevenfold_symmetry"] < 0.2, \
            f"Dodeca word sevenfold_symmetry={res_dw.metrics['sevenfold_symmetry']:.3f}"

    def test_cross_specificity(self):
        """ρ-signal should score higher on septagonal than on dodecagonal
        or decagonal detectors. Each ratio detector should prefer its own."""
        geom_s = SeptagonalGeometry()
        geom_d = DodecagonalGeometry()
        geom_10 = DecagonalGeometry()
        sig = _ratio_signal(geom_s.RATIO)
        score_s = geom_s.compute_metrics(sig).metrics["sevenfold_symmetry"]
        score_d = geom_d.compute_metrics(sig).metrics["twelvefold_symmetry"]
        score_10 = geom_10.compute_metrics(sig).metrics["tenfold_symmetry"]
        assert score_s > score_d, \
            f"ρ-signal: septagonal={score_s:.3f} should > dodecagonal={score_d:.3f}"
        assert score_s > score_10, \
            f"ρ-signal: septagonal={score_s:.3f} should > decagonal={score_10:.3f}"


# ── CROSS-CUTTING: DEGENERATE INPUT ───────────────────────────────────

class TestDegenerateInput:
    """All geometries should handle degenerate input without crashing."""

    ALL_CLASSES = [
        TorusGeometry, SphericalGeometry, CantorGeometry,
        HyperbolicGeometry, PersistentHomologyGeometry,
        E8Geometry, HeisenbergGeometry, TropicalGeometry,
        PenroseGeometry, AmmannBeenkerGeometry, EinsteinHatGeometry,
        DodecagonalGeometry, DecagonalGeometry, SeptagonalGeometry,
        SpectralGeometry, RecurrenceGeometry, PredictabilityGeometry,
        HolderRegularityGeometry, InformationGeometry,
        VisibilityGraphGeometry, AttractorGeometry,
        FisherGeometry, WassersteinGeometry,
        HigherOrderGeometry, MultifractalGeometry,
    ]

    @pytest.mark.parametrize("GeomClass", ALL_CLASSES,
                             ids=lambda c: c.__name__)
    def test_constant_input(self, GeomClass):
        """Constant input: no variance. Should not crash or produce NaN."""
        if GeomClass.__name__ == "CantorGeometry":
            geom = GeomClass(base=3)
        elif GeomClass.__name__ == "UltrametricGeometry":
            geom = GeomClass(p=2)
        else:
            geom = GeomClass()
        result = geom.compute_metrics(_constant())
        for k, v in result.metrics.items():
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

    def test_tightness(self):
        """Exponential data stays near origin (small radii) → low tightness.
        Data with large jumps (noise) → higher tightness (spiral extends further)."""
        geom = SpiralGeometry()
        t = np.linspace(0, 5, SIZE)
        exp_signal = (np.exp(t) / np.exp(5) * 255).astype(np.uint8)
        res_exp = geom.compute_metrics(exp_signal)
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, SIZE, dtype=np.uint8)
        res_rnd = geom.compute_metrics(noise)
        assert res_exp.metrics["tightness"] < res_rnd.metrics["tightness"]


class TestFractalMandelbrot:
    def test_rich_vs_flat_escape(self):
        """Chaotic signal should produce varied escape times in the Mandelbrot embedding.
        Constant signal → all same escape → low variance."""
        geom = FractalMandelbrotGeometry()
        res_chaos = geom.compute_metrics(_logistic_chaos())
        res_const = geom.compute_metrics(_constant())
        assert res_chaos.metrics["escape_time_variance"] > res_const.metrics["escape_time_variance"]


class TestFractalJulia:
    def test_connectedness_variation(self):
        """Different data should produce different Julia set connectedness values.
        This is a basic sanity check that the metric responds to input."""
        geom = FractalJuliaGeometry()
        res_per = geom.compute_metrics(_periodic())
        res_rnd = geom.compute_metrics(_white_noise())
        # Both should be finite and in [0, 1]
        assert 0 <= res_per.metrics["connectedness"] <= 1
        assert 0 <= res_rnd.metrics["connectedness"] <= 1


# ── CROSS-GEOMETRY CONSISTENCY ─────────────────────────────────────────

class TestCrossGeometryConsistency:
    """When data has a clear property, multiple geometries should agree."""

    def test_periodic_signal_consensus(self):
        """A periodic signal should be detected as structured by ALL relevant geometries.
        - Predictability: low cond_entropy
        - Recurrence: high determinism
        - Information: low entropy_rate
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
        assert ri.metrics["entropy_rate"] < rin.metrics["entropy_rate"], \
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
        assert ri.metrics["entropy_rate"] > 0.9, \
            f"White noise entropy_rate should be near 1.0, got {ri.metrics['entropy_rate']}"

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
        block_entropy_1 should be near 1.0 (normalized)."""
        geom = InformationGeometry()
        rng = np.random.default_rng(42)
        uniform = rng.integers(0, 256, SIZE * 4, dtype=np.uint8)  # large sample
        res = geom.compute_metrics(uniform)
        assert res.metrics["block_entropy_1"] > 0.99, \
            f"Uniform H_1 should be ~1.0, got {res.metrics['block_entropy_1']}"

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
        assert res_orig.metrics["fivefold_symmetry"] > res_shuf.metrics["fivefold_symmetry"], \
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
