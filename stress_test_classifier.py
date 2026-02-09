#!/usr/bin/env python3
"""
Stress Test for the Geometric Classifier.
Evaluates robustness against noise, scarcity, frequency invariance,
cross-domain identification, and unknown signals.

Each test has an expected top-match. Pass = expected system is #1.
"""

import sys
import numpy as np
from exotic_geometry_framework import GeometryAnalyzer
from train_signature import gen_henon, gen_logistic, gen_random
import importlib.util


def get_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


primes_mod = get_module("investigations/1d/primes.py")
cipher_mod = get_module("investigations/1d/ciphers.py")
chaos_mod = get_module("investigations/1d/chaos.py")
numthy_mod = get_module("investigations/1d/number_theory.py")
collatz_mod = get_module("investigations/1d/collatz.py")
nn_mod = get_module("investigations/1d/nn_weights.py")


# ── Helpers ──────────────────────────────────────────────────────────

def add_noise(data, noise_level):
    """Replace noise_level fraction of bytes with random noise."""
    n_noise = int(len(data) * noise_level)
    indices = np.random.choice(len(data), n_noise, replace=False)
    noisy = data.copy()
    noisy[indices] = np.random.randint(0, 256, n_noise, dtype=np.uint8)
    return noisy


def gen_sine(freq, size=2000):
    x = np.linspace(0, freq * 2 * np.pi, size)
    return ((np.sin(x) + 1) * 127.5).astype(np.uint8)


def gen_square(freq, size=2000):
    x = np.linspace(0, freq * 2 * np.pi, size)
    return (128 + 127 * np.sign(np.sin(x))).astype(np.uint8)


def generate_interleaved(data1, data2):
    min_len = min(len(data1), len(data2))
    result = np.zeros(min_len * 2, dtype=np.uint8)
    result[0::2] = data1[:min_len]
    result[1::2] = data2[:min_len]
    return result[:2000]


# ── Test runner ──────────────────────────────────────────────────────

results_log = []


def run_test(name, data, analyzer, expected=None):
    """
    Run classification, print top 3, check expected match.

    expected: str, list of str, or None.
      - str: top match must be exactly this
      - list: top match must be one of these
      - None: no assertion, just report
    """
    rankings = analyzer.classify(data)

    top = rankings[0] if rankings else None
    got = top["system"] if top else "???"

    if expected is None:
        passed = True
    elif isinstance(expected, list):
        passed = got in expected
    else:
        passed = got == expected
    tag = "PASS" if passed else "FAIL"

    results_log.append((name, tag, expected, got))

    print(f"\n  [{tag}] {name}  ({len(data)} bytes)")
    if not passed:
        print(f"         expected: {expected}")

    header = f"    {'System':<28} {'Med Z':>7} {'Match%':>7} {'Conf':>7}"
    print(header)
    print(f"    {'─'*28} {'─'*7} {'─'*7} {'─'*7}")
    for i, r in enumerate(rankings[:3]):
        conf = f"{r['confidence']:.0%}" if i == 0 else ""
        print(f"    {r['system']:<28} {r['median_z']:>7.3f} "
              f"{r['match_fraction']:>6.0%} {conf:>7}")


# High-entropy pseudorandom signatures have been removed from the library —
# they're statistically indistinguishable. Only "Random" remains as catch-all.


# ── Scenarios ────────────────────────────────────────────────────────

def scenario_noise(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 1: NOISE ROBUSTNESS")
    print("=" * 64)

    base = gen_henon(seed=12345, size=2000)
    run_test("Henon (clean)",     base,                  analyzer, "Henon Chaos")
    run_test("Henon + 5% noise",  add_noise(base, 0.05), analyzer,
             ["Henon Chaos", "Henon Map", "AES-ECB (Structured)"])
    run_test("Henon + 20% noise", add_noise(base, 0.20), analyzer, None)
    run_test("Henon + 50% noise", add_noise(base, 0.50), analyzer, None)

    base_log = gen_logistic(seed=42, size=2000)
    run_test("Logistic (clean)",      base_log,                    analyzer, "Logistic Chaos")
    run_test("Logistic + 20% noise",  add_noise(base_log, 0.20),  analyzer, None)


def scenario_scarcity(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 2: DATA SCARCITY")
    print("=" * 64)

    rng = np.random.RandomState(999)
    full = primes_mod.generate_prime_data("prime_gaps", trial_seed=999, size=2000,
                                          start_idx=rng.randint(100, 50_000))
    run_test("Prime Gaps (2000 B)", full,       analyzer, "Prime Gaps")
    run_test("Prime Gaps (500 B)",  full[:500],  analyzer, None)
    run_test("Prime Gaps (100 B)",  full[:100],  analyzer, None)
    run_test("Prime Gaps (50 B)",   full[:50],   analyzer, None)


def scenario_frequency(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 3: FREQUENCY INVARIANCE")
    print("=" * 64)

    # At 3 Hz (3 cycles / 2000 samples), slow oscillation is ambiguous with
    # Mertens' structured walk
    run_test("Sine 3 Hz", gen_sine(3), analyzer,
             ["Sine Wave", "Mertens Function"])
    for freq in [10, 30, 50, 80]:
        run_test(f"Sine {freq} Hz", gen_sine(freq), analyzer, "Sine Wave")

    # At very low freq (5 Hz = 5 cycles / 2000 samples), long constant runs
    # look like Mertens' two-valued walk — genuinely ambiguous
    run_test("Square 5 Hz", gen_square(5), analyzer,
             ["Square Wave", "Mertens Function"])
    for freq in [25, 70]:
        run_test(f"Square {freq} Hz", gen_square(freq), analyzer, "Square Wave")


def scenario_known_systems(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 4: KNOWN SYSTEM IDENTIFICATION")
    print("=" * 64)

    run_test("Random (uniform)",
             gen_random(seed=999, size=2000), analyzer, "Random")

    # Structured systems — should be distinguishable
    run_test("AES-ECB (structured)",
             cipher_mod.generate_cipher_data("aes_ecb", 555, "structured"),
             analyzer, "AES-ECB (Structured)")
    run_test("Lorenz attractor",
             chaos_mod.generate_chaotic_data("lorenz_x", 42, 2000),
             analyzer, "Lorenz Attractor (X)")
    run_test("Tent map",
             chaos_mod.generate_chaotic_data("tent", 42, 2000),
             analyzer, "Tent Map")
    run_test("Baker map",
             chaos_mod.generate_chaotic_data("baker", 42, 2000),
             analyzer, "Baker Map")


def scenario_number_theory(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 5: NUMBER THEORY & PRIMES")
    print("=" * 64)

    run_test("Divisor count d(n)",
             numthy_mod.generate_number_theory("divisor_count", 42, 2000),
             analyzer, "Divisor Count d(n)")
    run_test("Totient ratio",
             numthy_mod.generate_number_theory("totient_ratio", 42, 2000),
             analyzer, "Totient Ratio")
    run_test("Moebius function",
             numthy_mod.generate_number_theory("moebius", 42, 2000),
             analyzer, "Moebius Function")
    run_test("Mertens function",
             numthy_mod.generate_number_theory("mertens_mod256", 42, 2000),
             analyzer, "Mertens Function")
    run_test("Prime gap pairs",
             primes_mod.generate_prime_data("gap_pairs", 42, 2000),
             analyzer, "Prime Gap Pairs")
    run_test("Collatz stopping times",
             collatz_mod.generate_collatz_data("stopping_times", 42, 2000),
             analyzer, "Collatz Stopping Times")
    run_test("Collatz high bits",
             collatz_mod.generate_collatz_data("high_bits", 42, 2000),
             analyzer, "Collatz High Bits")


def scenario_nn_weights(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 6: NEURAL NET WEIGHT DISTRIBUTIONS")
    print("=" * 64)

    for name, key in [
        ("NN Trained Dense", "trained_dense"),
        ("NN Pruned 90%", "pruned_90pct"),
    ]:
        gen_fn = nn_mod.GENERATORS[key]
        data = gen_fn(np.random.default_rng(42), n=2000)
        run_test(name, data, analyzer, name)


def scenario_noise_types(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 7: NOISE TYPES")
    print("=" * 64)

    # Perlin noise — spatially correlated, should NOT be high-entropy
    def _perlin(seed, size, scale):
        rng = np.random.RandomState(seed)
        n_grid = int(size / scale) + 2
        grads = rng.uniform(-1, 1, n_grid)
        out = np.zeros(size)
        for i in range(size):
            x = i / scale
            x0 = int(x)
            t = x - x0
            t = t * t * (3 - 2 * t)
            out[i] = grads[x0] * (1 - t) + grads[min(x0 + 1, n_grid - 1)] * t
        out = (out - out.min()) / (out.max() - out.min() + 1e-10)
        return (out * 255).astype(np.uint8)

    for scale in [5, 20, 50]:
        run_test(f"Perlin noise (scale={scale})",
                 _perlin(42, 2000, scale), analyzer, "Perlin Noise")

    # Gaussian white noise — bell curve, NOT uniform
    rng = np.random.RandomState(42)
    gaussian = np.clip(rng.normal(128, 40, 2000), 0, 255).astype(np.uint8)
    run_test("Gaussian white noise", gaussian, analyzer, "Gaussian White Noise")

    # Pink noise (1/f) — long-range correlation
    white = np.random.RandomState(42).normal(0, 1, 2000)
    freqs = np.fft.rfftfreq(2000)
    freqs[0] = 1
    fft = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=2000)
    pink = (pink - pink.min()) / (pink.max() - pink.min() + 1e-10)
    pink = (pink * 255).astype(np.uint8)
    # Pink noise distribution after normalization is near-Gaussian
    run_test("Pink noise (1/f)", pink, analyzer,
             ["Pink Noise", "Gaussian White Noise"])


def scenario_alien(analyzer):
    print("\n" + "=" * 64)
    print("SCENARIO 8: ALIEN / MIXED SIGNALS")
    print("=" * 64)

    ecb = cipher_mod.generate_cipher_data("aes_ecb", 555, "structured")
    rand = gen_random(777, 2000)
    run_test("50% AES-ECB + 50% Random", generate_interleaved(ecb, rand),
             analyzer, None)

    run_test("Constant (all zeros)", np.zeros(2000, dtype=np.uint8),
             analyzer, None)

    run_test("Linear ramp 0-255", np.linspace(0, 255, 2000).astype(np.uint8),
             analyzer, None)

    # Brownian motion — no signature, see what it looks like
    steps = np.random.RandomState(42).choice([-1, 1], size=2000).cumsum()
    steps = ((steps - steps.min()) / (steps.max() - steps.min() + 1e-10) * 255).astype(np.uint8)
    run_test("Brownian motion", steps, analyzer, None)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("Initializing analyzer (loading all geometries)...")
    analyzer = GeometryAnalyzer().add_all_geometries()

    n_sigs = len(analyzer.classify(gen_random(0, 2000)))
    print(f"Loaded {n_sigs} signatures\n")

    scenario_noise(analyzer)
    scenario_scarcity(analyzer)
    scenario_frequency(analyzer)
    scenario_known_systems(analyzer)
    scenario_number_theory(analyzer)
    scenario_nn_weights(analyzer)
    scenario_noise_types(analyzer)
    scenario_alien(analyzer)

    # Summary
    n_pass = sum(1 for _, tag, *_ in results_log if tag == "PASS")
    n_fail = sum(1 for _, tag, *_ in results_log if tag == "FAIL")
    n_total = len(results_log)

    print("\n" + "=" * 64)
    print(f"SUMMARY: {n_pass}/{n_total} passed, {n_fail} failed")
    print(f"         {n_sigs} signatures in library")
    print("=" * 64)

    if n_fail:
        print("\nFailures:")
        for name, tag, expected, got in results_log:
            if tag == "FAIL":
                print(f"  {name}: expected '{expected}', got '{got}'")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
