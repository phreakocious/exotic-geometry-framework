#!/usr/bin/env python3
"""
Mathematical Constants: Geometric Fingerprints of Transcendence
===============================================================

Do exotic geometries see structure in mathematical constants? The answer
depends entirely on *representation*. Decimal digits of Pi are i.i.d.
uniform — indistinguishable from noise. But continued fraction terms
encode deep algebraic/transcendental structure that the framework detects.

DIRECTIONS:
D1: Digit Taxonomy — Base-256 bytes of Pi, e, Phi, Sqrt(2) vs white noise.
      Are any distinguishable from random? From each other?
D2: CF Taxonomy — Continued fraction terms: patterned (e) vs chaotic (Pi)
      vs periodic (algebraic irrationals) vs all-ones (Phi). Pairwise.
D3: CF vs Null Models — Pi CF vs Gauss-Kuzmin i.i.d. (distribution-matched).
      If significant: genuine sequential structure in Pi's CF.
D4: Representation Fingerprints — Same constant (Pi), four representations
      (base-256 digits, CF terms, binary digits, base-10 digits).
      How different do they look to the framework?
D5: The Algebraic Boundary — Periodic CFs (sqrt(n)) vs patterned
      transcendentals (e) vs chaotic transcendentals (Pi). Phase transition?
"""

import sys
import time
import numpy as np
from pathlib import Path
from mpmath import mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
DATA_SIZE = 2000
N_TRIALS = 25

# High precision for mpmath — enough for base-256 digits
# CF terms use their own precision scaling
mp.dps = 120000

# ==============================================================
# DATA GENERATORS: BASE-256 DIGITS
# ==============================================================

def _constant_to_base256(mpf_val, n_bytes):
    """Convert an mpmath float to its base-256 digit expansion."""
    # Work with the fractional part
    val = mpf_val - int(mpf_val)
    if val < 0:
        val = -val
    digits = []
    for _ in range(n_bytes + 100):  # extra headroom
        val *= 256
        d = int(val)
        digits.append(d % 256)
        val -= d
    return digits

# Pre-cache base-256 expansions (enough for 25 trials × 12000 + offsets)
_CACHE_SIZE = N_TRIALS * DATA_SIZE * 2 + 10000
print("Pre-computing base-256 expansions...", end=" ", flush=True)
_t0 = time.time()
PI_B256 = _constant_to_base256(mp.pi, _CACHE_SIZE)
E_B256 = _constant_to_base256(mp.e, _CACHE_SIZE)
PHI_B256 = _constant_to_base256(mp.phi, _CACHE_SIZE)
SQRT2_B256 = _constant_to_base256(mp.sqrt(2), _CACHE_SIZE)
SQRT3_B256 = _constant_to_base256(mp.sqrt(3), _CACHE_SIZE)
SQRT5_B256 = _constant_to_base256(mp.sqrt(5), _CACHE_SIZE)
LN2_B256 = _constant_to_base256(mp.ln(2), _CACHE_SIZE)
print(f"{time.time() - _t0:.1f}s")


def _gen_base256(cached_digits):
    """Generator that samples windows from pre-cached base-256 digits."""
    def gen(rng, size):
        max_off = len(cached_digits) - size - 1
        off = rng.integers(0, max(1, max_off))
        return np.array(cached_digits[off:off + size], dtype=np.uint8)
    return gen

gen_pi_b256 = _gen_base256(PI_B256)
gen_e_b256 = _gen_base256(E_B256)
gen_phi_b256 = _gen_base256(PHI_B256)
gen_sqrt2_b256 = _gen_base256(SQRT2_B256)
gen_sqrt3_b256 = _gen_base256(SQRT3_B256)
gen_sqrt5_b256 = _gen_base256(SQRT5_B256)
gen_ln2_b256 = _gen_base256(LN2_B256)


# ==============================================================
# DATA GENERATORS: BASE-10 DIGITS (for D4 comparison)
# ==============================================================

PI_STR = str(mp.pi).replace('.', '')
E_STR = str(mp.e).replace('.', '')

def _gen_base10(const_str):
    def gen(rng, size):
        max_off = len(const_str) - size - 10
        off = rng.integers(0, max(1, max_off))
        chunk = const_str[off:off + size]
        return np.array([int(d) for d in chunk if d.isdigit()][:size],
                        dtype=np.uint8)
    return gen

gen_pi_b10 = _gen_base10(PI_STR)


# ==============================================================
# DATA GENERATORS: BINARY DIGITS (for D4 comparison)
# ==============================================================

def _constant_to_binary(mpf_val, n_bits):
    """Convert fractional part to binary digit sequence (0/1 as uint8)."""
    val = mpf_val - int(mpf_val)
    if val < 0:
        val = -val
    bits = []
    for _ in range(n_bits + 100):
        val *= 2
        b = int(val)
        bits.append(b)
        val -= b
    return bits

PI_BIN = _constant_to_binary(mp.pi, _CACHE_SIZE)

def gen_pi_binary(rng, size):
    max_off = len(PI_BIN) - size - 1
    off = rng.integers(0, max(1, max_off))
    return np.array(PI_BIN[off:off + size], dtype=np.uint8)


# ==============================================================
# DATA GENERATORS: CONTINUED FRACTIONS
# ==============================================================

def _compute_cf_terms(mpf_val, n_terms, extra_dps=2000):
    """Compute continued fraction terms with sufficient precision."""
    old_dps = mp.dps
    # Precision needed: each CF step loses roughly log10(a_k) digits.
    # For Gauss-Kuzmin (avg term ~3.4), budget ~1.5 digits/term + safety margin.
    # Use exactly this — do NOT inherit the high global dps from base-256 cache.
    needed = int(n_terms * 1.5) + extra_dps
    mp.dps = needed
    val = mpf_val
    terms = []
    curr = val
    for _ in range(n_terms):
        a = int(curr)
        terms.append(a)
        curr = curr - a
        if curr == 0:
            break
        curr = 1 / curr
    mp.dps = old_dps
    return terms

# Pre-cache CF terms — use a pool of 5000 terms (windows overlap, which is fine
# since each trial gets a different random offset). This is much cheaper than
# computing N_TRIALS * DATA_SIZE terms, especially for Pi where each CF step
# involves expensive mpmath division at high precision.
_CF_POOL = 5000
print("Pre-computing continued fractions...", end=" ", flush=True)
_t0 = time.time()
CF_PI = _compute_cf_terms(mp.pi, _CF_POOL)
CF_E = _compute_cf_terms(mp.e, _CF_POOL)
CF_PHI = _compute_cf_terms(mp.phi, _CF_POOL)  # all 1s
CF_SQRT2 = _compute_cf_terms(mp.sqrt(2), _CF_POOL)  # [1, 2, 2, 2, ...]
CF_SQRT3 = _compute_cf_terms(mp.sqrt(3), _CF_POOL)  # [1, 1, 2, 1, 2, ...]
CF_SQRT5 = _compute_cf_terms(mp.sqrt(5), _CF_POOL)  # [2, 4, 4, 4, ...]
CF_LN2 = _compute_cf_terms(mp.ln(2), _CF_POOL)
print(f"{time.time() - _t0:.1f}s")


def _gen_cf(cached_terms):
    """Generator that samples windows from pre-cached CF terms, mod 256."""
    def gen(rng, size):
        max_off = len(cached_terms) - size - 1
        if max_off < 1:
            max_off = 1
        off = rng.integers(0, max_off)
        chunk = cached_terms[off:off + size]
        return np.array([int(t) % 256 for t in chunk], dtype=np.uint8)
    return gen

gen_cf_pi = _gen_cf(CF_PI)
gen_cf_e = _gen_cf(CF_E)
gen_cf_phi = _gen_cf(CF_PHI)
gen_cf_sqrt2 = _gen_cf(CF_SQRT2)
gen_cf_sqrt3 = _gen_cf(CF_SQRT3)
gen_cf_sqrt5 = _gen_cf(CF_SQRT5)
gen_cf_ln2 = _gen_cf(CF_LN2)


def gen_gauss_kuzmin(rng, size):
    """Generate i.i.d. samples from the Gauss-Kuzmin distribution.
    The Gauss map invariant measure is mu([0,x]) = log2(1+x).
    Sample x from this measure via x = 2^U - 1, then k = floor(1/x).
    """
    u = rng.random(size)
    x = 2.0 ** u - 1.0       # x ~ Gauss measure on (0,1)
    x = np.maximum(x, 1e-15)  # avoid division by zero
    k = np.floor(1.0 / x).astype(np.int64)
    k = np.maximum(k, 1)
    return (k % 256).astype(np.uint8)


# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Base-256 Digit Taxonomy — constants vs each other and vs random."""
    print("\n" + "=" * 60)
    print("D1: BASE-256 DIGIT TAXONOMY")
    print("=" * 60)

    conditions = {}
    gens = [
        ("Pi", gen_pi_b256),
        ("e", gen_e_b256),
        ("Phi", gen_phi_b256),
        ("Sqrt2", gen_sqrt2_b256),
        ("White Noise", lambda rng, size: rng.integers(0, 256, size, dtype=np.uint8)),
    ]
    for name, gen_fn in gens:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_2(runner):
    """D2: CF Taxonomy — pairwise comparison of CF term sequences."""
    print("\n" + "=" * 60)
    print("D2: CONTINUED FRACTION TAXONOMY")
    print("=" * 60)

    conditions = {}
    gens = [
        ("CF(Pi)", gen_cf_pi),
        ("CF(e)", gen_cf_e),
        ("CF(ln2)", gen_cf_ln2),
        ("CF(Phi)", gen_cf_phi),       # all 1s
        ("CF(Sqrt2)", gen_cf_sqrt2),   # periodic [2,2,2,...]
        ("CF(Sqrt3)", gen_cf_sqrt3),   # periodic [1,2,1,2,...]
    ]
    for name, gen_fn in gens:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_3(runner):
    """D3: CF(Pi) vs Gauss-Kuzmin i.i.d. — is there sequential structure?"""
    print("\n" + "=" * 60)
    print("D3: CF(PI) vs GAUSS-KUZMIN NULL")
    print("=" * 60)

    # Real CF(Pi)
    pi_chunks = [gen_cf_pi(rng, runner.data_size) for rng in runner.trial_rngs()]
    pi_met = runner.collect(pi_chunks)

    # Gauss-Kuzmin i.i.d. (same marginal distribution, no sequential structure)
    gk_chunks = [gen_gauss_kuzmin(rng, runner.data_size)
                 for rng in runner.trial_rngs(offset=100)]
    gk_met = runner.collect(gk_chunks)

    # Shuffled CF(Pi) (destroys sequence, preserves exact distribution)
    shuf_chunks = runner.shuffle_chunks(pi_chunks)
    shuf_met = runner.collect(shuf_chunks)

    ns_gk, details_gk = runner.compare(pi_met, gk_met)
    ns_shuf, details_shuf = runner.compare(pi_met, shuf_met)

    print(f"  CF(Pi) vs Gauss-Kuzmin i.i.d. = {ns_gk:3d} sig")
    print(f"  CF(Pi) vs shuffled CF(Pi)      = {ns_shuf:3d} sig")

    # Also test CF(e) and CF(ln2) vs their shuffled versions
    results = {"Pi vs GK": ns_gk, "Pi vs Shuf": ns_shuf}
    for name, gen_fn in [("e", gen_cf_e), ("ln2", gen_cf_ln2)]:
        chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[f"{name} vs Shuf"] = ns
        print(f"  CF({name}) vs shuffled          = {ns:3d} sig")

    return dict(results=results, details_gk=details_gk)


def direction_4(runner):
    """D4: Representation Fingerprints — Pi in 4 representations."""
    print("\n" + "=" * 60)
    print("D4: REPRESENTATION FINGERPRINTS (PI)")
    print("=" * 60)

    conditions = {}
    gens = [
        ("Base-256", gen_pi_b256),
        ("Base-10", gen_pi_b10),
        ("Binary", gen_pi_binary),
        ("CF terms", gen_cf_pi),
    ]
    for name, gen_fn in gens:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_5(runner):
    """D5: The Algebraic Boundary — periodic CF (algebraic) vs transcendental."""
    print("\n" + "=" * 60)
    print("D5: ALGEBRAIC BOUNDARY")
    print("=" * 60)

    # Compare each CF to shuffled — how much sequential structure?
    results = {}
    gens = [
        ("CF(Sqrt2)", gen_cf_sqrt2),   # period 1: [2,2,2,...]
        ("CF(Sqrt3)", gen_cf_sqrt3),   # period 2: [1,2,1,2,...]
        ("CF(Sqrt5)", gen_cf_sqrt5),   # period 1: [4,4,4,...]
        ("CF(Phi)", gen_cf_phi),       # all 1s (simplest)
        ("CF(e)", gen_cf_e),           # patterned transcendental
        ("CF(Pi)", gen_cf_pi),         # chaotic transcendental
        ("CF(ln2)", gen_cf_ln2),       # chaotic transcendental
        ("Gauss-Kuzmin", gen_gauss_kuzmin),  # theoretical null
    ]
    for name, gen_fn in gens:
        chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name:20s} vs shuffled = {ns:3d} sig")

    return dict(results=results)


# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5,
        "Mathematical Constants: Fingerprints of Transcendence")

    # D1: Base-256 taxonomy
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'],
                        "D1: Base-256 Digits")

    # D2: CF taxonomy
    runner.plot_heatmap(axes[1], d2['matrix'], d2['names'],
                        "D2: CF Taxonomy")

    # D3: Pi CF vs null models
    names3 = list(d3['results'].keys())
    vals3 = [d3['results'][n] for n in names3]
    runner.plot_bars(axes[2], names3, vals3,
                     "D3: CF Sequential Structure")

    # D4: Representation fingerprints
    runner.plot_heatmap(axes[3], d4['matrix'], d4['names'],
                        "D4: Pi Representations")

    # D5: Algebraic boundary
    names5 = list(d5['results'].keys())
    vals5 = [d5['results'][n] for n in names5]
    runner.plot_bars(axes[4], names5, vals5,
                     "D5: CF vs Shuffled")

    runner.save(fig, "math_constants")


# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Math Constants", mode="1d", data_size=DATA_SIZE)

    print("=" * 60)
    print("MATHEMATICAL CONSTANTS: FINGERPRINTS OF TRANSCENDENCE")
    print(f"  size={runner.data_size}, trials={runner.n_trials}, "
          f"metrics={runner.n_metrics}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    runner.print_summary({
        'D1': f"Digits taxonomy: {d1['matrix'].min()}-{d1['matrix'].max()} sig pairwise",
        'D2': f"CF taxonomy: {d2['matrix'].min()}-{d2['matrix'].max()} sig pairwise",
        'D3': f"Pi CF vs GK: {d3['results']['Pi vs GK']}, "
              f"vs shuf: {d3['results']['Pi vs Shuf']}",
        'D4': f"Pi representations: {d4['matrix'].min()}-{d4['matrix'].max()} sig",
        'D5': f"CF(e)={d5['results']['CF(e)']}, CF(Pi)={d5['results']['CF(Pi)']}, "
              f"CF(Sqrt2)={d5['results']['CF(Sqrt2)']}"
    })


if __name__ == "__main__":
    main()
