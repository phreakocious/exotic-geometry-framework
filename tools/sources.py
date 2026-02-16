"""
Unified source registry for the Exotic Geometry Framework.

Every self-contained data generator lives here.  File-based generators
(bearing, ECG, EEG, etc.) stay in their investigation scripts and call
register() to join the registry at import time.

Canonical generator signature:
    (rng: np.random.Generator, size: int) -> np.ndarray[uint8]

Usage:
    from tools.sources import get_sources, register, seed_adapter, DOMAIN_COLORS

    # All atlas sources (self-contained + file-based after their module loads)
    for s in get_sources(atlas=True):
        data = s.gen_fn(rng, 16384)

    # For train_signature.py (needs (seed, size) signature)
    for s in get_sources(signature=True):
        fn = seed_adapter(s.gen_fn)
        data = fn(42, 2000)
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional


# ================================================================
# Registry infrastructure
# ================================================================


@dataclass
class Source:
    name: str
    gen_fn: Callable  # (rng, size) -> uint8[]
    domain: str
    description: str = ""
    atlas: bool = True
    signature: bool = True


_REGISTRY: List[Source] = []


def source(name, domain, description="", atlas=True, signature=True):
    """Decorator that registers a generator function."""

    def decorator(fn):
        _REGISTRY.append(
            Source(
                name=name,
                gen_fn=fn,
                domain=domain,
                description=description,
                atlas=atlas,
                signature=signature,
            )
        )
        return fn

    return decorator


def register(name, gen_fn, domain, description="", atlas=True, signature=True):
    """Imperative registration for generators defined elsewhere."""
    _REGISTRY.append(
        Source(
            name=name,
            gen_fn=gen_fn,
            domain=domain,
            description=description,
            atlas=atlas,
            signature=signature,
        )
    )


def get_sources(atlas=None, signature=None):
    """Filter registry.  None = no filter on that flag."""
    result = _REGISTRY
    if atlas is not None:
        result = [s for s in result if s.atlas == atlas]
    if signature is not None:
        result = [s for s in result if s.signature == signature]
    return result


def seed_adapter(gen_fn):
    """Wrap (rng, size) -> (seed, size) for train_signature.py."""

    def adapted(seed, size):
        rng = np.random.default_rng(seed)
        return gen_fn(rng, size)

    return adapted


# ================================================================
# Domain colours (shared by atlas figures and CLI)
# ================================================================

DOMAIN_COLORS = {
    "chaos": "#e74c3c",
    "number_theory": "#3498db",
    "noise": "#95a5a6",
    "waveform": "#2ecc71",
    "bearing": "#f39c12",
    "binary": "#9b59b6",
    "bio": "#1abc9c",
    "medical": "#e84393",
    "financial": "#fdcb6e",
    "motion": "#6c5ce7",
    "astro": "#fd79a8",
    "climate": "#74b9ff",
    "speech": "#a29bfe",
    "exotic": "#ff6b6b",
    "geophysics": "#a52a2a",
}


# ================================================================
# Shared helpers
# ================================================================

_PRIMES = None


def _get_primes():
    global _PRIMES
    if _PRIMES is None:
        limit = 2_000_000
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i * i :: i] = False
        _PRIMES = np.where(sieve)[0]
    return _PRIMES


def _to_uint8(signal):
    """Normalize a float array to uint8 [0, 255]."""
    lo, hi = signal.min(), signal.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((signal - lo) / (hi - lo) * 255).astype(np.uint8)


# ================================================================
# Self-contained generators --- from structure_atlas.py
# ================================================================

# --- Chaos ---


@source(
    "Logistic Chaos",
    domain="chaos",
    description="The simplest chaotic system --- one-line recurrence x(n+1) = 4x(1-x) at r=4, "
    "filling the unit interval ergodically with a beta(½,½) distribution",
)
def gen_logistic(rng, size):
    x = 0.1 + 0.8 * rng.random()
    for _ in range(1000):
        x = 4.0 * x * (1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 4.0 * x * (1.0 - x)
        vals[i] = int(x * 255)
    return vals


@source(
    "Henon Map",
    domain="chaos",
    description="Classic 2D chaotic map --- the strange attractor has fractal cross-sections "
    "and correlation dimension ~1.25, a benchmark for testing dimension estimators",
)
def gen_henon(rng, size):
    x, y = 0.1 * rng.random(), 0.1 * rng.random()
    for _ in range(1000):
        x, y = 1.0 - 1.4 * x**2 + y, 0.3 * x
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x, y = 1.0 - 1.4 * x**2 + y, 0.3 * x
        vals[i] = int(np.clip((x + 1.5) / 3.0 * 255, 0, 255))
    return vals


@source(
    "Tent Map",
    domain="chaos",
    description="Piecewise-linear chaos --- a V-shaped map near slope 2, "
    "producing uniform invariant density with maximal topological entropy",
)
def gen_tent(rng, size):
    x = rng.random()
    for _ in range(500):
        x = 1.999 * min(x, 1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 1.999 * min(x, 1.0 - x)
        vals[i] = int(x * 255)
    return vals


@source(
    "Lorenz Attractor",
    domain="chaos",
    description="Lorenz '63 system x-coordinate --- the butterfly attractor, sensitive dependence on initial conditions. "
    "Two-lobe structure with unpredictable switching",
)
def gen_lorenz(rng, size):
    x, y, z = 0.1 * rng.standard_normal(3)
    dt = 0.01
    for _ in range(5000):
        dx = 10.0 * (y - x) * dt
        dy = (x * (28.0 - z) - y) * dt
        dz = (x * y - (8.0 / 3.0) * z) * dt
        x, y, z = x + dx, y + dy, z + dz
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        dx = 10.0 * (y - x) * dt
        dy = (x * (28.0 - z) - y) * dt
        dz = (x * y - (8.0 / 3.0) * z) * dt
        x, y, z = x + dx, y + dy, z + dz
        vals[i] = int(np.clip((x + 25) / 50 * 255, 0, 255))
    return vals


@source(
    "Rossler Attractor",
    domain="chaos",
    description="Rossler system x-coordinate --- the simplest 3D continuous chaotic flow, "
    "a single folded band with period-doubling route to chaos",
)
def gen_rossler(rng, size):
    x, y, z = 0.1 * rng.standard_normal(3)
    dt = 0.02
    a, b, c = 0.2, 0.2, 5.7
    for _ in range(5000):
        dx = (-y - z) * dt
        dy = (x + a * y) * dt
        dz = (b + z * (x - c)) * dt
        x, y, z = x + dx, y + dy, z + dz
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        dx = (-y - z) * dt
        dy = (x + a * y) * dt
        dz = (b + z * (x - c)) * dt
        x, y, z = x + dx, y + dy, z + dz
        vals[i] = int(np.clip((x + 12) / 24 * 255, 0, 255))
    return vals


# --- Number theory ---


@source(
    "Prime Gaps",
    domain="number_theory",
    description="Gaps between consecutive primes --- locally erratic but statistically governed by the "
    "prime number theorem, with conjectured connections to random matrix theory",
)
def gen_prime_gaps(rng, size):
    primes = _get_primes()
    max_start = len(primes) - size - 100
    start = rng.integers(0, max_start)
    gaps = np.diff(primes[start : start + size + 1])
    return np.clip(gaps, 0, 255).astype(np.uint8)


@source(
    "Collatz Trajectory",
    domain="number_theory",
    description="Collatz trajectory mod 256, restarting from consecutive integers "
    "to avoid 4-2-1 cycle collapse",
)
def gen_collatz(rng, size):
    start = int(rng.integers(10_000, 10_000_000))
    vals = np.empty(size, dtype=np.uint8)
    idx = 0
    n_start = start
    while idx < size:
        n = n_start
        # Follow trajectory until it drops below starting value or hits cycle
        while idx < size and n >= 2:
            vals[idx] = n % 256
            idx += 1
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            if n <= 4:  # approaching cycle, restart
                break
        n_start += 1
    return vals


@source(
    "Divisor Count",
    domain="number_theory",
    description="Divisor count d(n) --- erratic arithmetic function averaging ~ln(n), "
    "spikes at highly composite numbers, modulated by prime factorization structure",
)
def gen_divisor_count(rng, size):
    start = int(rng.integers(10_000, 500_000))
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        n = start + i
        d = 0
        for j in range(1, int(n**0.5) + 1):
            if n % j == 0:
                d += 2 if j != n // j else 1
        vals[i] = min(d, 255)
    return vals


@source(
    "Pi Digits",
    domain="number_theory",
    description="Digits of pi in base 256 --- conjectured normal (equidistributed), "
    "but entirely deterministic. Passes most statistical randomness tests",
)
def gen_pi_digits(rng, size):
    try:
        from mpmath import mp

        # Generate extra digits so trials can window into different positions
        n_total = size * 4
        mp.dps = n_total * 4 + 100
        pi_str = mp.nstr(mp.pi, n_total * 3 + 50, strip_zeros=False)
        digits = pi_str.replace(".", "")[1:]
        vals = []
        for i in range(0, len(digits) - 1, 3):
            chunk = digits[i : i + 3]
            if len(chunk) == 3:
                vals.append(int(chunk) % 256)
            if len(vals) >= n_total:
                break
        all_vals = np.array(vals[:n_total], dtype=np.uint8)
        start = int(rng.integers(0, max(1, len(all_vals) - size)))
        return all_vals[start : start + size]
    except ImportError:
        return rng.integers(0, 256, size, dtype=np.uint8)


# --- Noise ---


@source(
    "White Noise",
    domain="noise",
    description="IID uniform random bytes --- the null model, maximum entropy with zero sequential correlation",
)
def gen_white_noise(rng, size):
    return rng.integers(0, 256, size, dtype=np.uint8)


@source(
    "Blue Noise",
    domain="noise",
    description="High-pass filtered noise (PSD ~ f) --- anti-correlated increments that suppress low frequencies, "
    "common in error-diffusion dithering and spatial statistics",
)
def gen_blue_noise(rng, size):
    white = rng.standard_normal(size)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(size)
    freqs[0] = 1e-10
    fft *= np.sqrt(freqs)
    blue = np.fft.irfft(fft, n=size)
    return _to_uint8(blue)


@source(
    "Pink Noise",
    domain="noise",
    description="1/f noise (PSD ~ 1/f) --- equal energy per octave, ubiquitous in nature from "
    "heartbeat intervals to semiconductor flicker noise. Between white noise and Brownian motion",
)
def gen_pink_noise(rng, size):
    white = rng.standard_normal(size)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(size)
    freqs[0] = 1.0
    fft /= np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=size)
    pink = (pink - pink.min()) / (pink.max() - pink.min() + 1e-15) * 255
    return pink.astype(np.uint8)


@source(
    "Brownian Walk",
    domain="noise",
    description="Cumulative sum of Gaussian increments --- the canonical random walk, "
    "self-affine with Hurst exponent H=0.5 and PSD ~ 1/f²",
)
def gen_brownian(rng, size):
    steps = rng.standard_normal(size)
    walk = np.cumsum(steps)
    walk = (walk - walk.min()) / (walk.max() - walk.min() + 1e-15) * 255
    return walk.astype(np.uint8)


def _gen_fbm(H):
    """Factory for fractional Brownian motion generators."""

    def gen(rng, size):
        freqs = np.fft.rfftfreq(size, d=1.0)
        freqs[0] = 1.0
        psd = freqs ** (-(2 * H + 1))
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        fft_vals = np.sqrt(psd) * np.exp(1j * phases)
        fft_vals[0] = 0
        signal = np.fft.irfft(fft_vals, n=size)
        signal = np.cumsum(signal)
        return _to_uint8(signal)

    return gen


_fbm_03 = _gen_fbm(0.3)
_fbm_07 = _gen_fbm(0.7)
register(
    "fBm (Antipersistent)",
    _fbm_03,
    "noise",
    "Fractional Brownian motion with H=0.3 --- antipersistent (trend-reversing), rougher than standard Brownian motion",
)
register(
    "fBm (Persistent)",
    _fbm_07,
    "noise",
    "Fractional Brownian motion with H=0.7 --- persistent (trend-following), smoother trajectories with long-range dependence",
)


@source(
    "ARMA(2,1)",
    domain="noise",
    description="Autoregressive moving-average process --- linear short-memory dynamics, "
    "the workhorse model of classical time-series analysis",
)
def gen_arma(rng, size):
    ar = np.array([0.7, -0.2])
    ma = np.array([0.5])
    noise = rng.standard_normal(size + 100)
    x = np.zeros(size + 100)
    for t in range(2, len(x)):
        x[t] = ar[0] * x[t - 1] + ar[1] * x[t - 2] + noise[t] + ma[0] * noise[t - 1]
    x = x[100:]
    return _to_uint8(x)


@source(
    "Regime Switching",
    domain="noise",
    description="Two-state hidden Markov process --- alternates between calm (σ=0.5) and volatile (σ=3.0) regimes "
    "with asymmetric transition rates",
)
def gen_regime_switch(rng, size):
    state = 0
    vals = np.zeros(size)
    x = 0.0
    for i in range(size):
        if state == 0:
            x += rng.standard_normal() * 0.5
            if rng.random() < 0.01:
                state = 1
        else:
            x += rng.standard_normal() * 3.0
            if rng.random() < 0.05:
                state = 0
        vals[i] = x
    return _to_uint8(vals)


# --- Waveforms ---


@source(
    "Sine Wave",
    domain="waveform",
    description="Pure sinusoid --- perfectly periodic, single-frequency, minimal complexity. "
    "The baseline for all oscillatory comparisons",
)
def gen_sine(rng, size):
    freq = rng.uniform(5, 50)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, 1, size)
    wave = np.sin(2 * np.pi * freq * t + phase)
    return ((wave + 1) / 2 * 255).astype(np.uint8)


@source(
    "Sawtooth Wave",
    domain="waveform",
    description="Sawtooth wave --- linear ramp with discontinuous reset, rich in harmonics (1/n Fourier decay)",
)
def gen_sawtooth(rng, size):
    freq = rng.uniform(5, 50)
    t = np.linspace(0, 1, size)
    wave = 2 * (t * freq - np.floor(t * freq + 0.5))
    return ((wave + 1) / 2 * 255).astype(np.uint8)


# --- Binary / encrypted ---


def _gen_file_binary(path, label):
    """Create a generator that reads from a fixed binary file."""

    def gen(rng, size):
        from pathlib import Path

        p = Path(__file__).resolve().parents[1] / path
        if not p.exists():
            return rng.integers(0, 256, size, dtype=np.uint8)
        data = p.read_bytes()
        if len(data) <= size:
            arr = np.frombuffer(data, dtype=np.uint8).copy()
            if len(arr) < size:
                arr = np.pad(arr, (0, size - len(arr)), mode="wrap")
            return arr
        start = rng.integers(0, len(data) - size)
        return np.frombuffer(data[start : start + size], dtype=np.uint8).copy()

    gen.__name__ = f"gen_{label}"
    return gen


register(
    "Linux ELF x86-64",
    _gen_file_binary("data/binary/linux_elf_lsb_x86-64", "linux_elf"),
    "binary",
    "Stripped Linux x86-64 PIE executable (3.7 MB) --- ELF with .text, .rodata, "
    "relocation tables, and x86-64 instruction byte patterns",
)
register(
    "OpenBSD ELF x86-64",
    _gen_file_binary("data/binary/openbsd_elf_lsb_x86-64", "openbsd_elf"),
    "binary",
    "Stripped OpenBSD x86-64 PIE executable (1.1 MB) --- ELF with W^X enforcement, "
    "subtly different instruction mix from Linux due to compiler and ABI differences",
)
register(
    "Windows PE x86-64",
    _gen_file_binary("data/binary/windows_pe32_x86-64", "windows_pe"),
    "binary",
    "Windows PE32+ console executable (2.3 MB) --- x86-64 with PE headers, "
    "import tables, and different code generation patterns from Unix compilers",
)
_MACHO_CODE = None


def _get_macho_code():
    """Extract __TEXT,__text sections from all 3 architectures in the Mach-O fat binary."""
    global _MACHO_CODE
    if _MACHO_CODE is not None:
        return _MACHO_CODE
    from pathlib import Path
    import struct

    p = Path(__file__).resolve().parents[1] / "data" / "binary" / "macos_mach-o_universal"
    if not p.exists():
        _MACHO_CODE = np.array([], dtype=np.uint8)
        return _MACHO_CODE
    data = p.read_bytes()
    magic = struct.unpack(">I", data[:4])[0]
    if magic != 0xCAFEBABE:
        _MACHO_CODE = np.frombuffer(data, dtype=np.uint8).copy()
        return _MACHO_CODE
    code = b""
    narch = struct.unpack(">I", data[4:8])[0]
    for i in range(narch):
        off = 8 + i * 20
        _, _, arch_offset, arch_size, _ = struct.unpack(">5I", data[off : off + 20])
        sl = data[arch_offset : arch_offset + arch_size]
        mh_magic = struct.unpack("<I", sl[:4])[0]
        is64 = mh_magic == 0xFEEDFACF
        if is64:
            ncmds = struct.unpack("<I", sl[16:20])[0]
            pos = 32
        else:
            ncmds = struct.unpack("<I", sl[12:16])[0]
            pos = 28
        for _ in range(ncmds):
            cmd, cmdsize = struct.unpack("<II", sl[pos : pos + 8])
            if (cmd == 0x19 and is64) or (cmd == 1 and not is64):
                segname = sl[pos + 8 : pos + 24].split(b"\x00")[0]
                if segname == b"__TEXT":
                    if is64:
                        nsects = struct.unpack("<I", sl[pos + 64 : pos + 68])[0]
                        sect_off = pos + 72
                        sect_size_struct = 80
                    else:
                        nsects = struct.unpack("<I", sl[pos + 48 : pos + 52])[0]
                        sect_off = pos + 56
                        sect_size_struct = 68
                    for _ in range(nsects):
                        sectname = sl[sect_off : sect_off + 16].split(b"\x00")[0]
                        if sectname == b"__text":
                            if is64:
                                _, sz, soff = struct.unpack(
                                    "<QII", sl[sect_off + 32 : sect_off + 48]
                                )
                            else:
                                _, sz, soff = struct.unpack(
                                    "<III", sl[sect_off + 32 : sect_off + 44]
                                )
                            code += sl[soff : soff + sz]
                        sect_off += sect_size_struct
            pos += cmdsize
    _MACHO_CODE = (
        np.frombuffer(code, dtype=np.uint8).copy() if code else np.array([], dtype=np.uint8)
    )
    return _MACHO_CODE


def _gen_macho_code(rng, size):
    code = _get_macho_code()
    if len(code) == 0:
        return rng.integers(0, 256, size, dtype=np.uint8)
    max_start = max(0, len(code) - size)
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    chunk = code[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="wrap")
    return chunk.copy()


register(
    "macOS Mach-O (dyld)",
    _gen_macho_code,
    "binary",
    "macOS dyld universal binary (2.6 MB) --- i386 + x86-64 + arm64e __text sections, pure machine code",
)


@source(
    "AES Encrypted",
    domain="binary",
    description="Simulated AES-CTR ciphertext --- keyed PRNG output indistinguishable from random "
    "without the key, the gold standard for pseudorandom streams",
)
def gen_aes_encrypted(rng, size):
    key = rng.integers(0, 2**32)
    cipher_rng = np.random.default_rng(key)
    return cipher_rng.integers(0, 256, size, dtype=np.uint8)


def _gen_file_binary_center(path, label):
    """Create a generator that reads from the center of a binary file (avoids headers/footers)."""

    def gen(rng, size):
        from pathlib import Path

        p = Path(__file__).resolve().parents[1] / path
        if not p.exists():
            return rng.integers(0, 256, size, dtype=np.uint8)
        data = p.read_bytes()
        if len(data) <= size:
            arr = np.frombuffer(data, dtype=np.uint8).copy()
            if len(arr) < size:
                arr = np.pad(arr, (0, size - len(arr)), mode="wrap")
            return arr
        # Sample from the middle 80% to avoid format headers and footers
        margin = len(data) // 10
        lo = margin
        hi = len(data) - margin - size
        if hi <= lo:
            lo, hi = 0, len(data) - size
        start = rng.integers(lo, max(lo + 1, hi))
        return np.frombuffer(data[start : start + size], dtype=np.uint8).copy()

    gen.__name__ = f"gen_{label}"
    return gen


register(
    "Gzip (level 1)",
    _gen_file_binary_center("data/binary/gzip_level_1", "gzip1"),
    "binary",
    "Gzip-compressed data (1.4 MB, level 1) --- fast DEFLATE with more residual structure",
)
register(
    "Gzip (level 9)",
    _gen_file_binary_center("data/binary/gzip_level_9", "gzip9"),
    "binary",
    "Gzip-compressed data (1.3 MB, level 9) --- maximum DEFLATE compression, near-entropy stream",
)
register(
    "Bzip2 (level 1)",
    _gen_file_binary_center("data/binary/bzip2_level_1", "bzip2_1"),
    "binary",
    "Bzip2-compressed data (1.2 MB, level 1, 100k blocks) --- Burrows-Wheeler transform with "
    "small block size preserves more local structure from the original data",
)
register(
    "Bzip2 (level 9)",
    _gen_file_binary_center("data/binary/bzip2_level_9", "bzip2_9"),
    "binary",
    "Bzip2-compressed data (1.1 MB, level 9, 900k blocks) --- BWT with maximum block size, "
    "approaching the entropy floor of the source data",
)


# --- Bio (synthetic) ---


# --- Exotic structure (7D breakers + adversarial) ---


@source(
    "Devil's Staircase",
    domain="exotic",
    description="Cantor function --- continuous and monotone but with derivative zero almost everywhere. "
    "A singular measure concentrated on a fractal set of measure zero",
)
def gen_devils_staircase(rng, size):
    t = np.linspace(0, 1, size)
    vals = np.zeros(size, dtype=np.float64)
    for i, x in enumerate(t):
        result = 0.0
        power = 0.5
        x = (x + rng.uniform(0, 0.01)) % 1.0
        for _ in range(30):
            digit = int(x * 3)
            if digit == 1:
                result += power
                break
            elif digit == 0:
                x = x * 3
            else:
                result += power
                x = x * 3 - 2
            power *= 0.5
        vals[i] = result
    return (vals * 255).astype(np.uint8)


@source(
    "Levy Flight",
    domain="exotic",
    description="Random walk with Cauchy-distributed jumps --- infinite variance, "
    "occasional extreme leaps create fractal clustering of visited sites",
)
def gen_levy_flight(rng, size):
    # Clip increments to avoid degenerate quantization from extreme Cauchy tails
    increments = rng.standard_cauchy(size)
    increments = np.clip(increments, -100, 100)
    walk = np.cumsum(increments)
    lo, hi = np.percentile(walk, [1, 99])
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((walk - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


@source(
    "Thue-Morse",
    domain="exotic",
    description="Binary substitution sequence (0→01, 1→10) --- perfectly balanced yet aperiodic, "
    "with flat Fourier spectrum like random noise despite being entirely deterministic",
)
def gen_thue_morse(rng, size):
    n_total = size * 4
    vals = np.zeros(n_total, dtype=np.uint8)
    for i in range(n_total):
        n = i
        bits = 0
        while n:
            bits += n & 1
            n >>= 1
        vals[i] = 255 if (bits % 2) else 0
    start = int(rng.integers(0, max(1, n_total - size)))
    return vals[start : start + size]


@source(
    "Fibonacci Word",
    domain="exotic",
    description="Sturmian sequence --- the simplest aperiodic word, with golden-ratio quasiperiodicity "
    "and complexity function p(n) = n + 1 (minimal for non-periodic sequences)",
)
def gen_fibonacci_word(rng, size):
    n_total = size * 4
    a, b = "1", "0"
    word = a
    while len(word) < n_total + 100:
        word_new = ""
        for c in word:
            word_new += "10" if c == "1" else "1"
        word = word_new
    word = word[:n_total]
    vals = np.array([255 if c == "1" else 0 for c in word], dtype=np.uint8)
    start = int(rng.integers(0, max(1, len(vals) - size)))
    return vals[start : start + size]


@source(
    "Hilbert Walk",
    domain="exotic",
    description="1D trace of Hilbert space-filling curve --- preserves 2D locality, "
    "creating structured self-similar fluctuations at every scale",
)
def gen_hilbert_walk(rng, size):
    order = int(np.ceil(np.log2(np.sqrt(size)))) + 1

    def d2xy(n, d):
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (d & 2) else 0
            ry = 1 if ((d & 1) ^ rx) else 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y

    n_grid = 2**order
    n_points = 4**order
    offset = int(rng.integers(0, max(1, n_points - size)))
    xs = np.zeros(size, dtype=np.float64)
    for i in range(size):
        x, y = d2xy(n_grid, offset + i)
        xs[i] = x
    return _to_uint8(xs)


@source(
    "Random Steps",
    domain="exotic",
    description="Random piecewise-constant signal --- flat plateaus of random height with geometrically-distributed durations, "
    "creating a sparse derivative and blocky texture",
)
def gen_step_function(rng, size):
    # Geometric step lengths: mean ~50 samples, giving ~300 steps for 16384
    vals = np.empty(size, dtype=np.uint8)
    idx = 0
    while idx < size:
        level = rng.integers(0, 256)
        length = min(rng.geometric(p=1 / 50), size - idx)
        vals[idx : idx + length] = level
        idx += length
    return vals


@source(
    "Weierstrass",
    domain="exotic",
    description="Weierstrass function --- continuous everywhere but differentiable nowhere, "
    "the first known mathematical 'monster'. Self-similar roughness at all scales",
)
def gen_weierstrass(rng, size):
    a = 0.5
    b = 7
    t = np.linspace(0, 4 * np.pi, size)
    phase = rng.uniform(0, 2 * np.pi)
    vals = np.zeros(size, dtype=np.float64)
    for n in range(50):
        vals += a**n * np.cos(b**n * np.pi * (t + phase))
    return _to_uint8(vals)


@source(
    "Exponential Chirp",
    domain="exotic",
    description="Exponential frequency sweep --- instantaneous frequency grows geometrically, "
    "creating equal time per octave. Used in acoustic measurements and radar",
)
def gen_chirp_exp(rng, size):
    t = np.linspace(0, 1, size)
    f0 = 1.0 + rng.uniform(0, 2)
    f1 = 80.0 + rng.uniform(0, 40)
    phase = rng.uniform(0, 2 * np.pi)
    k = f1 / f0
    instantaneous_phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k) + phase
    vals = np.sin(instantaneous_phase)
    return ((vals + 1) / 2 * 255).astype(np.uint8)


# ================================================================
# Additional generators
# ================================================================


@source(
    "Baker Map",
    domain="chaos",
    description="Baker's map --- stretching and folding like kneading dough, "
    "the canonical model of chaotic mixing. Re-seeded every 40 iterations to avoid float collapse",
)
def gen_baker(rng, size):
    vals = np.empty(size, dtype=np.uint8)
    # Re-seed every 40 iterations to avoid float precision collapse
    # (doubling map loses all mantissa bits after ~53 steps)
    chunk = 40
    idx = 0
    while idx < size:
        x, y = rng.random(), rng.random()
        n = min(chunk, size - idx)
        for j in range(n):
            if x < 0.5:
                x, y = 2 * x, y / 2
            else:
                x, y = 2 - 2 * x, 1 - y / 2
            vals[idx] = int(x * 255) % 256
            idx += 1
    return vals


@source(
    "Chirikov Standard Map",
    domain="chaos",
    description="Chirikov standard map at K=0.9716 --- mixed phase space with coexisting regular islands "
    "and chaotic sea, the paradigm of Hamiltonian chaos",
)
def gen_standard_map(rng, size):
    K = 0.9716
    theta, p = rng.random() * 2 * np.pi, rng.random() * 2 * np.pi
    for _ in range(500):
        p = (p + K * np.sin(theta)) % (2 * np.pi)
        theta = (theta + p) % (2 * np.pi)
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        p = (p + K * np.sin(theta)) % (2 * np.pi)
        theta = (theta + p) % (2 * np.pi)
        vals[i] = int(theta / (2 * np.pi) * 255) % 256
    return vals


@source(
    "Logistic Edge-of-Chaos",
    domain="chaos",
    description="Logistic map at the Feigenbaum point r=3.5699 --- the boundary between order and chaos, "
    "where the period-doubling cascade accumulates",
)
def gen_logistic_edge(rng, size):
    x = 0.1 + 0.8 * rng.random()
    r = 3.5699
    for _ in range(1000):
        x = r * x * (1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = r * x * (1.0 - x)
        vals[i] = int(x * 255)
    return vals


# ── Logistic bifurcation sweep ──────────────────────────────────


def _gen_logistic_r(rng, size, r):
    """Logistic map x(n+1) = r*x*(1-x) at a given r."""
    x = 0.1 + 0.8 * rng.random()
    for _ in range(1000):
        x = r * x * (1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = r * x * (1.0 - x)
        vals[i] = int(x * 255)
    return vals


@source(
    "Logistic r=3.2 (Period-2)",
    domain="chaos",
    description="Logistic map in stable period-2 --- output alternates between two fixed values, "
    "the simplest oscillation after the first bifurcation",
)
def gen_logistic_p2(rng, size):
    return _gen_logistic_r(rng, size, 3.2)


@source(
    "Logistic r=3.5 (Period-4)",
    domain="chaos",
    description="Logistic map in period-4 --- four-value cycle after the second bifurcation, "
    "output concentrates on four narrow bands",
)
def gen_logistic_p4(rng, size):
    return _gen_logistic_r(rng, size, 3.5)


@source(
    "Logistic r=3.68 (Banded Chaos)",
    domain="chaos",
    description="Logistic map in banded chaos --- chaotic within disjoint bands that the orbit cycles through, "
    "mixing local unpredictability with global periodicity",
)
def gen_logistic_banded(rng, size):
    return _gen_logistic_r(rng, size, 3.68)


@source(
    "Logistic r=3.83 (Period-3 Window)",
    domain="chaos",
    description="Logistic map in the period-3 window --- a stable island inside the chaotic sea. "
    "By Sharkovskii's theorem, period-3 implies all periods exist nearby",
)
def gen_logistic_p3(rng, size):
    return _gen_logistic_r(rng, size, 3.83)


@source(
    "Logistic r=3.9 (Near-Full Chaos)",
    domain="chaos",
    description="Logistic map at r=3.9 --- nearly full chaos but with thin gaps in the attractor "
    "revealing remnants of periodic windows",
)
def gen_logistic_near_full(rng, size):
    return _gen_logistic_r(rng, size, 3.9)


@source(
    "Critical Circle Map",
    domain="chaos",
    description="Sine circle map at K=1 critical coupling with golden-mean bare frequency --- "
    "sits on the boundary of mode-locking, a different universality class from Feigenbaum's period-doubling",
)
def gen_circle_critical(rng, size):
    omega = (np.sqrt(5) - 1) / 2  # golden mean bare frequency
    K = 1.0  # critical coupling
    theta = rng.random()
    for _ in range(1000):
        theta = (theta + omega - K / (2 * np.pi) * np.sin(2 * np.pi * theta)) % 1.0
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        theta = (theta + omega - K / (2 * np.pi) * np.sin(2 * np.pi * theta)) % 1.0
        vals[i] = int(theta * 255)
    return vals


@source(
    "Perlin Noise",
    domain="noise",
    description="Spectral Perlin-like noise --- coherent smooth fluctuations with 1/f amplitude falloff, "
    "widely used for procedural terrain and texture generation",
)
def gen_perlin(rng, size):
    fft = np.zeros(size // 2 + 1, dtype=complex)
    freqs = np.fft.rfftfreq(size)
    freqs[0] = 1e-10
    amplitudes = 1.0 / (freqs**1.0)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    fft = amplitudes * np.exp(1j * phases)
    fft[0] = 0
    signal = np.fft.irfft(fft, n=size)
    return _to_uint8(signal)


@source(
    "Square Wave",
    domain="waveform",
    description="Square wave --- binary oscillation (0 or 255), maximal harmonic content with 1/n odd-harmonic Fourier series",
)
def gen_square(rng, size):
    freq = rng.uniform(5, 50)
    t = np.linspace(0, 1, size)
    wave = np.sign(np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi)))
    return ((wave + 1) / 2 * 255).astype(np.uint8)


@source(
    "Collatz Parity",
    domain="number_theory",
    description="Collatz parity sequence --- binary trace (even/odd) of the 3n+1 orbit, "
    "conjectured to behave like a biased coin flip with P(odd) ≈ log₂(3)/(1+log₂(3))",
)
def gen_collatz_parity(rng, size):
    n = int(rng.integers(10_000, 100_000_000))
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        vals[i] = n % 2 * 255
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    return vals


@source(
    "Collatz Stopping Times",
    domain="number_theory",
    description="Collatz stopping times --- how many steps each integer takes to reach 1. "
    "Wildly erratic, with typical values around 3.5·log₂(n) but enormous outliers",
)
def gen_collatz_stopping(rng, size):
    start = int(rng.integers(10_000, 500_000))
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        n = start + i
        steps = 0
        while n > 1 and steps < 1000:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        vals[i] = min(steps, 255)
    return vals


@source(
    "Mertens Function",
    domain="number_theory",
    description="Cumulative sum of the Moebius function --- a random-walk-like sequence whose growth rate "
    "is equivalent to the Riemann Hypothesis (O(n^(½+ε)) iff RH)",
)
def gen_mertens(rng, size):
    start = int(rng.integers(1000, 100_000))
    limit = start + size + 10
    mu = np.zeros(limit + 1, dtype=np.int8)
    mu[1] = 1
    for i in range(1, limit + 1):
        if mu[i] == 0 and i > 1:
            continue
        for j in range(2 * i, limit + 1, i):
            mu[j] -= mu[i]
    mertens = np.cumsum(mu[1 : start + size + 1])
    chunk = mertens[start : start + size]
    return _to_uint8(chunk.astype(np.float64))


@source(
    "Euler Totient Ratio",
    domain="number_theory",
    description="Euler totient ratio φ(n)/n --- the fraction of integers below n that are coprime to n, "
    "dense in (0,1) with a fractal structure governed by prime factorization",
)
def gen_totient_ratio(rng, size):
    start = int(rng.integers(1000, 100_000))
    limit = start + size + 10
    phi = np.arange(limit + 1, dtype=np.float64)
    for i in range(2, limit + 1):
        if phi[i] == i:  # prime
            for j in range(i, limit + 1, i):
                phi[j] *= 1 - 1.0 / i
    ratios = phi[start : start + size] / np.arange(start, start + size, dtype=np.float64)
    return (ratios * 255).astype(np.uint8)


@source(
    "RANDU",
    domain="binary",
    description="IBM's infamous RANDU (1968) --- the worst widely-deployed RNG in history. "
    "Points fall on just 15 parallel planes in 3D. Multiplier 65539 = 2¹⁶+3",
)
def gen_randu(rng, size):
    state = 65539 * (int(rng.integers(1, 2**16)) + 1) % (2**31)
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        state = (65539 * state) % (2**31)
        vals[i] = (state >> 16) & 0xFF
    return vals


@source(
    "glibc LCG",
    domain="binary",
    description="The C standard library LCG --- used by glibc rand() for decades. "
    "Low bits cycle with short period, high bits have lattice structure in dimensions >5",
)
def gen_glibc_lcg(rng, size):
    state = int(rng.integers(1, 2**31))
    if state == 0:
        state = 1
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        state = (1103515245 * state + 12345) % (2**31)
        vals[i] = (state >> 16) & 0xFF
    return vals


@source(
    "Middle-Square (von Neumann)",
    domain="binary",
    description="Von Neumann's 1949 middle-square PRNG --- famously degenerates to short cycles. "
    "Square the state, extract middle digits. Exhibits visible lattice structure.",
)
def gen_middle_square(rng, size):
    # Use 8-digit (32-bit-ish) state, extract middle 8 digits after squaring
    state = int(rng.integers(10_000_000, 99_999_999))
    vals = np.empty(size, dtype=np.uint8)
    for i in range(size):
        state = state * state
        s = f"{state:016d}"  # zero-pad to 16 digits
        state = int(s[4:12])  # extract middle 8 digits
        vals[i] = state & 0xFF
        if state == 0:
            state = int(rng.integers(10_000_000, 99_999_999))
    return vals


@source(
    "Neural Net (Dense)",
    domain="binary",
    description="Synthetic neural network weights --- Gaussian-initialized dense layer, "
    "resembling the near-random parameters of an untrained network",
)
def gen_nn_dense(rng, size):
    weights = rng.standard_normal(size) * 0.1
    return _to_uint8(weights)


@source(
    "Neural Net (Pruned 90%)",
    domain="binary",
    description="Pruned neural network weights (90% zero) --- extreme sparsity typical of compressed models, "
    "creates a near-zero-dominated distribution with scattered non-zero entries",
)
def gen_nn_pruned(rng, size):
    weights = rng.standard_normal(size) * 0.1
    mask = rng.random(size) < 0.9
    weights[mask] = 0
    return _to_uint8(weights)


# --- Void-filling generators ---


def _gen_ca(rule_num):
    """Factory for elementary cellular automaton generators."""
    rule_table = np.array([(rule_num >> i) & 1 for i in range(8)], dtype=np.uint8)

    def gen(rng, size):
        width = 2 * size + 1
        row = np.zeros(width, dtype=np.uint8)
        if rule_num == 110:
            row[size:] = rng.integers(0, 2, size + 1, dtype=np.uint8)
        else:
            row[size] = 1
        vals = np.empty(size, dtype=np.uint8)
        for i in range(size):
            vals[i] = row[size]
            neighborhood = (row[:-2] << 2) | (row[1:-1] << 1) | row[2:]
            new = np.zeros_like(row)
            new[1:-1] = rule_table[neighborhood]
            row = new
        return vals * 255

    return gen


register(
    "Rule 30",
    _gen_ca(30),
    "exotic",
    "Wolfram Rule 30 center column --- produces output indistinguishable from random despite simple local rules. "
    "Used as Mathematica's default random generator",
)
register(
    "Rule 110",
    _gen_ca(110),
    "exotic",
    "Wolfram Rule 110 center column --- proven Turing-complete, the simplest known universal computer. "
    "Supports gliders and localized structures like a 1D Game of Life",
)


@source(
    "Gaussian Noise",
    domain="noise",
    description="Gaussian noise scaled to fill uint8 --- bell-curve density with 3σ mapped to [0,255], "
    "concentrating samples around the midpoint",
)
def gen_gaussian_noise(rng, size):
    samples = rng.standard_normal(size)
    # Scale to use most of uint8 range (mean=128, ~3 sigma fills 0-255)
    scaled = 128 + samples * 42
    return np.clip(scaled, 0, 255).astype(np.uint8)


@source(
    "Triangle Wave",
    domain="waveform",
    description="Triangle wave --- linear ramps up and down, smooth peaks with 1/n² Fourier decay (only odd harmonics)",
)
def gen_triangle(rng, size):
    period = rng.uniform(20, 200)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(size) / period + phase / (2 * np.pi)
    # Triangle: linear ramp up and down
    vals = 2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1
    return ((vals + 1) * 127.5).astype(np.uint8)


@source(
    "Dice Rolls",
    domain="exotic",
    description="Simulated dice rolls --- uniform over just 6 levels (0, 51, 102, 153, 204, 255), "
    "creating a maximally discrete distribution with IID independence",
)
def gen_dice(rng, size):
    rolls = rng.integers(1, 7, size=size)
    # Map 1-6 to spread across uint8: 0, 51, 102, 153, 204, 255
    mapping = np.array([0, 51, 102, 153, 204, 255], dtype=np.uint8)
    return mapping[rolls - 1]


_MIDI_BYTES = None


def _get_midi_bytes():
    """Load concatenated raw MIDI bytes from 4 classical pieces (365KB total)."""
    global _MIDI_BYTES
    if _MIDI_BYTES is not None:
        return _MIDI_BYTES
    from pathlib import Path

    mdir = Path(__file__).resolve().parents[1] / "data" / "midi"
    if not mdir.exists():
        _MIDI_BYTES = np.array([], dtype=np.uint8)
        return _MIDI_BYTES
    raw = b""
    for mid in sorted(mdir.glob("*.mid")):
        raw += mid.read_bytes()
    _MIDI_BYTES = np.frombuffer(raw, dtype=np.uint8).copy()
    return _MIDI_BYTES


@source(
    "Classical MIDI",
    domain="binary",
    description="Real MIDI files --- Bach (chorale, Brandenburg 3), Beethoven (5th), Mozart (Figaro). "
    "Raw Standard MIDI format 1 bytes: delta-time VLQs, note-on/off events, multi-track. 365KB.",
)
def gen_midi(rng, size):
    data = _get_midi_bytes()
    if len(data) == 0:
        return rng.integers(0, 256, size, dtype=np.uint8)
    max_start = max(0, len(data) - size)
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    chunk = data[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="wrap")
    return chunk.copy()


@source(
    "Clipped Sine",
    domain="waveform",
    description="Sine wave hard-clipped at 70% --- flat tops and bottoms create a hybrid of smooth oscillation "
    "and constant segments, concentrating the distribution at extremes",
)
def gen_clipped_sine(rng, size):
    period = rng.uniform(20, 200)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(size) * 2 * np.pi / period + phase
    vals = np.sin(t)
    clip = 0.7
    vals = np.clip(vals, -clip, clip) / clip
    return ((vals + 1) * 127.5).astype(np.uint8)


# --- L-Systems ---


@source(
    "L-System (Algae)",
    domain="exotic",
    description="Lindenmayer system (A→AB, B→A) --- the simplest L-system, producing Fibonacci-length words. "
    "Deterministic, self-similar, with growth rate equal to the golden ratio",
)
def gen_lsystem_algae(rng, size):
    """
    Classic Algae L-system.
    Rules: A -> AB, B -> A
    Mapped to: A=255, B=0
    """
    s = "A"
    # Grow string until it's at least 'size' long
    # Growth is exponential (Fibonacci lengths), so this is fast
    while len(s) < size:
        s = "".join(["AB" if c == "A" else "A" for c in s])

    # Take a random starting window to allow trial diversity
    start = rng.integers(0, max(1, len(s) - size))
    chunk = s[start : start + size]

    # Map symbols to uint8
    vals = np.array([255 if c == "A" else 0 for c in chunk], dtype=np.uint8)
    return vals


# --- Symbolic Dynamics ---


@source(
    "Symbolic Lorenz",
    domain="exotic",
    description="Lorenz attractor reduced to a binary symbol stream (left lobe vs right lobe), "
    "one symbol per half-orbit. The symbolic itinerary encodes the topological structure of chaos",
)
def gen_symbolic_lorenz(rng, size):
    """
    Symbolic dynamics of the Lorenz system.
    Standard parameters: sigma=10, rho=28, beta=8/3
    Partition: x < 0 -> 0, x > 0 -> 255
    Subsampled every 50 Euler steps (~1 per lobe visit) to capture
    the switching dynamics rather than long constant runs.
    """
    x, y, z = rng.standard_normal(3) * 5.0
    dt = 0.01
    subsample = 50  # ~1 sample per lobe visit

    # Warm up to reach the attractor
    for _ in range(2000):
        dx = 10.0 * (y - x) * dt
        dy = (x * (28.0 - z) - y) * dt
        dz = (x * y - (8.0 / 3.0) * z) * dt
        x, y, z = x + dx, y + dy, z + dz

    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        for _ in range(subsample):
            dx = 10.0 * (y - x) * dt
            dy = (x * (28.0 - z) - y) * dt
            dz = (x * y - (8.0 / 3.0) * z) * dt
            x, y, z = x + dx, y + dy, z + dz
        vals[i] = 255 if x > 0 else 0

    return vals


@source(
    "Symbolic Henon",
    domain="exotic",
    description="Henon map reduced to a binary symbol stream (x < 0 or x > 0) --- the symbolic itinerary of a 2D strange attractor, "
    "encoding its topological horseshoe structure",
)
def gen_symbolic_henon(rng, size):
    """
    Symbolic dynamics of the Henon map.
    Standard parameters: a=1.4, b=0.3
    Partition: x < 0 -> 0, x > 0 -> 255
    """
    x, y = rng.random() * 0.2, rng.random() * 0.2

    # Warm up
    for _ in range(1000):
        x, y = 1.0 - 1.4 * x**2 + y, 0.3 * x

    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x, y = 1.0 - 1.4 * x**2 + y, 0.3 * x
        vals[i] = 255 if x > 0 else 0

    return vals


# --- Bio (Protein) ---

_PROTEIN_SEQ = None


def _get_protein_seq():
    """Load concatenated amino acid sequence from Swiss-Prot FASTA (11.4M residues).

    Uses raw ASCII byte values (A=65, C=67, ..., Y=89) so the framework
    sees the same bytes a real binary analysis tool would.
    """
    global _PROTEIN_SEQ
    if _PROTEIN_SEQ is not None:
        return _PROTEIN_SEQ
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "protein" / "human_swissprot.fa"
    if not p.exists():
        _PROTEIN_SEQ = np.array([], dtype=np.uint8)
        return _PROTEIN_SEQ
    # Parse FASTA: skip header lines, concatenate AA sequences as raw ASCII
    valid = set(b"ACDEFGHIKLMNPQRSTVWY")
    residues = []
    with open(p, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                continue
            for b in line.strip().upper():
                if b in valid:
                    residues.append(b)
    _PROTEIN_SEQ = np.array(residues, dtype=np.uint8)
    return _PROTEIN_SEQ


@source(
    "Human Proteome",
    domain="bio",
    description="Real human proteome --- 20,431 reviewed Swiss-Prot sequences (UniProt 2025), "
    "raw ASCII bytes (20 amino acid letters in the 65-89 range). 11.4M residues.",
)
def gen_protein_globular(rng, size):
    seq = _get_protein_seq()
    if len(seq) == 0:
        return rng.integers(0, 256, size, dtype=np.uint8)
    max_start = max(0, len(seq) - size)
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    chunk = seq[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="wrap")
    return chunk.copy()


# --- Text / Linguistic ---

_GUTENBERG_TEXT = None


def _get_gutenberg_text():
    """Load concatenated Gutenberg text as raw bytes (2.8MB, 4 novels)."""
    global _GUTENBERG_TEXT
    if _GUTENBERG_TEXT is not None:
        return _GUTENBERG_TEXT
    from pathlib import Path

    gdir = Path(__file__).resolve().parents[1] / "data" / "gutenberg"
    if not gdir.exists():
        _GUTENBERG_TEXT = np.array([], dtype=np.uint8)
        return _GUTENBERG_TEXT
    text = b""
    for name in sorted(gdir.glob("*.txt")):
        text += name.read_bytes()
    _GUTENBERG_TEXT = np.frombuffer(text, dtype=np.uint8).copy()
    return _GUTENBERG_TEXT


@source(
    "English Literature",
    domain="speech",
    description="Real Project Gutenberg text --- Austen, Carroll, Doyle, Melville (2.8MB). "
    "Raw ASCII bytes with natural English letter frequencies and word structure.",
)
def gen_english_text(rng, size):
    text = _get_gutenberg_text()
    if len(text) == 0:
        return rng.integers(0, 256, size, dtype=np.uint8)
    max_start = max(0, len(text) - size)
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    chunk = text[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="wrap")
    return chunk.copy()


# --- Audio / Quantization ---


@source(
    "μ-law Sine",
    domain="waveform",
    description="Sine wave with non-linear mu-law quantization (G.711). Creates a structural bridge between analogue waves and discrete logic.",
)
def gen_mulaw_sine(rng, size):
    """
    mu-law encoded sine wave.
    mu = 255.
    """
    freq = rng.uniform(5, 50)
    t = np.linspace(0, 1, size)
    x = np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

    # mu-law transform: y = sign(x) * ln(1 + mu|x|) / ln(1 + mu)
    mu = 255.0
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

    # Map to uint8
    return ((y + 1) * 127.5).astype(np.uint8)


# ==============================================================
# VOID-FILLING GENERATORS
# Targeting the largest internal void at PCA center (-9.6, -6.6)
# Need: constrained distribution + low sequential correlation
# ==============================================================

_EARTHQUAKE_DATA = None


def _get_earthquake_data():
    """Load CORGIS earthquake catalog (8395 events, USGS 2016)."""
    global _EARTHQUAKE_DATA
    if _EARTHQUAKE_DATA is not None:
        return _EARTHQUAKE_DATA
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "seismic" / "earthquakes.csv"
    if not p.exists():
        _EARTHQUAKE_DATA = {}
        return _EARTHQUAKE_DATA
    import csv

    mags, depths, epochs, gaps = [], [], [], []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mags.append(float(row["impact.magnitude"]))
                depths.append(float(row["location.depth"]))
                epochs.append(float(row["time.epoch"]))
                gaps.append(float(row["impact.gap"]))
            except (ValueError, KeyError):
                continue
    # Sort by time
    order = np.argsort(epochs)
    _EARTHQUAKE_DATA = {
        "magnitude": np.array(mags)[order],
        "depth": np.array(depths)[order],
        "epoch": np.array(epochs)[order],
        "gap": np.array(gaps)[order],
    }
    return _EARTHQUAKE_DATA


@source(
    "Earthquake Magnitudes",
    domain="geophysics",
    description="Real USGS earthquake magnitudes --- Gutenberg-Richter distribution, 8395 events (CORGIS/USGS)",
)
def gen_earthquake_magnitudes(rng, size):
    d = _get_earthquake_data()
    if not d:
        return rng.integers(0, 256, size, dtype=np.uint8)
    return _real_data_gen(d["magnitude"], rng, size)


@source(
    "Earthquake Depths",
    domain="geophysics",
    description="Real USGS earthquake depths --- bimodal: shallow crustal (<30km) and deep subduction (>100km)",
)
def gen_earthquake_depths(rng, size):
    d = _get_earthquake_data()
    if not d:
        return rng.integers(0, 256, size, dtype=np.uint8)
    return _real_data_gen(d["depth"], rng, size)


@source(
    "Earthquake Intervals",
    domain="geophysics",
    description="Real USGS interevent times --- Omori-law clustering after mainshocks, Poisson background (CORGIS/USGS)",
)
def gen_earthquake_intervals(rng, size):
    d = _get_earthquake_data()
    if not d or len(d["epoch"]) < 2:
        return rng.integers(0, 256, size, dtype=np.uint8)
    intervals = np.diff(d["epoch"]) / 1000.0  # ms -> seconds
    intervals = np.log1p(np.maximum(intervals, 0))  # log-transform
    return _real_data_gen(intervals, rng, size)


@source(
    "El Centro 1940",
    domain="geophysics",
    description="1940 Imperial Valley NS strong motion --- the most-studied earthquake record in structural engineering",
)
def gen_el_centro(rng, size):
    """El Centro 1940 NS component: 31s of ground acceleration at 50Hz.
    1562 samples --- wraps to fill size. The classic benchmark seismogram."""
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "seismic" / "el_centro.dat"
    if not p.exists():
        return rng.integers(0, 256, size, dtype=np.uint8)
    vals = []
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    continue
    arr = np.array(vals, dtype=np.float64)
    if len(arr) < size:
        arr = np.tile(arr, (size // len(arr)) + 1)[:size]
    return _to_uint8(arr[:size])


_RAINFALL = None


def _get_rainfall():
    """Load ORD 2020-2023 hourly precipitation (35k observations)."""
    global _RAINFALL
    if _RAINFALL is not None:
        return _RAINFALL
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "rainfall" / "ord_2020_2023_hourly.csv"
    if not p.exists():
        _RAINFALL = np.array([], dtype=np.float64)
        return _RAINFALL
    vals = []
    with open(p) as f:
        for line in f:
            if line.startswith("station") or line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    vals.append(float(parts[2]))
                except ValueError:
                    continue
    _RAINFALL = np.array(vals, dtype=np.float64)
    return _RAINFALL


@source(
    "Rainfall (ORD Hourly)",
    domain="climate",
    description="Real hourly precipitation at Chicago O'Hare 2020--2023 (IEM/ASOS). "
    "35k observations, ~85% zero (dry hours), heavy-tailed wet spells. Public domain.",
)
def gen_rainfall(rng, size):
    data = _get_rainfall()
    return _real_data_gen(data, rng, size)


def _load_data_file(relpath, parser):
    """Load a data file relative to the project root. Returns numpy array or None."""
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / relpath
    if not p.exists():
        return None
    return parser(p)


def _real_data_gen(data_array, rng, size):
    """Sample a contiguous block from a real data array."""
    if data_array is None or len(data_array) == 0:
        return rng.integers(0, 256, size, dtype=np.uint8)
    max_start = max(0, len(data_array) - size)
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    chunk = data_array[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="wrap")
    return _to_uint8(chunk.astype(np.float64))


# --- Cached real data arrays ---
_BUOY_WSPD = None
_BUOY_WVHT = None
_BUOY_PRES = None
_KP_INDEX = None
_RIVER_DISCHARGE = None
_SEISMOGRAPH = None


def _get_buoy_data():
    global _BUOY_WSPD, _BUOY_WVHT, _BUOY_PRES
    if _BUOY_WSPD is not None:
        return _BUOY_WSPD, _BUOY_WVHT, _BUOY_PRES

    def parse_buoy(path):
        wspd, wvht, pres = [], [], []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 14:
                    try:
                        w = float(parts[6])  # WSPD
                        h = float(parts[8])  # WVHT
                        p = float(parts[12])  # PRES
                        if w < 90:
                            wspd.append(w)
                        if h < 90:
                            wvht.append(h)
                        if p < 9000:
                            pres.append(p)
                    except ValueError:
                        continue
        return np.array(wspd), np.array(wvht), np.array(pres)

    result = _load_data_file("data/geophysics/buoy_46042_2023.txt", parse_buoy)
    if result is not None:
        _BUOY_WSPD, _BUOY_WVHT, _BUOY_PRES = result
    else:
        _BUOY_WSPD = _BUOY_WVHT = _BUOY_PRES = np.array([])
    return _BUOY_WSPD, _BUOY_WVHT, _BUOY_PRES


def _get_kp_index():
    global _KP_INDEX
    if _KP_INDEX is not None:
        return _KP_INDEX

    def parse_kp(path):
        vals = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 22:
                    try:
                        # 8 ap values per day (cols index 15-22)
                        for i in range(15, 23):
                            v = int(parts[i])
                            if v >= 0:
                                vals.append(v)
                    except (ValueError, IndexError):
                        continue
        return np.array(vals, dtype=np.float64)

    _KP_INDEX = _load_data_file("data/geophysics/kp_index.txt", parse_kp)
    if _KP_INDEX is None:
        _KP_INDEX = np.array([])
    return _KP_INDEX


def _get_river_discharge():
    global _RIVER_DISCHARGE
    if _RIVER_DISCHARGE is not None:
        return _RIVER_DISCHARGE

    def parse_rdb(path):
        vals = []
        with open(path) as f:
            for line in f:
                if line.startswith("#") or line.startswith("agency") or line.startswith("5s"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    try:
                        v = float(parts[4])
                        vals.append(v)
                    except ValueError:
                        continue
        return np.array(vals, dtype=np.float64)

    _RIVER_DISCHARGE = _load_data_file("data/geophysics/potomac_discharge.rdb", parse_rdb)
    if _RIVER_DISCHARGE is None:
        _RIVER_DISCHARGE = np.array([])
    return _RIVER_DISCHARGE


def _get_seismograph():
    global _SEISMOGRAPH
    if _SEISMOGRAPH is not None:
        return _SEISMOGRAPH

    def parse_iris(path):
        vals = []
        with open(path) as f:
            for line in f:
                if line.startswith("TIMESERIES"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        vals.append(float(parts[1]))
                    except ValueError:
                        continue
                elif len(parts) == 1:
                    try:
                        vals.append(float(parts[0]))
                    except ValueError:
                        continue
        return np.array(vals, dtype=np.float64)

    _SEISMOGRAPH = _load_data_file("data/seismograph/anmo_bhz.txt", parse_iris)
    if _SEISMOGRAPH is None:
        _SEISMOGRAPH = np.array([])
    return _SEISMOGRAPH


@source(
    "Ocean Wind (Buoy)",
    domain="climate",
    description="NOAA buoy 46042 wind speed --- real 10-min observations off Monterey Bay 2023, "
    "capturing marine boundary layer turbulence and diurnal sea breeze cycles (CC0)",
)
def gen_buoy_wind(rng, size):
    wspd, _, _ = _get_buoy_data()
    return _real_data_gen(wspd, rng, size)


@source(
    "Wave Height (Buoy)",
    domain="geophysics",
    description="NOAA buoy 46042 significant wave height --- real ocean swell measurements, 10-min intervals (CC0)",
)
def gen_buoy_waves(rng, size):
    _, wvht, _ = _get_buoy_data()
    return _real_data_gen(wvht, rng, size)


@source(
    "Barometric Pressure (Buoy)",
    domain="climate",
    description="NOAA buoy 46042 atmospheric pressure --- real synoptic weather patterns, 10-min intervals (CC0)",
)
def gen_buoy_pressure(rng, size):
    _, _, pres = _get_buoy_data()
    return _real_data_gen(pres, rng, size)


@source(
    "Geomagnetic ap Index",
    domain="geophysics",
    description="Real 3-hourly geomagnetic ap index 1932-present --- solar-driven magnetic storms, 93 years (CC BY 4.0, GFZ)",
)
def gen_kp_index(rng, size):
    data = _get_kp_index()
    return _real_data_gen(data, rng, size)


@source(
    "Potomac River Flow",
    domain="geophysics",
    description="USGS gauge 01646500 --- real 15-min discharge, Potomac near DC, 2024 (public domain)",
)
def gen_river_discharge(rng, size):
    data = _get_river_discharge()
    if len(data) > 0:
        data = np.log1p(data)  # log-transform to tame flood spikes
    return _real_data_gen(data, rng, size)


@source(
    "Seismograph (ANMO)",
    domain="geophysics",
    description="IRIS IU.ANMO broadband vertical --- real 40Hz seismograph, Albuquerque NM, 2024-01-01 (CC0)",
)
def gen_seismograph(rng, size):
    data = _get_seismograph()
    return _real_data_gen(data, rng, size)


_GOES_XRAY = None


def _get_goes_xray():
    global _GOES_XRAY
    if _GOES_XRAY is not None:
        return _GOES_XRAY
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "astro" / "goes_xray.csv"
    if not p.exists():
        _GOES_XRAY = np.array([])
        return _GOES_XRAY
    vals = []
    with open(p) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    v = float(parts[1])
                    if v > 0:
                        vals.append(v)
                except ValueError:
                    continue
    _GOES_XRAY = np.array(vals, dtype=np.float64)
    return _GOES_XRAY


@source(
    "GOES X-Ray Flux",
    domain="astro",
    description="GOES satellite 1-min X-ray flux --- solar flare monitoring, log-scale A/B/C/M/X classes (NOAA)",
)
def gen_goes_xray(rng, size):
    data = _get_goes_xray()
    if len(data) > 0:
        data = np.log10(data)  # log-scale spans ~6 orders of magnitude
    return _real_data_gen(data, rng, size)


@source(
    "Codon Usage",
    domain="bio",
    description="Synthetic coding DNA --- nucleotide triplets (codons) with Zipf-like usage bias "
    "and weak Markov transitions, emitted as raw ASCII bases (A=65, C=67, G=71, T=84)",
)
def gen_codon_usage(rng, size):
    """64 codons with realistic usage bias, emitted as raw ASCII nucleotides."""
    # Build the 64 codons as byte triplets
    bases = [65, 67, 71, 84]  # A, C, G, T in ASCII
    codons = []
    for a in bases:
        for b in bases:
            for c in bases:
                codons.append((a, b, c))

    # Zipf-like codon frequencies (some codons used 5x more than synonyms)
    freqs = 1.0 / (np.arange(1, 65) ** 0.6)
    rng.shuffle(freqs)
    freqs /= freqs.sum()

    # Generate codon indices with weak Markov transitions
    n_codons = (size + 2) // 3  # enough triplets to fill size bytes
    indices = np.empty(n_codons, dtype=int)
    current = rng.choice(64, p=freqs)
    for i in range(n_codons):
        if rng.random() < 0.15:
            current = (current + rng.choice([-3, -2, -1, 1, 2, 3])) % 64
        else:
            current = rng.choice(64, p=freqs)
        indices[i] = current

    # Expand to raw ASCII bytes
    data = np.empty(n_codons * 3, dtype=np.uint8)
    for i, idx in enumerate(indices):
        a, b, c = codons[idx]
        data[i * 3] = a
        data[i * 3 + 1] = b
        data[i * 3 + 2] = c
    return data[:size]


@source(
    "Poker Hands",
    domain="exotic",
    description="Stream of poker hand ranks (High Card through Royal Flush) --- extremely skewed: "
    "~50% High Card, ~42% Pair, <0.002% Straight Flush or better",
)
def gen_poker_hands(rng, size):
    """
    Stream of Poker hand ranks (0=High Card, 9=Royal Flush).
    Heavily skewed distribution: High Card ~50%, Pair ~42%, Royal Flush <0.001%.
    """
    probs = np.array(
        [0.501, 0.422, 0.047, 0.021, 0.0039, 0.0019, 0.0014, 0.00024, 0.000013, 0.000001]
    )
    probs /= probs.sum()
    ranks = rng.choice(10, size=size, p=probs)
    mapping = np.linspace(0, 255, 10, dtype=np.uint8)
    return mapping[ranks]


@source(
    "Pulse-Width Modulation",
    domain="waveform",
    description="Pulse-width modulation --- binary carrier (0/255) whose duty cycle sweeps smoothly via sine modulation, "
    "encoding an analog signal in digital on/off timing",
)
def gen_pwm_signal(rng, size):
    """
    Pulse-Width Modulation.
    Carrier frequency ~1% of sampling rate.
    Duty cycle modulated by a slow sine wave.
    """
    period = 100
    t = np.arange(size)
    mod_freq = 2 * np.pi / (size * 0.3)
    duty = 0.5 + 0.4 * np.sin(t * mod_freq + rng.uniform(0, 2 * np.pi))
    phase = t % period
    threshold = duty * period
    vals = (phase < threshold).astype(np.uint8) * 255
    return vals


@source(
    "Morse Code",
    domain="waveform",
    description="Synthetic Morse code --- dots (1 unit), dashes (3 units), and hierarchical gaps (1/3/7 units). "
    "Creates multi-scale on/off structure encoding random letter sequences",
)
def gen_morse_code(rng, size):
    """
    Morse code stream.
    Dot = 1 unit, Dash = 3 units.
    Intra-char gap = 1 unit.
    Letter gap = 3 units.
    Word gap = 7 units.
    """
    morse_map = [
        [1, 3],
        [3, 1, 1, 1],
        [3, 1, 3, 1],
        [3, 1, 1],
        [1],  # A-E
        [1, 1, 3, 1],
        [3, 3, 1],
        [1, 1, 1, 1],
        [1, 1],
        [1, 3, 3, 3],  # F-J
        [3, 1, 3],
        [1, 3, 1, 1],
        [3, 3],
        [3, 1],
        [3, 3, 3],  # K-O
        [1, 3, 3, 1],
        [3, 3, 1, 3],
        [1, 3, 1],
        [1, 1, 1],
        [3],  # P-T
        [1, 1, 3],
        [1, 1, 1, 3],
        [1, 3, 3],
        [3, 1, 1, 3],
        [3, 1, 3, 3],
        [3, 3, 1, 1],  # U-Z
    ]
    data = np.zeros(size, dtype=np.uint8)
    idx = 0
    while idx < size:
        char_seq = morse_map[rng.integers(0, 26)]
        for duration in char_seq:
            end = min(idx + duration * 4, size)
            data[idx:end] = 255
            idx = end
            idx += 4  # Intra-char gap
        idx += 8  # Letter gap
        if rng.random() < 0.2:
            idx += 16  # Word gap
    return data[:size]


@source(
    "Sensor Event Stream",
    domain="exotic",
    description="Categorical sensor event stream --- door open/close, motion, temperature alerts "
    "with state-dependent transitions. Mostly quiet (value 0) with coupled event bursts",
)
def gen_iot_sensor(rng, size):
    """
    Categorical event stream: "Door Open", "Door Close", "Motion", "Temp High".
    Highly coupled events (Door Open -> Door Close).
    """
    data = np.zeros(size, dtype=np.uint8)
    i = 0
    door_open = False
    while i < size:
        r = rng.random()
        if door_open:
            if r < 0.1:
                data[i] = 200  # Close
                door_open = False
            elif r < 0.6:
                data[i] = 150
                if rng.random() < 0.5:
                    data[i] = 50  # Motion while open
            else:
                data[i] = 0
        else:
            if r < 0.02:
                data[i] = 150  # Open
                door_open = True
            elif r < 0.1:
                data[i] = 50  # Motion
            elif r < 0.105:
                data[i] = 250  # Temp alert
            else:
                data[i] = 0
        i += 1
    return data


# ================================================================
# Cross-domain chaotic dynamics --- diversify the chaos cluster
# ================================================================


@source(
    "Chua's Circuit",
    domain="exotic",
    description="Double-scroll electronic chaos --- bimodal attractor with different topology from Lorenz",
)
def gen_chua_circuit(rng, size):
    """Chua's circuit: piecewise-linear ODE producing double-scroll attractor."""
    alpha, beta = 15.6, 28.0
    m0, m1 = -1.143, -0.714

    def h(x):
        return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))

    x = rng.uniform(-0.5, 0.5)
    y = rng.uniform(-0.5, 0.5)
    z = rng.uniform(-0.5, 0.5)
    dt = 0.005
    for _ in range(10000):  # transient
        dx = alpha * (y - x - h(x))
        dy = x - y + z
        dz = -beta * y
        x, y, z = x + dx * dt, y + dy * dt, z + dz * dt
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        dx = alpha * (y - x - h(x))
        dy = x - y + z
        dz = -beta * y
        x, y, z = x + dx * dt, y + dy * dt, z + dz * dt
        vals[i] = x
    return _to_uint8(vals)


@source(
    "Double Pendulum",
    domain="motion",
    description="Angular velocity of second arm --- mechanical chaos with mixed regular/chaotic regions",
)
def gen_double_pendulum(rng, size):
    """Double pendulum simulation outputting angular velocity of lower arm."""
    g = 9.81
    L1 = L2 = 1.0
    m1 = m2 = 1.0
    th1 = rng.uniform(1.0, 3.0)
    th2 = rng.uniform(1.0, 3.0)
    w1 = rng.uniform(-0.5, 0.5)
    w2 = rng.uniform(-0.5, 0.5)
    dt = 0.005
    for _ in range(5000):
        delta = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        den2 = (L2 / L1) * den1
        a1 = (
            m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(th2) * np.cos(delta)
            + m2 * L2 * w2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th1)
        ) / den1
        a2 = (
            -m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
            - (m1 + m2) * L1 * w1**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th2)
        ) / den2
        w1 += a1 * dt
        w2 += a2 * dt
        th1 += w1 * dt
        th2 += w2 * dt
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        delta = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        den2 = (L2 / L1) * den1
        a1 = (
            m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(th2) * np.cos(delta)
            + m2 * L2 * w2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th1)
        ) / den1
        a2 = (
            -m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
            - (m1 + m2) * L1 * w1**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th2)
        ) / den2
        w1 += a1 * dt
        w2 += a2 * dt
        th1 += w1 * dt
        th2 += w2 * dt
        vals[i] = w2
    return _to_uint8(vals)


@source(
    "Mackey-Glass",
    domain="medical",
    description="Delay-differential equation modeling blood cell regulation --- high-dimensional chaos",
)
def gen_mackey_glass(rng, size):
    """Mackey-Glass delay-differential equation.
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    Chaotic for tau >= 17. Higher tau = higher-dimensional attractor."""
    tau = rng.integers(17, 30)  # delay in the chaotic regime
    beta = 0.2
    gamma_mg = 0.1
    n_exp = 10
    dt = 1.0
    history_len = tau + 1
    # Initialize with random history around the fixed point x* ≈ 1
    history = 1.0 + 0.1 * rng.standard_normal(history_len)
    history = np.clip(history, 0.1, 2.0)
    x = history[-1]
    # Transient
    for _ in range(5000):
        x_delayed = history[0]
        dx = beta * x_delayed / (1 + x_delayed**n_exp) - gamma_mg * x
        x = max(x + dx * dt, 0.001)
        history = np.roll(history, -1)
        history[-1] = x
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        x_delayed = history[0]
        dx = beta * x_delayed / (1 + x_delayed**n_exp) - gamma_mg * x
        x = max(x + dx * dt, 0.001)
        history = np.roll(history, -1)
        history[-1] = x
        vals[i] = x
    return _to_uint8(vals)


_TURBULENCE_WIND = None


def _get_turbulence_wind():
    """Load ORD 2023 5-minute wind speed (114k observations)."""
    global _TURBULENCE_WIND
    if _TURBULENCE_WIND is not None:
        return _TURBULENCE_WIND
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "data" / "turbulence" / "ord_2023_wind_5min.csv"
    if not p.exists():
        _TURBULENCE_WIND = np.array([], dtype=np.float64)
        return _TURBULENCE_WIND
    vals = []
    with open(p) as f:
        for line in f:
            if line.startswith("station") or line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[2]:
                try:
                    vals.append(float(parts[2]))
                except ValueError:
                    continue
    _TURBULENCE_WIND = np.array(vals, dtype=np.float64)
    return _TURBULENCE_WIND


@source(
    "Surface Wind (ORD 5-min)",
    domain="climate",
    description="Real 5-minute surface wind speed at Chicago O'Hare 2023 (IEM/ASOS). "
    "114k observations, quantized to ~1-knot steps (31 levels). "
    "Captures gusts, diurnal cycles, and frontal passages. Public domain.",
)
def gen_turbulent_wind(rng, size):
    data = _get_turbulence_wind()
    return _real_data_gen(data, rng, size)


# ================================================================
# Void-filling generators --- targeting empty regions of PC1-PC3 space
# ================================================================

# --- Group A: Bursty-oscillatory desert (peaked + empirical/bursty) ---


@source(
    "Sandpile",
    domain="exotic",
    description="Bak-Tang-Wiesenfeld sandpile --- self-organized criticality produces power-law avalanche sizes "
    "without parameter tuning. Slow buildup with sudden cascading collapses",
)
def gen_sandpile(rng, size):
    """2D BTW sandpile: drop grains, record avalanche sizes (total topples).
    Threshold=4 on a 2D grid (matching 4 neighbors). Each topple sends 1 grain
    to each neighbor; grains are lost at boundaries (open boundary conditions).
    Avalanche sizes follow a power law at the SOC critical state."""
    L = 64
    grid = rng.integers(0, 4, (L, L))
    threshold = 4
    # Burn-in: drive to critical state
    for _ in range(L * L):
        r, c = rng.integers(0, L), rng.integers(0, L)
        grid[r, c] += 1
        while np.any(grid >= threshold):
            toppling = np.argwhere(grid >= threshold)
            for r2, c2 in toppling:
                grid[r2, c2] -= 4
                if r2 > 0:
                    grid[r2 - 1, c2] += 1
                if r2 < L - 1:
                    grid[r2 + 1, c2] += 1
                if c2 > 0:
                    grid[r2, c2 - 1] += 1
                if c2 < L - 1:
                    grid[r2, c2 + 1] += 1
    # Record avalanche sizes
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        r, c = rng.integers(0, L), rng.integers(0, L)
        grid[r, c] += 1
        topples = 0
        while np.any(grid >= threshold):
            toppling = np.argwhere(grid >= threshold)
            topples += len(toppling)
            for r2, c2 in toppling:
                grid[r2, c2] -= 4
                if r2 > 0:
                    grid[r2 - 1, c2] += 1
                if r2 < L - 1:
                    grid[r2 + 1, c2] += 1
                if c2 > 0:
                    grid[r2, c2 - 1] += 1
                if c2 < L - 1:
                    grid[r2, c2 + 1] += 1
        vals[i] = topples
    return _to_uint8(vals)


@source(
    "Stochastic Resonance",
    domain="exotic",
    description="Weak periodic signal amplified by noise in a bistable potential --- intermittent well-hopping",
)
def gen_stochastic_resonance(rng, size):
    """Kramers bistable potential V(x) = -x^2/2 + x^4/4 with weak
    periodic forcing and tuned noise. Output is the particle position."""
    dt = 0.01
    A = rng.uniform(0.1, 0.3)  # weak signal amplitude
    omega = rng.uniform(0.005, 0.02)  # slow driving frequency
    D = rng.uniform(0.3, 0.6)  # noise intensity (tuned for resonance)
    x = rng.choice([-1.0, 1.0])  # start in one well
    # Transient
    for i in range(10000):
        force = x - x**3 + A * np.sin(omega * i * dt)
        x += force * dt + np.sqrt(2 * D * dt) * rng.standard_normal()
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        force = x - x**3 + A * np.sin(omega * (i + 10000) * dt)
        x += force * dt + np.sqrt(2 * D * dt) * rng.standard_normal()
        vals[i] = x
    return _to_uint8(vals)


@source(
    "Coupled Map Lattice",
    domain="chaos",
    description="Spatiotemporal chaos in coupled logistic maps --- chimera states and coherence-incoherence patterns",
)
def gen_coupled_map_lattice(rng, size):
    """Chain of N coupled logistic maps with nearest-neighbor diffusive coupling.
    Record the value at one site over time. Near the spatiotemporal chaos
    transition, shows intermittent bursts of coherence."""
    N_sites = 50
    epsilon = rng.uniform(0.2, 0.5)  # coupling strength
    r = 4.0  # fully chaotic logistic parameter
    x = rng.random(N_sites)
    # Transient
    for _ in range(5000):
        f = r * x * (1 - x)
        coupled = (1 - epsilon) * f + (epsilon / 2) * (np.roll(f, 1) + np.roll(f, -1))
        x = np.clip(coupled, 0, 1)
    # Record spatial mean (collective observable)
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        f = r * x * (1 - x)
        coupled = (1 - epsilon) * f + (epsilon / 2) * (np.roll(f, 1) + np.roll(f, -1))
        x = np.clip(coupled, 0, 1)
        vals[i] = x.mean()
    return _to_uint8(vals)


# --- Group B: Smooth-peaked gap (oscillatory + moderate) ---


@source(
    "Kuramoto Oscillators",
    domain="exotic",
    description="Order parameter of coupled oscillators near synchronization transition --- partial coherence",
)
def gen_kuramoto(rng, size):
    """N coupled oscillators with natural frequencies drawn from Cauchy
    distribution. Near critical coupling, the order parameter R(t)
    fluctuates between coherence and incoherence."""
    N_osc = 50
    omega = rng.standard_cauchy(N_osc) * 0.5  # natural frequencies
    theta = rng.uniform(0, 2 * np.pi, N_osc)
    # Critical coupling for Cauchy is K_c = 2*gamma where gamma is scale
    K = rng.uniform(0.8, 1.2)  # near critical
    dt = 0.05
    # Transient
    for _ in range(5000):
        z = np.exp(1j * theta)
        R = np.abs(z.mean())
        psi = np.angle(z.mean())
        theta += (omega + K * R * np.sin(psi - theta)) * dt
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        z = np.exp(1j * theta)
        R = np.abs(z.mean())
        psi = np.angle(z.mean())
        theta += (omega + K * R * np.sin(psi - theta)) * dt
        vals[i] = R
    return _to_uint8(vals)


@source(
    "Van der Pol Oscillator",
    domain="exotic",
    description="Van der Pol relaxation oscillations --- slow drift along cubic nullcline punctuated by fast jumps, "
    "the canonical model of nonlinear self-sustained oscillation",
)
def gen_van_der_pol(rng, size):
    """Van der Pol oscillator x'' - mu*(1-x^2)*x' + x = 0.
    Large mu gives relaxation oscillations: slow drift + fast jumps."""
    mu = rng.uniform(3.0, 8.0)  # strong nonlinearity
    x = rng.uniform(-0.5, 0.5)
    y = rng.uniform(-0.5, 0.5)
    dt = 0.01
    # Transient
    for _ in range(20000):
        dx = y
        dy = mu * (1 - x * x) * y - x
        x += dx * dt
        y += dy * dt
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        dx = y
        dy = mu * (1 - x * x) * y - x
        x += dx * dt
        y += dy * dt
        vals[i] = x
    return _to_uint8(vals)


# --- Group C: Discrete-peaked desert (symbolic + formal) ---


@source(
    "Fibonacci Quasicrystal",
    domain="number_theory",
    description="1D Fibonacci quasicrystal --- aperiodic tiling with long-range order, discrete diffraction",
)
def gen_quasicrystal(rng, size):
    """Cut-and-project method: project 2D square lattice points near a
    line of slope 1/phi onto that line. The gaps are L or S (two tiles),
    giving an aperiodic sequence with discrete Fourier spectrum."""
    phi = (1 + np.sqrt(5)) / 2
    # Generate Fibonacci word: L=1, S=0
    # Use substitution: S->L, L->LS
    seq = [1]  # start with L
    while len(seq) < size * 2:
        new = []
        for s in seq:
            if s == 1:
                new.extend([1, 0])  # L -> LS
            else:
                new.append(1)  # S -> L
        seq = new
    seq = np.array(seq[:size], dtype=np.float64)
    # Add phase offset and modulate to get richer structure
    offset = rng.uniform(0, 2 * np.pi)
    # Map through a quasiperiodic function
    t = np.arange(size, dtype=np.float64)
    # Superimpose the Fibonacci tiling with a quasiperiodic modulation
    vals = seq * 128 + 64 * np.sin(2 * np.pi * t / phi + offset)
    vals += 32 * np.sin(2 * np.pi * t / (phi * phi) + rng.uniform(0, 2 * np.pi))
    return _to_uint8(vals)


@source(
    "Langton's Ant",
    domain="exotic",
    description="Multi-state turmite on small torus --- complex emergent trajectories from simple rules",
)
def gen_langtons_ant(rng, size):
    """Generalized Langton's ant with rule string (e.g. RLR, LLRR).
    On a small torus, the position wraps to create a rich signal.
    Multi-state rules produce much more complex behavior than basic RL."""
    rules = ["RLR", "RLLR", "LLRR", "LRRL", "RLRL", "RRLLL"]
    rule = rules[rng.integers(0, len(rules))]
    n_states = len(rule)
    grid_size = 64
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    x, y = grid_size // 2, grid_size // 2
    dx, dy = 0, 1
    # Random initial configuration
    n_init = rng.integers(100, 500)
    for _ in range(n_init):
        ix, iy = rng.integers(0, grid_size, 2)
        grid[ix, iy] = rng.integers(0, n_states)
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        state = grid[x, y]
        if rule[state] == "R":
            dx, dy = dy, -dx
        else:
            dx, dy = -dy, dx
        grid[x, y] = (state + 1) % n_states
        x = (x + dx) % grid_size
        y = (y + dy) % grid_size
        # Combine x and y into a single value for richer signal
        vals[i] = x + y * 0.618  # irrational weight avoids degeneracy
    return _to_uint8(vals)


@source(
    "Continued Fractions",
    domain="number_theory",
    description="CF digits of random quadratic irrationals --- bounded partial quotients with hidden periodicity",
)
def gen_continued_fractions(rng, size):
    """Continued fraction expansion of sqrt(D) for random D.
    These are eventually periodic (Lagrange theorem) with constrained
    digit distribution. Concatenate several to fill the buffer."""
    vals = []
    while len(vals) < size:
        # Pick a non-square D
        D = rng.integers(2, 10000)
        sqrt_D = np.sqrt(D)
        if sqrt_D == int(sqrt_D):
            continue
        a0 = int(sqrt_D)
        # Generate CF digits
        m, d, a = 0, 1, a0
        seen = {}
        cf_digits = [a0]
        for _ in range(size):
            m = d * a - m
            d = (D - m * m) // d
            if d == 0:
                break
            a = (a0 + m) // d
            cf_digits.append(a)
            state = (m, d)
            if state in seen:
                break
            seen[state] = True
        vals.extend(cf_digits)
    arr = np.array(vals[:size], dtype=np.float64)
    # Log transform to compress the heavy tail of partial quotients
    arr = np.log1p(arr)
    return _to_uint8(arr)


# --- Group D/E: Mixed + very bursty ---


@source(
    "Forest Fire",
    domain="exotic",
    description="Drossel-Schwabl forest fire model --- tree density fluctuates as growth and burns compete",
)
def gen_forest_fire(rng, size):
    """1D forest fire: trees grow with probability p, lightning strikes
    with probability f. Burns propagate to all connected trees.
    Output is tree density over time --- slow buildup with sharp drops."""
    L = 500
    p = rng.uniform(0.03, 0.08)
    f = rng.uniform(0.0005, 0.002)
    grid = np.zeros(L, dtype=np.uint8)
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        # Grow trees
        empty = np.where(grid == 0)[0]
        if len(empty) > 0:
            grow_mask = rng.random(len(empty)) < p
            grid[empty[grow_mask]] = 1
        # Lightning
        if rng.random() < f:
            trees = np.where(grid == 1)[0]
            if len(trees) > 0:
                strike = rng.choice(trees)
                to_burn = [strike]
                burned = set()
                while to_burn:
                    s = to_burn.pop()
                    if s in burned or s < 0 or s >= L or grid[s] == 0:
                        continue
                    burned.add(s)
                    grid[s] = 0
                    to_burn.extend([s - 1, s + 1])
        vals[i] = grid.sum()
    return _to_uint8(vals)


# ================================================================
# Wave 2: targeting remaining voids
# ================================================================

# --- Voids 5, 7: broad + oscillatory (high entropy, smooth) ---


@source(
    "Duffing Oscillator",
    domain="chaos",
    description="Driven damped nonlinear oscillator --- strange attractor with broad amplitude visits",
)
def gen_duffing(rng, size):
    """x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t).
    Chaotic regime visits broad amplitude range while remaining smooth."""
    alpha = -1.0
    beta = 1.0
    delta = rng.uniform(0.15, 0.4)
    gamma = rng.uniform(0.3, 0.5)
    omega = rng.uniform(1.0, 1.4)
    dt = 0.02
    x = rng.uniform(-0.5, 0.5)
    y = rng.uniform(-0.5, 0.5)
    for i in range(20000):
        t = i * dt
        dx = y
        dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
        x += dx * dt
        y += dy * dt
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        t = (i + 20000) * dt
        dx = y
        dy = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
        x += dx * dt
        y += dy * dt
        vals[i] = x
    return _to_uint8(vals)


@source(
    "Pomeau-Manneville",
    domain="chaos",
    description="Pomeau-Manneville type-I intermittency --- near a tangent bifurcation, the orbit lingers in long "
    "laminar phases before erupting into chaotic bursts. Power-law laminar length distribution",
)
def gen_pomeau_manneville(rng, size):
    """x(n+1) = x + x^z (mod 1), z near 2.
    Near the tangent bifurcation: long laminar stretches + chaotic bursts."""
    z = rng.uniform(1.8, 2.2)
    epsilon = rng.uniform(0.001, 0.01)
    x = rng.random()
    for _ in range(5000):
        x = (x + (1 + epsilon) * x**z) % 1.0
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        x = (x + (1 + epsilon) * x**z) % 1.0
        vals[i] = x
    return _to_uint8(vals)


# --- Voids 1, 2, 6: broad + discrete + formal ---


@source(
    "Rudin-Shapiro",
    domain="number_theory",
    description="Partial sums of Rudin-Shapiro --- deterministic random walk with flat spectral density",
)
def gen_rudin_shapiro(rng, size):
    """Cumulative sum of r(n) = (-1)^{f(n)} where f(n) counts overlapping
    11-pairs in binary(n). Grows as O(sqrt(n*log(n))), creating a
    random-walk-like signal that is entirely deterministic but spectrally flat."""
    offset = rng.integers(0, 1000000)
    cumsum = 0.0
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        n = i + offset
        f = 0
        prev = 0
        temp = n
        while temp > 0:
            curr = temp & 1
            if curr == 1 and prev == 1:
                f += 1
            prev = curr
            temp >>= 1
        cumsum += 1.0 if (f % 2 == 0) else -1.0
        vals[i] = cumsum
    return _to_uint8(vals)


@source(
    "De Bruijn Sequence",
    domain="number_theory",
    description="B(4,4) De Bruijn cycle --- every 4-symbol window over alphabet 0-3 appears exactly once",
)
def gen_de_bruijn(rng, size):
    """B(4,4): cycle of length 256 over alphabet {0,1,2,3} where every
    4-gram appears exactly once. Sliding windows of 4 symbols give
    values 0-255. Tile with random offsets per seed."""
    n = 4
    k = 4  # alphabet size
    a = [0] * (n + 1)
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    db_seq = np.array(sequence, dtype=np.uint8)  # length 256
    # Sliding window of 4 symbols -> byte value
    offset = rng.integers(0, len(db_seq))
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        byte = 0
        for d in range(4):
            idx = (offset + i + d) % len(db_seq)
            byte |= db_seq[idx] << (d * 2)
        vals[i] = byte
    return vals


@source(
    "Champernowne",
    domain="number_theory",
    description="Provably normal number in base 256 --- uniform digit distribution but deterministic concatenation",
)
def gen_champernowne(rng, size):
    """Concatenate 0,1,2,... in base 256. Provably normal: every byte
    pattern appears equally often in the limit. But deterministic."""
    offset = rng.integers(0, 1000000)
    vals = []
    n = offset
    while len(vals) < size:
        if n == 0:
            vals.append(0)
        else:
            digits = []
            temp = n
            while temp > 0:
                digits.append(temp & 0xFF)
                temp >>= 8
            vals.extend(reversed(digits))
        n += 1
    return np.array(vals[:size], dtype=np.uint8)


# --- Voids 9, 10: between chaos and symbolic ---


@source(
    "Collatz Flights",
    domain="number_theory",
    description="Peak altitude of Collatz orbits --- how high consecutive integers fly before crashing to 1",
)
def gen_collatz_flights(rng, size):
    """For consecutive integers, compute the maximum value reached during
    the Collatz orbit (the 'excursion height'). These vary wildly ---
    some numbers barely climb, others spike to millions before falling."""
    start = rng.integers(1000, 1000000)
    vals = np.zeros(size, dtype=np.float64)
    for i in range(size):
        x = start + i
        peak = x
        steps = 0
        while x != 1 and steps < 1000:
            if x % 2 == 0:
                x //= 2
            else:
                x = 3 * x + 1
            if x > peak:
                peak = x
            steps += 1
        vals[i] = peak
    # Log transform to compress the enormous range
    vals = np.log1p(vals)
    return _to_uint8(vals)


@source(
    "Stern-Brocot Walk",
    domain="number_theory",
    description="Stern diatomic sequence --- every positive rational visited once, fractal self-similar structure",
)
def gen_stern_brocot(rng, size):
    """s(1)=s(2)=1, s(2n)=s(n), s(2n+1)=s(n)+s(n+1).
    Fractal structure encoding the Stern-Brocot tree of rationals."""
    offset = rng.integers(1, 100000)
    length = offset + size + 1
    s = np.ones(length + 1, dtype=np.int64)
    for i in range(2, length + 1):
        if i % 2 == 0:
            s[i] = s[i // 2]
        else:
            s[i] = s[i // 2] + s[i // 2 + 1]
    chunk = s[offset : offset + size].astype(np.float64)
    return _to_uint8(np.log1p(chunk))


# ================================================================
# Wave 3: Filling the broad+transition voids (PC1 +4 to +13)
# ================================================================

# --- Targeting Voids 1, 9, 13: correlated + moderately constrained ---


@source(
    "Random Telegraph",
    domain="exotic",
    description="Two-state Markov process with exponential holding times --- binary correlation structure",
)
def gen_random_telegraph(rng, size):
    """Symmetric random telegraph signal: stays in state 0 or 1 for
    exponentially distributed durations.  The holding rate controls
    how 'blocky' the signal looks --- between white noise and square wave."""
    rate = rng.uniform(0.005, 0.05)  # mean run length 20-200
    vals = np.empty(size, dtype=np.uint8)
    state = rng.integers(0, 2)
    i = 0
    while i < size:
        hold = int(rng.exponential(1.0 / rate))
        hold = max(hold, 1)
        end = min(i + hold, size)
        vals[i:end] = state
        state = 1 - state
        i = end
    # Scale to full range for richer distribution: add analog noise within each state
    out = vals.astype(np.float64) * 200 + rng.normal(0, 15, size)
    return _to_uint8(out)


@source(
    "Geometric Brownian Motion",
    domain="noise",
    description="Geometric Brownian motion --- multiplicative random walk (dS/S = μdt + σdW), "
    "the basis of Black-Scholes option pricing. Log-normal distribution with positive skew",
)
def gen_geometric_brownian(rng, size):
    """dS/S = mu*dt + sigma*dW.  The log transform makes this a regular
    Brownian motion, but in raw space the distribution is right-skewed
    and the trajectory has multiplicative scaling."""
    mu = rng.uniform(-0.001, 0.001)
    sigma = rng.uniform(0.02, 0.08)
    log_returns = mu + sigma * rng.standard_normal(size)
    log_price = np.cumsum(log_returns)
    return _to_uint8(log_price)


@source(
    "Ornstein-Uhlenbeck",
    domain="noise",
    description="Ornstein-Uhlenbeck process --- mean-reverting diffusion with stationary Gaussian distribution, "
    "the continuous-time analog of an AR(1) process. Bounded excursions unlike Brownian motion",
)
def gen_ornstein_uhlenbeck(rng, size):
    """dx = theta*(mu - x)*dt + sigma*dW.  Stationary distribution is
    Gaussian with variance sigma^2/(2*theta).  The mean-reversion creates
    moderate correlation with bounded excursions."""
    theta = rng.uniform(0.01, 0.1)  # mean-reversion speed
    sigma = rng.uniform(0.5, 2.0)
    x = 0.0
    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        x += theta * (0 - x) + sigma * rng.standard_normal()
        vals[i] = x
    return _to_uint8(vals)


# --- Targeting Voids 5, 6, 8, 20: iid-like + moderately constrained ---


@source(
    "Beta Noise",
    domain="noise",
    description="IID Beta(0.3, 0.3) noise --- U-shaped distribution concentrating mass at both extremes "
    "with a valley in the middle, structurally opposite to Gaussian",
)
def gen_beta_noise(rng, size):
    """Beta distribution with both parameters < 1 gives a U-shaped density
    that concentrates mass near 0 and 1, with relatively little in the
    middle.  This is neither uniform nor peaked --- structurally unique."""
    a = rng.uniform(0.2, 0.5)
    b = rng.uniform(0.2, 0.5)
    vals = rng.beta(a, b, size=size)
    return (vals * 255).astype(np.uint8)


@source(
    "Zipf Distribution",
    domain="exotic",
    description="IID draws from Zipf/power-law distribution --- steep rank-frequency, heavy right tail",
)
def gen_zipf_samples(rng, size):
    """Zipf distribution: P(k) ~ k^{-alpha}.  Most samples cluster at small
    values with occasional large spikes.  We use a mix of Zipf draws and
    their squares to spread the distribution more broadly."""
    alpha = rng.uniform(1.3, 2.0)  # lower alpha = heavier tail = more spread
    vals = rng.zipf(alpha, size=size).astype(np.float64)
    # Double log to compress extreme range while preserving spread
    vals = np.log1p(np.log1p(vals))
    # Add small uniform jitter to break ties at low values
    vals += rng.uniform(0, 0.3, size)
    return _to_uint8(vals)


@source(
    "Benford's Law",
    domain="number_theory",
    description="Benford-distributed significands --- leading-digit law P(d) = log₁₀(1+1/d), "
    "the distribution of first digits in real-world datasets spanning multiple orders of magnitude",
)
def gen_benford(rng, size):
    """Generate numbers whose leading digits follow Benford's law by
    taking 10^U for uniform U, then extracting significands.
    Distribution is logarithmically concentrated at small values."""
    u = rng.uniform(0, rng.uniform(4, 8), size=size)
    vals = 10.0**u
    # Extract significand (first 3 significant digits)
    log_vals = np.log10(np.maximum(vals, 1e-30))
    frac = log_vals - np.floor(log_vals)
    significands = 10**frac  # in [1, 10)
    return _to_uint8(significands)


# --- Targeting Voids 2, 3, 11, 16: near period-doubling but bursty ---


@source(
    "Noisy Period-2",
    domain="chaos",
    description="Period-2 oscillation corrupted by additive noise --- deterministic alternation partially "
    "washed out by Gaussian perturbation, bridging periodic and stochastic regimes",
)
def gen_noisy_period2(rng, size):
    """Deterministic alternation between two levels, perturbed by noise.
    At low noise the sequential structure dominates; at high noise it
    washes out.  We pick an intermediate regime."""
    level_a = rng.uniform(30, 80)
    level_b = rng.uniform(170, 220)
    noise_std = rng.uniform(15, 40)
    vals = np.where(np.arange(size) % 2 == 0, level_a, level_b)
    vals = vals + rng.normal(0, noise_std, size)
    return np.clip(vals, 0, 255).astype(np.uint8)


@source(
    "Intermittent Silence",
    domain="exotic",
    description="Signal with long silent stretches punctuated by active bursts --- on/off intermittency",
)
def gen_intermittent_silence(rng, size):
    """Models processes like neural spike trains or rainfall: mostly zero/quiet
    with occasional active episodes.  The ratio of active to quiet determines
    where it lands on the burstiness axis."""
    p_active = rng.uniform(0.1, 0.3)  # fraction of time active
    burst_len_mean = rng.uniform(20, 100)
    quiet_len_mean = burst_len_mean * (1 - p_active) / p_active
    vals = np.zeros(size, dtype=np.float64)
    i = 0
    active = rng.random() < p_active
    while i < size:
        if active:
            length = max(1, int(rng.exponential(burst_len_mean)))
            end = min(i + length, size)
            vals[i:end] = rng.uniform(100, 255, end - i)
            i = end
        else:
            length = max(1, int(rng.exponential(quiet_len_mean)))
            end = min(i + length, size)
            # Silent period: small noise around zero
            vals[i:end] = rng.uniform(0, 10, end - i)
            i = end
        active = not active
    return vals.astype(np.uint8)


# --- Targeting Voids 4, 7, 17: extremely negative PC3 (deeply bursty) ---


@source(
    "Hawkes Process",
    domain="exotic",
    description="Self-exciting point process --- events trigger more events, creating clustered bursts",
)
def gen_hawkes(rng, size):
    """Hawkes process: intensity lambda(t) = mu + sum of exponential kernels.
    Simulated via discrete-time approximation for speed.
    Records the running intensity, which captures the bursty excitation dynamics."""
    mu = rng.uniform(0.3, 1.5)  # baseline rate
    alpha = rng.uniform(0.4, 0.9)  # excitation strength (< 1 for stability)
    beta = rng.uniform(0.05, 0.2)  # decay rate per step
    # Discrete-time simulation
    intensity = np.empty(size, dtype=np.float64)
    lam = mu
    for i in range(size):
        intensity[i] = lam
        # Event occurs with probability proportional to intensity
        if rng.random() < min(lam * 0.1, 0.95):
            lam += alpha  # excitation kick
        lam = mu + (lam - mu) * (1 - beta)  # exponential decay toward baseline
    return _to_uint8(intensity)


@source(
    "Multiplicative Cascade",
    domain="exotic",
    description="Random multiplicative process on dyadic tree --- multifractal burstiness at all scales",
)
def gen_multiplicative_cascade(rng, size):
    """Binary multiplicative cascade: start with uniform mass, repeatedly
    split each interval with random weight W and (1-W).  The result is a
    multifractal measure with intermittent spikes --- the canonical model
    for turbulence intermittency."""
    # Build cascade on 2^n grid
    n_levels = int(np.ceil(np.log2(max(size, 16))))
    n_bins = 2**n_levels
    measure = np.ones(n_bins, dtype=np.float64)
    for level in range(n_levels):
        step = n_bins >> level
        half = step >> 1
        for start in range(0, n_bins, step):
            w = rng.beta(2, 2)  # weight in [0,1], centered
            measure[start : start + half] *= 2 * w
            measure[start + half : start + step] *= 2 * (1 - w)
    # Downsample to requested size
    if n_bins > size:
        chunk = n_bins // size
        measure = measure[: size * chunk].reshape(size, chunk).mean(axis=1)
    else:
        measure = measure[:size]
    return _to_uint8(np.log1p(measure))


@source(
    "Spike Train",
    domain="exotic",
    description="Integrate-and-fire neuron model --- quiescent charging with sudden discharge spikes",
)
def gen_spike_train(rng, size):
    """Leaky integrate-and-fire neuron: membrane voltage charges toward
    threshold, fires a spike, resets.  Input current has noisy fluctuations.
    The output is raw membrane voltage --- mostly sub-threshold with sharp spikes."""
    tau = rng.uniform(10, 50)  # membrane time constant
    threshold = rng.uniform(0.8, 1.2)
    reset = 0.0
    I_mean = rng.uniform(0.02, 0.06)  # mean input current (just above threshold)
    I_std = rng.uniform(0.01, 0.04)
    v = 0.0
    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        I = I_mean + I_std * rng.standard_normal()
        v += -v / tau + I
        if v >= threshold:
            vals[i] = threshold * 1.5  # spike overshoot
            v = reset
        else:
            vals[i] = v
    return _to_uint8(vals)


# ================================================================
# Financial --- real market data from gemini-alpha
# ================================================================

_BTC_CSV_DATA = None


def _get_btc_csv():
    """Load BTC/USD hourly OHLCV from CSV (~96K rows, 2015-2026). Cached."""
    global _BTC_CSV_DATA
    if _BTC_CSV_DATA is not None:
        return _BTC_CSV_DATA
    from pathlib import Path

    csv_path = (
        Path(__file__).resolve().parents[1]
        / "gemini-alpha"
        / "data"
        / "qc_official"
        / "btcusd.csv"
    )
    if not csv_path.exists():
        csv_path = Path("/Volumes/chonk/projects/gemini-alpha/data/qc_official/btcusd.csv")
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                try:
                    rows.append(
                        [
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                            float(parts[5]),
                        ]
                    )
                except ValueError:
                    continue
    # columns: open, high, low, close, volume
    _BTC_CSV_DATA = np.array(rows, dtype=np.float64)
    return _BTC_CSV_DATA


def _btc_chunk(rng, size, col_idx, transform=None):
    """Extract a contiguous chunk from BTC CSV data."""
    d = _get_btc_csv()
    if d is None:
        return rng.integers(0, 256, size, dtype=np.uint8)
    series = d[:, col_idx]
    if transform:
        series = transform(series)
    max_start = max(0, len(series) - size)
    start = rng.integers(0, max_start + 1)
    chunk = series[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="edge")
    return _to_uint8(chunk)


@source(
    "BTC Close Price",
    domain="financial",
    description="Bitcoin hourly closing prices --- contiguous 16KB block sampled from 96K bars (2015-2026)",
)
def gen_btc_close(rng, size):
    return _btc_chunk(rng, size, 3)


@source(
    "BTC Returns",
    domain="financial",
    description="Bitcoin hourly log-returns --- heavy tails (kurtosis ~18), volatility clustering, and leverage effects. "
    "The canonical non-Gaussian financial time series",
)
def gen_btc_returns(rng, size):
    def log_returns(close):
        return np.diff(np.log(np.maximum(close, 1e-8)))

    return _btc_chunk(rng, size, 3, transform=log_returns)


@source(
    "BTC Volatility",
    domain="financial",
    description="Bitcoin realized volatility --- rolling 24h std of log-returns, exhibits long memory",
)
def gen_btc_volatility(rng, size):
    def rolling_vol(close):
        lr = np.diff(np.log(np.maximum(close, 1e-8)))
        # Vectorized rolling std via cumsum trick
        w = 24
        cs = np.cumsum(np.concatenate([[0], lr]))
        cs2 = np.cumsum(np.concatenate([[0], lr**2]))
        n = len(lr)
        vol = np.zeros(n)
        for i in range(w, n):
            s = cs[i] - cs[i - w]
            s2 = cs2[i] - cs2[i - w]
            vol[i] = np.sqrt(max(0, s2 / w - (s / w) ** 2))
        return vol[w:]  # drop warmup

    return _btc_chunk(rng, size, 3, transform=rolling_vol)


@source(
    "BTC Volume",
    domain="financial",
    description="Bitcoin hourly trading volume (log-transformed) --- extreme burstiness with power-law spikes, "
    "strong autocorrelation, and intraday seasonality",
)
def gen_btc_volume(rng, size):
    def log_volume(vol):
        return np.log1p(np.maximum(vol, 0))

    return _btc_chunk(rng, size, 4, transform=log_volume)


@source(
    "BTC Range",
    domain="financial",
    description="Bitcoin hourly high-low range --- intrabar volatility proxy with characteristic clustering",
)
def gen_btc_range(rng, size):
    d = _get_btc_csv()
    if d is None:
        return rng.integers(0, 256, size, dtype=np.uint8)
    hl_range = np.log1p(np.maximum(d[:, 1] - d[:, 2], 0))
    max_start = max(0, len(hl_range) - size)
    start = rng.integers(0, max_start + 1)
    chunk = hl_range[start : start + size]
    if len(chunk) < size:
        chunk = np.pad(chunk, (0, size - len(chunk)), mode="edge")
    return _to_uint8(chunk)


@source(
    "ETH/BTC Ratio",
    domain="financial",
    description="Ethereum-to-Bitcoin price ratio --- cross-asset relative strength with regime shifts",
)
def gen_eth_btc_ratio(rng, size):
    try:
        import pandas as pd

        btc_path = "/Volumes/chonk/projects/gemini-alpha/data/historical/BTC_USD/1h_20240126_20260124.parquet"
        eth_path = "/Volumes/chonk/projects/gemini-alpha/data/historical/ETH_USD/1h_20240126_20260124.parquet"
        btc = pd.read_parquet(btc_path)["close"].values
        eth = pd.read_parquet(eth_path)["close"].values
        n = min(len(btc), len(eth))
        ratio = eth[:n] / np.maximum(btc[:n], 1e-8)
        max_start = max(0, len(ratio) - size)
        start = rng.integers(0, max_start + 1)
        chunk = ratio[start : start + size]
        if len(chunk) < size:
            chunk = np.pad(chunk, (0, size - len(chunk)), mode="edge")
        return _to_uint8(chunk)
    except Exception:
        return rng.integers(0, 256, size, dtype=np.uint8)


# ================================================================
# Quantum --- simulated quantum systems
# ================================================================

DOMAIN_COLORS["quantum"] = "#00cec9"


@source(
    "GUE Spacings",
    domain="quantum",
    description="Gaussian Unitary Ensemble spacings --- level repulsion P(s)~s² exp(-4s²/π), broken time-reversal symmetry",
)
def gen_gue_spacings(rng, size):
    """GUE: complex Hermitian random matrices. Eigenvalue spacings exhibit
    quadratic level repulsion --- nearby levels repel strongly, unlike
    uncorrelated Poisson spectra. Models quantum systems without
    time-reversal symmetry (e.g. in magnetic fields)."""
    spacings = []
    while len(spacings) < size:
        dim = 64
        A = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(
            2 * dim
        )
        H = (A + A.conj().T) / 2
        eigs = np.sort(np.linalg.eigvalsh(H))
        s = np.diff(eigs)
        mean_s = np.mean(s)
        if mean_s > 1e-15:
            s /= mean_s  # unfold
        spacings.extend(s.tolist())
    return _to_uint8(np.array(spacings[:size]))


@source(
    "GOE Spacings",
    domain="quantum",
    description="Gaussian Orthogonal Ensemble spacings --- linear repulsion P(s)~s exp(-πs²/4), time-reversal symmetric",
)
def gen_goe_spacings(rng, size):
    """GOE: real symmetric random matrices. Describes nuclear energy levels
    and other time-reversal invariant quantum systems. Spacing repulsion
    is linear (weaker than GUE's quadratic)."""
    spacings = []
    while len(spacings) < size:
        dim = 64
        A = rng.standard_normal((dim, dim)) / np.sqrt(2 * dim)
        H = (A + A.T) / 2
        eigs = np.sort(np.linalg.eigvalsh(H))
        s = np.diff(eigs)
        mean_s = np.mean(s)
        if mean_s > 1e-15:
            s /= mean_s
        spacings.extend(s.tolist())
    return _to_uint8(np.array(spacings[:size]))


@source(
    "Quantum Walk",
    domain="quantum",
    description="Hadamard walk on Z --- ballistic spreading (σ~t not √t), interference creates asymmetric peaks",
)
def gen_quantum_walk(rng, size):
    """Discrete quantum walk: evolve for N steps on a lattice of 2N+1 sites,
    then output the final probability distribution resampled to `size` points.
    The distribution has characteristic peaks at ±N/√2 from ballistic spreading."""
    N = 2048  # walk steps --- enough for rich structure
    n_sites = 2 * N + 1
    mid = N
    psi = np.zeros((n_sites, 2), dtype=np.complex128)
    theta = rng.uniform(0, 2 * np.pi)
    psi[mid, 0] = np.cos(theta)
    psi[mid, 1] = np.sin(theta) * np.exp(1j * rng.uniform(0, 2 * np.pi))

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    for t in range(N):
        coined = psi @ H.T
        psi_new = np.zeros_like(psi)
        psi_new[:-1, 0] += coined[1:, 0]
        psi_new[1:, 1] += coined[:-1, 1]
        psi = psi_new

    # Final probability distribution
    prob = np.sum(np.abs(psi) ** 2, axis=1)
    # Resample to output size via interpolation
    x_orig = np.linspace(0, 1, len(prob))
    x_new = np.linspace(0, 1, size)
    vals = np.interp(x_new, x_orig, prob)
    return _to_uint8(vals)


@source(
    "Kicked Rotor",
    domain="quantum",
    description="Quantum kicked rotor --- dynamical localization freezes classical chaos, Anderson localization in momentum space",
)
def gen_kicked_rotor(rng, size):
    """QKR: quantum version of the Chirikov standard map. For K>0.97 the
    classical map is chaotic with diffusive momentum growth. Quantum
    mechanically, interference causes Anderson-like localization after
    a break time t* ~ D_cl/ℏ², freezing the energy growth."""
    N = 256
    K = rng.uniform(3.0, 8.0)
    hbar_eff = rng.uniform(0.5, 2.0)
    ns = np.arange(N) - N // 2
    free_phase = np.exp(-1j * hbar_eff * ns**2 / 2)
    thetas = 2 * np.pi * np.arange(N) / N
    kick_phase = np.exp(-1j * K * np.cos(thetas) / hbar_eff)

    psi = np.zeros(N, dtype=np.complex128)
    psi[N // 2] = 1.0

    vals = np.zeros(size)
    for t in range(size):
        psi *= free_phase
        psi_x = np.fft.fft(psi)
        psi_x *= kick_phase
        psi = np.fft.ifft(psi_x)
        vals[t] = np.sum(ns**2 * np.abs(psi) ** 2)
    return _to_uint8(vals)


@source(
    "Berry Random Wave",
    domain="quantum",
    description="Berry's random plane wave --- superposition model for chaotic eigenstate statistics, Gaussian amplitude distribution",
)
def gen_berry_wave(rng, size):
    """Berry's conjecture: eigenstates of classically chaotic billiards
    look like random superpositions of plane waves. Amplitude follows
    Gaussian distribution, spatial correlations decay as Bessel J_0."""
    n_waves = rng.integers(50, 200)
    k = rng.uniform(5.0, 20.0)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    angles = rng.uniform(0, 2 * np.pi, n_waves)
    x = np.linspace(0, 10 * np.pi, size)
    wave = np.zeros(size)
    for i in range(n_waves):
        wave += np.cos(k * np.cos(angles[i]) * x + phases[i])
    wave /= np.sqrt(n_waves)
    return _to_uint8(wave)


@source(
    "Wigner Semicircle",
    domain="quantum",
    description="Eigenvalue density from large random matrices --- converges to Wigner semicircle law ρ(x)=√(4-x²)/(2π)",
)
def gen_wigner_semicircle(rng, size):
    """Direct eigenvalue samples from GOE. Empirical density converges to
    the semicircle distribution --- a fundamental universality result."""
    eigs = []
    while len(eigs) < size:
        dim = 128
        A = rng.standard_normal((dim, dim)) / np.sqrt(2 * dim)
        H = (A + A.T) / 2
        eigs.extend(np.linalg.eigvalsh(H).tolist())
    return _to_uint8(np.array(eigs[:size]))


@source(
    "Entanglement Entropy",
    domain="quantum",
    description="Page curve entanglement entropy of random bipartite states --- ⟨S⟩ ≈ log(d_A) - d_A/(2d_B)",
)
def gen_entanglement_entropy(rng, size):
    """For a random state in H_A⊗H_B, the entanglement entropy
    S = -Tr(ρ_A log ρ_A) follows the Page distribution."""
    d_A, d_B = 4, 8
    vals = np.zeros(size)
    for i in range(size):
        psi = rng.standard_normal(d_A * d_B) + 1j * rng.standard_normal(d_A * d_B)
        psi /= np.linalg.norm(psi)
        M = psi.reshape(d_A, d_B)
        rho_A = M @ M.conj().T
        eigs = np.linalg.eigvalsh(rho_A)
        eigs = eigs[eigs > 1e-15]
        vals[i] = -np.sum(eigs * np.log(eigs))
    return _to_uint8(vals)


@source(
    "Poisson Spacings",
    domain="quantum",
    description="Uncorrelated Poisson spacings P(s)=exp(-s) --- integrable quantum systems, no level repulsion",
)
def gen_poisson_spacings(rng, size):
    """Spacing distribution for integrable (non-chaotic) quantum systems
    where energy levels are uncorrelated. Contrast with GOE/GUE."""
    return _to_uint8(rng.exponential(1.0, size))


# ================================================================
# Bio --- dynamical systems models
# ================================================================


@source(
    "Lotka-Volterra",
    domain="bio",
    description="Predator-prey oscillations --- nonlinear limit cycles with phase-shifted population waves",
)
def gen_lotka_volterra(rng, size):
    """Lotka-Volterra predator-prey: dx/dt = αx - βxy, dy/dt = δxy - γy.
    Produces conservative oscillations around a fixed point. We output the
    prey population x(t), which shows characteristic asymmetric cycles ---
    fast crashes followed by slow recoveries."""
    alpha = rng.uniform(0.8, 1.5)  # prey growth rate
    beta = rng.uniform(0.4, 1.0)  # predation rate
    delta = rng.uniform(0.2, 0.6)  # predator growth from eating
    gamma = rng.uniform(0.6, 1.2)  # predator death rate
    x = rng.uniform(1.0, 4.0)  # initial prey
    y = rng.uniform(0.5, 2.0)  # initial predator
    dt = 0.02
    # Warmup to settle onto orbit
    for _ in range(2000):
        dx = (alpha * x - beta * x * y) * dt
        dy = (delta * x * y - gamma * y) * dt
        x = max(x + dx, 1e-6)
        y = max(y + dy, 1e-6)
    # Record prey population
    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        dx = (alpha * x - beta * x * y) * dt
        dy = (delta * x * y - gamma * y) * dt
        x = max(x + dx, 1e-6)
        y = max(y + dy, 1e-6)
        vals[i] = x
    return _to_uint8(vals)


@source(
    "SIR Epidemic",
    domain="bio",
    description="Stochastic SIR model --- infection waves with exponential rise, peak overshoot, and power-law decay",
)
def gen_sir_epidemic(rng, size):
    """Stochastic SIR (Susceptible-Infected-Recovered) epidemic model using
    Gillespie-like tau-leaping. Output is the infected fraction I(t).
    Multiple outbreaks can occur as susceptible pool regenerates (SIRS
    with waning immunity)."""
    beta_param = rng.uniform(0.3, 0.8)  # transmission rate
    gamma_param = rng.uniform(0.05, 0.2)  # recovery rate
    xi = rng.uniform(0.001, 0.01)  # immunity waning rate
    N = 10000
    S = int(N * rng.uniform(0.7, 0.95))
    I = int(N * rng.uniform(0.01, 0.05))
    R = N - S - I
    dt = 0.1
    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        # Stochastic rates
        new_infected = rng.poisson(max(0, beta_param * S * I / N * dt))
        new_recovered = rng.poisson(max(0, gamma_param * I * dt))
        new_susceptible = rng.poisson(max(0, xi * R * dt))
        new_infected = min(new_infected, S)
        new_recovered = min(new_recovered, I)
        new_susceptible = min(new_susceptible, R)
        S = S - new_infected + new_susceptible
        I = I + new_infected - new_recovered
        R = R + new_recovered - new_susceptible
        vals[i] = I / N
    return _to_uint8(vals)


@source(
    "Hodgkin-Huxley",
    domain="bio",
    description="Biophysical neuron --- action potentials with Na⁺/K⁺ channel kinetics, refractory periods",
)
def gen_hodgkin_huxley(rng, size):
    """Hodgkin-Huxley model: 4 coupled ODEs for membrane voltage V and
    gating variables m, h, n. The original 1952 squid giant axon model.
    We inject noisy current to produce irregular spike trains."""
    # Constants (Hodgkin & Huxley 1952)
    C_m = 1.0  # membrane capacitance (μF/cm²)
    g_Na = 120.0  # max Na conductance
    g_K = 36.0  # max K conductance
    g_L = 0.3  # leak conductance
    E_Na = 50.0  # Na reversal potential (mV)
    E_K = -77.0  # K reversal potential
    E_L = -54.387  # leak reversal potential

    def alpha_m(V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)) if abs(V + 40) > 1e-7 else 1.0

    def beta_m(V):
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(V):
        return 1.0 / (1 + np.exp(-(V + 35) / 10))

    def alpha_n(V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)) if abs(V + 55) > 1e-7 else 0.1

    def beta_n(V):
        return 0.125 * np.exp(-(V + 65) / 80)

    V = -65.0
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))

    I_mean = rng.uniform(6.0, 12.0)  # mean injected current (above threshold ~6 μA/cm²)
    I_std = rng.uniform(1.0, 4.0)
    dt = 0.04  # ms

    # Warmup
    for _ in range(5000):
        I_ext = I_mean + I_std * rng.standard_normal()
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)
        dV = (I_ext - I_Na - I_K - I_L) / C_m * dt
        V += dV
        m += (alpha_m(V) * (1 - m) - beta_m(V) * m) * dt
        h += (alpha_h(V) * (1 - h) - beta_h(V) * h) * dt
        n += (alpha_n(V) * (1 - n) - beta_n(V) * n) * dt
        m = np.clip(m, 0, 1)
        h = np.clip(h, 0, 1)
        n = np.clip(n, 0, 1)

    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        I_ext = I_mean + I_std * rng.standard_normal()
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)
        dV = (I_ext - I_Na - I_K - I_L) / C_m * dt
        V += dV
        m += (alpha_m(V) * (1 - m) - beta_m(V) * m) * dt
        h += (alpha_h(V) * (1 - h) - beta_h(V) * h) * dt
        n += (alpha_n(V) * (1 - n) - beta_n(V) * n) * dt
        m = np.clip(m, 0, 1)
        h = np.clip(h, 0, 1)
        n = np.clip(n, 0, 1)
        vals[i] = V
    return _to_uint8(vals)


# ================================================================
# Motion --- physical dynamics models
# ================================================================


@source(
    "Damped Pendulum",
    domain="motion",
    description="Nonlinear pendulum with friction --- transient chaos at high amplitude, decay to limit cycle or fixed point",
)
def gen_damped_pendulum(rng, size):
    """Nonlinear pendulum: d²θ/dt² = -γ dθ/dt - (g/L) sin(θ) + A cos(ωt).
    With driving force A, exhibits period-doubling route to chaos.
    Output is angular velocity dθ/dt."""
    gamma = rng.uniform(0.1, 0.5)  # damping
    g_L = rng.uniform(1.0, 3.0)  # g/L
    A = rng.uniform(0.5, 2.0)  # driving amplitude
    omega = rng.uniform(0.5, 2.0)  # driving frequency
    theta = rng.uniform(-np.pi, np.pi)
    omega_dot = rng.uniform(-1.0, 1.0)
    dt = 0.02
    t = 0.0
    # Warmup past transients
    for _ in range(5000):
        acc = -gamma * omega_dot - g_L * np.sin(theta) + A * np.cos(omega * t)
        omega_dot += acc * dt
        theta += omega_dot * dt
        t += dt
    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        acc = -gamma * omega_dot - g_L * np.sin(theta) + A * np.cos(omega * t)
        omega_dot += acc * dt
        theta += omega_dot * dt
        t += dt
        vals[i] = omega_dot
    return _to_uint8(vals)


@source(
    "Projectile with Drag",
    domain="motion",
    description="Ballistic arcs with quadratic air resistance --- asymmetric trajectories, terminal velocity approach",
)
def gen_projectile_drag(rng, size):
    """Repeated ballistic launches with quadratic drag: F_drag = -½ρCdA|v|v.
    Each launch has random initial angle and speed. Output is the altitude
    sequence across many trajectories, producing a quasi-periodic sawtooth."""
    g = 9.81
    drag_coeff = rng.uniform(0.01, 0.05)  # combined ½ρCdA/m
    vals = np.empty(size, dtype=np.float64)
    idx = 0
    while idx < size:
        v0 = rng.uniform(20.0, 80.0)
        angle = rng.uniform(0.3, 1.2)  # radians
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        x, y = 0.0, 0.0
        dt = 0.01
        while idx < size:
            speed = np.sqrt(vx**2 + vy**2)
            ax = -drag_coeff * speed * vx
            ay = -g - drag_coeff * speed * vy
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            if y < 0 and vy < 0:
                vals[idx] = 0.0
                idx += 1
                break
            vals[idx] = max(y, 0.0)
            idx += 1
    return _to_uint8(vals)


@source(
    "Langevin Double-Well",
    domain="motion",
    description="Brownian particle in double-well potential --- noise-driven switching between metastable states (Kramers problem)",
)
def gen_langevin_double_well(rng, size):
    """Overdamped Langevin dynamics: dx/dt = -V'(x) + √(2D)ξ(t), where
    V(x) = x⁴/4 - x²/2 is a symmetric double-well. The particle hops
    between minima at x=±1 with Kramers rate ~ exp(-ΔV/D). The resulting
    time series is a telegraph-like signal with stochastic switching."""
    D = rng.uniform(0.15, 0.5)  # noise intensity (controls switching rate)
    x = rng.choice([-1.0, 1.0])  # start in one well
    dt = 0.01
    sqrt_2D_dt = np.sqrt(2 * D * dt)
    # Warmup
    for _ in range(2000):
        x += (-(x**3) + x) * dt + sqrt_2D_dt * rng.standard_normal()
    vals = np.empty(size, dtype=np.float64)
    for i in range(size):
        x += (-(x**3) + x) * dt + sqrt_2D_dt * rng.standard_normal()
        vals[i] = x
    return _to_uint8(vals)
