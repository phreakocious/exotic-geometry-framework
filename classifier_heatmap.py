#!/usr/bin/env python3
"""
Generate a differentiation heatmap: for each signature in the library,
generate a fresh sample, classify it against all 48 signatures, and
plot the median z-scores as a heatmap.

Diagonal = self-match (should be low). Off-diagonal = cross-match.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from exotic_geometry_framework import GeometryAnalyzer
from train_signature import gen_random, gen_henon, gen_logistic
import importlib.util
import json
import os


def get_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chaos_mod = get_module("investigations/1d/chaos.py")
cipher_mod = get_module("investigations/1d/ciphers.py")
prng_mod = get_module("investigations/1d/prng.py")
hash_mod = get_module("investigations/1d/hashes.py")
numthy_mod = get_module("investigations/1d/number_theory.py")
primes_mod = get_module("investigations/1d/primes.py")
collatz_mod = get_module("investigations/1d/collatz.py")
nn_mod = get_module("investigations/1d/nn_weights.py")
dna_mod = get_module("investigations/1d/dna.py")
comp_mod = get_module("investigations/1d/compression_algos.py")


# ── Generators for each signature ────────────────────────────────────
# Maps signature name → callable(seed, size) → uint8 array

def _perlin(seed, size):
    rng = np.random.RandomState(seed)
    scale = rng.uniform(5, 50)
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

def _gaussian(seed, size):
    rng = np.random.RandomState(seed)
    return np.clip(rng.normal(128, 40, size), 0, 255).astype(np.uint8)

def _pink(seed, size):
    rng = np.random.RandomState(seed)
    white = rng.normal(0, 1, size)
    freqs = np.fft.rfftfreq(size); freqs[0] = 1
    fft = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=size)
    pink = (pink - pink.min()) / (pink.max() - pink.min() + 1e-10)
    return (pink * 255).astype(np.uint8)

def _sine(seed, size):
    rng = np.random.RandomState(seed)
    freq = rng.uniform(5, 80)
    x = np.linspace(0, freq * 2 * np.pi, size)
    return ((np.sin(x) + 1) * 127.5).astype(np.uint8)

def _square(seed, size):
    rng = np.random.RandomState(seed)
    freq = rng.uniform(5, 80)
    x = np.linspace(0, freq * 2 * np.pi, size)
    return (128 + 127 * np.sign(np.sin(x))).astype(np.uint8)

def _sawtooth(seed, size):
    rng = np.random.RandomState(seed)
    freq = rng.uniform(5, 80)
    x = np.linspace(0, 1, size)
    return ((((x * freq) + rng.uniform()) % 1.0) * 255).astype(np.uint8)

def _dna_wrapped(dna_type, seed, size):
    rng = np.random.default_rng(seed)
    if dna_type == 'ecoli':
        seq = dna_mod.gen_ecoli(size, rng)
    elif dna_type == 'human':
        seq = dna_mod.gen_human(size, rng)
    elif dna_type == 'viral':
        seq = dna_mod.gen_viral(size, rng)
    return dna_mod.dna_to_bytes(seq)

def _compressed(alg, seed, size):
    source = comp_mod.generate_source('english', seed, size=10000)
    compressed = comp_mod.compress_data(source, alg)
    arr = np.frombuffer(compressed, dtype=np.uint8)
    if len(arr) < size:
        res = np.zeros(size, dtype=np.uint8)
        res[:len(arr)] = arr
        return res
    return arr[:size]

def _nn(key, seed, size):
    return nn_mod.GENERATORS[key](np.random.default_rng(seed), n=size)


GENERATORS = {
    # Original set
    "Random":                    gen_random,
    "Henon Chaos":               gen_henon,
    "Logistic Chaos":            gen_logistic,
    "RANDU":                     lambda s, sz: prng_mod.generate_prng_data('randu', s, sz),
    "glibc LCG":                 lambda s, sz: prng_mod.generate_prng_data('lcg_glibc', s, sz),
    "AES-ECB (Structured)":      lambda s, sz: cipher_mod.generate_cipher_data('aes_ecb', s, 'structured'),
    "Viral DNA":                 lambda s, sz: _dna_wrapped('viral', s, sz),
    "Bacterial DNA (E. coli)":   lambda s, sz: _dna_wrapped('ecoli', s, sz),
    "Eukaryotic DNA (Human)":    lambda s, sz: _dna_wrapped('human', s, sz),
    "Collatz (Hailstone)":       lambda s, sz: collatz_mod.generate_collatz_data('hailstone_small', s, sz),
    "Collatz (Parity)":          lambda s, sz: collatz_mod.generate_collatz_data('parity', s, sz),
    "Prime Gaps":                lambda s, sz: primes_mod.generate_prime_data('prime_gaps', s, sz,
                                     start_idx=np.random.RandomState(s).randint(100, 50_000)),
    "Lorenz Attractor (X)":      lambda s, sz: chaos_mod.generate_chaotic_data('lorenz_x', s, sz),
    "Rossler Attractor":         lambda s, sz: chaos_mod.generate_chaotic_data('rossler', s, sz),
    "Sine Wave":                 _sine,
    "Square Wave":               _square,
    "Sawtooth Wave":             _sawtooth,
    "Zlib Compressed":           lambda s, sz: _compressed('zlib', s, sz),
    "BZip2 Compressed":          lambda s, sz: _compressed('bz2', s, sz),
    # New set
    "Tent Map":                  lambda s, sz: chaos_mod.generate_chaotic_data('tent', s, sz),
    "Logistic Edge-of-Chaos":    lambda s, sz: chaos_mod.generate_chaotic_data('logistic_edge', s, sz),
    "Standard Map (Chirikov)":   lambda s, sz: chaos_mod.generate_chaotic_data('standard_map', s, sz),
    "Baker Map":                 lambda s, sz: chaos_mod.generate_chaotic_data('baker', s, sz),
    "Divisor Count d(n)":        lambda s, sz: numthy_mod.generate_number_theory('divisor_count', s, sz,
                                     start_idx=np.random.RandomState(s).randint(1000, 500_000)),
    "Totient Ratio":             lambda s, sz: numthy_mod.generate_number_theory('totient_ratio', s, sz),
    "Moebius Function":          lambda s, sz: numthy_mod.generate_number_theory('moebius', s, sz),
    "Mertens Function":          lambda s, sz: numthy_mod.generate_number_theory('mertens_mod256', s, sz),
    "Prime Gap Pairs":           lambda s, sz: primes_mod.generate_prime_data('gap_pairs', s, sz),
    "Prime Gap Diffs":           lambda s, sz: primes_mod.generate_prime_data('gap_diff', s, sz),
    "Collatz Stopping Times":    lambda s, sz: collatz_mod.generate_collatz_data('stopping_times', s, sz),
    "Collatz High Bits":         lambda s, sz: collatz_mod.generate_collatz_data('high_bits', s, sz),
    "NN Trained Dense":          lambda s, sz: _nn('trained_dense', s, sz),
    "NN Pruned 90%":             lambda s, sz: _nn('pruned_90pct', s, sz),
    "Perlin Noise":              _perlin,
    "Gaussian White Noise":      _gaussian,
    "Pink Noise":                _pink,
}


def main():
    print("Initializing analyzer...")
    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get signature names in library order
    sig_dir = "signatures"
    sig_names = []
    for fn in sorted(os.listdir(sig_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(sig_dir, fn)) as f:
                sig_names.append(json.load(f)["name"])

    n_sigs = len(sig_names)
    print(f"Library: {n_sigs} signatures")

    # Only generate samples for signatures we have generators for
    gen_names = [s for s in sig_names if s in GENERATORS]
    n_gen = len(gen_names)
    print(f"Generators available: {n_gen}/{n_sigs}")
    missing = [s for s in sig_names if s not in GENERATORS]
    if missing:
        print(f"  Missing generators: {missing}")

    # Build z-score matrix: rows = generated sample, cols = signature
    z_matrix = np.full((n_gen, n_sigs), np.nan)
    match_matrix = np.full((n_gen, n_sigs), np.nan)

    for i, gen_name in enumerate(gen_names):
        print(f"  [{i+1}/{n_gen}] Classifying {gen_name}...", flush=True)
        data = GENERATORS[gen_name](777, 2000)
        rankings = analyzer.classify(data)

        rank_map = {r["system"]: r for r in rankings}
        for j, sig_name in enumerate(sig_names):
            if sig_name in rank_map:
                z_matrix[i, j] = rank_map[sig_name]["median_z"]
                match_matrix[i, j] = rank_map[sig_name]["match_fraction"]

    # ── Print summary stats ──────────────────────────────────────────
    print(f"\n{'='*64}")
    print("DIFFERENTIATION SUMMARY")
    print(f"{'='*64}")

    n_correct = 0
    n_top3 = 0
    for i, gen_name in enumerate(gen_names):
        row = z_matrix[i]
        best_j = np.nanargmin(row)
        best_name = sig_names[best_j]
        self_j = sig_names.index(gen_name)
        self_z = row[self_j]
        best_z = row[best_j]

        sorted_idx = np.argsort(row)
        rank = np.where(sorted_idx == self_j)[0][0] + 1

        correct = (best_name == gen_name)
        if correct:
            n_correct += 1
        if rank <= 3:
            n_top3 += 1

        marker = "ok" if correct else f"MISS (got {best_name})"
        print(f"  {gen_name:<30} self_z={self_z:.2f}  rank={rank:<3} {marker}")

    print(f"\nExact top-1: {n_correct}/{n_gen} ({100*n_correct/n_gen:.0f}%)")
    print(f"Within top-3: {n_top3}/{n_gen} ({100*n_top3/n_gen:.0f}%)")

    # ── Heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 18))
    fig.patch.set_facecolor('#181818')
    ax.set_facecolor('#181818')

    # Clip for display
    z_display = np.clip(z_matrix, 0.1, 10)

    im = ax.imshow(z_display, cmap='RdYlGn_r', aspect='auto',
                   norm=LogNorm(vmin=0.2, vmax=8))

    # Mark the diagonal (self-match) and top-1 match
    for i in range(n_gen):
        self_j = sig_names.index(gen_names[i])
        best_j = np.nanargmin(z_matrix[i])
        # Outline self-match cell
        ax.add_patch(plt.Rectangle((self_j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='white', linewidth=1.5))
        # Star the actual top match if different from self
        if best_j != self_j:
            ax.plot(best_j, i, marker='x', color='white', markersize=6, markeredgewidth=1.5)

    ax.set_xticks(range(n_sigs))
    ax.set_xticklabels(sig_names, rotation=90, fontsize=7, color='#cccccc')
    ax.set_yticks(range(n_gen))
    ax.set_yticklabels(gen_names, fontsize=7, color='#cccccc')

    ax.set_xlabel("Signature in library", color='#cccccc', fontsize=10)
    ax.set_ylabel("Generated sample", color='#cccccc', fontsize=10)
    ax.set_title(f"Classifier Differentiation: {n_sigs} Signatures\n"
                 f"(green = strong match, red = poor match, "
                 f"white box = self, x = top match if not self)",
                 color='#cccccc', fontsize=12, pad=15)
    ax.tick_params(colors='#888888')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label='Median |z-score|')
    cbar.ax.yaxis.label.set_color('#cccccc')
    cbar.ax.tick_params(colors='#888888')

    plt.tight_layout()
    plt.savefig("figures/classifier_heatmap.png", dpi=180,
                facecolor='#181818', bbox_inches='tight')
    print(f"\nSaved figures/classifier_heatmap.png")


if __name__ == "__main__":
    main()
