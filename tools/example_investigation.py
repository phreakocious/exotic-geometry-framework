#!/usr/bin/env python3
"""
Example Investigation: White Noise vs Colored Noise
====================================================

Minimal example showing the Runner API. Tests whether exotic geometries
can distinguish white noise from pink noise and brown noise.

DIRECTIONS:
D1: Noise taxonomy — white, pink, brown pairwise
D2: Sequential structure — each vs shuffled
D3: Spectral slope sweep — detection vs spectral exponent
D4: Length sensitivity — detection vs chunk size
D5: Delay embedding — does tau amplify noise structure?
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.investigation_runner import Runner
from exotic_geometry_framework import delay_embed


# ==============================================================
# DATA GENERATORS
# ==============================================================
def generate_white(rng, size):
    """Uniform random bytes."""
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def generate_colored(rng, size, exponent=1.0):
    """Generate 1/f^exponent noise, quantized to uint8."""
    freqs = np.fft.rfftfreq(size * 2, d=1.0)[1:]
    power = freqs ** (-exponent / 2)
    phases = rng.uniform(0, 2 * np.pi, len(power))
    spectrum = power * np.exp(1j * phases)
    spectrum = np.concatenate([[0], spectrum])
    signal = np.fft.irfft(spectrum, n=size * 2)[:size]
    # Normalize to 0-255
    signal = signal - signal.min()
    if signal.max() > 0:
        signal = signal / signal.max() * 255
    return signal.astype(np.uint8)


def generate_pink(rng, size):
    return generate_colored(rng, size, exponent=1.0)


def generate_brown(rng, size):
    return generate_colored(rng, size, exponent=2.0)


# ==============================================================
# DIRECTIONS
# ==============================================================
def direction_1(runner):
    """D1: Noise taxonomy — white, pink, brown pairwise."""
    print("\n" + "=" * 60)
    print("D1: NOISE TAXONOMY — white, pink, brown")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [("white", generate_white),
                         ("pink", generate_pink),
                         ("brown", generate_brown)]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, findings = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_2(runner):
    """D2: Sequential structure — each vs shuffled."""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE — real vs shuffled")
    print("=" * 60)

    results = {}
    for name, gen_fn in [("white", generate_white),
                         ("pink", generate_pink),
                         ("brown", generate_brown)]:
        chunks = [gen_fn(rng, runner.data_size)
                  for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name:8s} vs shuffled = {ns:3d} sig")

    return dict(results=results)


def direction_3(runner):
    """D3: Spectral slope sweep — detection vs spectral exponent."""
    print("\n" + "=" * 60)
    print("D3: SPECTRAL SLOPE SWEEP — exponent vs detection")
    print("=" * 60)

    exponents = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # Compare each against white noise
    white_chunks = [generate_white(rng, runner.data_size)
                    for rng in runner.trial_rngs()]
    white_met = runner.collect(white_chunks)

    results = {}
    for exp in exponents:
        with runner.timed(f"exp={exp:.1f}"):
            chunks = [generate_colored(rng, runner.data_size, exponent=exp)
                      for rng in runner.trial_rngs(offset=100)]
            met = runner.collect(chunks)
            ns, _ = runner.compare(white_met, met)
            results[exp] = ns
            print(f"    → {ns:3d} sig vs white")

    return dict(results=results, exponents=exponents)


def direction_4(runner):
    """D4: Length sensitivity — pink detection at varying N."""
    print("\n" + "=" * 60)
    print("D4: LENGTH SENSITIVITY — pink vs white at varying N")
    print("=" * 60)

    sizes = [500, 1000, 2000, 4000, 8000]
    results = {}

    for sz in sizes:
        with runner.timed(f"N={sz}"):
            white = [generate_white(rng, sz)
                     for rng in runner.trial_rngs()]
            pink = [generate_pink(rng, sz)
                    for rng in runner.trial_rngs(offset=200)]
            w_met = runner.collect(white)
            p_met = runner.collect(pink)
            ns, _ = runner.compare(w_met, p_met)
            results[sz] = ns
            print(f"    → {ns:3d} sig")

    return dict(results=results, sizes=sizes)


def direction_5(runner):
    """D5: Delay embedding — does tau amplify pink noise structure?"""
    print("\n" + "=" * 60)
    print("D5: DELAY EMBEDDING — pink noise")
    print("=" * 60)

    pink_chunks = [generate_pink(rng, runner.data_size)
                   for rng in runner.trial_rngs()]
    shuf_chunks = runner.shuffle_chunks(pink_chunks)

    taus = [1, 2, 3, 4, 5]
    results = {}

    for tau in taus:
        with runner.timed(f"tau={tau}"):
            de_real = [delay_embed(c, tau) for c in pink_chunks]
            de_shuf = [delay_embed(s, tau) for s in shuf_chunks]
            r_met = runner.collect(de_real)
            s_met = runner.collect(de_shuf)
            ns, _ = runner.compare(r_met, s_met)
            results[tau] = ns

    # Raw baseline
    raw_met = runner.collect(pink_chunks)
    shuf_met = runner.collect(shuf_chunks)
    ns_raw, _ = runner.compare(raw_met, shuf_met)
    results['raw'] = ns_raw
    print(f"  raw (no DE): {ns_raw:3d} sig")

    return dict(results=results, taus=taus)


# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(
        5, "Colored Noise: Exotic Geometry Detection")

    # D1: heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'],
                        "D1: Noise Taxonomy")

    # D2: bars
    names2 = list(d2['results'].keys())
    vals2 = [d2['results'][n] for n in names2]
    runner.plot_bars(axes[1], names2, vals2, "D2: vs Shuffled")

    # D3: spectral slope sweep
    exps = d3['exponents']
    sigs3 = [d3['results'][e] for e in exps]
    runner.plot_line(axes[2], exps, sigs3, "D3: Spectral Slope",
                     xlabel="Exponent", color='#e74c3c')

    # D4: length sensitivity
    sizes = d4['sizes']
    sigs4 = [d4['results'][s] for s in sizes]
    runner.plot_line(axes[3], sizes, sigs4, "D4: Length Sensitivity",
                     xlabel="Chunk size N", color='#3498db')

    # D5: delay embedding
    taus = d5['taus']
    de_vals = [d5['results'][t] for t in taus]
    raw_val = d5['results']['raw']
    runner.plot_line(axes[4], taus, de_vals, "D5: Delay Embedding",
                     xlabel="Delay tau", color='#2ecc71', label='DE')
    axes[4].axhline(y=raw_val, color='#e74c3c', ls='--', lw=1.5,
                     label=f'Raw ({raw_val})')
    axes[4].legend(fontsize=8, facecolor='#222222', edgecolor='#444444',
                   labelcolor='#cccccc')

    runner.save(fig, "example_colored_noise")


# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Colored Noise", mode="1d")

    print("=" * 60)
    print("COLORED NOISE INVESTIGATION")
    print(f"size={runner.data_size}, trials={runner.n_trials}, "
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
        'D1': [f"{d1['names'][i]} vs {d1['names'][j]} = "
               f"{d1['matrix'][i,j]} sig"
               for i in range(len(d1['names']))
               for j in range(i + 1, len(d1['names']))],
        'D2': [f"{n} vs shuffled = {v}"
               for n, v in d2['results'].items()],
        'D3': f"Exponent sweep: " + ", ".join(
            f"{e}→{d3['results'][e]}" for e in d3['exponents']),
        'D4': f"Length: " + ", ".join(
            f"N={s}→{d4['results'][s]}" for s in d4['sizes']),
        'D5': f"Raw={d5['results']['raw']}, " + ", ".join(
            f"τ={t}:{d5['results'][t]}" for t in d5['taus']),
    })


if __name__ == "__main__":
    main()
