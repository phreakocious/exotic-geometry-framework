#!/usr/bin/env python3
"""
Quantum Geometry Investigation: Wavefunction Signatures
=======================================================

Can exotic geometries distinguish between different quantum probability
distributions and detect the signature of coherence vs noise?

DIRECTIONS:
D1: Wavefunction Taxonomy — Box, Oscillator, Hydrogen-like, Quantum Walk.
D2: Coherence vs Decoherence — Pure state vs noise-perturbed state.
D3: Energy Levels — Evolution of signature across n=0 to n=4 states.
D4: Superposition — Single eigenstate vs interference patterns.
D5: Wavepacket Spread — Time-evolution snapshots of a Gaussian packet.

NOTE: Generators sample positions from |psi|^2 to produce independent
trials (histograms of quantum measurements). This is physically motivated:
each trial is one experimental run of N position measurements.
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.special import eval_hermite, genlaguerre
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
SEED = 42
np.random.seed(SEED)


# ==============================================================
# DATA GENERATORS
# ==============================================================
def to_uint8(data):
    """Normalize and convert to uint8."""
    mn, mx = data.min(), data.max()
    if mx - mn < 1e-10:
        return np.full(data.shape, 128, dtype=np.uint8)
    return (255 * (data - mn) / (mx - mn)).astype(np.uint8)


def _sample_from_prob(prob, rng, size):
    """Sample positions from |psi|^2, return histogram as uint8.

    Each call gives independent shot noise — like a real experiment
    measuring N particle positions from the same wavefunction.
    """
    prob = np.abs(prob).astype(float)
    prob /= prob.sum() + 1e-15
    indices = rng.choice(len(prob), size=size * 10, p=prob)
    hist, _ = np.histogram(indices, bins=size, range=(0, len(prob)))
    return to_uint8(hist.astype(float))


def gen_particle_in_box(rng, size, n=1):
    """n-th eigenstate of a particle in a 1D box — sampled."""
    x = np.linspace(0, 1, size * 4)
    prob = np.sin(n * np.pi * x)**2
    return _sample_from_prob(prob, rng, size)


def gen_harmonic_oscillator(rng, size, n=0):
    """n-th eigenstate of a 1D harmonic oscillator — sampled."""
    x = np.linspace(-5, 5, size * 4)
    psi = eval_hermite(n, x) * np.exp(-x**2 / 2)
    prob = psi**2
    return _sample_from_prob(prob, rng, size)


def gen_hydrogen_radial(rng, size, n=1, l=0):
    """Simplified radial probability for Hydrogen-like atom — sampled."""
    r = np.linspace(0.01, 20, size * 4)
    rho = (2*r/n)**l * np.exp(-r/n) * genlaguerre(n-l-1, 2*l+1)(2*r/n)
    prob = (r**2) * (rho**2)
    return _sample_from_prob(prob, rng, size)


def gen_quantum_walk(rng, size):
    """Simulated 1D Quantum Walk — sampled from Hadamard distribution."""
    x = np.linspace(-0.99, 0.99, size * 4)
    prob = 1.0 / (np.sqrt(np.abs(1 - x**2)) + 1e-5)
    prob *= (1 + 0.5 * x)  # asymmetry
    return _sample_from_prob(prob, rng, size)


def gen_coherent_superposition(rng, size):
    """Superposition with interference — sampled."""
    x = np.linspace(-5, 5, size * 4)
    psi0 = eval_hermite(0, x) * np.exp(-x**2 / 2)
    psi1 = eval_hermite(1, x) * np.exp(-x**2 / 2)
    prob = (psi0 + psi1)**2
    return _sample_from_prob(prob, rng, size)


def gen_decoherent_mixed(rng, size):
    """Decoherent mixed state (no interference) — sampled."""
    x = np.linspace(-5, 5, size * 4)
    psi0 = eval_hermite(0, x) * np.exp(-x**2 / 2)
    psi1 = eval_hermite(1, x) * np.exp(-x**2 / 2)
    prob = psi0**2 + psi1**2  # no cross-term
    return _sample_from_prob(prob, rng, size)


def gen_wavepacket(rng, size, t=0):
    """Gaussian wavepacket spreading over time — sampled."""
    x = np.linspace(-10, 10, size * 4)
    sigma = 1.0 + t * 0.5
    prob = norm.pdf(x, loc=0, scale=sigma)
    return _sample_from_prob(prob, rng, size)


# ==============================================================
# DIRECTIONS
# ==============================================================
def direction_1(runner):
    """D1: Wavefunction Taxonomy"""
    print("\n" + "=" * 60)
    print("D1: WAVEFUNCTION TAXONOMY")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [("Box (n=3)", lambda r, s: gen_particle_in_box(r, s, n=3)),
                         ("Oscillator (n=2)", lambda r, s: gen_harmonic_oscillator(r, s, n=2)),
                         ("Hydrogen (3s)", lambda r, s: gen_hydrogen_radial(r, s, n=3, l=0)),
                         ("Quantum Walk", gen_quantum_walk)]:
        with runner.timed(name):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names, conditions=conditions)


def direction_2(runner):
    """D2: Coherence vs Decoherence"""
    print("\n" + "=" * 60)
    print("D2: COHERENCE vs DECOHERENCE")
    print("=" * 60)

    with runner.timed("coherent"):
        coh_chunks = [gen_coherent_superposition(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
        coh_met = runner.collect(coh_chunks)

    with runner.timed("decoherent"):
        dec_chunks = [gen_decoherent_mixed(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
        dec_met = runner.collect(dec_chunks)

    ns, findings = runner.compare(coh_met, dec_met)
    print(f"  Coherent vs Decoherent = {ns:3d} sig")
    if findings:
        for m, d, p in findings[:3]:
            print(f"    {m:45s} d={d:+8.2f}")

    return dict(ns=ns, findings=findings)


def direction_3(runner):
    """D3: Energy Level Sweep"""
    print("\n" + "=" * 60)
    print("D3: ENERGY LEVELS (Harmonic Oscillator n=0..4)")
    print("=" * 60)

    results = []
    base_chunks = [gen_harmonic_oscillator(rng, runner.data_size, n=0)
                   for rng in runner.trial_rngs()]
    base_met = runner.collect(base_chunks)

    for n in range(1, 5):
        print(f"  Testing n={n} vs n=0...", end=" ", flush=True)
        t0 = time.time()
        test_chunks = [gen_harmonic_oscillator(rng, runner.data_size, n=n)
                       for rng in runner.trial_rngs(offset=n * 100)]
        test_met = runner.collect(test_chunks)
        ns, _ = runner.compare(base_met, test_met)
        results.append(ns)
        print(f"{ns:3d} sig ({time.time()-t0:.1f}s)")

    return dict(ns_list=results, levels=list(range(1, 5)))


def direction_4(runner):
    """D4: Superposition vs Eigenstates"""
    print("\n" + "=" * 60)
    print("D4: SUPERPOSITION vs EIGENSTATES")
    print("=" * 60)

    conditions = {}
    for name, gen_fn in [
        ("n=1", lambda r, s: gen_particle_in_box(r, s, n=1)),
        ("n=2", lambda r, s: gen_particle_in_box(r, s, n=2)),
        ("n=5", lambda r, s: gen_particle_in_box(r, s, n=5)),
    ]:
        with runner.timed(f"box_{name}"):
            chunks = [gen_fn(rng, runner.data_size)
                      for rng in runner.trial_rngs()]
            conditions[name] = runner.collect(chunks)

    results = {}
    for name in conditions:
        ns, _ = runner.compare(conditions["n=1"], conditions[name])
        results[name] = ns
        if name != "n=1":
            print(f"  Box n=1 vs {name} = {ns:3d} sig")

    return dict(results=results)


def direction_5(runner):
    """D5: Wavepacket Evolution"""
    print("\n" + "=" * 60)
    print("D5: WAVEPACKET EVOLUTION (t=0..8)")
    print("=" * 60)

    results = []
    times = [1, 2, 4, 8]
    base_chunks = [gen_wavepacket(rng, runner.data_size, t=0)
                   for rng in runner.trial_rngs()]
    base_met = runner.collect(base_chunks)

    for t in times:
        print(f"  t={t} vs t=0...", end=" ", flush=True)
        t0_clock = time.time()
        test_chunks = [gen_wavepacket(rng, runner.data_size, t=t)
                       for rng in runner.trial_rngs(offset=t * 100)]
        test_met = runner.collect(test_chunks)
        ns, _ = runner.compare(base_met, test_met)
        results.append(ns)
        print(f"{ns:3d} sig ({time.time()-t0_clock:.1f}s)")

    return dict(ns_list=results, times=times)


# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(
        5, "Quantum Geometry: Wavefunction Signatures")

    # D1: Taxonomy Heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'],
                        "D1: Wavefunction Taxonomy")

    # D2: Coherence Bar + top findings
    runner.plot_bars(axes[1], ["Coh vs Dec"], [d2['ns']],
                     "D2: Coherence Signature")
    if d2.get('findings'):
        lines = ["Top discriminators:"]
        for m, d_val, p in d2['findings'][:5]:
            short = m.split(':')[1] if ':' in m else m
            lines.append(f"  {short}: d={d_val:+.1f}")
        axes[1].text(0.98, 0.95, "\n".join(lines), transform=axes[1].transAxes,
                     fontsize=7, va='top', ha='right', color='#aaa',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#222',
                               edgecolor='#444'))

    # D3: Energy Sweep Line
    runner.plot_line(axes[2], d3['levels'], d3['ns_list'],
                     "D3: Energy Level Sweep",
                     xlabel="Oscillator n", ylabel="Sig metrics vs n=0")

    # D4: Eigenstate comparison
    names4 = [k for k in d4['results'] if k != "n=1"]
    vals4 = [d4['results'][k] for k in names4]
    runner.plot_bars(axes[3], names4, vals4,
                     "D4: Box Eigenstates vs n=1")

    # D5: Evolution Line
    runner.plot_line(axes[4], d5['times'], d5['ns_list'],
                     "D5: Wavepacket Spread",
                     xlabel="Time (t)", ylabel="Sig metrics vs t=0",
                     color='#3498db')

    runner.save(fig, "quantum_geometry")


# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("Quantum Geometry", mode="1d")

    print("=" * 60)
    print("QUANTUM GEOMETRY: WAVEFUNCTION SIGNATURES")
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
        'D1': f"Wavefunction taxonomy: "
              f"{np.mean(d1['matrix'][np.triu_indices_from(d1['matrix'], k=1)]):.0f} avg sig",
        'D2': f"Coherent vs decoherent: {d2['ns']} sig",
        'D3': f"Energy levels vs n=0: " +
              ", ".join(f"n={n}:{s}" for n, s in zip(d3['levels'], d3['ns_list'])),
        'D4': f"Box eigenstates vs n=1: " +
              ", ".join(f"{k}={v}" for k, v in d4['results'].items() if k != "n=1"),
        'D5': f"Wavepacket spread: " +
              ", ".join(f"t={t}:{s}" for t, s in zip(d5['times'], d5['ns_list'])),
    })


if __name__ == "__main__":
    main()
