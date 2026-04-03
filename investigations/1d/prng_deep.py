#!/usr/bin/env python3
"""
Investigation: Deep PRNG Analysis — Catching the "Indistinguishable"
====================================================================

Follow-up to rng.py, which showed that modern PRNGs (MT19937, XorShift128)
are nearly indistinguishable from os.urandom at the byte level. This
investigation uses advanced preprocessing to try and catch them.

Directions:
  D1: Multiscale Delay Embedding — Combine signatures from τ=1, 2, 3, 5, 8.
      Linear correlations in LFSR-based generators often appear at specific lags.
  D2: Bitplane Decomposition — Analyze each of the 8 bitplanes separately.
      The LSB of some generators is known to be less random than the MSB.
  D3: 2D Spatial Analysis — Reshape PRNG bytes into a square image and
      analyze with spatial metrics. Structural patterns invisible in 1D
      may emerge as 2D texture/topology anomalies.
  D4: Spectral Anomalies — Use spectral_preprocess to see if MT19937 has
      high-frequency bias invisible in the time domain.
  D5: Determinism Detection (Shuffle Test) — If a PRNG is deterministic,
      shuffling its output destroys sequential correlations that geometry
      can detect. Truly random sequences are indistinguishable from their
      shuffles; deterministic ones may not be.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from exotic_geometry_framework import delay_embed, bitplane_extract, spectral_preprocess
from tools.investigation_runner import Runner

N_TRIALS = 25
DATA_SIZE = 16384
N_WORKERS = 8


# =========================================================================
# GENERATORS
# =========================================================================

def gen_urandom(rng, size):
    return rng.integers(0, 256, size, dtype=np.uint8)

def gen_mt19937(rng, size):
    seed = int(rng.integers(0, 2**32))
    r = np.random.Generator(np.random.MT19937(seed))
    return r.integers(0, 256, size, dtype=np.uint8)

class XorShift128:
    def __init__(self, seed):
        self.x = seed & 0xFFFFFFFF or 1
        self.y = ((seed >> 32) & 0xFFFFFFFF) or 362436069
        self.z = 521288629
        self.w = 88675123
    def next_u32(self):
        t = self.x ^ ((self.x << 11) & 0xFFFFFFFF)
        self.x, self.y, self.z = self.y, self.z, self.w
        self.w = (self.w ^ (self.w >> 19)) ^ (t ^ (t >> 8))
        return self.w & 0xFFFFFFFF

def gen_xorshift128(rng, size):
    seed = int(rng.integers(0, 2**63))
    xs = XorShift128(seed)
    return np.array([xs.next_u32() & 0xFF for _ in range(size)], dtype=np.uint8)

def gen_minstd(rng, size):
    state = int(rng.integers(1, 2**31 - 1))
    out = np.empty(size, dtype=np.uint8)
    for i in range(size):
        state = (16807 * state) % (2**31 - 1)
        out[i] = state & 0xFF
    return out


# =========================================================================
# D1: MULTISCALE DELAY EMBEDDING
# =========================================================================

def direction_1(runner):
    print("\n" + "=" * 78)
    print("D1: MULTISCALE DELAY EMBEDDING (τ = 1, 2, 3, 5, 8)")
    print("=" * 78)

    taus = [1, 2, 3, 5, 8]
    gens = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128, 'MINSTD': gen_minstd}

    # Reference: urandom at each tau
    ref_by_tau = {}
    for tau in taus:
        ref_chunks = [delay_embed(gen_urandom(rng, DATA_SIZE), tau)
                      for rng in runner.trial_rngs()]
        ref_by_tau[tau] = runner.collect(ref_chunks)

    results = {}
    for gname, gen_fn in gens.items():
        print(f"  Testing {gname}:")
        results[gname] = {}
        for tau in taus:
            print(f"    tau={tau}...", end=" ", flush=True)
            target_chunks = [delay_embed(gen_fn(rng, DATA_SIZE), tau)
                             for rng in runner.trial_rngs(offset=500)]
            target_met = runner.collect(target_chunks)
            n_sig, findings = runner.compare(target_met, ref_by_tau[tau])
            results[gname][tau] = n_sig
            print(f"{n_sig} sig")
            if findings:
                print(f"      Top: {findings[0][0]} d={findings[0][1]:.2f}")

    return taus, results


# =========================================================================
# D2: BITPLANE DECOMPOSITION
# =========================================================================

def direction_2(runner):
    print("\n" + "=" * 78)
    print("D2: BITPLANE DECOMPOSITION (Plane 0 = LSB, Plane 7 = MSB)")
    print("=" * 78)

    planes = [0, 7]
    gens = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128}
    BS = DATA_SIZE * 2  # Larger data because bitplane reduces N by 8x

    results = {}
    for gname, gen_fn in gens.items():
        print(f"  Testing {gname}:")
        results[gname] = {}
        for p in planes:
            print(f"    plane={p}...", end=" ", flush=True)
            ref_chunks = [bitplane_extract(gen_urandom(rng, BS), p)
                          for rng in runner.trial_rngs()]
            target_chunks = [bitplane_extract(gen_fn(rng, BS), p)
                             for rng in runner.trial_rngs(offset=500)]
            ref_met = runner.collect(ref_chunks)
            target_met = runner.collect(target_chunks)
            n_sig, _ = runner.compare(target_met, ref_met)
            results[gname][p] = n_sig
            print(f"{n_sig} sig")

    return results


# =========================================================================
# D3: 2D SPATIAL ANALYSIS
# =========================================================================

def direction_3():
    print("\n" + "=" * 78)
    print("D3: 2D SPATIAL ANALYSIS (PRNG bytes → square image)")
    print("=" * 78)

    side = int(np.sqrt(DATA_SIZE))
    n_pixels = side * side
    print(f"  Reshaping {DATA_SIZE} bytes → {side}x{side} image ({n_pixels} pixels)")

    spatial_runner = Runner("PRNG Spatial", mode="2d",
                            n_workers=N_WORKERS, data_size=side,
                            n_trials=N_TRIALS)

    gens = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128, 'MINSTD': gen_minstd}

    try:
        # Reference
        print("  Collecting urandom reference...", flush=True)
        ref_chunks = [gen_urandom(rng, n_pixels).reshape(side, side).astype(float)
                      for rng in spatial_runner.trial_rngs()]
        ref_met = spatial_runner.collect(ref_chunks)

        results = {}
        for gname, gen_fn in gens.items():
            print(f"  Testing {gname}...", end=" ", flush=True)
            target_chunks = [gen_fn(rng, n_pixels).reshape(side, side).astype(float)
                             for rng in spatial_runner.trial_rngs(offset=500)]
            target_met = spatial_runner.collect(target_chunks)
            n_sig, findings = spatial_runner.compare(target_met, ref_met)
            results[gname] = {'n_sig': n_sig, 'findings': findings}
            print(f"{n_sig}/{spatial_runner.n_metrics} sig")
            if findings:
                for m, d, _ in findings[:3]:
                    print(f"    {m:50s} d={d:+.2f}")
    finally:
        spatial_runner.close()

    return results


# =========================================================================
# D4: SPECTRAL ANOMALIES
# =========================================================================

def direction_4(runner):
    print("\n" + "=" * 78)
    print("D4: SPECTRAL ANOMALIES")
    print("=" * 78)

    gens = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128}

    results = {}
    for gname, gen_fn in gens.items():
        print(f"  Testing {gname} spectral...", end=" ", flush=True)
        ref_chunks = [spectral_preprocess(gen_urandom(rng, DATA_SIZE))
                      for rng in runner.trial_rngs()]
        target_chunks = [spectral_preprocess(gen_fn(rng, DATA_SIZE))
                         for rng in runner.trial_rngs(offset=500)]
        ref_met = runner.collect(ref_chunks)
        target_met = runner.collect(target_chunks)
        n_sig, findings = runner.compare(target_met, ref_met)
        results[gname] = n_sig
        print(f"{n_sig} sig")
        if findings:
            for m, d, _ in findings[:3]:
                print(f"    {m:50s} d={d:+.2f}")

    return results


# =========================================================================
# D5: DETERMINISM DETECTION (SHUFFLE TEST)
# =========================================================================

def direction_5(runner):
    print("\n" + "=" * 78)
    print("D5: DETERMINISM DETECTION (original vs shuffled)")
    print("=" * 78)
    print("  If geometry detects difference → sequential structure exists")
    print("  If 0 sig → indistinguishable from random permutation")

    gens = {
        'urandom': gen_urandom,
        'MT19937': gen_mt19937,
        'XorShift128': gen_xorshift128,
        'MINSTD': gen_minstd,
    }

    results = {}
    for gname, gen_fn in gens.items():
        print(f"  Testing {gname}...", end=" ", flush=True)
        rngs = runner.trial_rngs(offset=700)
        orig_chunks = [gen_fn(rng, DATA_SIZE) for rng in rngs]
        shuf_chunks = []
        for i, chunk in enumerate(orig_chunks):
            s = chunk.copy()
            np.random.default_rng(42000 + i).shuffle(s)
            shuf_chunks.append(s)

        orig_met = runner.collect(orig_chunks)
        shuf_met = runner.collect(shuf_chunks)
        n_sig, findings = runner.compare(orig_met, shuf_met)
        results[gname] = {'n_sig': n_sig, 'findings': findings}
        print(f"{n_sig}/{runner.n_metrics} sig (orig vs shuf)")
        if findings:
            for m, d, _ in findings[:3]:
                print(f"    {m:50s} d={d:+.2f}")

    return results


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(runner, taus, d1_res, d5_res):
    fig, axes = runner.create_figure(2, "Deep PRNG Analysis")

    # D1: Multiscale delay
    ax1 = axes[0]
    for gname in d1_res:
        vals = [d1_res[gname][tau] for tau in taus]
        ax1.plot(taus, vals, 'o-', label=gname, linewidth=2)
    ax1.set_xlabel('Delay tau', fontsize=9)
    ax1.set_ylabel('Significant metrics', fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.set_title('D1: Detection vs Delay Tau', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)

    # D5: Shuffle test
    ax5 = axes[1]
    gens5 = list(d5_res.keys())
    x5 = np.arange(len(gens5))
    vals5 = [d5_res[g]['n_sig'] for g in gens5]
    colors5 = ['#607D8B', '#FF9800', '#00BCD4', '#8BC34A']
    ax5.bar(x5, vals5, 0.6, color=colors5[:len(gens5)], alpha=0.85)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(gens5, fontsize=8)
    ymax5 = max(max(vals5), 5) * 1.15
    ax5.set_ylim(bottom=0, top=ymax5)
    ax5.set_ylabel('Significant metrics (orig vs shuf)', fontsize=9)
    ax5.set_title('D5: Determinism Detection (Shuffle Test)', fontsize=11, fontweight='bold')
    for i, v in enumerate(vals5):
        ax5.text(i, v + ymax5 * 0.03, str(v), ha='center', fontsize=10,
                 fontweight='bold', color='white')

    runner.save(fig, "prng_deep")


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()
    runner = Runner("PRNG Deep", mode="1d",
                    n_workers=N_WORKERS, data_size=DATA_SIZE,
                    n_trials=N_TRIALS)

    print("=" * 78)
    print("DEEP PRNG ANALYSIS: CATCHING THE INDISTINGUISHABLE")
    print("=" * 78)

    try:
        taus, d1_res = direction_1(runner)
        d2_res = direction_2(runner)
        d3_res = direction_3()
        d4_res = direction_4(runner)
        d5_res = direction_5(runner)

        make_figure(runner, taus, d1_res, d5_res)

        elapsed = time.time() - t0
        print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for gname in ['MT19937', 'XorShift128', 'MINSTD']:
            max_sig = max(d1_res.get(gname, {0: 0}).values())
            best_tau = [t for t, v in d1_res.get(gname, {}).items() if v == max_sig]
            best_tau = best_tau[0] if best_tau else '?'
            bp0 = d2_res.get(gname, {}).get(0, '-')
            bp7 = d2_res.get(gname, {}).get(7, '-')
            spatial = d3_res.get(gname, {}).get('n_sig', '-')
            spec = d4_res.get(gname, '-')
            shuf = d5_res.get(gname, {}).get('n_sig', '-')
            print(f"  {gname:15s}: D1={max_sig:2d}(tau={best_tau}), "
                  f"D2=LSB:{bp0}/MSB:{bp7}, D3={spatial}, D4={spec}, D5={shuf}")

        shuf_ur = d5_res.get('urandom', {}).get('n_sig', '-')
        print(f"  {'urandom':15s}: D5 self-check = {shuf_ur} (should be ~0)")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
