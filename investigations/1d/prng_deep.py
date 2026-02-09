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
      analyze with 80 spatial metrics. Structural patterns invisible in 1D
      may emerge as 2D texture/topology anomalies.
  D4: Spectral Anomalies — Use spectral_preprocess to see if MT19937 has
      high-frequency bias invisible in the time domain.
  D5: Determinism Detection (Shuffle Test) — If a PRNG is deterministic,
      shuffling its output destroys sequential correlations that geometry
      can detect. Truly random sequences are indistinguishable from their
      shuffles; deterministic ones may not be.

Methodology: N_TRIALS=25, DATA_SIZE=4000 (larger for more power), Cohen's d.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import stats
from exotic_geometry_framework import (GeometryAnalyzer, delay_embed, 
                                       bitplane_extract, spectral_preprocess)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

N_TRIALS = 25
DATA_SIZE = 4000
ALPHA = 0.05

# --- Discover metric names ---
_analyzer = GeometryAnalyzer().add_all_geometries()
_dummy = _analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
METRIC_NAMES = []
for _r in _dummy.results:
    for _mn in sorted(_r.metrics.keys()):
        METRIC_NAMES.append(f"{_r.geometry_name}:{_mn}")
N_METRICS = len(METRIC_NAMES)
BONF = ALPHA / N_METRICS
del _analyzer, _dummy, _r, _mn


# =========================================================================
# GENERATORS (from rng.py)
# =========================================================================

def gen_urandom(trial, size):
    rng = np.random.default_rng(1000 + trial)
    return rng.integers(0, 256, size, dtype=np.uint8)

def gen_mt19937(trial, size):
    rng = np.random.Generator(np.random.MT19937(4000 + trial))
    return rng.integers(0, 256, size, dtype=np.uint8)

def gen_xorshift128(trial, size):
    # Simplified XorShift128 for testing
    state = np.array([6000 + trial, 362436069, 521288629, 88675123], dtype=np.uint32)
    def next_u32():
        nonlocal state
        t = state[0] ^ (state[0] << 11)
        state[0], state[1], state[2] = state[1], state[2], state[3]
        state[3] = (state[3] ^ (state[3] >> 19)) ^ (t ^ (t >> 8))
        return state[3]
    return np.array([next_u32() & 0xFF for _ in range(size)], dtype=np.uint8)

def gen_minstd(trial, size):
    state = 7000 + trial
    if state == 0: state = 1
    def next_u8():
        nonlocal state
        state = (16807 * state) % (2**31 - 1)
        return state & 0xFF
    return np.array([next_u8() for _ in range(size)], dtype=np.uint8)


# =========================================================================
# UTILITIES
# =========================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    if not (np.isfinite(sa) and np.isfinite(sb)):
        return float('nan')
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if not np.isfinite(ps) or ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * 1e6
    return (np.mean(a) - np.mean(b)) / ps

def collect_metrics(analyzer, data_arrays):
    out = defaultdict(list)
    for arr in data_arrays:
        res = analyzer.analyze(arr)
        for r in res.results:
            for mn, mv in r.metrics.items():
                key = f"{r.geometry_name}:{mn}"
                if np.isfinite(mv):
                    out[key].append(mv)
    return out

def compare_to_ref(target_data, ref_data):
    sig_count = 0
    findings = []
    for m in METRIC_NAMES:
        a, b = target_data[m], ref_data[m]
        if len(a) < 3 or len(b) < 3: continue
        d = cohens_d(a, b)
        if not np.isfinite(d): continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        if not np.isfinite(p): continue
        if p < BONF and abs(d) > 0.8:
            sig_count += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig_count, findings


# =========================================================================
# D1: MULTISCALE DELAY EMBEDDING
# =========================================================================

def direction_1(analyzer):
    print("\n" + "=" * 78)
    print("D1: MULTISCALE DELAY EMBEDDING (τ = 1, 2, 3, 5, 8)")
    print("=" * 78)
    
    taus = [1, 2, 3, 5, 8]
    gens = ['MT19937', 'XorShift128', 'MINSTD']
    gen_fns = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128, 'MINSTD': gen_minstd}
    
    results = {}
    
    for gname in gens:
        print(f"  Testing {gname}:")
        results[gname] = {}
        for tau in taus:
            print(f"    tau={tau}...", end=" ", flush=True)
            
            ref_arrays = [delay_embed(gen_urandom(t, DATA_SIZE), tau) for t in range(N_TRIALS)]
            target_arrays = [delay_embed(gen_fns[gname](t, DATA_SIZE), tau) for t in range(N_TRIALS)]
            
            ref_metrics = collect_metrics(analyzer, ref_arrays)
            target_metrics = collect_metrics(analyzer, target_arrays)
            
            n_sig, findings = compare_to_ref(target_metrics, ref_metrics)
            results[gname][tau] = n_sig
            print(f"{n_sig} sig")
            if findings:
                print(f"      Top: {findings[0][0]} d={findings[0][1]:.2f}")
                
    return taus, results


# =========================================================================
# D2: BITPLANE DECOMPOSITION
# =========================================================================

def direction_2(analyzer):
    print("\n" + "=" * 78)
    print("D2: BITPLANE DECOMPOSITION (Plane 0 = LSB, Plane 7 = MSB)")
    print("=" * 78)
    
    planes = [0, 7]
    gens = ['MT19937', 'XorShift128']
    gen_fns = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128}
    
    results = {}
    
    for gname in gens:
        print(f"  Testing {gname}:")
        results[gname] = {}
        for p in planes:
            print(f"    plane={p}...", end=" ", flush=True)
            
            # Larger data for bitplane because it reduces N by 8x
            BS = DATA_SIZE * 2
            ref_arrays = [bitplane_extract(gen_urandom(t, BS), p) for t in range(N_TRIALS)]
            target_arrays = [bitplane_extract(gen_fns[gname](t, BS), p) for t in range(N_TRIALS)]
            
            ref_metrics = collect_metrics(analyzer, ref_arrays)
            target_metrics = collect_metrics(analyzer, target_arrays)
            
            n_sig, findings = compare_to_ref(target_metrics, ref_metrics)
            results[gname][p] = n_sig
            print(f"{n_sig} sig")
            
    return results


# =========================================================================
# D3: 2D SPATIAL ANALYSIS
# =========================================================================

def direction_3(analyzer_spatial):
    print("\n" + "=" * 78)
    print("D3: 2D SPATIAL ANALYSIS (PRNG bytes → square image)")
    print("=" * 78)

    gens = ['MT19937', 'XorShift128', 'MINSTD']
    gen_fns = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128, 'MINSTD': gen_minstd}

    # Determine image size: nearest square from DATA_SIZE bytes
    side = int(np.sqrt(DATA_SIZE))
    n_pixels = side * side
    print(f"  Reshaping {DATA_SIZE} bytes → {side}x{side} image ({n_pixels} pixels)")

    # Discover spatial metric names from a dummy analysis
    dummy_img = np.random.default_rng(0).integers(0, 256, (side, side), dtype=np.uint8).astype(float)
    dummy_res = analyzer_spatial.analyze(dummy_img)
    spatial_metric_names = []
    for r in dummy_res.results:
        for mn in sorted(r.metrics.keys()):
            spatial_metric_names.append(f"{r.geometry_name}:{mn}")
    n_spatial = len(spatial_metric_names)
    bonf_spatial = ALPHA / n_spatial if n_spatial > 0 else ALPHA
    print(f"  {n_spatial} spatial metrics, Bonferroni α={bonf_spatial:.2e}")

    # Collect reference (urandom) spatial metrics
    print("  Collecting urandom reference...", flush=True)
    ref_metrics = defaultdict(list)
    for t in range(N_TRIALS):
        raw = gen_urandom(t, DATA_SIZE)
        img = raw[:n_pixels].reshape(side, side).astype(float)
        res = analyzer_spatial.analyze(img)
        for r in res.results:
            for mn, mv in r.metrics.items():
                if np.isfinite(mv):
                    ref_metrics[f"{r.geometry_name}:{mn}"].append(mv)

    results = {}
    for gname in gens:
        print(f"  Testing {gname}...", end=" ", flush=True)
        target_metrics = defaultdict(list)
        for t in range(N_TRIALS):
            raw = gen_fns[gname](t, DATA_SIZE)
            img = raw[:n_pixels].reshape(side, side).astype(float)
            res = analyzer_spatial.analyze(img)
            for r in res.results:
                for mn, mv in r.metrics.items():
                    if np.isfinite(mv):
                        target_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        sig_count = 0
        findings = []
        for m in spatial_metric_names:
            a = target_metrics.get(m, [])
            b = ref_metrics.get(m, [])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            if not np.isfinite(d): continue
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if not np.isfinite(p): continue
            if p < bonf_spatial and abs(d) > 0.8:
                sig_count += 1
                findings.append((m, d, p))
        findings.sort(key=lambda x: -abs(x[1]))
        results[gname] = {'n_sig': sig_count, 'findings': findings}
        print(f"{sig_count}/{n_spatial} sig")
        for m, d, p in findings[:3]:
            print(f"    {m:50s} d={d:+.2f}")

    return results


# =========================================================================
# D4: SPECTRAL ANOMALIES
# =========================================================================

def direction_4(analyzer):
    print("\n" + "=" * 78)
    print("D4: SPECTRAL ANOMALIES")
    print("=" * 78)
    
    gens = ['MT19937', 'XorShift128']
    gen_fns = {'MT19937': gen_mt19937, 'XorShift128': gen_xorshift128}
    
    results = {}
    
    for gname in gens:
        print(f"  Testing {gname} spectral...", end=" ", flush=True)
        
        ref_arrays = [spectral_preprocess(gen_urandom(t, DATA_SIZE)) for t in range(N_TRIALS)]
        target_arrays = [spectral_preprocess(gen_fns[gname](t, DATA_SIZE)) for t in range(N_TRIALS)]
        
        ref_metrics = collect_metrics(analyzer, ref_arrays)
        target_metrics = collect_metrics(analyzer, target_arrays)
        
        n_sig, findings = compare_to_ref(target_metrics, ref_metrics)
        results[gname] = n_sig
        print(f"{n_sig} sig")
        if findings:
            for m, d, p in findings[:3]:
                print(f"    {m:50s} d={d:+.2f}")
                
    return results


# =========================================================================
# D5: DETERMINISM DETECTION (SHUFFLE TEST)
# =========================================================================

def direction_5(analyzer):
    print("\n" + "=" * 78)
    print("D5: DETERMINISM DETECTION (original vs shuffled)")
    print("=" * 78)
    print("  If geometry detects difference → sequential structure exists")
    print("  If 0 sig → indistinguishable from random permutation")

    gens = ['urandom', 'MT19937', 'XorShift128', 'MINSTD']
    gen_fns = {
        'urandom': gen_urandom,
        'MT19937': gen_mt19937,
        'XorShift128': gen_xorshift128,
        'MINSTD': gen_minstd,
    }

    results = {}

    for gname in gens:
        print(f"  Testing {gname}...", end=" ", flush=True)

        orig_metrics = defaultdict(list)
        shuf_metrics = defaultdict(list)

        for t in range(N_TRIALS):
            raw = gen_fns[gname](t, DATA_SIZE)
            shuffled = raw.copy()
            np.random.default_rng(42000 + t).shuffle(shuffled)

            res_orig = analyzer.analyze(raw)
            for r in res_orig.results:
                for mn, mv in r.metrics.items():
                    if np.isfinite(mv):
                        orig_metrics[f"{r.geometry_name}:{mn}"].append(mv)

            res_shuf = analyzer.analyze(shuffled)
            for r in res_shuf.results:
                for mn, mv in r.metrics.items():
                    if np.isfinite(mv):
                        shuf_metrics[f"{r.geometry_name}:{mn}"].append(mv)

        sig_count = 0
        findings = []
        for m in METRIC_NAMES:
            a = orig_metrics.get(m, [])
            b = shuf_metrics.get(m, [])
            if len(a) < 3 or len(b) < 3:
                continue
            d = cohens_d(a, b)
            if not np.isfinite(d): continue
            _, p = stats.ttest_ind(a, b, equal_var=False)
            if not np.isfinite(p): continue
            if p < BONF and abs(d) > 0.8:
                sig_count += 1
                findings.append((m, d, p))
        findings.sort(key=lambda x: -abs(x[1]))
        results[gname] = {'n_sig': sig_count, 'findings': findings}
        print(f"{sig_count}/{N_METRICS} sig (orig vs shuf)")
        for m, d, p in findings[:3]:
            print(f"    {m:50s} d={d:+.2f}")

    return results


# =========================================================================
# FIGURE
# =========================================================================

def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=8)


def make_figure(taus, d1_res, d2_res, d3_res, d4_res, d5_res):
    print("\nGenerating figure...", flush=True)

    plt.rcParams.update({
        'figure.facecolor': '#181818',
        'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 22), facecolor='#181818')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- (0,0) D1: Multiscale Delay ---
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)
    for gname in d1_res:
        vals = [d1_res[gname][tau] for tau in taus]
        ax1.plot(taus, vals, 'o-', label=gname, linewidth=2)
    ax1.set_xlabel('Delay tau', fontsize=9)
    ax1.set_ylabel('Significant metrics', fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.set_title('D1: Detection vs Delay Tau', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # --- (0,1) D2: Bitplane LSB vs MSB ---
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)
    gens = list(d2_res.keys())
    x = np.arange(len(gens))
    p0 = [d2_res[g][0] for g in gens]
    p7 = [d2_res[g][7] for g in gens]
    ax2.bar(x - 0.2, p0, 0.4, label='Plane 0 (LSB)', color='#E91E63', alpha=0.8)
    ax2.bar(x + 0.2, p7, 0.4, label='Plane 7 (MSB)', color='#2196F3', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gens)
    ax2.set_ylim(bottom=0, top=max(max(p0 + p7), 5) * 1.15)
    ax2.set_ylabel('Significant metrics', fontsize=9)
    ax2.set_title('D2: Bitplane Detection (LSB vs MSB)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # --- (1,0) D3: 2D Spatial detections ---
    ax3 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax3)
    gens3 = list(d3_res.keys())
    x3 = np.arange(len(gens3))
    vals3 = [d3_res[g]['n_sig'] for g in gens3]
    colors3 = ['#FF9800', '#00BCD4', '#8BC34A']
    ax3.bar(x3, vals3, 0.6, color=colors3[:len(gens3)], alpha=0.85)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(gens3)
    ymax3 = max(max(vals3), 5) * 1.15
    ax3.set_ylim(bottom=0, top=ymax3)
    ax3.set_ylabel('Significant spatial metrics', fontsize=9)
    ax3.set_title('D3: 2D Spatial Analysis (bytes → image)', fontsize=11, fontweight='bold')
    for i, v in enumerate(vals3):
        ax3.text(i, v + ymax3 * 0.03, str(v), ha='center', fontsize=10, fontweight='bold', color='white')

    # --- (1,1) D4: Spectral vs Raw ---
    ax4 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax4)
    gens4 = list(d4_res.keys())
    spec_vals = [d4_res[g] for g in gens4]
    raw_vals = [d1_res[g][1] for g in gens4]
    ax4.bar(np.arange(len(gens4)) - 0.2, raw_vals, 0.4, label='Raw', color='#4CAF50', alpha=0.8)
    ax4.bar(np.arange(len(gens4)) + 0.2, spec_vals, 0.4, label='Spectral', color='#9C27B0', alpha=0.8)
    ax4.set_xticks(np.arange(len(gens4)))
    ax4.set_xticklabels(gens4)
    ax4.set_ylim(bottom=0, top=max(max(raw_vals + spec_vals), 5) * 1.15)
    ax4.set_ylabel('Significant metrics', fontsize=9)
    ax4.set_title('D4: Raw vs Spectral Detection', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, facecolor='#333', edgecolor='#666')

    # --- (2,0) D5: Shuffle test ---
    ax5 = fig.add_subplot(gs[2, 0])
    _dark_ax(ax5)
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
        ax5.text(i, v + ymax5 * 0.03, str(v), ha='center', fontsize=10, fontweight='bold', color='white')

    # --- (2,1) Summary text ---
    ax6 = fig.add_subplot(gs[2, 1])
    _dark_ax(ax6)
    ax6.axis('off')
    lines = ['SUMMARY', '']
    for gname in ['MT19937', 'XorShift128', 'MINSTD']:
        best_tau_sig = max(d1_res.get(gname, {0: 0}).values())
        best_tau = [t for t, v in d1_res.get(gname, {}).items() if v == best_tau_sig]
        best_tau = best_tau[0] if best_tau else '?'
        spatial = d3_res.get(gname, {}).get('n_sig', '-')
        spec = d4_res.get(gname, '-')
        shuf = d5_res.get(gname, {}).get('n_sig', '-')
        lines.append(f'{gname}:')
        lines.append(f'  D1 best: {best_tau_sig} sig (tau={best_tau})')
        lines.append(f'  D3 spatial: {spatial} sig')
        lines.append(f'  D4 spectral: {spec} sig')
        lines.append(f'  D5 shuffle: {shuf} sig')
        lines.append('')
    shuf_ur = d5_res.get('urandom', {}).get('n_sig', '-')
    lines.append(f'urandom self-check (D5): {shuf_ur} sig')
    ax6.text(0.05, 0.95, '\n'.join(lines), transform=ax6.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             color='white')

    fig.suptitle('Deep PRNG Analysis: Catching the Indistinguishable',
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'prng_deep.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#181818')
    print(f"  Saved {out_path}")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    analyzer = GeometryAnalyzer().add_all_geometries()
    analyzer_spatial = GeometryAnalyzer().add_spatial_geometries()

    taus, d1_res = direction_1(analyzer)
    d2_res = direction_2(analyzer)
    d3_res = direction_3(analyzer_spatial)
    d4_res = direction_4(analyzer)
    d5_res = direction_5(analyzer)

    make_figure(taus, d1_res, d2_res, d3_res, d4_res, d5_res)

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

    print("\n[Investigation complete]")
