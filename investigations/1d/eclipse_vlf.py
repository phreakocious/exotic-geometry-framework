#!/usr/bin/env python3
"""
Investigation: Solar Eclipse VLF Radio — Geometric Analysis
============================================================

Does the April 8, 2024 total solar eclipse create detectable geometric
structure changes in VLF radio recordings?

Data:
  - Eclipse Research Group sensor ET0001 (Cleveland, OH — path of totality)
  - 20 kHz sample rate, 0-10 kHz bandwidth (ELF/VLF)
  - 15.5 days continuous: April 1-16, 2024
  - Eclipse day: April 8, totality ~19:13-19:17 UTC

Methodology:
  - Spectral profiles: FFT in 10-min windows, log-power, rebin to 2000, uint8
  - Time series: envelope RMS in 2000-record blocks (~33 min each)
  - Compare eclipse day (April 8, 16:00-22:00 UTC) vs control days (Apr 9-12)
  - Bonferroni-corrected Welch t-test with |Cohen's d| > 0.8

Physics:
  During a solar eclipse the ionospheric D-layer collapses as solar UV drops,
  changing VLF propagation — effectively creating nighttime conditions mid-day.
  The framework asks whether this produces geometric structure changes beyond
  what broadband power statistics reveal.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import hashlib
import numpy as np
import h5py
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_FILE = os.path.join(_ROOT, 'data', 'eclipse', 'ET0001.nc')
ENVELOPE_FILE = os.path.join(_ROOT, 'data', 'eclipse', 'envelope.npz')
CACHE_DIR = os.path.join(_ROOT, 'data', 'eclipse', '.cache')
FIG_DIR = os.path.join(_ROOT, 'figures')

# ── Configuration ─────────────────────────────────────────────
SAMPLE_RATE = 20000
DATA_SIZE = 2000
ALPHA = 0.05
WINDOW_MIN = 10          # minutes per spectral window
TS_BLOCK = 2000           # records per time-series extraction

# Eclipse timing — Cleveland OH, April 8 2024 (UTC epoch seconds)
ECLIPSE_DAY = 1712534400                        # 00:00 UTC
CONTACT_1 = ECLIPSE_DAY + 17 * 3600 + 59 * 60  # 17:59 UTC
TOTALITY_S = ECLIPSE_DAY + 19 * 3600 + 13 * 60 # 19:13 UTC
TOTALITY_E = ECLIPSE_DAY + 19 * 3600 + 17 * 60 # 19:17 UTC
CONTACT_4 = ECLIPSE_DAY + 20 * 3600 + 29 * 60  # 20:29 UTC

ANALYSIS_H0, ANALYSIS_H1 = 16, 22  # UTC hours
CONTROL_DAYS = [ECLIPSE_DAY + d * 86400 for d in range(1, 5)]  # Apr 9-12


# ── Helpers ───────────────────────────────────────────────────
def to_uint8(arr, p_lo=0.5, p_hi=99.5):
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    if hi <= lo:
        return np.full(len(arr), 128, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def extract_spectral_profile(block, target_len=DATA_SIZE):
    """Average power spectra of all rows, log-scale, rebin to target_len."""
    ps = np.abs(np.fft.rfft(block.astype(np.float64), axis=1)) ** 2
    avg = np.log1p(np.mean(ps, axis=0))
    trim = (len(avg) // target_len) * target_len
    return to_uint8(avg[:trim].reshape(target_len, -1).mean(axis=1))


def classify_phase(t):
    if t < CONTACT_1:      return 'pre-eclipse'
    if t < TOTALITY_S:      return 'partial-in'
    if t <= TOTALITY_E:     return 'totality'
    if t <= CONTACT_4:      return 'partial-out'
    return 'post-eclipse'


def _fw_hash():
    p = os.path.join(_ROOT, 'exotic_geometry_framework.py')
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 1e-15 else 0.0


# ── Analysis ──────────────────────────────────────────────────
def analyze(data, analyzer, metric_names, cache_key, fw_hash):
    """Run framework, caching per extraction."""
    cp = os.path.join(CACHE_DIR, f"{cache_key}_{fw_hash}.npz")
    if os.path.exists(cp):
        return np.load(cp)['profile']

    result = analyzer.analyze(data)
    profile = np.full(len(metric_names), np.nan)
    for r in result.results:
        for mn, mv in r.metrics.items():
            fn = f"{r.geometry_name}:{mn}"
            if fn in metric_names:
                v = float(mv)
                if np.isfinite(v):
                    profile[metric_names.index(fn)] = v
    np.savez_compressed(cp, profile=profile)
    return profile


def compare(eclipse_p, control_p, metric_names):
    """Bonferroni-corrected Welch t-test + Cohen's d threshold."""
    bonf = ALPHA / len(metric_names)
    out = []
    for i, name in enumerate(metric_names):
        ev = eclipse_p[:, i][~np.isnan(eclipse_p[:, i])]
        cv = control_p[:, i][~np.isnan(control_p[:, i])]
        if len(ev) < 3 or len(cv) < 3:
            continue
        _, p = stats.ttest_ind(ev, cv, equal_var=False)
        d = cohens_d(ev, cv)
        out.append({'metric': name, 'd': d, 'p': p,
                    'sig': p < bonf and abs(d) > 0.8})
    out.sort(key=lambda r: abs(r['d']), reverse=True)
    return out


# ── Figure ────────────────────────────────────────────────────
def make_figure(spec_res, ts_res, n_es, n_cs, n_et, n_ct):
    bg, fg, dim = '#0d1117', '#c9d1d9', '#8b949e'

    fig = plt.figure(figsize=(16, 10), facecolor=bg)
    fig.suptitle('Solar Eclipse VLF Radio — Geometric Analysis',
                 color=fg, fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.935,
             'ET0001 Cleveland OH (path of totality) · April 8 2024 vs Apr 9-12 control',
             color=dim, fontsize=10, ha='center')

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.95, top=0.88, bottom=0.08)

    def style(ax):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=9)
        for s in ax.spines.values(): s.set_color('#30363d')
        ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)

    # P1: summary bar
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    ns = sum(r['sig'] for r in spec_res)
    nt = sum(r['sig'] for r in ts_res)
    bars = ax.bar(['Spectral\nProfiles', 'Time\nSeries'], [ns, nt],
                  color=['#58a6ff', '#f0883e'], width=0.5)
    for b, v in zip(bars, [ns, nt]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                str(v), ha='center', color=fg, fontsize=14, fontweight='bold')
    ax.set_ylabel('Significant metrics')
    ax.set_title(f'Significant metrics  ({n_es} vs {n_cs} spec, {n_et} vs {n_ct} ts)')
    ax.set_ylim(0, max(ns, nt, 5) * 1.3)

    # P2: top spectral
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    top = spec_res[:15]
    if top:
        nm = [r['metric'].split(':')[-1][:22] for r in top]
        ds = [r['d'] for r in top]
        cl = ['#f85149' if r['sig'] else dim for r in top]
        ax.barh(range(len(nm)), ds, color=cl, height=0.7)
        ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='#30363d', lw=0.5)
        for v in [0.8, -0.8]:
            ax.axvline(v, color='#f85149', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel("Cohen's d"); ax.set_title('Top Spectral Discriminators')

    # P3: top time series
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    top = ts_res[:15]
    if top:
        nm = [r['metric'].split(':')[-1][:22] for r in top]
        ds = [r['d'] for r in top]
        cl = ['#f85149' if r['sig'] else dim for r in top]
        ax.barh(range(len(nm)), ds, color=cl, height=0.7)
        ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='#30363d', lw=0.5)
        for v in [0.8, -0.8]:
            ax.axvline(v, color='#f85149', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel("Cohen's d"); ax.set_title('Top Time Series Discriminators')

    # P4: combined heatmap
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    combined = sorted(spec_res + ts_res, key=lambda r: abs(r['d']), reverse=True)[:20]
    if combined:
        nm = [r['metric'].split(':')[-1][:25] for r in combined]
        ds = np.array([r['d'] for r in combined])
        im = ax.imshow(ds.reshape(-1, 1), cmap='RdBu_r', aspect='auto',
                       vmin=-2, vmax=2)
        ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=7)
        ax.set_xticks([])
        for i, r in enumerate(combined):
            if r['sig']:
                ax.text(0, i, '*', ha='center', va='center',
                        color='white', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)
    ax.set_title('Combined Top-20 Effect Sizes')

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'eclipse_vlf.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("Solar Eclipse VLF Radio Investigation")
    print("=" * 50)

    # Framework init
    print("Initializing framework...")
    analyzer = GeometryAnalyzer().add_all_geometries()
    dummy = analyzer.analyze(np.zeros(100, dtype=np.uint8))
    metric_names = []
    for r in dummy.results:
        for mn in sorted(r.metrics.keys()):
            metric_names.append(f"{r.geometry_name}:{mn}")
    n_metrics = len(metric_names)
    fw = _fw_hash()
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  {n_metrics} metrics, framework hash {fw}")

    # Load data handles
    print("Loading data...")
    env = np.load(ENVELOPE_FILE)
    env_t, env_rms = env['gps_time'], env['rms']
    hf = h5py.File(DATA_FILE, 'r')
    gps_t = hf['gps_time'][:]
    samples = hf['samples']

    days = [('eclipse', ECLIPSE_DAY)] + [('control', d) for d in CONTROL_DAYS]

    # ── Spectral profiles ─────────────────────────────────
    print("\n--- Spectral Profiles ---")
    all_spec = []
    n_win = (ANALYSIS_H1 - ANALYSIS_H0) * 60 // WINDOW_MIN

    for label, day0 in days:
        day_n = 8 + (day0 - ECLIPSE_DAY) // 86400
        w0 = day0 + ANALYSIS_H0 * 3600
        count = 0

        for w in range(n_win):
            t0 = w0 + w * WINDOW_MIN * 60
            t1 = t0 + WINDOW_MIN * 60
            mask = (gps_t >= t0) & (gps_t < t1)
            if mask.sum() < 10:
                continue

            idx = np.where(mask)[0]
            ck = f"spec_d{day_n}_w{w:02d}"

            block = samples[idx[0]:idx[-1] + 1, :]
            data = extract_spectral_profile(block)
            profile = analyze(data, analyzer, metric_names, ck, fw)

            phase = classify_phase((t0 + t1) / 2) if label == 'eclipse' else 'control'
            all_spec.append({'label': label, 'day': day_n, 'phase': phase,
                             'profile': profile})
            count += 1

        print(f"  Apr {day_n} ({label}): {count} windows")

    hf.close()

    # ── Time series (envelope RMS) ────────────────────────
    print("\n--- RMS Time Series ---")
    all_ts = []

    for label, day0 in days:
        day_n = 8 + (day0 - ECLIPSE_DAY) // 86400
        w0 = day0 + ANALYSIS_H0 * 3600
        w1 = day0 + ANALYSIS_H1 * 3600
        mask = (env_t >= w0) & (env_t < w1)
        day_idx = np.where(mask)[0]
        n_blk = len(day_idx) // TS_BLOCK
        count = 0

        for b in range(n_blk):
            bi = day_idx[b * TS_BLOCK:(b + 1) * TS_BLOCK]
            data = to_uint8(env_rms[bi])
            ck = f"ts_d{day_n}_b{b}"
            profile = analyze(data, analyzer, metric_names, ck, fw)
            all_ts.append({'label': label, 'day': day_n, 'block': b,
                           'profile': profile})
            count += 1

        print(f"  Apr {day_n} ({label}): {count} blocks")

    # ── Compare ───────────────────────────────────────────
    print("\n--- Eclipse vs Control ---")
    e_s = np.array([e['profile'] for e in all_spec if e['label'] == 'eclipse'])
    c_s = np.array([e['profile'] for e in all_spec if e['label'] == 'control'])
    spec_r = compare(e_s, c_s, metric_names)
    ns = sum(r['sig'] for r in spec_r)
    print(f"  Spectral: {len(e_s)} eclipse vs {len(c_s)} control → {ns} significant")

    e_t = np.array([e['profile'] for e in all_ts if e['label'] == 'eclipse'])
    c_t = np.array([e['profile'] for e in all_ts if e['label'] == 'control'])
    ts_r = compare(e_t, c_t, metric_names)
    nt = sum(r['sig'] for r in ts_r)
    print(f"  Time series: {len(e_t)} eclipse vs {len(c_t)} control → {nt} significant")

    print(f"\n  Top spectral (by |d|):")
    for r in spec_r[:10]:
        s = " ***" if r['sig'] else ""
        print(f"    {r['metric']:50s}  d={r['d']:+.3f}  p={r['p']:.2e}{s}")

    print(f"\n  Top time series (by |d|):")
    for r in ts_r[:10]:
        s = " ***" if r['sig'] else ""
        print(f"    {r['metric']:50s}  d={r['d']:+.3f}  p={r['p']:.2e}{s}")

    # ── Phase breakdown (eclipse day only) ────────────────
    print("\n--- Eclipse Phase Breakdown (spectral) ---")
    phases = ['pre-eclipse', 'partial-in', 'totality', 'partial-out', 'post-eclipse']
    for ph in phases:
        n = sum(1 for e in all_spec if e['phase'] == ph)
        print(f"  {ph}: {n} windows")

    # ── Figure ────────────────────────────────────────────
    print("\n--- Figure ---")
    make_figure(spec_r, ts_r, len(e_s), len(c_s), len(e_t), len(c_t))

    # ── Summary ───────────────────────────────────────────
    total = ns + nt
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  Spectral:    {ns} significant / {n_metrics}")
    print(f"  Time series: {nt} significant / {n_metrics}")
    if total == 0:
        print("  → No geometric structure difference detected")
    elif total < 10:
        print("  → Mild difference (likely instrumental/environmental)")
    elif total < 50:
        print("  → Moderate difference (warrants investigation)")
    else:
        print("  → Strong geometric difference detected")


if __name__ == '__main__':
    main()
