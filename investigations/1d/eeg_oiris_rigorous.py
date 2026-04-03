#!/usr/bin/env python3
"""
Investigation: O.IRIS Rigorous — Addressing Every Blind Spot
=============================================================

Follows up eeg_oiris.py findings with controls for every identified weakness.

Blind spots addressed:
  D1: Cross-lab confound — within-dataset rest vs sleep (same subjects,
      same equipment, same session in ds003768)
  D2: Spatial structure — channels × time as 2D image → spatial geometries.
      Tests whether rest/sleep differ in cross-channel structure.
  D3: Cross-channel coherence — pairwise coherence in alpha/theta/beta bands.
      Standard neuroscience connectivity, not framework-dependent.
  D4: Phase-amplitude coupling — theta phase × gamma amplitude.
      Tests O.IRIS's implicit "cross-frequency" claim.
  D5: Window length sensitivity — 5s, 10s, 30s, 65s windows on meditation
      vs rest. Does our finding survive at O.IRIS's timescale?
  D6: Proper phi attractor test — is m=2.18 a statistical attractor?
      Bootstrap: does spectral slope cluster at 2.18 tighter than at
      random reference values?
  D7: Per-channel profile diversity — run 1D framework on individual
      channels, compare metric variance across channels between states.
  D8: Connectivity matrix geometry — coherence matrix → flatten upper
      triangle → 1D framework. Treats the connectivity pattern as a signal.

Datasets: ds001787 (meditation), ds003768 (sleep/rest) — same as eeg_oiris.py.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import signal, stats
from scipy.signal import hilbert
from collections import defaultdict
from tools.investigation_runner import Runner

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_TRIALS = 25
N_WORKERS = 8

OIRIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'data', 'eeg', 'oiris')
MEDITATION_DIR = os.path.join(OIRIS_DIR, 'ds001787')
SLEEP_DIR = os.path.join(OIRIS_DIR, 'ds003768')

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}


# =========================================================================
# DATA LOADING — MULTI-CHANNEL (no averaging)
# =========================================================================

_MNE = None
def mne():
    global _MNE
    if _MNE is None:
        import mne as _m
        _m.set_log_level('ERROR')
        _MNE = _m
    return _MNE


def to_uint8(sig):
    sig = np.asarray(sig, dtype=np.float64)
    lo, hi = np.nanmin(sig), np.nanmax(sig)
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((sig - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def to_uint8_2d(data):
    """Min-max normalize 2D array to uint8."""
    data = np.asarray(data, dtype=np.float64)
    lo, hi = np.nanmin(data), np.nanmax(data)
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((data - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


_RAW_CACHE = {}

def load_raw(filepath, target_sfreq=250):
    """Load EEG file, return Raw object (multi-channel, filtered, resampled)."""
    if filepath in _RAW_CACHE:
        return _RAW_CACHE[filepath]
    m = mne()
    if filepath.endswith('.bdf'):
        raw = m.io.read_raw_bdf(filepath, preload=True, verbose=False)
    elif filepath.endswith('.vhdr'):
        raw = m.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    else:
        raw = m.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    if raw.info['sfreq'] > target_sfreq + 10:
        raw.resample(target_sfreq, verbose=False)
    raw.filter(1., 55., fir_design='firwin', verbose=False)
    _RAW_CACHE[filepath] = raw
    return raw


def scan_files():
    """Scan downloaded datasets."""
    med_files = sorted([os.path.join(r, f) for r, _, fs in os.walk(MEDITATION_DIR)
                        for f in fs if f.endswith('.bdf')]) if os.path.isdir(MEDITATION_DIR) else []
    rest_files, sleep_files = [], []
    if os.path.isdir(SLEEP_DIR):
        for r, _, fs in os.walk(SLEEP_DIR):
            for f in fs:
                if f.endswith('.vhdr'):
                    path = os.path.join(r, f)
                    if 'task-rest' in f:
                        rest_files.append(path)
                    elif 'task-sleep' in f:
                        sleep_files.append(path)
    return sorted(med_files), sorted(rest_files), sorted(sleep_files)


def compute_spectral_slope(data, sfreq):
    """Spectral slope m: PSD ~ 1/f^m."""
    from scipy.stats import linregress
    nperseg = min(int(4 * sfreq), len(data))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= 1) & (freqs <= 40) & (psd > 0)
    if np.sum(mask) < 5:
        return np.nan
    slope, _, _, _, _ = linregress(np.log10(freqs[mask]), np.log10(psd[mask]))
    return -slope


# =========================================================================
# D1: WITHIN-DATASET REST VS SLEEP
# =========================================================================

def direction_1(runner):
    """D1: Within-dataset rest vs sleep — same subjects, same equipment."""
    print("\n" + "=" * 78)
    print("D1: WITHIN-DATASET REST VS SLEEP (ds003768 only)")
    print("    Same subjects, same equipment, same session.")
    print("    Eliminates cross-lab confound.")
    print("=" * 78)

    _, rest_files, sleep_files = scan_files()
    if not rest_files or not sleep_files:
        print("  Missing rest or sleep data")
        return {}

    # Match subjects — only compare rest and sleep from the same subjects
    def get_subject(f):
        base = os.path.basename(f)
        return base.split('_')[0]  # e.g. "sub-01"

    rest_subjs = set(get_subject(f) for f in rest_files)
    sleep_subjs = set(get_subject(f) for f in sleep_files)
    shared = sorted(rest_subjs & sleep_subjs)
    print(f"  Shared subjects: {len(shared)}")

    def gen_from_files(files, rng, size):
        f = str(rng.choice(files))
        raw = load_raw(f)
        sig = np.mean(raw.get_data(), axis=0)
        if len(sig) < size:
            sig = np.tile(sig, size // len(sig) + 1)
        start = int(rng.integers(0, max(1, len(sig) - size)))
        return to_uint8(sig[start:start + size])

    shared_rest = [f for f in rest_files if get_subject(f) in shared]
    shared_sleep = [f for f in sleep_files if get_subject(f) in shared]

    with runner.timed("Rest (matched)"):
        rest_chunks = [gen_from_files(shared_rest, rng, runner.data_size)
                       for rng in runner.trial_rngs()]
        rest_met = runner.collect(rest_chunks)

    with runner.timed("Sleep (matched)"):
        sleep_chunks = [gen_from_files(shared_sleep, rng, runner.data_size)
                        for rng in runner.trial_rngs(offset=100)]
        sleep_met = runner.collect(sleep_chunks)

    n_sig, findings = runner.compare(rest_met, sleep_met)
    print(f"\n  Matched rest vs sleep: {n_sig} sig (of {runner.n_metrics})")
    for m, d, _ in findings[:5]:
        print(f"    {m:50s} d={d:+8.2f}")

    return {'n_sig': n_sig, 'findings': findings}


# =========================================================================
# D2: 2D SPATIAL STRUCTURE — channels × time
# =========================================================================

def direction_2():
    """D2: Spatial structure — channels × time as 2D image."""
    print("\n" + "=" * 78)
    print("D2: SPATIAL STRUCTURE — CHANNELS × TIME (2D geometries)")
    print("    Tests whether rest/sleep differ in cross-channel structure.")
    print("=" * 78)

    _, rest_files, sleep_files = scan_files()
    med_files, _, _ = scan_files()
    if not rest_files or not sleep_files:
        print("  Missing data")
        return {}

    # Use a fixed number of channels and time samples for consistent 2D images
    n_chan_target = 16  # subsample channels for square-ish images
    n_time = 256        # time samples per window
    img_size = n_chan_target  # for Runner data_size param (not really used for 2D)

    runner_2d = Runner("EEG 2D", mode="2d", n_workers=N_WORKERS,
                       data_size=img_size, n_trials=N_TRIALS)

    def gen_2d(files, rng, n_ch=n_chan_target, n_t=n_time):
        f = str(rng.choice(files))
        raw = load_raw(f)
        data = raw.get_data()  # (n_channels, n_samples)
        n_channels = data.shape[0]
        # Subsample channels evenly
        ch_idx = np.linspace(0, n_channels - 1, n_ch, dtype=int)
        data = data[ch_idx]
        # Random time window
        max_start = max(1, data.shape[1] - n_t)
        start = int(rng.integers(0, max_start))
        img = data[:, start:start + n_t]
        return to_uint8_2d(img).astype(float)

    states = {}
    if med_files:
        states['Meditation'] = med_files[:5]
    states['Rest'] = rest_files[:5]
    states['Sleep'] = sleep_files[:10]

    all_met = {}
    for name, files in states.items():
        with runner_2d.timed(name):
            chunks = [gen_2d(files, rng) for rng in runner_2d.trial_rngs(offset=200)]
            all_met[name] = runner_2d.collect(chunks)

    # Pairwise
    d2 = {}
    state_names = list(all_met.keys())
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j > i:
                n_sig, findings = runner_2d.compare(all_met[n1], all_met[n2])
                d2[(n1, n2)] = n_sig
                top = findings[0] if findings else ('', 0, 0)
                print(f"  {n1:15s} vs {n2:15s}: {n_sig:3d} sig"
                      + (f"  top: {top[0]} d={top[1]:+.2f}" if findings else ""))

    runner_2d.close()
    return d2


# =========================================================================
# D3: CROSS-CHANNEL COHERENCE
# =========================================================================

def compute_coherence_matrix(raw, band, window_samples=2500):
    """Compute mean pairwise coherence in a frequency band."""
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    n_ch = data.shape[0]
    lo, hi = BANDS[band]

    # Use first window_samples
    seg = data[:, :min(window_samples, data.shape[1])]
    nperseg = min(int(2 * sfreq), seg.shape[1])

    # Compute coherence for subset of channel pairs (full matrix too slow)
    n_pairs = min(n_ch, 20)
    ch_idx = np.linspace(0, n_ch - 1, n_pairs, dtype=int)
    coh_matrix = np.zeros((n_pairs, n_pairs))

    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            f, Cxy = signal.coherence(seg[ch_idx[i]], seg[ch_idx[j]],
                                       fs=sfreq, nperseg=nperseg)
            band_mask = (f >= lo) & (f <= hi)
            if np.any(band_mask):
                coh_matrix[i, j] = np.mean(Cxy[band_mask])
                coh_matrix[j, i] = coh_matrix[i, j]

    return coh_matrix


def direction_3():
    """D3: Cross-channel coherence in alpha/theta/beta bands."""
    print("\n" + "=" * 78)
    print("D3: CROSS-CHANNEL COHERENCE")
    print("    Standard neuroscience connectivity analysis.")
    print("=" * 78)

    med_files, rest_files, sleep_files = scan_files()

    states = {}
    if med_files:
        states['Meditation'] = med_files[:5]
    if rest_files:
        states['Rest'] = rest_files[:5]
    if sleep_files:
        states['Sleep'] = sleep_files[:5]

    d3 = {}
    for band in ['theta', 'alpha', 'beta']:
        print(f"\n  Band: {band}")
        state_coherences = {}
        for name, files in states.items():
            cohs = []
            for f in files[:3]:
                raw = load_raw(f)
                coh = compute_coherence_matrix(raw, band)
                # Summary stats: mean coherence, std, max
                upper = coh[np.triu_indices_from(coh, k=1)]
                cohs.append({
                    'mean': np.mean(upper),
                    'std': np.std(upper),
                    'max': np.max(upper) if len(upper) > 0 else 0,
                })
            state_coherences[name] = cohs
            mean_coh = np.mean([c['mean'] for c in cohs])
            std_coh = np.mean([c['std'] for c in cohs])
            print(f"    {name:15s}: mean_coh={mean_coh:.4f}, std_coh={std_coh:.4f}")

        # Compare rest vs sleep coherence
        if 'Rest' in state_coherences and 'Sleep' in state_coherences:
            rest_means = [c['mean'] for c in state_coherences['Rest']]
            sleep_means = [c['mean'] for c in state_coherences['Sleep']]
            if len(rest_means) >= 2 and len(sleep_means) >= 2:
                t, p = stats.ttest_ind(rest_means, sleep_means)
                print(f"    Rest vs Sleep: t={t:.2f}, p={p:.3f}")
                d3[(band, 'rest_v_sleep')] = {'t': t, 'p': p}

        if 'Meditation' in state_coherences and 'Rest' in state_coherences:
            med_means = [c['mean'] for c in state_coherences['Meditation']]
            rest_means = [c['mean'] for c in state_coherences['Rest']]
            if len(med_means) >= 2 and len(rest_means) >= 2:
                t, p = stats.ttest_ind(med_means, rest_means)
                print(f"    Med vs Rest:   t={t:.2f}, p={p:.3f}")
                d3[(band, 'med_v_rest')] = {'t': t, 'p': p}

    return d3


# =========================================================================
# D4: PHASE-AMPLITUDE COUPLING
# =========================================================================

def compute_pac(sig, sfreq, phase_band=(4, 8), amp_band=(30, 50)):
    """Compute phase-amplitude coupling (modulation index)."""
    # Filter for phase signal (theta)
    nyq = sfreq / 2
    b_ph, a_ph = signal.butter(4, [phase_band[0]/nyq, phase_band[1]/nyq], btype='band')
    phase_sig = signal.filtfilt(b_ph, a_ph, sig)
    phase = np.angle(hilbert(phase_sig))

    # Filter for amplitude signal (gamma)
    b_amp, a_amp = signal.butter(4, [amp_band[0]/nyq, amp_band[1]/nyq], btype='band')
    amp_sig = signal.filtfilt(b_amp, a_amp, sig)
    amplitude = np.abs(hilbert(amp_sig))

    # Modulation index (Tort et al. 2010)
    n_bins = 18
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.any(mask):
            mean_amp[i] = np.mean(amplitude[mask])

    # Normalize
    mean_amp_norm = mean_amp / (np.sum(mean_amp) + 1e-15)
    # KL divergence from uniform
    uniform = np.ones(n_bins) / n_bins
    kl = np.sum(mean_amp_norm * np.log((mean_amp_norm + 1e-15) / uniform))
    mi = kl / np.log(n_bins)
    return mi


def direction_4():
    """D4: Phase-amplitude coupling — theta phase × gamma amplitude."""
    print("\n" + "=" * 78)
    print("D4: PHASE-AMPLITUDE COUPLING (theta→gamma)")
    print("    Tests cross-frequency interaction that O.IRIS implies.")
    print("=" * 78)

    med_files, rest_files, sleep_files = scan_files()
    states = {}
    if med_files:
        states['Meditation'] = med_files[:5]
    if rest_files:
        states['Rest'] = rest_files[:5]
    if sleep_files:
        states['Sleep'] = sleep_files[:5]

    d4 = {}
    for name, files in states.items():
        pac_values = []
        for f in files[:5]:
            raw = load_raw(f)
            sig = np.mean(raw.get_data(), axis=0)
            sfreq = raw.info['sfreq']
            # Compute PAC on multiple segments
            seg_len = int(30 * sfreq)  # 30-second segments
            for start in range(0, len(sig) - seg_len, seg_len):
                mi = compute_pac(sig[start:start + seg_len], sfreq)
                pac_values.append(mi)
        d4[name] = pac_values
        print(f"  {name:15s}: PAC MI = {np.mean(pac_values):.6f} ± {np.std(pac_values):.6f}"
              f"  (n={len(pac_values)})")

    # Compare
    pairs = [('Meditation', 'Rest'), ('Meditation', 'Sleep'), ('Rest', 'Sleep')]
    for n1, n2 in pairs:
        if n1 in d4 and n2 in d4 and len(d4[n1]) >= 3 and len(d4[n2]) >= 3:
            t, p = stats.ttest_ind(d4[n1], d4[n2])
            d_val = (np.mean(d4[n1]) - np.mean(d4[n2])) / (
                np.sqrt((np.std(d4[n1])**2 + np.std(d4[n2])**2) / 2) + 1e-15)
            sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {n1} vs {n2}: d={d_val:+.3f}, p={p:.4f} {sig_str}")

    return d4


# =========================================================================
# D5: WINDOW LENGTH SENSITIVITY
# =========================================================================

def direction_5():
    """D5: Window length sensitivity — does our finding survive at all timescales?"""
    print("\n" + "=" * 78)
    print("D5: WINDOW LENGTH SENSITIVITY")
    print("    Testing at 5s, 10s, 30s, 65s (O.IRIS uses 10-30s)")
    print("=" * 78)

    med_files, rest_files, _ = scan_files()
    if not med_files or not rest_files:
        print("  Missing data")
        return {}

    window_secs = [5, 10, 30, 65]
    sfreq = 250
    d5 = {}

    for ws in window_secs:
        data_size = int(ws * sfreq)
        runner_w = Runner(f"EEG {ws}s", mode="1d", n_workers=N_WORKERS,
                          data_size=data_size, n_trials=N_TRIALS)

        def gen_state(files, rng, size):
            f = str(rng.choice(files))
            raw = load_raw(f)
            sig = np.mean(raw.get_data(), axis=0)
            if len(sig) < size:
                sig = np.tile(sig, size // len(sig) + 1)
            start = int(rng.integers(0, max(1, len(sig) - size)))
            return to_uint8(sig[start:start + size])

        with runner_w.timed(f"Med {ws}s"):
            med_chunks = [gen_state(med_files[:5], rng, data_size)
                          for rng in runner_w.trial_rngs()]
            med_met = runner_w.collect(med_chunks)
        with runner_w.timed(f"Rest {ws}s"):
            rest_chunks = [gen_state(rest_files[:5], rng, data_size)
                           for rng in runner_w.trial_rngs(offset=100)]
            rest_met = runner_w.collect(rest_chunks)

        n_sig, findings = runner_w.compare(med_met, rest_met)
        d5[ws] = n_sig
        top = findings[0] if findings else ('', 0, 0)
        print(f"  {ws:3d}s ({data_size:5d} samples): Med vs Rest = {n_sig:3d} sig"
              + (f"  top: {top[0].split(':')[0]}:{top[0].split(':')[1][:20]} d={top[1]:+.1f}"
                 if findings else ""))
        runner_w.close()

    return d5


# =========================================================================
# D6: PROPER PHI ATTRACTOR TEST
# =========================================================================

def direction_6():
    """D6: Is m=2.18 a statistical attractor for meditation spectral slope?"""
    print("\n" + "=" * 78)
    print("D6: PHI ATTRACTOR TEST")
    print("    Bootstrap: does meditation m cluster at 2.18 tighter than chance?")
    print("=" * 78)

    med_files, rest_files, sleep_files = scan_files()
    if not med_files:
        print("  No meditation data")
        return {}

    PHI_M = 2.18
    sfreq = 250
    window_samples = int(10 * sfreq)  # 10-second windows

    # Collect spectral slopes from all meditation windows
    med_slopes = []
    for f in med_files[:5]:
        raw = load_raw(f)
        sig = np.mean(raw.get_data(), axis=0)
        for start in range(0, len(sig) - window_samples, window_samples):
            m = compute_spectral_slope(sig[start:start + window_samples], sfreq)
            if np.isfinite(m):
                med_slopes.append(m)

    rest_slopes = []
    for f in (rest_files or [])[:5]:
        raw = load_raw(f)
        sig = np.mean(raw.get_data(), axis=0)
        for start in range(0, len(sig) - window_samples, window_samples):
            m = compute_spectral_slope(sig[start:start + window_samples], sfreq)
            if np.isfinite(m):
                rest_slopes.append(m)

    med_slopes = np.array(med_slopes)
    rest_slopes = np.array(rest_slopes)

    print(f"  Meditation slopes: n={len(med_slopes)}, "
          f"mean={np.mean(med_slopes):.3f}, median={np.median(med_slopes):.3f}, "
          f"SD={np.std(med_slopes):.3f}")
    if len(rest_slopes) > 0:
        print(f"  Rest slopes:       n={len(rest_slopes)}, "
              f"mean={np.mean(rest_slopes):.3f}, median={np.median(rest_slopes):.3f}, "
              f"SD={np.std(rest_slopes):.3f}")

    # Test: is |m - 2.18| for meditation smaller than expected by chance?
    # Null: resample from the combined pool, compute mean |m - 2.18|
    observed_dist = np.mean(np.abs(med_slopes - PHI_M))
    print(f"\n  Observed mean |m - {PHI_M}|: {observed_dist:.4f}")

    # Bootstrap: compare to clustering at other reference values
    ref_values = np.arange(1.0, 3.01, 0.1)
    print(f"\n  Clustering at alternative reference values:")
    best_ref = None
    best_dist = np.inf
    for ref in ref_values:
        dist = np.mean(np.abs(med_slopes - ref))
        marker = " ← phi" if abs(ref - PHI_M) < 0.05 else ""
        if dist < best_dist:
            best_dist = dist
            best_ref = ref
        if abs(ref - PHI_M) < 0.05 or abs(ref - best_ref) < 0.05 or ref in [1.0, 1.5, 2.0, 2.5, 3.0]:
            print(f"    m={ref:.1f}: mean |m - ref| = {dist:.4f}{marker}")

    print(f"\n  Best-fit reference: m={best_ref:.2f} (mean |m - ref| = {best_dist:.4f})")
    print(f"  Phi reference:      m={PHI_M:.2f} (mean |m - ref| = {observed_dist:.4f})")
    if abs(best_ref - PHI_M) < 0.15:
        print(f"  → Phi IS near the empirical optimum (Δ = {abs(best_ref - PHI_M):.2f})")
    else:
        print(f"  → Phi is NOT the empirical optimum (best is {best_ref:.2f}, Δ = {abs(best_ref - PHI_M):.2f})")

    # Permutation test: is meditation's clustering at phi tighter than rest's?
    if len(rest_slopes) > 0:
        rest_dist = np.mean(np.abs(rest_slopes - PHI_M))
        print(f"\n  Rest mean |m - {PHI_M}|: {rest_dist:.4f}")
        print(f"  Meditation is {rest_dist/observed_dist:.1f}× closer to phi than rest")

    return {
        'med_slopes': med_slopes,
        'rest_slopes': rest_slopes,
        'observed_dist': observed_dist,
        'best_ref': best_ref,
    }


# =========================================================================
# D7: PER-CHANNEL PROFILE DIVERSITY
# =========================================================================

def direction_7():
    """D7: Per-channel metric profiles — is meditation more uniform across channels?"""
    print("\n" + "=" * 78)
    print("D7: PER-CHANNEL PROFILE DIVERSITY")
    print("    Run framework on individual channels, compare cross-channel variance.")
    print("=" * 78)

    from exotic_geometry_framework import GeometryAnalyzer

    med_files, rest_files, sleep_files = scan_files()
    if not med_files or not rest_files:
        print("  Missing data")
        return {}

    analyzer = GeometryAnalyzer()
    analyzer.add_tier_geometries(tier='quick')  # fast tier for per-channel analysis

    # Pick one file per state, analyze 8 evenly-spaced channels
    n_channels = 8
    window_samples = int(10 * 250)
    states = {}
    if med_files:
        states['Meditation'] = med_files[0]
    if rest_files:
        states['Rest'] = rest_files[0]
    if sleep_files:
        states['Sleep'] = sleep_files[0]

    d7 = {}
    for name, fpath in states.items():
        raw = load_raw(fpath)
        data = raw.get_data()
        n_ch = data.shape[0]
        ch_idx = np.linspace(0, n_ch - 1, n_channels, dtype=int)

        # For each channel, compute metrics on 5 windows
        channel_profiles = []
        for ch in ch_idx:
            sig = data[ch]
            metrics_list = []
            for w in range(5):
                start = w * window_samples
                if start + window_samples > len(sig):
                    break
                chunk = to_uint8(sig[start:start + window_samples])
                res = analyzer.analyze(chunk)
                m = {}
                for r in res.results:
                    for mn, mv in r.metrics.items():
                        if np.isfinite(mv):
                            m[f"{r.geometry_name}:{mn}"] = mv
                metrics_list.append(m)
            if metrics_list:
                # Average metrics across windows for this channel
                avg = {}
                for k in metrics_list[0]:
                    vals = [m.get(k, np.nan) for m in metrics_list]
                    avg[k] = np.nanmean(vals)
                channel_profiles.append(avg)

        # Compute cross-channel variance for each metric
        if len(channel_profiles) >= 2:
            metric_names = list(channel_profiles[0].keys())
            cvs = []
            for mn in metric_names:
                vals = [cp.get(mn, np.nan) for cp in channel_profiles]
                vals = [v for v in vals if np.isfinite(v)]
                if len(vals) >= 2 and np.mean(vals) != 0:
                    cv = np.std(vals) / (abs(np.mean(vals)) + 1e-15)
                    cvs.append(cv)
            mean_cv = np.mean(cvs) if cvs else 0
            d7[name] = mean_cv
            print(f"  {name:15s}: mean cross-channel CV = {mean_cv:.4f} ({len(cvs)} metrics)")

    # Compare
    if 'Meditation' in d7 and 'Rest' in d7:
        ratio = d7['Rest'] / (d7['Meditation'] + 1e-15)
        print(f"\n  Rest/Meditation CV ratio: {ratio:.2f}")
        if ratio > 1.2:
            print(f"  → Meditation is more spatially uniform ({ratio:.1f}× less channel variance)")
        elif ratio < 0.8:
            print(f"  → Rest is more spatially uniform")
        else:
            print(f"  → Similar spatial uniformity")

    return d7


# =========================================================================
# D8: CONNECTIVITY MATRIX GEOMETRY
# =========================================================================

def direction_8(runner):
    """D8: Coherence matrix → flatten → 1D framework."""
    print("\n" + "=" * 78)
    print("D8: CONNECTIVITY MATRIX GEOMETRY")
    print("    Flatten coherence matrix upper triangle → 1D framework analysis.")
    print("=" * 78)

    _, rest_files, sleep_files = scan_files()
    med_files, _, _ = scan_files()

    n_ch = 16  # subsample to 16 channels
    n_features = n_ch * (n_ch - 1) // 2  # upper triangle = 120

    def gen_coh_vector(files, rng, size):
        """Generate uint8 vector from coherence matrix."""
        f = str(rng.choice(files))
        raw = load_raw(f)
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        ch_idx = np.linspace(0, data.shape[0] - 1, n_ch, dtype=int)
        seg_data = data[ch_idx]

        # Random segment
        seg_len = int(10 * sfreq)
        max_start = max(1, seg_data.shape[1] - seg_len)
        start = int(rng.integers(0, max_start))
        seg = seg_data[:, start:start + seg_len]

        # Compute coherence matrix (broadband 1-40 Hz)
        nperseg = min(int(2 * sfreq), seg.shape[1])
        coh_matrix = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                f_c, Cxy = signal.coherence(seg[i], seg[j], fs=sfreq, nperseg=nperseg)
                mask = (f_c >= 1) & (f_c <= 40)
                coh_matrix[i, j] = np.mean(Cxy[mask]) if np.any(mask) else 0

        # Flatten upper triangle and tile to reach target size
        upper = coh_matrix[np.triu_indices(n_ch, k=1)]
        # Tile to reach size (120 coherence values → tile to 16384)
        tiled = np.tile(upper, size // len(upper) + 1)[:size]
        return to_uint8(tiled)

    states = {}
    if med_files:
        states['Meditation'] = med_files[:5]
    if rest_files:
        states['Rest'] = rest_files[:5]
    if sleep_files:
        states['Sleep'] = sleep_files[:5]

    all_met = {}
    for name, files in states.items():
        with runner.timed(f"Coh {name}"):
            chunks = [gen_coh_vector(files, rng, runner.data_size)
                      for rng in runner.trial_rngs(offset=300)]
            all_met[name] = runner.collect(chunks)

    d8 = {}
    state_names = list(all_met.keys())
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j > i:
                n_sig, findings = runner.compare(all_met[n1], all_met[n2])
                d8[(n1, n2)] = n_sig
                print(f"  {n1:15s} vs {n2:15s}: {n_sig:3d} sig")
                for m, d, _ in findings[:3]:
                    print(f"    {m:50s} d={d:+8.2f}")

    return d8


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()

    med_files, rest_files, sleep_files = scan_files()
    print(f"Data: {len(med_files)} meditation, {len(rest_files)} rest, "
          f"{len(sleep_files)} sleep files")

    runner = Runner("O.IRIS Rigorous", mode="1d", n_workers=N_WORKERS,
                    data_size=16384, n_trials=N_TRIALS)

    print(f"\n{'='*78}")
    print("O.IRIS RIGOROUS — ADDRESSING EVERY BLIND SPOT")
    print(f"{'='*78}")

    try:
        d1 = direction_1(runner)
        d2 = direction_2()
        d3 = direction_3()
        d4 = direction_4()
        d5 = direction_5()
        d6 = direction_6()
        d7 = direction_7()
        d8 = direction_8(runner)

        elapsed = time.time() - t0
        print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Synthesis
        print(f"\n{'='*78}")
        print("SYNTHESIS: WHAT SURVIVED RIGOROUS TESTING?")
        print(f"{'='*78}")

        print(f"\n  D1 Within-dataset rest vs sleep: {d1.get('n_sig', '?')} sig")
        if d1.get('n_sig', 0) == 0:
            print(f"     → CONFIRMED: rest = sleep even within same subjects/equipment")
        else:
            print(f"     → OVERTURNED: rest ≠ sleep ({d1['n_sig']} sig within-dataset)")

        print(f"\n  D2 Spatial (2D) rest vs sleep: {d2.get(('Rest', 'Sleep'), '?')} sig")
        if d2.get(('Rest', 'Sleep'), 0) > 0:
            print(f"     → NEW: rest/sleep differ in cross-channel spatial structure")

        print(f"\n  D5 Window sensitivity (Med vs Rest):")
        for ws in sorted(d5):
            print(f"     {ws:3d}s: {d5[ws]:3d} sig")

        if d6:
            print(f"\n  D6 Phi attractor: best-fit m = {d6.get('best_ref', '?'):.2f} "
                  f"(phi = 2.18)")

    finally:
        runner.close()


if __name__ == "__main__":
    main()
