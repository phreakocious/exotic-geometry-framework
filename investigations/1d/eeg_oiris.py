#!/usr/bin/env python3
"""
Investigation: O.IRIS Brain State Geometry — Full Framework Analysis
=====================================================================

Applies the full 278-metric exotic geometry framework to the actual O.IRIS
datasets (OpenNeuro ds001787 meditation, ds003768 sleep/rest) to test the
O.IRIS hypotheses with 54 geometric lenses instead of 3 hand-crafted metrics.

Key questions:
  1. Does meditation EEG have nonlinear structure beyond the power spectrum?
     (IAAFT surrogate test — our eeg_deep.py found 135 nonlinear sig for
     seizure but only 11 for resting. Where does meditation land?)
  2. Can Penrose/AB quasicrystal geometries detect golden-ratio structure
     in meditation EEG? (Falsification of the phi focal point hypothesis)
  3. Which of 54 geometries best separate meditation from sleep?
     (O.IRIS uses 3 metrics; we have 278)
  4. Does the band structure differ between meditation and rest/sleep?
     (O.IRIS claims meditation has steeper spectral slope m≈1.82)

Datasets (downloaded by download_oiris_data.py):
  ds001787: Meditation EEG — 24 subjects, 64ch BDF, 2048 Hz
  ds003768: Sleep/Rest EEG — 33 subjects, 32ch BrainVision, 5000 Hz

Directions:
  D1: Structure detection — meditation/sleep/rest vs shuffled
  D2: Cross-state discrimination — meditation vs rest vs sleep (pairwise)
  D3: Per-geometry heatmap — which geometric lenses detect which states
  D4: IAAFT nonlinear structure — the KEY test (meditation vs surrogate)
  D5: Spectral band decomposition — per-band geometric signatures
  D6: Quasicrystal phi test — Penrose/AB metrics on meditation vs rest
  D7: Top discriminating metrics — what should O.IRIS be measuring?
  D8: Temporal trajectory — geometric signature stability within sessions
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from scipy import signal, stats
from collections import defaultdict
from tools.investigation_runner import Runner

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_TRIALS = 25
DATA_SIZE = 16384
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
# ENCODING & PREPROCESSING
# =========================================================================

def to_uint8(sig):
    sig = np.asarray(sig, dtype=np.float64)
    lo, hi = np.nanmin(sig), np.nanmax(sig)
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((sig - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def bandpass(sig, sfreq, lo, hi, order=4):
    nyq = sfreq / 2
    lo_n, hi_n = max(lo / nyq, 0.001), min(hi / nyq, 0.999)
    b, a = signal.butter(order, [lo_n, hi_n], btype='band')
    return signal.filtfilt(b, a, sig)


def iaaft_surrogate(data, n_iter=100):
    """IAAFT surrogate: preserves spectrum + amplitude distribution."""
    n = len(data)
    sorted_data = np.sort(data.astype(np.float64))
    amp_orig = np.abs(np.fft.rfft(data))
    phase = np.random.uniform(0, 2 * np.pi, len(amp_orig))
    surr = np.fft.irfft(amp_orig * np.exp(1j * phase), n=n)
    for _ in range(n_iter):
        ranks = np.argsort(np.argsort(surr))
        surr = sorted_data[ranks]
        fft_surr = np.fft.rfft(surr)
        surr = np.fft.irfft(amp_orig * np.exp(1j * np.angle(fft_surr)), n=n)
    ranks = np.argsort(np.argsort(surr))
    return sorted_data[ranks]


# =========================================================================
# DATA LOADERS
# =========================================================================

_MNE_IMPORTED = False
_MNE = None

def _get_mne():
    global _MNE_IMPORTED, _MNE
    if not _MNE_IMPORTED:
        import mne
        mne.set_log_level('ERROR')
        _MNE = mne
        _MNE_IMPORTED = True
    return _MNE


def _scan_meditation_files():
    """Scan for downloaded meditation BDF files."""
    files = []
    if not os.path.isdir(MEDITATION_DIR):
        return files
    for root, dirs, fnames in os.walk(MEDITATION_DIR):
        for f in fnames:
            if f.endswith('.bdf'):
                files.append(os.path.join(root, f))
    files.sort()
    return files

_MED_FILES = None
def get_meditation_files():
    global _MED_FILES
    if _MED_FILES is None:
        _MED_FILES = _scan_meditation_files()
    return _MED_FILES


def _scan_sleep_files():
    """Scan for downloaded sleep/rest BrainVision files."""
    rest_files = []
    sleep_files = []
    if not os.path.isdir(SLEEP_DIR):
        return rest_files, sleep_files
    for root, dirs, fnames in os.walk(SLEEP_DIR):
        for f in fnames:
            if f.endswith('.vhdr'):
                path = os.path.join(root, f)
                if 'task-rest' in f:
                    rest_files.append(path)
                elif 'task-sleep' in f:
                    sleep_files.append(path)
    rest_files.sort()
    sleep_files.sort()
    return rest_files, sleep_files

_REST_FILES = None
_SLEEP_FILES = None
def get_sleep_rest_files():
    global _REST_FILES, _SLEEP_FILES
    if _REST_FILES is None:
        _REST_FILES, _SLEEP_FILES = _scan_sleep_files()
    return _REST_FILES, _SLEEP_FILES


def load_eeg_signal(filepath, target_sfreq=250):
    """Load EEG file, average channels, resample, filter, return (signal, sfreq)."""
    mne = _get_mne()
    if filepath.endswith('.bdf'):
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
    elif filepath.endswith('.vhdr'):
        raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    elif filepath.endswith('.edf'):
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    else:
        raise ValueError(f"Unknown format: {filepath}")

    raw.pick_types(eeg=True, verbose=False)
    if raw.info['sfreq'] > target_sfreq + 10:
        raw.resample(target_sfreq, verbose=False)
    raw.filter(1., 55., fir_design='firwin', verbose=False)
    sig = np.mean(raw.get_data(), axis=0)
    return sig, raw.info['sfreq']


# Cache loaded signals to avoid re-reading files
_SIG_CACHE = {}

def get_signal(filepath, target_sfreq=250):
    if filepath not in _SIG_CACHE:
        _SIG_CACHE[filepath] = load_eeg_signal(filepath, target_sfreq)
    return _SIG_CACHE[filepath]


def gen_meditation(rng, size):
    """Generate uint8 chunk from meditation EEG."""
    files = get_meditation_files()
    if not files:
        raise RuntimeError("No meditation files found. Run download_oiris_data.py first.")
    fpath = str(rng.choice(files))
    sig, sfreq = get_signal(fpath)
    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


def gen_rest(rng, size):
    """Generate uint8 chunk from resting EEG."""
    rest_files, _ = get_sleep_rest_files()
    if not rest_files:
        raise RuntimeError("No rest files found. Run download_oiris_data.py first.")
    fpath = str(rng.choice(rest_files))
    sig, sfreq = get_signal(fpath)
    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


def gen_sleep(rng, size):
    """Generate uint8 chunk from sleep EEG."""
    _, sleep_files = get_sleep_rest_files()
    if not sleep_files:
        raise RuntimeError("No sleep files found. Run download_oiris_data.py first.")
    fpath = str(rng.choice(sleep_files))
    sig, sfreq = get_signal(fpath)
    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


def gen_meditation_band(band_name, rng, size):
    """Generate uint8 chunk from meditation EEG filtered to a specific band."""
    files = get_meditation_files()
    fpath = str(rng.choice(files))
    sig, sfreq = get_signal(fpath)
    lo, hi = BANDS[band_name]
    sig = bandpass(sig, sfreq, lo, hi)
    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


def gen_sleep_band(band_name, rng, size):
    """Generate uint8 chunk from sleep EEG filtered to a specific band."""
    _, sleep_files = get_sleep_rest_files()
    fpath = str(rng.choice(sleep_files))
    sig, sfreq = get_signal(fpath)
    lo, hi = BANDS[band_name]
    sig = bandpass(sig, sfreq, lo, hi)
    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


# =========================================================================
# DIRECTIONS
# =========================================================================

def direction_1(runner):
    """D1: Structure detection — each state vs shuffled."""
    print("\n" + "=" * 78)
    print("D1: STRUCTURE DETECTION — BRAIN STATE VS SHUFFLED")
    print("=" * 78)

    states = {}
    rest_files, sleep_files = get_sleep_rest_files()
    med_files = get_meditation_files()

    if med_files:
        states['Meditation'] = gen_meditation
    if rest_files:
        states['Rest'] = gen_rest
    if sleep_files:
        states['Sleep'] = gen_sleep

    d1 = {}
    for name, gen_fn in states.items():
        with runner.timed(name):
            chunks = [gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs()]
            met = runner.collect(chunks)

        shuf_chunks = []
        for i, chunk in enumerate(chunks):
            s = chunk.copy()
            np.random.default_rng(42000 + i).shuffle(s)
            shuf_chunks.append(s)
        shuf_met = runner.collect(shuf_chunks)

        n_sig, findings = runner.compare(met, shuf_met)
        d1[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"  {name:<15s}: {n_sig:3d} sig (vs shuffled)")
        for m, d, _ in findings[:3]:
            print(f"    {m:50s} d={d:+8.2f}")

    return d1


def direction_2(runner):
    """D2: Cross-state discrimination — pairwise."""
    print("\n" + "=" * 78)
    print("D2: CROSS-STATE DISCRIMINATION")
    print("=" * 78)

    state_gens = {}
    rest_files, sleep_files = get_sleep_rest_files()
    med_files = get_meditation_files()
    if med_files:
        state_gens['Meditation'] = gen_meditation
    if rest_files:
        state_gens['Rest'] = gen_rest
    if sleep_files:
        state_gens['Sleep'] = gen_sleep

    all_met = {}
    for name, gen_fn in state_gens.items():
        with runner.timed(name):
            chunks = [gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=200)]
            all_met[name] = runner.collect(chunks)

    state_names = list(state_gens.keys())
    d2 = {}
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j > i:
                n_sig, findings = runner.compare(all_met[n1], all_met[n2])
                d2[(n1, n2)] = n_sig
                top = findings[0] if findings else ('', 0, 0)
                print(f"  {n1:15s} vs {n2:15s}: {n_sig:3d} sig"
                      + (f"  top: {top[0]:30s} d={top[1]:+.2f}" if findings else ""))

    return d2, all_met


def direction_3(runner, all_met):
    """D3: Per-geometry heatmap — which lenses detect which states."""
    print("\n" + "=" * 78)
    print("D3: PER-GEOMETRY DETECTION — TOP DISCRIMINATORS")
    print("=" * 78)

    state_names = list(all_met.keys())
    if len(state_names) < 2:
        print("  Need at least 2 states for comparison")
        return {}, []

    # Compare each pair and track per-geometry sig counts
    geom_names = sorted(set(m.split(':')[0] for m in runner.metric_names))
    heatmap = {}

    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j <= i:
                continue
            for geom in geom_names:
                geom_metrics = [m for m in runner.metric_names
                                if m.startswith(geom + ':')]
                n_sig = 0
                for m in geom_metrics:
                    a = np.array(all_met[n1].get(m, []))
                    b = np.array(all_met[n2].get(m, []))
                    if len(a) < 3 or len(b) < 3:
                        continue
                    _, p = stats.ttest_ind(a, b, equal_var=False)
                    if np.isfinite(p) and p < runner.bonf_alpha:
                        n_sig += 1
                heatmap[(geom, n1, n2)] = n_sig

    # Print top geometries for each pair
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j <= i:
                continue
            detectors = [(g, heatmap.get((g, n1, n2), 0)) for g in geom_names]
            detectors.sort(key=lambda x: -x[1])
            top = [f"{g}={n}" for g, n in detectors[:5] if n > 0]
            print(f"  {n1} vs {n2}: {', '.join(top)}")

    return heatmap, geom_names


def direction_4(runner):
    """D4: IAAFT nonlinear structure — THE KEY TEST."""
    print("\n" + "=" * 78)
    print("D4: IAAFT SURROGATES — NONLINEAR STRUCTURE (THE KEY TEST)")
    print("=" * 78)
    print("  If meditation has high nonlinear sig → qualitatively different state")
    print("  If meditation ≈ rest (both low) → structure is mostly spectral")

    state_gens = {}
    rest_files, sleep_files = get_sleep_rest_files()
    med_files = get_meditation_files()
    if med_files:
        state_gens['Meditation'] = gen_meditation
    if rest_files:
        state_gens['Rest'] = gen_rest
    if sleep_files:
        state_gens['Sleep'] = gen_sleep

    d4 = {}
    for name, gen_fn in state_gens.items():
        print(f"  {name}:")
        with runner.timed(f"{name} original"):
            orig_chunks = [gen_fn(rng, DATA_SIZE)
                           for rng in runner.trial_rngs(offset=400)]
            orig_met = runner.collect(orig_chunks)

        with runner.timed(f"{name} IAAFT"):
            surr_chunks = []
            for chunk in orig_chunks:
                s = iaaft_surrogate(chunk)
                surr_chunks.append(np.clip(s, 0, 255).astype(np.uint8))
            surr_met = runner.collect(surr_chunks)

        n_sig, findings = runner.compare(orig_met, surr_met)
        d4[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"    Original vs IAAFT: {n_sig:3d} sig")
        for m, d, _ in findings[:5]:
            print(f"      {m:50s} d={d:+8.2f}")

    return d4


def direction_5(runner):
    """D5: Spectral band decomposition."""
    print("\n" + "=" * 78)
    print("D5: SPECTRAL BAND DECOMPOSITION")
    print("=" * 78)

    d5 = {}
    med_files = get_meditation_files()
    _, sleep_files = get_sleep_rest_files()

    state_gens = []
    if med_files:
        state_gens.append(('Meditation', gen_meditation_band))
    if sleep_files:
        state_gens.append(('Sleep', gen_sleep_band))

    for state_name, band_gen in state_gens:
        print(f"  {state_name}:")
        for band_name in BANDS:
            with runner.timed(f"{state_name}/{band_name}"):
                chunks = [band_gen(band_name, rng, DATA_SIZE)
                          for rng in runner.trial_rngs(offset=500)]
                met = runner.collect(chunks)

            shuf_chunks = [np.random.default_rng(i).permutation(c)
                           for i, c in enumerate(chunks)]
            shuf_met = runner.collect(shuf_chunks)
            n_sig, findings = runner.compare(met, shuf_met)
            d5[(state_name, band_name)] = n_sig
            top_str = f"  top: {findings[0][0].split(':')[0]}" if findings else ""
            print(f"    {band_name:8s}: {n_sig:3d} sig{top_str}")

    return d5


def direction_6(runner):
    """D6: Quasicrystal phi test — does Penrose detect golden-ratio structure?"""
    print("\n" + "=" * 78)
    print("D6: QUASICRYSTAL PHI TEST")
    print("=" * 78)
    print("  Testing: do Penrose/AB metrics detect elevated phi-structure")
    print("  in meditation EEG vs rest/sleep?")

    med_files = get_meditation_files()
    rest_files, sleep_files = get_sleep_rest_files()

    if not med_files:
        print("  No meditation data available")
        return {}

    # Collect meditation and rest/sleep metrics
    all_met = {}
    with runner.timed("Meditation"):
        chunks = [gen_meditation(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=600)]
        all_met['Meditation'] = runner.collect(chunks)
    if rest_files:
        with runner.timed("Rest"):
            chunks = [gen_rest(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=700)]
            all_met['Rest'] = runner.collect(chunks)
    if sleep_files:
        with runner.timed("Sleep"):
            chunks = [gen_sleep(rng, DATA_SIZE) for rng in runner.trial_rngs(offset=800)]
            all_met['Sleep'] = runner.collect(chunks)

    # Extract Penrose and AB metrics specifically
    qc_metrics = [m for m in runner.metric_names
                  if m.startswith('Penrose') or m.startswith('Ammann-Beenker')]

    d6 = {}
    for other in ['Rest', 'Sleep']:
        if other not in all_met:
            continue
        print(f"\n  Meditation vs {other} — QC metrics:")
        for m in qc_metrics:
            a = np.array(all_met['Meditation'].get(m, []))
            b = np.array(all_met[other].get(m, []))
            if len(a) < 3 or len(b) < 3:
                continue
            d_val = (np.mean(a) - np.mean(b)) / (np.sqrt((np.std(a)**2 + np.std(b)**2) / 2) + 1e-15)
            _, p = stats.ttest_ind(a, b, equal_var=False)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            metric_short = m.split(':')[1]
            direction = "Med > " + other if d_val > 0 else other + " > Med"
            print(f"    {metric_short:35s} d={d_val:+6.2f}  p={p:.4f} {sig:3s} ({direction})")
            d6[(m, other)] = d_val

    return d6


def direction_7(runner, all_met):
    """D7: Top discriminating metrics — what should O.IRIS be measuring?"""
    print("\n" + "=" * 78)
    print("D7: TOP DISCRIMINATING METRICS")
    print("=" * 78)
    print("  These are the metrics O.IRIS should consider adopting.")

    state_names = list(all_met.keys())
    if len(state_names) < 2:
        print("  Need at least 2 states")
        return {}

    # For each pair, find top 10 metrics by Cohen's d
    d7 = {}
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j <= i:
                continue
            print(f"\n  {n1} vs {n2} — Top 10:")
            results = []
            for m in runner.metric_names:
                a = np.array(all_met[n1].get(m, []))
                b = np.array(all_met[n2].get(m, []))
                if len(a) < 3 or len(b) < 3:
                    continue
                pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
                if pooled < 1e-15:
                    continue
                d_val = (np.mean(a) - np.mean(b)) / pooled
                _, p = stats.ttest_ind(a, b, equal_var=False)
                if np.isfinite(d_val) and np.isfinite(p):
                    results.append((m, d_val, p))
            results.sort(key=lambda x: -abs(x[1]))
            for m, d_val, p in results[:10]:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {m:50s} d={d_val:+8.2f} {sig}")
            d7[(n1, n2)] = results[:10]

    return d7


def direction_8(runner):
    """D8: Temporal trajectory — signature stability within meditation sessions."""
    print("\n" + "=" * 78)
    print("D8: TEMPORAL TRAJECTORY — WITHIN-SESSION STABILITY")
    print("=" * 78)

    med_files = get_meditation_files()
    if not med_files:
        print("  No meditation data available")
        return {}

    # Pick one long session, slice into early/middle/late thirds
    fpath = med_files[0]
    sig, sfreq = get_signal(fpath)
    n = len(sig)
    third = n // 3

    d8 = {}
    segments = {
        'Early':  sig[:third],
        'Middle': sig[third:2*third],
        'Late':   sig[2*third:],
    }

    seg_met = {}
    for seg_name, seg_sig in segments.items():
        with runner.timed(seg_name):
            # Slice into N_TRIALS chunks
            chunk_size = min(DATA_SIZE, len(seg_sig) // N_TRIALS)
            chunks = []
            for i in range(N_TRIALS):
                start = i * (len(seg_sig) // N_TRIALS)
                chunk = to_uint8(seg_sig[start:start + chunk_size])
                if len(chunk) >= 1000:
                    chunks.append(chunk)
            if chunks:
                seg_met[seg_name] = runner.collect(chunks)
        print(f"  {seg_name}: {len(chunks)} chunks collected")

    # Compare early vs late
    seg_names = list(seg_met.keys())
    for i, n1 in enumerate(seg_names):
        for j, n2 in enumerate(seg_names):
            if j > i:
                n_sig, findings = runner.compare(seg_met[n1], seg_met[n2])
                d8[(n1, n2)] = n_sig
                print(f"  {n1:8s} vs {n2:8s}: {n_sig:3d} sig")
                for m, d, _ in findings[:3]:
                    print(f"    {m:50s} d={d:+8.2f}")

    return d8


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(runner, d1, d2, d4, d5, d6):
    fig, axes = runner.create_figure(4, "O.IRIS Brain State Geometry",
                                     rows=2, cols=2, figsize=(18, 14))

    # D1: Structure detection
    ax = axes[0]
    names = list(d1.keys())
    sigs = [d1[n]['n_sig'] for n in names]
    colors = {'Meditation': '#FF9800', 'Rest': '#2196F3', 'Sleep': '#9C27B0'}
    ax.bar(names, sigs, color=[colors.get(n, '#666') for n in names], alpha=0.85)
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D1: Structure vs Shuffled', fontsize=11, fontweight='bold')
    for i, v in enumerate(sigs):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10, color='white',
                fontweight='bold')

    # D2: Cross-state heatmap
    ax = axes[1]
    state_names = list(set(n for pair in d2 for n in pair))
    state_names.sort()
    n = len(state_names)
    mat = np.zeros((n, n))
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if i != j:
                val = d2.get((n1, n2), d2.get((n2, n1), 0))
                mat[i, j] = val
    runner.plot_heatmap(ax, mat, state_names, 'D2: Cross-State Discrimination')

    # D4: IAAFT nonlinear structure
    ax = axes[2]
    iaaft_names = list(d4.keys())
    iaaft_sigs = [d4[n]['n_sig'] for n in iaaft_names]
    ax.bar(iaaft_names, iaaft_sigs,
           color=[colors.get(n, '#666') for n in iaaft_names], alpha=0.85)
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D4: IAAFT Nonlinear Structure', fontsize=11, fontweight='bold')
    for i, v in enumerate(iaaft_sigs):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10, color='white',
                fontweight='bold')

    # D5: Band decomposition
    ax = axes[3]
    band_names = list(BANDS.keys())
    for state_name in ['Meditation', 'Sleep']:
        vals = [d5.get((state_name, b), 0) for b in band_names]
        if any(v > 0 for v in vals):
            ax.plot(band_names, vals, 'o-', linewidth=2, markersize=6,
                    color=colors.get(state_name, '#666'), label=state_name)
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D5: Spectral Band Structure', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    runner.save(fig, "eeg_oiris")


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()

    # Check data availability
    med_files = get_meditation_files()
    rest_files, sleep_files = get_sleep_rest_files()
    print(f"Data check:")
    print(f"  Meditation (ds001787): {len(med_files)} BDF files")
    print(f"  Rest (ds003768):       {len(rest_files)} BrainVision files")
    print(f"  Sleep (ds003768):      {len(sleep_files)} BrainVision files")

    if not med_files and not rest_files and not sleep_files:
        print("\nNo O.IRIS data found. Run download_oiris_data.py first.")
        return

    runner = Runner("O.IRIS EEG", mode="1d",
                    n_workers=N_WORKERS, data_size=DATA_SIZE,
                    n_trials=N_TRIALS)

    print(f"\n{'='*78}")
    print("O.IRIS BRAIN STATE GEOMETRY — FULL FRAMEWORK ANALYSIS")
    print(f"  {runner.n_metrics} metrics, {N_WORKERS} workers")
    print(f"{'='*78}")

    try:
        d1 = direction_1(runner)
        d2, all_met = direction_2(runner)
        d3, d3_geoms = direction_3(runner, all_met)
        d4 = direction_4(runner)
        d5 = direction_5(runner)
        d6 = direction_6(runner)
        d7 = direction_7(runner, all_met)
        d8 = direction_8(runner)

        make_figure(runner, d1, d2, d4, d5, d6)

        elapsed = time.time() - t0
        print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Key results summary
        print(f"\n{'='*78}")
        print("KEY RESULTS FOR O.IRIS")
        print(f"{'='*78}")

        if 'Meditation' in d4:
            med_nl = d4['Meditation']['n_sig']
            rest_nl = d4.get('Rest', {}).get('n_sig', '?')
            sleep_nl = d4.get('Sleep', {}).get('n_sig', '?')
            print(f"\n  IAAFT Nonlinear Structure (the key test):")
            print(f"    Meditation: {med_nl} sig")
            print(f"    Rest:       {rest_nl} sig")
            print(f"    Sleep:      {sleep_nl} sig")
            if isinstance(med_nl, int) and isinstance(rest_nl, int):
                if med_nl > rest_nl * 2:
                    print(f"    → Meditation has {med_nl/max(rest_nl,1):.1f}× more "
                          f"nonlinear structure than rest")
                    print(f"    → The phi focal point may reflect genuine nonlinear dynamics")
                elif med_nl <= rest_nl:
                    print(f"    → Meditation ≈ rest in nonlinear structure")
                    print(f"    → The O.IRIS signal is predominantly spectral, not nonlinear")

        if d2:
            print(f"\n  Cross-state discrimination:")
            for (n1, n2), n_sig in sorted(d2.items()):
                print(f"    {n1} vs {n2}: {n_sig} sig")

        runner.print_summary({
            'D1 Structure': ', '.join(f"{n}={d1[n]['n_sig']}" for n in d1),
            'D4 IAAFT': ', '.join(f"{n}={d4[n]['n_sig']}" for n in d4),
            'D2 Discrimination': ', '.join(f"{n1}v{n2}={v}" for (n1,n2),v in d2.items()),
        })
    finally:
        runner.close()


if __name__ == "__main__":
    main()
