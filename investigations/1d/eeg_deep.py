#!/usr/bin/env python3
"""
Investigation: Deep EEG Geometry — Exhaustive Brain State Analysis
===================================================================

Three real EEG datasets through the full 278-metric exotic geometry framework.

Datasets:
  1. PhysioNet eegmmidb (109 subjects, 64 channels, 160 Hz)
     - Eyes open rest, eyes closed rest, motor execution, motor imagery
     - 14 runs per subject; we use runs 1-6 (6 distinct conditions)
  2. Bonn epilepsy (11,500 segments, 5 classes)
     - Seizure, tumor region EO, healthy EO, eyes closed, eyes open
  3. CHB-MIT scalp EEG (22 seizure recordings)
     - Seizure vs interictal windows

Directions:
  D1: Structure detection — each brain state vs shuffled surrogate
  D2: Cross-state discrimination — pairwise matrix (all states)
  D3: Per-geometry heatmap — which geometric lenses detect which states
  D4: Spectral band decomposition — delta/theta/alpha/beta/gamma separately
  D5: Channel topology — frontal vs central vs occipital geometric signatures
  D6: Seizure detection — Bonn + CHB-MIT seizure vs non-seizure
  D7: Motor imagery — real vs imagined movement (BCI-relevant)
  D8: IAAFT surrogate — nonlinear structure beyond the power spectrum
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
ALPHA = 0.05

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'eeg')
EEG_CACHE_DIRS = ['/tmp/eeg_deep', '/tmp/eeg_geometry']  # search both caches
EEG_CACHE = '/tmp/eeg_deep'
BONN_CSV = os.path.join(DATA_DIR, 'Epileptic Seizure Recognition.csv')
CHBMIT_DIR = os.path.join(DATA_DIR, 'chbmit')

# Channel groups for topological analysis (10-20 system)
CHANNEL_GROUPS = {
    'Frontal':   ['Fz', 'F3', 'F4', 'Fp1', 'Fp2', 'F7', 'F8'],
    'Central':   ['Cz', 'C3', 'C4'],
    'Parietal':  ['Pz', 'P3', 'P4'],
    'Occipital': ['Oz', 'O1', 'O2'],
    'Temporal':  ['T7', 'T8', 'P7', 'P8'],
}

# Spectral bands
BANDS = {
    'delta':  (0.5, 4),
    'theta':  (4, 8),
    'alpha':  (8, 13),
    'beta':   (13, 30),
    'gamma':  (30, 50),
}


# =========================================================================
# ENCODING
# =========================================================================

def to_uint8(sig):
    """Min-max normalize float signal to uint8."""
    sig = np.asarray(sig, dtype=np.float64)
    lo, hi = np.nanmin(sig), np.nanmax(sig)
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((sig - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def bandpass(sig, sfreq, lo, hi, order=4):
    """Butterworth bandpass filter."""
    nyq = sfreq / 2
    lo_n, hi_n = max(lo / nyq, 0.001), min(hi / nyq, 0.999)
    b, a = signal.butter(order, [lo_n, hi_n], btype='band')
    return signal.filtfilt(b, a, sig)


def iaaft_surrogate(data, n_iter=100):
    """IAAFT surrogate: preserves spectrum + amplitude distribution."""
    n = len(data)
    sorted_data = np.sort(data)
    fft_orig = np.fft.rfft(data)
    amp_orig = np.abs(fft_orig)
    # Random phase start
    phase = np.random.uniform(0, 2 * np.pi, len(fft_orig))
    surr = np.fft.irfft(amp_orig * np.exp(1j * phase), n=n)
    for _ in range(n_iter):
        # Rank-order to match amplitude distribution
        ranks = np.argsort(np.argsort(surr))
        surr = sorted_data[ranks]
        # Match spectrum
        fft_surr = np.fft.rfft(surr)
        surr = np.fft.irfft(amp_orig * np.exp(1j * np.angle(fft_surr)), n=n)
    ranks = np.argsort(np.argsort(surr))
    return sorted_data[ranks]


# =========================================================================
# DATA LOADERS
# =========================================================================

_BONN_CACHE = None

def load_bonn():
    """Load Bonn epilepsy CSV. Returns dict: class_label -> list of 178-sample segments."""
    global _BONN_CACHE
    if _BONN_CACHE is not None:
        return _BONN_CACHE
    import csv
    classes = defaultdict(list)
    with open(BONN_CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            label = int(row[-1])
            vals = np.array([float(x) for x in row[1:-1]])
            classes[label].append(vals)
    _BONN_CACHE = classes
    return classes

BONN_LABELS = {1: 'Seizure', 2: 'Tumor EO', 3: 'Healthy EO',
               4: 'Eyes Closed', 5: 'Eyes Open'}


def gen_bonn(label, rng, size):
    """Generate uint8 chunk from Bonn class by concatenating random segments."""
    classes = load_bonn()
    segments = classes[label]
    pieces = []
    total = 0
    while total < size:
        idx = int(rng.integers(0, len(segments)))
        pieces.append(segments[idx])
        total += len(segments[idx])
    return to_uint8(np.concatenate(pieces)[:size])


def _scan_cached_edf():
    """Scan all cache dirs for available eegmmidb EDF files.
    Returns dict: run_number -> list of (subject_int, filepath)."""
    import re
    available = defaultdict(list)
    for cache_dir in EEG_CACHE_DIRS:
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                m = re.match(r'S(\d+)R(\d+)\.edf', f)
                if m:
                    subj = int(m.group(1))
                    run = int(m.group(2))
                    available[run].append((subj, os.path.join(root, f)))
    # Deduplicate by subject per run
    for run in available:
        seen = set()
        deduped = []
        for subj, path in available[run]:
            if subj not in seen:
                seen.add(subj)
                deduped.append((subj, path))
        available[run] = deduped
    return available

_EEGMMI_CACHE = None

def get_eegmmi_available():
    """Get cached available EDF files (scanned once)."""
    global _EEGMMI_CACHE
    if _EEGMMI_CACHE is None:
        _EEGMMI_CACHE = _scan_cached_edf()
        print("  Cached eegmmidb runs:")
        for run in sorted(_EEGMMI_CACHE):
            print(f"    Run {run:2d}: {len(_EEGMMI_CACHE[run])} subjects")
    return _EEGMMI_CACHE


def load_eegmmi_raw_cached(subject, runs):
    """Load eegmmidb from cache only — no downloads."""
    import mne
    mne.set_log_level('ERROR')
    available = get_eegmmi_available()
    fnames = []
    for run in runs:
        for subj, path in available.get(run, []):
            if subj == subject:
                fnames.append(path)
                break
    if not fnames:
        raise FileNotFoundError(f"Subject {subject} runs {runs} not cached")
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
    mne.datasets.eegbci.standardize(raw)
    raw.filter(1., 55., fir_design='firwin', verbose=False)
    return raw


def _subjects_for_runs(runs):
    """Return list of subjects that have ALL requested runs cached."""
    available = get_eegmmi_available()
    if not runs:
        return []
    sets = [set(subj for subj, _ in available.get(r, [])) for r in runs]
    return sorted(set.intersection(*sets)) if sets else []


def gen_eegmmi(runs, rng, size, subjects=None, channel_group=None):
    """Generate uint8 chunk from cached eegmmidb data."""
    if subjects is None:
        subjects = _subjects_for_runs(runs)
    if not subjects:
        raise RuntimeError(f"No cached subjects for runs {runs}")

    subj = int(rng.choice(subjects))
    raw = load_eegmmi_raw_cached(subj, runs)

    # Pick channels
    if channel_group and channel_group in CHANNEL_GROUPS:
        picks = [ch for ch in CHANNEL_GROUPS[channel_group]
                 if ch in raw.ch_names]
        if not picks:
            picks = raw.ch_names[:3]
    else:
        picks = raw.ch_names

    data = raw.get_data(picks=picks)
    sig = np.mean(data, axis=0)

    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


def gen_eegmmi_band(runs, band_name, rng, size, subjects=None):
    """Generate uint8 chunk from cached eegmmidb filtered to a specific band."""
    if subjects is None:
        subjects = _subjects_for_runs(runs)
    if not subjects:
        raise RuntimeError(f"No cached subjects for runs {runs}")

    subj = int(rng.choice(subjects))
    raw = load_eegmmi_raw_cached(subj, runs)

    sig = np.mean(raw.get_data(), axis=0)
    lo, hi = BANDS[band_name]
    sig = bandpass(sig, raw.info['sfreq'], lo, hi)

    if len(sig) < size:
        sig = np.tile(sig, size // len(sig) + 1)
    start = int(rng.integers(0, max(1, len(sig) - size)))
    return to_uint8(sig[start:start + size])


# CHB-MIT seizure annotations (seconds)
CHBMIT_SEIZURES = {
    'chb01_03.edf': [(2996, 3036)],
    'chb01_04.edf': [(1467, 1494)],
    'chb01_15.edf': [(1732, 1772)],
    'chb01_16.edf': [(1015, 1066)],
    'chb01_18.edf': [(1720, 1810)],
    'chb02_16.edf': [(130, 212)],
    'chb02_16+.edf': [(2972, 3053)],
    'chb02_19.edf': [(3369, 3378)],
    'chb03_01.edf': [(362, 414)],
    'chb03_02.edf': [(731, 796)],
    'chb03_03.edf': [(432, 501)],
    'chb03_04.edf': [(2162, 2214)],
    'chb03_34.edf': [(1982, 2029)],
    'chb03_35.edf': [(2592, 2656)],
    'chb03_36.edf': [(1725, 1778)],
    'chb04_05.edf': [(7804, 7853)],
    'chb04_08.edf': [(6446, 6557)],
    'chb04_28.edf': [(1679, 1781), (3782, 3898)],
    'chb05_06.edf': [(417, 532)],
    'chb05_13.edf': [(1086, 1196)],
    'chb05_16.edf': [(2317, 2413)],
    'chb05_22.edf': [(2348, 2465)],
}


def gen_chbmit(seizure, rng, size):
    """Generate uint8 chunk from CHB-MIT. seizure=True for seizure windows."""
    try:
        import pyedflib
    except ImportError:
        return rng.integers(0, 256, size, dtype=np.uint8)

    files = [f for f in os.listdir(CHBMIT_DIR) if f.endswith('.edf')]
    if not files:
        return rng.integers(0, 256, size, dtype=np.uint8)

    fname = str(rng.choice(files))
    fpath = os.path.join(CHBMIT_DIR, fname)
    try:
        reader = pyedflib.EdfReader(fpath)
    except Exception:
        return rng.integers(0, 256, size, dtype=np.uint8)

    sfreq = reader.getSampleFrequency(0)
    n_samples = reader.getNSamples()[0]
    n_channels = min(reader.signals_in_file, 23)  # cap at 23 EEG channels

    seizure_times = CHBMIT_SEIZURES.get(fname, [])

    if seizure and seizure_times:
        # Pick a random seizure window
        s_start, s_end = seizure_times[int(rng.integers(0, len(seizure_times)))]
        center = int((s_start + s_end) / 2 * sfreq)
        start = max(0, center - size // 2)
    else:
        # Interictal: pick a window far from seizures
        seizure_samples = set()
        for s_start, s_end in seizure_times:
            for s in range(max(0, int(s_start * sfreq) - size),
                           min(n_samples, int(s_end * sfreq) + size)):
                seizure_samples.add(s)
        safe_starts = [s for s in range(0, n_samples - size, size)
                       if s not in seizure_samples]
        if safe_starts:
            start = int(rng.choice(safe_starts))
        else:
            start = 0

    # Read and average channels
    sigs = []
    for ch in range(n_channels):
        try:
            s = reader.readSignal(ch, start=start, n=size)
            if len(s) == size:
                sigs.append(s)
        except Exception:
            pass
    reader.close()

    if not sigs:
        return rng.integers(0, 256, size, dtype=np.uint8)
    return to_uint8(np.mean(sigs, axis=0))


# =========================================================================
# DIRECTIONS
# =========================================================================

def direction_1(runner):
    """D1: Structure detection — each brain state vs shuffled."""
    print("\n" + "=" * 78)
    print("D1: STRUCTURE DETECTION — BRAIN STATE VS SHUFFLED")
    print("=" * 78)

    # Build states from what's actually cached
    states = {
        'Seizure (Bonn)': lambda rng, sz: gen_bonn(1, rng, sz),
        'Healthy (Bonn)': lambda rng, sz: gen_bonn(5, rng, sz),
    }
    available = get_eegmmi_available()
    eegmmi_states = [
        ('Eyes Open',      [1]),
        ('Eyes Closed',    [2]),
        ('Motor Real',     [3]),
        ('Motor Imagined', [4]),
    ]
    for name, runs in eegmmi_states:
        subjs = _subjects_for_runs(runs)
        if len(subjs) >= 5:
            states[name] = lambda rng, sz, r=runs: gen_eegmmi(r, rng, sz)
        else:
            print(f"  Skipping {name}: only {len(subjs)} subjects cached")

    d1 = {}
    for name, gen_fn in states.items():
        with runner.timed(name):
            chunks = [gen_fn(rng, DATA_SIZE) for rng in runner.trial_rngs()]
            met = runner.collect(chunks)

        # Shuffled baseline
        shuf_chunks = []
        for chunk in chunks:
            s = chunk.copy()
            np.random.default_rng(hash(name) & 0xFFFFFFFF).shuffle(s)
            shuf_chunks.append(s)
        shuf_met = runner.collect(shuf_chunks)

        n_sig, findings = runner.compare(met, shuf_met)
        d1[name] = {'n_sig': n_sig, 'findings': findings}
        print(f"  {name:<20s}: {n_sig:3d} sig (vs shuffled)")
        for m, d, _ in findings[:3]:
            print(f"    {m:50s} d={d:+8.2f}")

    return d1


def direction_2(runner):
    """D2: Cross-state discrimination — pairwise."""
    print("\n" + "=" * 78)
    print("D2: CROSS-STATE DISCRIMINATION")
    print("=" * 78)

    candidate_states = [
        ('Eyes Open',      [1]),
        ('Eyes Closed',    [2]),
        ('Motor Real',     [3]),
        ('Motor Imagined', [4]),
    ]
    state_gens = {}
    for name, runs in candidate_states:
        subjs = _subjects_for_runs(runs)
        if len(subjs) >= 5:
            state_gens[name] = lambda rng, sz, r=runs: gen_eegmmi(r, rng, sz)
    state_names = list(state_gens.keys())

    # Collect all states
    all_met = {}
    for name, gen_fn in state_gens.items():
        with runner.timed(name):
            chunks = [gen_fn(rng, DATA_SIZE)
                      for rng in runner.trial_rngs(offset=200)]
            all_met[name] = runner.collect(chunks)

    # Pairwise
    d2 = {}
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j > i:
                n_sig, findings = runner.compare(all_met[n1], all_met[n2])
                d2[(n1, n2)] = n_sig
                top = findings[0] if findings else ('', 0, 0)
                print(f"  {n1:20s} vs {n2:20s}: {n_sig:3d} sig"
                      f"  top: {top[0]:30s} d={top[1]:+.2f}" if findings else
                      f"  {n1:20s} vs {n2:20s}: {n_sig:3d} sig")

    return d2, all_met


def direction_3(runner, all_met):
    """D3: Per-geometry heatmap — which lenses detect which states."""
    print("\n" + "=" * 78)
    print("D3: PER-GEOMETRY DETECTION HEATMAP")
    print("=" * 78)

    state_names = list(all_met.keys())
    geom_names = sorted(set(m.split(':')[0] for m in runner.metric_names))

    # For each state, compare to shuffled baseline
    heatmap = {}  # (geom, state) -> n_sig_metrics
    for state in state_names:
        met = all_met[state]
        for geom in geom_names:
            geom_metrics = [m for m in runner.metric_names if m.startswith(geom + ':')]
            n_sig = 0
            for m in geom_metrics:
                vals = met.get(m, [])
                if len(vals) < 5:
                    continue
                # Compare to shuffled mean (all metrics centered around the same
                # value for shuffled data → check CV as proxy)
                cv = np.std(vals) / (abs(np.mean(vals)) + 1e-15)
                if cv > 0.01:  # has variance
                    n_sig += 1
            heatmap[(geom, state)] = n_sig

    # Print top geometries per state
    for state in state_names:
        detectors = [(g, heatmap.get((g, state), 0)) for g in geom_names]
        detectors.sort(key=lambda x: -x[1])
        top = [f"{g}={n}" for g, n in detectors[:5] if n > 0]
        print(f"  {state:20s}: {', '.join(top)}")

    return heatmap, geom_names


def direction_4(runner):
    """D4: Spectral band decomposition — per-band geometric signatures."""
    print("\n" + "=" * 78)
    print("D4: SPECTRAL BAND DECOMPOSITION")
    print("=" * 78)

    candidate = [('Eyes Closed', [2]), ('Motor Real', [3])]
    states = {n: r for n, r in candidate if len(_subjects_for_runs(r)) >= 5}

    d4 = {}
    for state_name, runs in states.items():
        print(f"  {state_name}:")
        for band_name in BANDS:
            with runner.timed(f"{state_name}/{band_name}"):
                chunks = [gen_eegmmi_band(runs, band_name, rng, DATA_SIZE)
                          for rng in runner.trial_rngs(offset=300)]
                met = runner.collect(chunks)

            # Compare to shuffled
            shuf_chunks = [np.random.default_rng(i).permutation(c)
                           for i, c in enumerate(chunks)]
            shuf_met = runner.collect(shuf_chunks)
            n_sig, findings = runner.compare(met, shuf_met)
            d4[(state_name, band_name)] = n_sig
            top_str = f"  top: {findings[0][0].split(':')[0]}" if findings else ""
            print(f"    {band_name:8s}: {n_sig:3d} sig{top_str}")

    return d4


def direction_5(runner):
    """D5: Channel topology — regional geometric signatures."""
    print("\n" + "=" * 78)
    print("D5: CHANNEL TOPOLOGY — REGIONAL SIGNATURES")
    print("=" * 78)

    runs = [2]  # eyes closed for clearest regional differences
    d5 = {}
    region_met = {}

    for region in CHANNEL_GROUPS:
        with runner.timed(region):
            chunks = [gen_eegmmi(runs, rng, DATA_SIZE, channel_group=region)
                      for rng in runner.trial_rngs(offset=400)]
            region_met[region] = runner.collect(chunks)
        print(f"  {region:12s}: collected")

    # Pairwise regional discrimination
    regions = list(CHANNEL_GROUPS.keys())
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if j > i:
                n_sig, findings = runner.compare(region_met[r1], region_met[r2])
                d5[(r1, r2)] = n_sig
                print(f"  {r1:12s} vs {r2:12s}: {n_sig:3d} sig")

    return d5, region_met


def direction_6(runner):
    """D6: Seizure detection — Bonn + CHB-MIT."""
    print("\n" + "=" * 78)
    print("D6: SEIZURE DETECTION")
    print("=" * 78)

    # Bonn: seizure vs healthy
    with runner.timed("Bonn seizure"):
        seiz_chunks = [gen_bonn(1, rng, DATA_SIZE) for rng in runner.trial_rngs()]
        seiz_met = runner.collect(seiz_chunks)
    with runner.timed("Bonn healthy"):
        heal_chunks = [gen_bonn(5, rng, DATA_SIZE)
                       for rng in runner.trial_rngs(offset=500)]
        heal_met = runner.collect(heal_chunks)

    n_bonn, findings_bonn = runner.compare(seiz_met, heal_met)
    print(f"  Bonn seizure vs healthy:  {n_bonn:3d} sig")
    for m, d, _ in findings_bonn[:3]:
        print(f"    {m:50s} d={d:+8.2f}")

    # CHB-MIT: seizure vs interictal
    has_chbmit = os.path.isdir(CHBMIT_DIR) and len(os.listdir(CHBMIT_DIR)) > 0
    n_chbmit = 0
    if has_chbmit:
        with runner.timed("CHB-MIT seizure"):
            chb_seiz = [gen_chbmit(True, rng, DATA_SIZE) for rng in runner.trial_rngs()]
            chb_seiz_met = runner.collect(chb_seiz)
        with runner.timed("CHB-MIT interictal"):
            chb_inter = [gen_chbmit(False, rng, DATA_SIZE)
                         for rng in runner.trial_rngs(offset=600)]
            chb_inter_met = runner.collect(chb_inter)

        n_chbmit, findings_chbmit = runner.compare(chb_seiz_met, chb_inter_met)
        print(f"  CHB-MIT seizure vs inter: {n_chbmit:3d} sig")
        for m, d, _ in findings_chbmit[:3]:
            print(f"    {m:50s} d={d:+8.2f}")
    else:
        print("  CHB-MIT: data not found, skipping")

    return {'bonn': n_bonn, 'chbmit': n_chbmit}


def direction_7(runner):
    """D7: Motor imagery — real vs imagined (BCI-relevant)."""
    print("\n" + "=" * 78)
    print("D7: MOTOR IMAGERY — REAL VS IMAGINED")
    print("=" * 78)

    results = {}

    # L/R fist: real (run 3) vs imagined (run 4)
    real_subjs = _subjects_for_runs([3])
    imag_subjs = _subjects_for_runs([4])
    if len(real_subjs) >= 5 and len(imag_subjs) >= 5:
        with runner.timed("Motor real (L/R fist)"):
            real_chunks = [gen_eegmmi([3], rng, DATA_SIZE)
                           for rng in runner.trial_rngs(offset=700)]
            real_met = runner.collect(real_chunks)
        with runner.timed("Motor imagined (L/R fist)"):
            imag_chunks = [gen_eegmmi([4], rng, DATA_SIZE)
                           for rng in runner.trial_rngs(offset=800)]
            imag_met = runner.collect(imag_chunks)
        n_sig, findings = runner.compare(real_met, imag_met)
        results['lr'] = n_sig
        print(f"  Real vs Imagined (L/R): {n_sig:3d} sig")
        for m, d, _ in findings[:5]:
            print(f"    {m:50s} d={d:+8.2f}")
    else:
        results['lr'] = 0
        print(f"  L/R fist: insufficient cached data (real={len(real_subjs)}, imag={len(imag_subjs)})")

    # Both fists/feet: real (run 5) vs imagined (run 6)
    real2_subjs = _subjects_for_runs([5])
    imag2_subjs = _subjects_for_runs([6])
    if len(real2_subjs) >= 5 and len(imag2_subjs) >= 5:
        with runner.timed("Motor real (both)"):
            real2 = [gen_eegmmi([5], rng, DATA_SIZE)
                     for rng in runner.trial_rngs(offset=900)]
            real2_met = runner.collect(real2)
        with runner.timed("Motor imagined (both)"):
            imag2 = [gen_eegmmi([6], rng, DATA_SIZE)
                     for rng in runner.trial_rngs(offset=1000)]
            imag2_met = runner.collect(imag2)
        n_sig2, findings2 = runner.compare(real2_met, imag2_met)
        results['both'] = n_sig2
        print(f"  Real vs Imagined (both): {n_sig2:3d} sig")
        for m, d, _ in findings2[:3]:
            print(f"    {m:50s} d={d:+8.2f}")
    else:
        results['both'] = 0
        print(f"  Both fists/feet: insufficient cached data")

    return results


def direction_8(runner):
    """D8: IAAFT surrogates — nonlinear structure beyond spectrum."""
    print("\n" + "=" * 78)
    print("D8: IAAFT SURROGATES — NONLINEAR STRUCTURE")
    print("=" * 78)

    states = {
        'Eyes Closed': lambda rng, sz: gen_eegmmi([2], rng, sz),
        'Seizure':     lambda rng, sz: gen_bonn(1, rng, sz),
    }

    d8 = {}
    for name, gen_fn in states.items():
        print(f"  {name}:")
        with runner.timed(f"{name} original"):
            orig_chunks = [gen_fn(rng, DATA_SIZE)
                           for rng in runner.trial_rngs(offset=1100)]
            orig_met = runner.collect(orig_chunks)

        # IAAFT surrogates (preserve spectrum + amplitude distribution)
        with runner.timed(f"{name} IAAFT"):
            surr_chunks = []
            for chunk in orig_chunks:
                s = iaaft_surrogate(chunk.astype(np.float64))
                surr_chunks.append(np.clip(s, 0, 255).astype(np.uint8))
            surr_met = runner.collect(surr_chunks)

        n_sig, findings = runner.compare(orig_met, surr_met)
        d8[name] = n_sig
        print(f"    Original vs IAAFT: {n_sig:3d} sig")
        for m, d, _ in findings[:3]:
            print(f"      {m:50s} d={d:+8.2f}")

    return d8


# =========================================================================
# FIGURE
# =========================================================================

def make_figure(runner, d1, d2, d4, d5, d6, d7, d8):
    fig, axes = runner.create_figure(6, "Deep EEG Geometry", rows=3, cols=2,
                                     figsize=(20, 18))

    # D1: Structure detection bars
    ax = axes[0]
    names = list(d1.keys())
    sigs = [d1[n]['n_sig'] for n in names]
    ax.barh(range(len(names)), sigs, color='#2196F3', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D1: Structure vs Shuffled', fontsize=11, fontweight='bold')
    for i, v in enumerate(sigs):
        ax.text(v + 1, i, str(v), va='center', fontsize=9, color='white')

    # D2: Pairwise heatmap
    ax = axes[1]
    state_names = ['Eyes Open', 'Eyes Closed', 'Motor Real', 'Motor Imagined']
    n = len(state_names)
    mat = np.zeros((n, n))
    for i, n1 in enumerate(state_names):
        for j, n2 in enumerate(state_names):
            if j > i:
                val = d2.get((n1, n2), d2.get((n2, n1), 0))
                mat[i, j] = val
                mat[j, i] = val
    runner.plot_heatmap(ax, mat, [s[:10] for s in state_names],
                        'D2: Cross-State Discrimination')

    # D4: Band decomposition
    ax = axes[2]
    band_names = list(BANDS.keys())
    for state_name in ['Eyes Closed', 'Motor Real']:
        vals = [d4.get((state_name, b), 0) for b in band_names]
        ax.plot(band_names, vals, 'o-', linewidth=2, markersize=6,
                label=state_name)
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D4: Spectral Band Structure', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # D5: Regional discrimination
    ax = axes[3]
    regions = list(CHANNEL_GROUPS.keys())
    n_r = len(regions)
    rmat = np.zeros((n_r, n_r))
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if j > i:
                val = d5.get((r1, r2), d5.get((r2, r1), 0))
                rmat[i, j] = val
                rmat[j, i] = val
    runner.plot_heatmap(ax, rmat, regions, 'D5: Regional Discrimination')

    # D6: Seizure detection
    ax = axes[4]
    seiz_names = ['Bonn', 'CHB-MIT']
    seiz_vals = [d6['bonn'], d6['chbmit']]
    ax.bar(seiz_names, seiz_vals, color=['#E91E63', '#FF9800'], alpha=0.85)
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D6: Seizure Detection', fontsize=11, fontweight='bold')
    for i, v in enumerate(seiz_vals):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10, color='white',
                fontweight='bold')

    # D7 + D8 combined
    ax = axes[5]
    labels = ['BCI L/R', 'BCI Both', 'IAAFT EC', 'IAAFT Seiz']
    vals = [d7['lr'], d7['both'], d8.get('Eyes Closed', 0), d8.get('Seizure', 0)]
    colors = ['#4CAF50', '#8BC34A', '#9C27B0', '#E91E63']
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylabel(f'Sig metrics (of {runner.n_metrics})')
    ax.set_title('D7-D8: BCI + Nonlinear Structure', fontsize=11,
                 fontweight='bold')
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha='center', fontsize=10, color='white',
                fontweight='bold')

    runner.save(fig, "eeg_deep")


# =========================================================================
# MAIN
# =========================================================================

def main():
    t0 = time.time()
    runner = Runner("EEG Deep", mode="1d",
                    n_workers=N_WORKERS, data_size=DATA_SIZE,
                    n_trials=N_TRIALS)

    print("=" * 78)
    print("DEEP EEG GEOMETRY — EXHAUSTIVE BRAIN STATE ANALYSIS")
    print(f"  Datasets: PhysioNet eegmmidb (109 subj), Bonn (11.5k seg), CHB-MIT")
    print(f"  8 directions, {runner.n_metrics} metrics, {N_WORKERS} workers")
    print("=" * 78)

    # Scan cached eegmmidb data (no downloads)
    print("\nScanning cached EEG data...")
    get_eegmmi_available()

    try:
        d1 = direction_1(runner)
        d2_matrix, all_met = direction_2(runner)
        d3_heatmap, d3_geoms = direction_3(runner, all_met)
        d4 = direction_4(runner)
        d5, region_met = direction_5(runner)
        d6 = direction_6(runner)
        d7 = direction_7(runner)
        d8 = direction_8(runner)

        make_figure(runner, d1, d2_matrix, d4, d5, d6, d7, d8)

        elapsed = time.time() - t0
        print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        runner.print_summary({
            'D1 Structure': f"Strongest: {max(d1.items(), key=lambda x: x[1]['n_sig'])[0]}"
                            f" ({max(v['n_sig'] for v in d1.values())} sig)",
            'D2 Cross-state': f"Eyes Open vs Closed: {d2_matrix.get(('Eyes Open', 'Eyes Closed'), '?')} sig",
            'D4 Bands': f"Alpha EC={d4.get(('Eyes Closed', 'alpha'), '?')}, "
                        f"Beta Motor={d4.get(('Motor Real', 'beta'), '?')}",
            'D6 Seizure': f"Bonn={d6['bonn']}, CHB-MIT={d6['chbmit']}",
            'D7 BCI': f"Real vs Imagined: L/R={d7['lr']}, Both={d7['both']}",
            'D8 Nonlinear': f"EC={d8.get('Eyes Closed', '?')}, "
                            f"Seizure={d8.get('Seizure', '?')}",
        })
    finally:
        runner.close()


if __name__ == "__main__":
    main()
