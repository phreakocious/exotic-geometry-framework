#!/usr/bin/env python3
"""
Investigation: The Structure Atlas
====================================

Map diverse data sources into the framework's 183-metric structure space.
Project into principal components to reveal how the framework organizes
the world — what it considers similar across domain boundaries.

~25 data types from completely different domains, all reduced to the same
geometric fingerprint. Position in structure space = type of structure.

DIRECTIONS:
D1: Collect profiles — 25 trials × ~25 data types → mean+std profiles
D2: Principal structure space — PCA of the profile matrix
D3: The atlas — 2D projection, colored by domain, labeled
D4: Cross-domain neighbors — what's closest to what? Surprises?
D5: Structure types — cluster the atlas, name what we find
"""

import sys
import os
import glob
import hashlib
import json as _json
from collections import Counter
import numpy as np
from scipy.io import loadmat
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.investigation_runner import Runner
from tools.sources import get_sources, register, DOMAIN_COLORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================
# CONFIG
# ==============================================================
DATA_SIZE = 16384
N_TRIALS = 25
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'cwru', 'raw')

# ==============================================================
# FILE-BASED DATA GENERATORS
# (These stay here because they depend on local data files.)
# ==============================================================

# --- Bearing vibration (real-world) ---
def _load_bearing(filename, key):
    path = os.path.join(DATA_DIR, filename)
    return loadmat(path)[key].flatten()

_BEARING_SIGNALS = {}

def _get_bearing(filename, key):
    cache_key = f"{filename}:{key}"
    if cache_key not in _BEARING_SIGNALS:
        _BEARING_SIGNALS[cache_key] = _load_bearing(filename, key)
    return _BEARING_SIGNALS[cache_key]

def gen_bearing_normal(rng, size):
    sig = _get_bearing('Time_Normal_1_098.mat', 'X098_DE_time')
    return _bearing_chunk(sig, rng, size)

def gen_bearing_ball(rng, size):
    sig = _get_bearing('B014_1_190.mat', 'X190_DE_time')
    return _bearing_chunk(sig, rng, size)

def gen_bearing_inner(rng, size):
    sig = _get_bearing('IR014_1_175.mat', 'X175_DE_time')
    return _bearing_chunk(sig, rng, size)

def gen_bearing_outer(rng, size):
    sig = _get_bearing('OR014_6_1_202.mat', 'X202_DE_time')
    return _bearing_chunk(sig, rng, size)

def _bearing_chunk(signal, rng, size):
    n_avail = len(signal) // size
    idx = rng.integers(0, n_avail)
    chunk = signal[idx * size:(idx + 1) * size]
    lo, hi = chunk.min(), chunk.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)


# --- ECG heartbeat (MIT-BIH, Kaggle) ---
_ECG_DATA = None

def _get_ecg():
    global _ECG_DATA
    if _ECG_DATA is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'ecg', 'mitbih_train.csv')
        raw = np.loadtxt(path, delimiter=',')
        signals = raw[:, :-1]  # 187 time points, already [0,1]
        labels = raw[:, -1].astype(int)
        _ECG_DATA = (signals, labels)
    return _ECG_DATA


def _gen_ecg_class(class_id):
    """Return a generator for a specific ECG class."""
    def gen(rng, size):
        signals, labels = _get_ecg()
        mask = labels == class_id
        pool = signals[mask]
        # Concatenate random beats to reach target size
        n_beats = size // 187 + 1
        indices = rng.choice(len(pool), size=n_beats, replace=True)
        concat = pool[indices].flatten()[:size]
        return (concat * 255).astype(np.uint8)
    return gen

gen_ecg_normal = _gen_ecg_class(0)
gen_ecg_svpb = _gen_ecg_class(1)    # Supraventricular premature beat
gen_ecg_vpb = _gen_ecg_class(2)     # Ventricular premature beat
gen_ecg_fusion = _gen_ecg_class(3)  # Fusion beat


# --- EEG seizure (Bonn/Andrzejak, Kaggle) ---
_EEG_DATA = None

def _get_eeg():
    global _EEG_DATA
    if _EEG_DATA is None:
        import csv
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'eeg',
                            'Epileptic Seizure Recognition.csv')
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for r in reader:
                rows.append([float(v) for v in r[1:]])  # skip unnamed col
        arr = np.array(rows)
        signals = arr[:, :-1]  # 178 time points
        labels = arr[:, -1].astype(int)
        _EEG_DATA = (signals, labels)
    return _EEG_DATA


def _gen_eeg_class(class_id):
    """Return a generator for a specific EEG class."""
    def gen(rng, size):
        signals, labels = _get_eeg()
        mask = labels == class_id
        pool = signals[mask]
        n_segs = size // 178 + 1
        indices = rng.choice(len(pool), size=n_segs, replace=True)
        concat = pool[indices].flatten()[:size]
        # Normalize to uint8
        lo, hi = concat.min(), concat.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((concat - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_eeg_seizure = _gen_eeg_class(1)     # Seizure activity
gen_eeg_tumor = _gen_eeg_class(2)       # Tumor area (eyes open)
gen_eeg_healthy = _gen_eeg_class(3)     # Healthy area (eyes open)
gen_eeg_eyes_closed = _gen_eeg_class(4) # Eyes closed


# --- Financial time series (stock indices, Kaggle) ---
_STOCK_DATA = {}

def _get_stock(index_name):
    if index_name not in _STOCK_DATA:
        import csv
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'stock', 'indexProcessed.csv')
        prices = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Index'] == index_name:
                    try:
                        prices.append(float(row['CloseUSD']))
                    except (ValueError, KeyError):
                        pass
        _STOCK_DATA[index_name] = np.array(prices)
    return _STOCK_DATA[index_name]


def _gen_stock(index_name):
    """Generate chunks from stock price returns."""
    def gen(rng, size):
        prices = _get_stock(index_name)
        # Log returns
        returns = np.diff(np.log(prices + 1e-10))
        if len(returns) < size:
            # Pad with noise if not enough data
            returns = np.concatenate([returns, rng.standard_normal(size)])
        start = rng.integers(0, max(1, len(returns) - size))
        chunk = returns[start:start + size]
        # Normalize to uint8
        lo, hi = chunk.min(), chunk.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_stock_sp500 = _gen_stock('NYA')     # NYSE Composite (longest)
gen_stock_nikkei = _gen_stock('N225')   # Nikkei 225
gen_stock_nasdaq = _gen_stock('IXIC')   # NASDAQ


# --- Real DNA sequences ---
_DNA_SEQS = {}

def _get_dna_seqs(species):
    if species not in _DNA_SEQS:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'dna', f'{species}.txt')
        seqs = []
        with open(path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    seqs.append(parts[0].upper())
        _DNA_SEQS[species] = seqs
    return _DNA_SEQS[species]


def _gen_dna_species(species):
    mapping = {ord('A'): 65, ord('T'): 84, ord('G'): 71, ord('C'): 67}
    def gen(rng, size):
        seqs = _get_dna_seqs(species)
        indices = rng.choice(len(seqs), size=size // 100 + 1, replace=True)
        concat = ''.join(seqs[i] for i in indices)[:size]
        vals = np.array([mapping.get(ord(c), 78) for c in concat], dtype=np.uint8)
        if len(vals) < size:
            vals = np.pad(vals, (0, size - len(vals)), constant_values=78)
        return vals[:size]
    return gen

gen_dna_human_legacy = _gen_dna_species('human')
gen_dna_chimp_legacy = _gen_dna_species('chimpanzee')
gen_dna_dog_legacy = _gen_dna_species('dog')


# --- Real DNA from FASTA files ---
_FASTA_CACHE = {}

def _load_fasta(filename):
    """Load a FASTA file, return a single concatenated uppercase sequence."""
    if filename not in _FASTA_CACHE:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'dna', filename)
        bases = []
        with open(path) as f:
            for line in f:
                if line.startswith('>'):
                    continue
                bases.append(line.strip().upper())
        _FASTA_CACHE[filename] = ''.join(bases)
    return _FASTA_CACHE[filename]


def _gen_fasta_source(filename):
    """Generator factory: random contiguous block from a FASTA sequence."""
    mapping = {ord('A'): 65, ord('T'): 84, ord('G'): 71, ord('C'): 67}
    def gen(rng, size):
        seq = _load_fasta(filename)
        if len(seq) < size:
            return rng.integers(0, 256, size, dtype=np.uint8)
        start = int(rng.integers(0, max(1, len(seq) - size)))
        block = seq[start:start + size]
        vals = np.array([mapping.get(ord(c), 78) for c in block], dtype=np.uint8)
        return vals
    return gen


gen_dna_human = _gen_fasta_source('human_chr1.fa')
gen_dna_chimp = _gen_fasta_source('chimp_chr1.fa')
gen_dna_dog = _gen_fasta_source('dog_chr1.fa')
gen_dna_sars = _gen_fasta_source('sars_cov2.fa')
gen_dna_phage = _gen_fasta_source('phage_lambda.fa')
gen_dna_plasmodium = _gen_fasta_source('plasmodium_chr1.fa')
gen_dna_thermus = _gen_fasta_source('thermus.fa')
gen_dna_centromere = _gen_fasta_source('alpha_satellite.fa')


# --- MotionSense accelerometer ---
_MOTION_DATA = {}

def _get_motion(activity_prefix):
    if activity_prefix not in _MOTION_DATA:
        import csv
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'motionsense',
                            'A_DeviceMotion_data', 'A_DeviceMotion_data')
        all_vals = []
        for d in sorted(os.listdir(base)):
            if d.startswith(activity_prefix):
                folder = os.path.join(base, d)
                for fname in sorted(os.listdir(folder)):
                    if fname.endswith('.csv'):
                        with open(os.path.join(folder, fname)) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    # userAcceleration.x is the main signal
                                    all_vals.append(float(row['userAcceleration.x']))
                                except (ValueError, KeyError):
                                    pass
        _MOTION_DATA[activity_prefix] = np.array(all_vals)
    return _MOTION_DATA[activity_prefix]


def _gen_motion(activity_prefix):
    def gen(rng, size):
        signal = _get_motion(activity_prefix)
        if len(signal) < size:
            return rng.integers(0, 256, size, dtype=np.uint8)
        start = rng.integers(0, max(1, len(signal) - size))
        chunk = signal[start:start + size]
        lo, hi = chunk.min(), chunk.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_motion_walk = _gen_motion('wlk')
gen_motion_jog = _gen_motion('jog')
gen_motion_sit = _gen_motion('sit')
gen_motion_stairs = _gen_motion('ups')


# --- Gravitational waves (LIGO GW150914) ---
_GW_DATA = {}

def _get_gw(detector):
    if detector not in _GW_DATA:
        import h5py
        fname = f'{detector}-{detector}1_LOSC_4_V1-1126259446-32.hdf5'
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'gw', fname)
        with h5py.File(path, 'r') as f:
            _GW_DATA[detector] = f['strain/Strain'][:].copy()
    return _GW_DATA[detector]


def _gen_gw(detector):
    def gen(rng, size):
        strain = _get_gw(detector)
        if len(strain) < size:
            return rng.integers(0, 256, size, dtype=np.uint8)
        start = rng.integers(0, max(1, len(strain) - size))
        chunk = strain[start:start + size]
        lo, hi = chunk.min(), chunk.max()
        if hi - lo < 1e-30:
            hi = lo + 1.0
        return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_gw_hanford = _gen_gw('H')
gen_gw_livingston = _gen_gw('L')


# --- Jena Climate sensor data ---
_CLIMATE_DATA = {}

def _get_climate(column):
    if column not in _CLIMATE_DATA:
        import csv
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'climate',
                            'jena_climate_2009_2016.csv')
        vals = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    vals.append(float(row[column]))
                except (ValueError, KeyError):
                    pass
        _CLIMATE_DATA[column] = np.array(vals)
    return _CLIMATE_DATA[column]


def _gen_climate(column):
    def gen(rng, size):
        signal = _get_climate(column)
        if len(signal) < size:
            return rng.integers(0, 256, size, dtype=np.uint8)
        start = rng.integers(0, max(1, len(signal) - size))
        chunk = signal[start:start + size]
        lo, hi = chunk.min(), chunk.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_climate_temp = _gen_climate('T (degC)')
gen_climate_pressure = _gen_climate('p (mbar)')
gen_climate_humidity = _gen_climate('rh (%)')
gen_climate_wind = _gen_climate('wv (m/s)')


# --- Free Spoken Digit Dataset (WAV audio) ---
_FSDD_FILES = {}

def _get_fsdd_files(digit):
    if digit not in _FSDD_FILES:
        rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', 'data', 'fsdd', 'recordings')
        files = [os.path.join(rec_dir, f)
                 for f in sorted(os.listdir(rec_dir))
                 if f.startswith(f'{digit}_') and f.endswith('.wav')]
        _FSDD_FILES[digit] = files
    return _FSDD_FILES[digit]


def _gen_speech_digit(digit):
    def gen(rng, size):
        import wave, struct
        files = _get_fsdd_files(digit)
        # Concatenate random recordings to reach target size
        samples = []
        while len(samples) < size:
            f = files[rng.integers(0, len(files))]
            with wave.open(f, 'rb') as w:
                raw = w.readframes(w.getnframes())
                vals = np.array(struct.unpack(f'<{w.getnframes()}h', raw),
                                dtype=np.float64)
                samples.extend(vals.tolist())
        chunk = np.array(samples[:size])
        lo, hi = chunk.min(), chunk.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_speech_zero = _gen_speech_digit(0)
gen_speech_five = _gen_speech_digit(5)
gen_speech_nine = _gen_speech_digit(9)


# --- Kepler exoplanet light curves ---
_KEPLER_DATA = None

def _get_kepler():
    global _KEPLER_DATA
    if _KEPLER_DATA is None:
        raw = np.loadtxt(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'data', 'kepler', 'exoTrain.csv'),
            delimiter=',', skiprows=1)
        labels = raw[:, 0].astype(int)  # 2=exoplanet, 1=non
        flux = raw[:, 1:]  # 3197 flux measurements per star
        _KEPLER_DATA = (flux, labels)
    return _KEPLER_DATA


def _gen_kepler(label):
    def gen(rng, size):
        flux, labels = _get_kepler()
        mask = labels == label
        pool = flux[mask]
        # Concatenate random stars, normalizing each star independently
        # to remove batch effects from different baseline flux levels
        n_stars = size // 3197 + 1
        indices = rng.choice(len(pool), size=n_stars, replace=True)
        pieces = []
        for idx in indices:
            star = pool[idx]
            mu, sigma = star.mean(), star.std()
            if sigma < 1e-15:
                sigma = 1.0
            pieces.append((star - mu) / sigma)
        concat = np.concatenate(pieces)[:size]
        lo, hi = np.percentile(concat, [0.5, 99.5])
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return np.clip((concat - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return gen

gen_kepler_exoplanet = _gen_kepler(2)  # confirmed exoplanet
gen_kepler_nonplanet = _gen_kepler(1)  # no exoplanet


# --- Tidal gauge (NOAA CO-OPS, San Francisco) ---
_TIDAL_DATA = None

def _get_tidal():
    global _TIDAL_DATA
    if _TIDAL_DATA is None:
        import csv
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'tidal', 'sf_2023.csv')
        vals = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                try:
                    vals.append(float(row[1]))  # Water Level column
                except (ValueError, IndexError):
                    pass
        _TIDAL_DATA = np.array(vals)
    return _TIDAL_DATA


def gen_tidal(rng, size):
    signal = _get_tidal()
    if len(signal) < size:
        return rng.integers(0, 256, size, dtype=np.uint8)
    start = rng.integers(0, max(1, len(signal) - size))
    chunk = signal[start:start + size]
    lo, hi = chunk.min(), chunk.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((chunk - lo) / (hi - lo) * 255).astype(np.uint8)


# --- Sunspot numbers (SIDC daily) ---
_SUNSPOT_DATA = None

def _get_sunspot():
    global _SUNSPOT_DATA
    if _SUNSPOT_DATA is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'sunspot', 'SN_d_tot_V2.0.csv')
        vals = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 5:
                    try:
                        v = float(parts[4])
                        if v >= 0:  # -1 = missing
                            vals.append(v)
                    except ValueError:
                        pass
        _SUNSPOT_DATA = np.array(vals)
    return _SUNSPOT_DATA


def gen_sunspot(rng, size):
    signal = _get_sunspot()
    if len(signal) < size:
        return rng.integers(0, 256, size, dtype=np.uint8)
    start = rng.integers(0, max(1, len(signal) - size))
    chunk = signal[start:start + size]
    # Sunspot numbers are zero-heavy with occasional spikes to ~400
    lo, hi = np.percentile(chunk, [0, 99])
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return np.clip((chunk - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


# --- Solar wind (NASA OMNI hourly) ---
_SOLARWIND_DATA = {}

def _get_solarwind(param='V'):
    """Load solar wind parameter. V=bulk speed (col 24), B=IMF magnitude (col 8).
    OMNI2 docs use 1-based indexing; these are 0-based.
    Filters to 1995+ for reliable coverage and interpolates short gaps."""
    if param not in _SOLARWIND_DATA:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'solarwind',
                            'omni2_all_years.dat')
        col_map = {'V': 24, 'B': 8}
        fill_map = {'V': 9999., 'B': 999.9}
        col = col_map[param]
        fill = fill_map[param]
        raw = []  # (year, value_or_nan)
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) > col:
                    try:
                        yr = int(parts[0])
                        if yr < 1995:
                            continue
                        v = float(parts[col])
                        raw.append(v if v < fill * 0.9 else np.nan)
                    except ValueError:
                        pass
        arr = np.array(raw)
        # Interpolate gaps up to 24 hours (consecutive NaNs)
        nans = np.isnan(arr)
        if nans.any():
            good = np.where(~nans)[0]
            if len(good) > 100:
                arr[nans] = np.interp(np.where(nans)[0], good, arr[good])
        _SOLARWIND_DATA[param] = arr[~np.isnan(arr)]
    return _SOLARWIND_DATA[param]


def _gen_solarwind(param):
    def gen(rng, size):
        signal = _get_solarwind(param)
        if len(signal) < size:
            return rng.integers(0, 256, size, dtype=np.uint8)
        start = rng.integers(0, max(1, len(signal) - size))
        chunk = signal[start:start + size]
        # Global normalization — use full-dataset percentiles so quiet
        # and active periods both have good dynamic range
        lo, hi = np.percentile(signal, [0.5, 99.5])
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return np.clip((chunk - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return gen

gen_solarwind_speed = _gen_solarwind('V')
gen_solarwind_imf = _gen_solarwind('B')


# ==============================================================
# REGISTER FILE-BASED SOURCES INTO THE GLOBAL REGISTRY
# ==============================================================

register("Bearing Normal", gen_bearing_normal, "bearing",
         "CWRU normal bearing vibration", signature=False)
register("Bearing Ball", gen_bearing_ball, "bearing",
         "CWRU ball fault vibration", signature=False)
register("Bearing Inner", gen_bearing_inner, "bearing",
         "CWRU inner race fault vibration", signature=False)
register("Bearing Outer", gen_bearing_outer, "bearing",
         "CWRU outer race fault vibration", signature=False)

register("ECG Normal", gen_ecg_normal, "medical",
         "MIT-BIH normal heartbeat", signature=False)
register("ECG Supraventr.", gen_ecg_svpb, "medical",
         "MIT-BIH supraventricular premature beat", signature=False)
register("ECG Ventricular", gen_ecg_vpb, "medical",
         "MIT-BIH ventricular premature beat", signature=False)
register("ECG Fusion", gen_ecg_fusion, "medical",
         "MIT-BIH fusion beat", signature=False)

register("EEG Seizure", gen_eeg_seizure, "medical",
         "Bonn EEG seizure activity", signature=False)
register("EEG Tumor", gen_eeg_tumor, "medical",
         "Bonn EEG tumor area", signature=False)
register("EEG Healthy", gen_eeg_healthy, "medical",
         "Bonn EEG healthy area", signature=False)
register("EEG Eyes Closed", gen_eeg_eyes_closed, "medical",
         "Bonn EEG eyes closed", signature=False)

register("NYSE Returns", gen_stock_sp500, "financial",
         "NYSE Composite log returns", signature=False)
register("Nikkei Returns", gen_stock_nikkei, "financial",
         "Nikkei 225 log returns", signature=False)
register("NASDAQ Returns", gen_stock_nasdaq, "financial",
         "NASDAQ Composite log returns", signature=False)

register("DNA Human", gen_dna_human, "bio",
         "Human chr1 genomic DNA (GRCh38) — 500kb contiguous block, GC ~41%",
         signature=False)
register("DNA Chimp", gen_dna_chimp, "bio",
         "Chimpanzee chr1 genomic DNA (Clint_PTRv2) — 500kb contiguous, GC ~41%",
         signature=False)
register("DNA Dog", gen_dna_dog, "bio",
         "Dog chr1 genomic DNA (ROS_Cfam_1.0) — 500kb contiguous, GC ~41%",
         signature=False)

register("DNA SARS-CoV-2", gen_dna_sars, "bio",
         "SARS-CoV-2 complete genome (NC_045512) — RNA virus, 30kb, GC 38%",
         signature=False)
register("DNA Phage Lambda", gen_dna_phage, "bio",
         "Bacteriophage lambda genome (NC_001416) — balanced GC 50%, 48kb",
         signature=False)
register("DNA Plasmodium", gen_dna_plasmodium, "bio",
         "P. falciparum chr1 (NC_004325) — extreme AT bias, GC 21%, 640kb",
         signature=False)
register("DNA Thermus", gen_dna_thermus, "bio",
         "Thermus thermophilus (NC_006461) — extreme GC 69%, thermophile",
         signature=False)
register("DNA Centromere", gen_dna_centromere, "bio",
         "Human alpha satellite (AC017075) — tandem 171bp repeats, centromeric",
         signature=False)

register("Accel Walk", gen_motion_walk, "motion",
         "MotionSense walking accelerometer", signature=False)
register("Accel Jog", gen_motion_jog, "motion",
         "MotionSense jogging accelerometer", signature=False)
register("Accel Sit", gen_motion_sit, "motion",
         "MotionSense sitting accelerometer", signature=False)
register("Accel Stairs", gen_motion_stairs, "motion",
         "MotionSense stairs accelerometer", signature=False)

register("LIGO Hanford", gen_gw_hanford, "astro",
         "GW150914 Hanford strain", signature=False)
register("LIGO Livingston", gen_gw_livingston, "astro",
         "GW150914 Livingston strain", signature=False)

register("Kepler Exoplanet", gen_kepler_exoplanet, "astro",
         "Kepler confirmed exoplanet light curve", signature=False)
register("Kepler Non-planet", gen_kepler_nonplanet, "astro",
         "Kepler non-planet light curve", signature=False)

register("Temperature", gen_climate_temp, "climate",
         "Jena temperature time series", signature=False)
register("Pressure", gen_climate_pressure, "climate",
         "Jena atmospheric pressure", signature=False)
register("Humidity", gen_climate_humidity, "climate",
         "Jena relative humidity", signature=False)
register("Wind Speed", gen_climate_wind, "climate",
         "Jena wind speed", signature=False)

register("Speech \"Zero\"", gen_speech_zero, "speech",
         "FSDD spoken digit 0", signature=False)
register("Speech \"Five\"", gen_speech_five, "speech",
         "FSDD spoken digit 5", signature=False)
register("Speech \"Nine\"", gen_speech_nine, "speech",
         "FSDD spoken digit 9", signature=False)

register("Tidal Gauge (SF)", gen_tidal, "geophysics",
         "NOAA CO-OPS 6-min water level, San Francisco", signature=False)
register("Sunspot Number", gen_sunspot, "astro",
         "SIDC daily international sunspot number", signature=False)
register("Solar Wind Speed", gen_solarwind_speed, "astro",
         "OMNI hourly solar wind bulk velocity", signature=False)
register("Solar Wind IMF", gen_solarwind_imf, "astro",
         "OMNI hourly interplanetary magnetic field magnitude", signature=False)

# ==============================================================
# ALL DATA SOURCES — built from the registry
# ==============================================================

_atlas_sources = get_sources(atlas=True)
SOURCES = [(s.name, s.gen_fn, s.domain) for s in _atlas_sources]
_SOURCE_DESCRIPTIONS = {s.name: (s.description or "") for s in _atlas_sources}

# ==============================================================
# COLLECTION
# ==============================================================

def _cache_key(metric_names, data_size, n_trials):
    """Hash of configuration and geometry code for cache invalidation.

    Hashes the framework (geometry implementations) and the metric/size
    config.  Does NOT hash the atlas script or sources.py — edits to
    collection logic, figures, or unrelated generators should not bust
    every cached profile.  Per-source files are keyed by source name,
    so a renamed/new source simply won't have a cache hit.
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..')
    h = hashlib.sha256()
    for rel in ['exotic_geometry_framework.py']:
        try:
            with open(os.path.join(base, rel), 'rb') as f:
                h.update(f.read())
        except OSError:
            h.update(rel.encode())
    code_hash = h.hexdigest()[:16]
    content = _json.dumps({
        "metrics": sorted(metric_names),
        "size": data_size,
        "trials": n_trials,
        "code": code_hash,
    })
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def collect_profiles(runner):
    """Collect mean metric profiles for all sources, with per-source caching."""
    print("\n" + "=" * 60)
    print("COLLECTING STRUCTURE PROFILES")
    print("=" * 60)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'figures', '.atlas_cache', 'profiles')
    os.makedirs(cache_dir, exist_ok=True)

    key = _cache_key(runner.metric_names, runner.data_size, runner.n_trials)
    meta_path = os.path.join(cache_dir, 'meta.json')

    cache_valid = False
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)
        cache_valid = meta.get('key') == key

    if not cache_valid:
        # Write meta now so Ctrl+C mid-run doesn't orphan .npz files
        with open(meta_path, 'w') as f:
            _json.dump({'key': key}, f)
        # Purge stale .npz files from previous key
        for old in glob.glob(os.path.join(cache_dir, '*.npz')):
            os.remove(old)
        cache_valid = True

    profiles = {}  # name -> {metric: mean_value}
    profiles_std = {}
    domains = {}
    descriptions = {}
    n_cached = 0

    for name, gen_fn, domain in SOURCES:
        safe_name = name.replace(' ', '_').replace('"', '').replace("'", '').replace('/', '_')
        cache_path = os.path.join(cache_dir, f'{safe_name}.npz')
        descriptions[name] = _SOURCE_DESCRIPTIONS.get(name, "")

        if cache_valid and os.path.exists(cache_path):
            cached = np.load(cache_path)
            mean_profile = dict(zip(runner.metric_names, cached['means']))
            std_profile = dict(zip(runner.metric_names, cached['stds']))
            profiles[name] = mean_profile
            profiles_std[name] = std_profile
            domains[name] = domain
            n_nonzero = sum(1 for v in mean_profile.values() if abs(v) > 1e-15)
            print(f"  {name:20s} [{domain:13s}]  {n_nonzero}/{len(runner.metric_names)} active metrics [cached]")
            n_cached += 1
            continue

        rngs = runner.trial_rngs()
        try:
            chunks = [gen_fn(rng, runner.data_size) for rng in rngs]
        except Exception as exc:
            print(f"  {name:20s} [{domain:13s}]  SKIPPED: {exc}")
            continue
        metrics = runner.collect(chunks)

        # Compute mean profile
        mean_profile = {}
        std_profile = {}
        for m in runner.metric_names:
            vals = metrics.get(m, [])
            if len(vals) > 0:
                mean_profile[m] = np.mean(vals)
                std_profile[m] = np.std(vals)
            else:
                mean_profile[m] = 0.0
                std_profile[m] = 0.0

        # Save to cache — include raw trial values so D6 can reuse them
        means_arr = np.array([mean_profile[m] for m in runner.metric_names])
        stds_arr = np.array([std_profile[m] for m in runner.metric_names])
        # Raw trial matrix: n_metrics × n_trials (ragged → pad with nan)
        max_trials = max((len(metrics.get(m, [])) for m in runner.metric_names), default=0)
        raw_arr = np.full((len(runner.metric_names), max_trials), np.nan)
        for i, m in enumerate(runner.metric_names):
            vals = metrics.get(m, [])
            raw_arr[i, :len(vals)] = vals
        np.savez(cache_path, means=means_arr, stds=stds_arr, raw=raw_arr)

        profiles[name] = mean_profile
        profiles_std[name] = std_profile
        domains[name] = domain
        n_nonzero = sum(1 for v in mean_profile.values() if abs(v) > 1e-15)
        print(f"  {name:20s} [{domain:13s}]  {n_nonzero}/{len(runner.metric_names)} active metrics")

    if n_cached > 0:
        print(f"  ({n_cached}/{len(SOURCES)} loaded from cache)")

    return profiles, profiles_std, domains, descriptions


# ==============================================================
# ANALYSIS
# ==============================================================

def build_matrix(profiles, metric_names):
    """Build (n_sources × n_metrics) matrix from profiles."""
    names = list(profiles.keys())
    X = np.array([[profiles[n].get(m, 0.0) for m in metric_names] for n in names])
    return names, X


def principal_projection(X):
    """Rank-normalize, SVD, project to principal components.

    Rank normalization (equivalent to PCA on the Spearman correlation matrix)
    is the right choice when metrics have incommensurate units and heavy-tailed
    distributions.  It guarantees each metric gets equal voice regardless of
    tail behavior, and is consistent with the rank-based distances already used
    for neighbor-finding and clustering.
    """
    Z = _rank_normalize(X)
    # Center (ranks already have equal scale, no need to re-standardize)
    Z = Z - Z.mean(axis=0)

    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    variance_explained = s**2 / np.sum(s**2)
    cumvar = np.cumsum(variance_explained)

    # Project to first k components
    coords = U * s  # (n_sources, n_components)
    return coords, variance_explained, cumvar, Z, Vt


def _rank_normalize(X):
    """Convert each column to percentile ranks.  Robust to extreme values."""
    from scipy.stats import rankdata  # noqa: F811
    R = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = np.nan_to_num(X[:, j], nan=0.0)
        if np.std(col) < 1e-15:
            R[:, j] = 0.5
        else:
            R[:, j] = rankdata(col) / len(col)
    return R


def find_neighbors(names, Z, k=3):
    """Find k nearest neighbors using rank-normalized euclidean distance."""
    R = _rank_normalize(Z)
    dist = squareform(pdist(R, metric='euclidean'))
    neighbors = {}
    for i, name in enumerate(names):
        dists = dist[i].copy()
        dists[i] = np.inf
        nearest = np.argsort(dists)[:k]
        neighbors[name] = [(names[j], dists[j]) for j in nearest]
    return neighbors, dist


def cluster_atlas(Z, names, n_clusters=8):
    """Hierarchical clustering using rank-normalized Ward linkage."""
    R = _rank_normalize(Z)
    condensed = pdist(R, metric='euclidean')
    link = linkage(condensed, method='ward')
    labels = fcluster(link, n_clusters, criterion='maxclust')
    order = leaves_list(link)

    clusters = {}
    for i, name in enumerate(names):
        c = int(labels[i])
        clusters.setdefault(c, []).append(name)

    return clusters, labels, order, link


# ==============================================================
# DIRECTIONS
# ==============================================================

def direction_1(runner):
    """D1: Collect all profiles."""
    profiles, profiles_std, domains, descriptions = collect_profiles(runner)
    return profiles, profiles_std, domains, descriptions


def direction_2(profiles, metric_names):
    """D2: Principal structure space."""
    print("\n" + "=" * 60)
    print("D2: PRINCIPAL STRUCTURE SPACE")
    print("=" * 60)

    names, X = build_matrix(profiles, metric_names)
    coords, varexp, cumvar, Z, Vt = principal_projection(X)

    print(f"  PC1: {varexp[0]*100:.1f}% variance")
    print(f"  PC2: {varexp[1]*100:.1f}% variance")
    print(f"  PC1+2: {cumvar[1]*100:.1f}% variance")
    print(f"  PC1-5: {cumvar[4]*100:.1f}% variance")

    # Participation ratio
    pr = np.sum(varexp**1)**2 / np.sum(varexp**2)
    print(f"  Effective dimensionality: {pr:.1f}")

    return names, X, coords, varexp, cumvar, Z


def direction_3(names, domains):
    """D3: Just labeling for the figure — handled in make_figure."""
    pass


def direction_4(names, Z, domains):
    """D4: Cross-domain neighbors."""
    print("\n" + "=" * 60)
    print("D4: CROSS-DOMAIN NEIGHBORS")
    print("=" * 60)

    neighbors, dist = find_neighbors(names, Z, k=3)

    for name in names:
        nn = neighbors[name]
        domain = domains[name]
        nn_str = ', '.join(f"{n} ({d:.3f})" for n, d in nn)
        cross = sum(1 for n, _ in nn if domains[n] != domain)
        marker = " ***" if cross > 0 else ""
        print(f"  {name:20s} [{domain:13s}] -> {nn_str}{marker}")

    return neighbors, dist


def direction_5(Z, names, domains):
    """D5: Cluster the atlas."""
    print("\n" + "=" * 60)
    print("D5: STRUCTURE TYPES (CLUSTERS)")
    print("=" * 60)

    clusters, labels, order, link = cluster_atlas(Z, names, n_clusters=8)

    for c_id in sorted(clusters.keys()):
        members = clusters[c_id]
        member_domains = set(domains[m] for m in members)
        print(f"\n  Cluster {c_id} ({len(members)} members, "
              f"domains: {', '.join(sorted(member_domains))}):")
        for m in members:
            print(f"    {m:20s} [{domains[m]}]")

    return clusters, labels, order, link


def direction_6(runner, domains):
    """D6: Surrogate Decomposition — what kind of structure does each source have?

    For each source, compare Real vs 3 surrogates:
    - Distribution-matched (shuffled): same histogram, no sequence
    - Rolled (cyclic shift): preserves local correlations, breaks global position
    - Reversed: same values + local stats, breaks time arrow

    The distance from Real to each surrogate reveals the structure type.
    """
    print("\n" + "=" * 60)
    print("D6: SURROGATE DECOMPOSITION")
    print("=" * 60)

    # Per-source caching
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'figures', '.atlas_cache', 'surrogates')
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(runner.metric_names, runner.data_size, runner.n_trials)
    meta_path = os.path.join(cache_dir, 'meta.json')

    cache_valid = False
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)
        cache_valid = meta.get('key') == key

    if not cache_valid:
        with open(meta_path, 'w') as f:
            _json.dump({'key': key}, f)
        for old in glob.glob(os.path.join(cache_dir, '*.json')):
            if os.path.basename(old) != 'meta.json':
                os.remove(old)
        cache_valid = True

    results = {}
    n_cached = 0

    def _count_sig(metrics_a, metrics_b):
        bonf = 0.05 / len(runner.metric_names)
        n_sig = 0
        total_d = 0.0
        for m in runner.metric_names:
            a = np.array(metrics_a.get(m, []))
            b = np.array(metrics_b.get(m, []))
            if len(a) < 3 or len(b) < 3:
                continue
            sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
            ps = np.sqrt((sa**2 + sb**2) / 2)
            if ps < 1e-15:
                continue
            d = abs((np.mean(a) - np.mean(b)) / ps)
            if not np.isfinite(d):
                continue
            _, p = sp_stats.ttest_ind(a, b, equal_var=False)
            if p < bonf and d > 0.8:
                n_sig += 1
                total_d += d
        return n_sig, total_d

    # D1 profile cache — reuse raw trial values for real data
    d1_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'figures', '.atlas_cache', 'profiles')

    for name, gen_fn, domain in SOURCES:
        safe_name = name.replace(' ', '_').replace('"', '').replace("'", '').replace('/', '_')
        cache_path = os.path.join(cache_dir, f'{safe_name}.json')

        # Check cache
        if cache_valid and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = _json.load(f)
            results[name] = cached
            print(f"  {name:20s}  shuf={cached['shuffled_sig']:3d}  "
                  f"roll={cached['rolled_sig']:3d}  rev={cached['reversed_sig']:3d}  "
                  f"seq={cached['seq_fraction']:.2f} [cached]")
            n_cached += 1
            continue

        rngs = runner.trial_rngs()

        # Try to reuse raw trial values from D1 profile cache
        d1_path = os.path.join(d1_cache_dir, f'{safe_name}.npz')
        real_metrics = None
        real_chunks = None
        if os.path.exists(d1_path):
            try:
                d1_data = np.load(d1_path)
                if 'raw' in d1_data:
                    raw = d1_data['raw']  # n_metrics × n_trials
                    real_metrics = {}
                    for i, m in enumerate(runner.metric_names):
                        vals = raw[i][np.isfinite(raw[i])]
                        if len(vals) > 0:
                            real_metrics[m] = list(vals)
                        else:
                            real_metrics[m] = []
            except Exception:
                real_metrics = None

        # Generate real chunks (needed for surrogates, and for analysis if no D1 cache)
        try:
            real_chunks = [gen_fn(rng, runner.data_size) for rng in rngs]
        except Exception as exc:
            print(f"  {name:20s}  SKIPPED: {exc}")
            continue

        # Build all surrogate chunks, then analyze in one batched collect()
        surr_rngs = runner.trial_rngs(offset=1000)
        shuf_chunks = []
        for chunk, srng in zip(real_chunks, surr_rngs):
            s = chunk.copy()
            srng.shuffle(s)
            shuf_chunks.append(s)

        roll_chunks = []
        for chunk, srng in zip(real_chunks, surr_rngs):
            shift = int(srng.integers(len(chunk) // 4, 3 * len(chunk) // 4))
            roll_chunks.append(np.roll(chunk, shift))

        rev_chunks = [chunk[::-1].copy() for chunk in real_chunks]

        # Batch: real (if needed) + shuf + roll + rev in one parallel call
        n_t = runner.n_trials
        need_real = real_metrics is None
        batch = []
        if need_real:
            batch.extend(real_chunks)
        batch.extend(shuf_chunks)
        batch.extend(roll_chunks)
        batch.extend(rev_chunks)

        batch_metrics = runner.collect(batch)

        # Split results back out
        def _split(combined, start, count, metric_names):
            """Extract trial slice [start:start+count] from combined metrics."""
            out = {}
            for m in metric_names:
                vals = combined.get(m, [])
                out[m] = vals[start:start + count]
            return out

        offset = 0
        if need_real:
            real_metrics = _split(batch_metrics, offset, n_t, runner.metric_names)
            offset += n_t
        shuf_metrics = _split(batch_metrics, offset, n_t, runner.metric_names)
        offset += n_t
        roll_metrics = _split(batch_metrics, offset, n_t, runner.metric_names)
        offset += n_t
        rev_metrics = _split(batch_metrics, offset, n_t, runner.metric_names)

        n_shuf, d_shuf = _count_sig(real_metrics, shuf_metrics)
        n_roll, d_roll = _count_sig(real_metrics, roll_metrics)
        n_rev, d_rev = _count_sig(real_metrics, rev_metrics)

        seq_frac = n_shuf / len(runner.metric_names)

        entry = {
            'shuffled_sig': n_shuf,
            'rolled_sig': n_roll,
            'reversed_sig': n_rev,
            'shuffled_d': d_shuf,
            'rolled_d': d_roll,
            'reversed_d': d_rev,
            'seq_fraction': seq_frac,
        }
        results[name] = entry

        # Save to cache
        with open(cache_path, 'w') as f:
            _json.dump(entry, f)

        print(f"  {name:20s}  shuf={n_shuf:3d}  roll={n_roll:3d}  "
              f"rev={n_rev:3d}  seq={seq_frac:.2f}")

    if n_cached:
        print(f"\n  ({n_cached} sources loaded from cache)")

    # Summary
    print(f"\n  Most sequential (disrupted by shuffling):")
    ranked = sorted(results.items(), key=lambda x: -x[1]['seq_fraction'])
    for name, r in ranked[:10]:
        print(f"    {name:20s}  {r['shuffled_sig']:3d} metrics disrupted "
              f"({r['seq_fraction']:.0%})")

    print(f"\n  Most distributional (shuffling changes nothing):")
    for name, r in ranked[-5:]:
        print(f"    {name:20s}  {r['shuffled_sig']:3d} metrics disrupted "
              f"({r['seq_fraction']:.0%})")

    print(f"\n  Most time-asymmetric (reversing changes structure):")
    ranked_rev = sorted(results.items(), key=lambda x: -x[1]['reversed_sig'])
    for name, r in ranked_rev[:5]:
        print(f"    {name:20s}  {r['reversed_sig']:3d} metrics disrupted")

    return results


def direction_7(runner, domains):
    """D7: Multi-Scale Filtration — how does structure change with observation scale?

    For a representative subset, sweep DATA_SIZE from 256 to 8192 and count
    how many metrics remain significant vs white noise at each scale.
    """
    print("\n" + "=" * 60)
    print("D7: MULTI-SCALE FILTRATION")
    print("=" * 60)

    from scipy import stats as sp_stats

    SCALES = [256, 512, 1024, 2048, 4096, 8192]

    # Representative subset — one per major structure type + key exemplars
    source_map = {name: (gen_fn, domain) for name, gen_fn, domain in SOURCES}
    SUBSET_NAMES = [
        'Logistic Chaos', 'Lorenz Attractor', 'Prime Gaps', 'White Noise',
        'Pink Noise', 'Brownian Walk', 'Bearing Ball', 'ECG Normal',
        'EEG Seizure', 'LIGO Hanford', 'Temperature', 'DNA Human',
        'Accel Walk', 'Speech "Five"', 'NYSE Returns',
    ]
    SUBSET = [(n, source_map[n][0], source_map[n][1])
              for n in SUBSET_NAMES if n in source_map]

    from exotic_geometry_framework import GeometryAnalyzer
    results = {}  # name -> {scale: n_sig}

    for name, gen_fn, domain in SUBSET:
        results[name] = {}
        for scale in SCALES:
            # Build a temporary analyzer for this scale
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', '.cache')
            analyzer = GeometryAnalyzer(cache_dir=cache_dir).add_tier_geometries(tier=runner.tier)
            rngs = [np.random.default_rng(42 + i) for i in range(15)]  # fewer trials for speed

            # Generate real and noise chunks at this scale
            try:
                real_chunks = [gen_fn(rng, scale) for rng in rngs]
            except Exception:
                results[name][scale] = 0
                continue
            noise_rngs = [np.random.default_rng(99000 + i) for i in range(15)]
            noise_chunks = [rng.integers(0, 256, scale, dtype=np.uint8) for rng in noise_rngs]

            # Collect metrics
            real_m = {}
            for chunk in real_chunks:
                r = analyzer.analyze(chunk)
                for res in r.results:
                    for mn, mv in res.metrics.items():
                        key = f"{res.geometry_name}:{mn}"
                        real_m.setdefault(key, []).append(mv if np.isfinite(mv) else 0.0)

            noise_m = {}
            for chunk in noise_chunks:
                r = analyzer.analyze(chunk)
                for res in r.results:
                    for mn, mv in res.metrics.items():
                        key = f"{res.geometry_name}:{mn}"
                        noise_m.setdefault(key, []).append(mv if np.isfinite(mv) else 0.0)

            # Count significant metrics
            all_keys = sorted(set(real_m.keys()) & set(noise_m.keys()))
            bonf = 0.05 / max(len(all_keys), 1)
            n_sig = 0
            for k in all_keys:
                a = np.array(real_m[k])
                b = np.array(noise_m[k])
                if len(a) < 3 or len(b) < 3:
                    continue
                ps = np.sqrt(((len(a)-1)*np.std(a,ddof=1)**2 +
                              (len(b)-1)*np.std(b,ddof=1)**2) / (len(a)+len(b)-2))
                if ps < 1e-15:
                    continue
                d = abs((np.mean(a) - np.mean(b)) / ps)
                _, p = sp_stats.ttest_ind(a, b, equal_var=False)
                if p < bonf and d > 0.8:
                    n_sig += 1

            results[name][scale] = n_sig

        scale_str = '  '.join(f'{results[name].get(s, 0):3d}' for s in SCALES)
        print(f"  {name:20s}  {scale_str}")

    print(f"\n  Scale:                {'  '.join(f'{s:3d}' for s in SCALES)}")

    return results, SCALES


# ==============================================================
# GEOMETRIC VIEWS — different lenses on the same data
# ==============================================================

def direction_8_views(names, X, metric_names, domains, view_lenses):
    """D8: Geometric Views — project data through each family of geometries.

    view_lenses is discovered from geometry metadata via GeometryAnalyzer.view_lenses().
    """
    print("\n" + "=" * 60)
    print("D8: GEOMETRIC VIEWS")
    print("=" * 60)

    # Map metric indices to their geometry prefix
    geo_to_cols = {}
    for j, mn in enumerate(metric_names):
        geo = mn.split(':')[0]
        geo_to_cols.setdefault(geo, []).append(j)

    # Degeneracy discount: sources with very low byte complexity produce
    # degenerate geometry embeddings.  Their extreme metric scores are
    # artifacts of having few distinct values, not genuine structure.
    # Compute complexity percentile from distributional metrics and apply
    # a gentle discount (floor at 0.3x) to non-distributional views.
    from scipy.stats import rankdata as _drd
    _complexity_names = {'Information Theory:block_entropy_2',
                         'Wasserstein:normalized_entropy',
                         'Cantor Set:coverage', 'Torus T^2:coverage'}
    _cidx = [j for j, mn in enumerate(metric_names) if mn in _complexity_names]
    if _cidx:
        _cranks = np.column_stack([_drd(X[:, j]) / len(X) for j in _cidx])
        _complexity = np.mean(_cranks, axis=1)
    else:
        _complexity = np.ones(len(X))
    _degeneracy_factor = np.clip(_complexity / 0.25, 0.3, 1.0)

    view_results = {}
    for vname, vinfo in view_lenses.items():
        cols = []
        found = []
        for geo in vinfo["geometries"]:
            if geo in geo_to_cols:
                cols.extend(geo_to_cols[geo])
                found.append(geo)

        if len(cols) < 5:
            print(f"  {vname:15s}  SKIP ({len(cols)} metrics)")
            continue

        X_view = X[:, cols]
        view_metric_names = [metric_names[j] for j in cols]

        # Deduplicate near-identical metrics (|Spearman r| > 0.95) so
        # shared metrics across geometries don't dominate PCA.
        # Keep one representative per cluster (highest column variance).
        if X_view.shape[1] > 3:
            rho = np.abs(sp_stats.spearmanr(X_view, axis=0).statistic)
            if rho.ndim == 0:
                rho = np.array([[1.0]])
            dist_m = np.clip(1.0 - rho, 0, None)
            np.fill_diagonal(dist_m, 0)
            dist_m = (dist_m + dist_m.T) / 2.0
            Z_corr = linkage(squareform(dist_m, checks=False), method='average')
            clust_labels = fcluster(Z_corr, t=0.05, criterion='distance')
            col_var = np.var(X_view, axis=0)
            keep = []
            for c in sorted(set(clust_labels)):
                members = np.where(clust_labels == c)[0]
                best = members[np.argmax(col_var[members])]
                keep.append(best)
            keep = sorted(keep)
            n_before = X_view.shape[1]
            X_view = X_view[:, keep]
            view_metric_names = [view_metric_names[k] for k in keep]
            n_after = len(keep)
            if n_after < n_before:
                print(f"    dedup: {n_before} → {n_after} metrics")

        coords_v, varexp_v, cumvar_v, Z_v, Vt_v = principal_projection(X_view)
        clusters_v, labels_v, order_v, link_v = cluster_atlas(Z_v, names)

        # Effective dimensionality
        pr = np.sum(varexp_v) ** 2 / np.sum(varexp_v ** 2)

        # Top-loading metrics for each PC axis (by absolute loading weight)
        axes_info = {}
        for pc_idx in range(min(3, len(Vt_v))):
            loadings = np.abs(Vt_v[pc_idx])
            top_idx = np.argsort(loadings)[::-1][:3]
            # Summarize by geometry: aggregate loading per geometry
            geo_load = {}
            for j in range(len(view_metric_names)):
                geo = view_metric_names[j].split(':')[0]
                geo_load[geo] = geo_load.get(geo, 0.0) + Vt_v[pc_idx, j] ** 2
            top_geo = max(geo_load, key=geo_load.get)
            top_metric = view_metric_names[top_idx[0]].split(':')[1]
            axes_info[f"pc{pc_idx+1}"] = {
                "top_geometry": top_geo,
                "top_metric": top_metric,
            }

        # Per-view cluster descriptions (lightweight: top domains + members)
        view_cluster_desc = {}
        for c_id, members in clusters_v.items():
            domain_counts = Counter(domains[m] for m in members)
            top_doms = [d for d, _ in domain_counts.most_common(2)]
            label = ' + '.join(d.replace('_', ' ').title() for d in top_doms)
            view_cluster_desc[c_id] = {
                "label": label,
                "size": len(members),
                "members": members,
                "top_domains": top_doms,
            }
        # Deduplicate labels
        seen = {}
        for c_id, desc in view_cluster_desc.items():
            if desc["label"] in seen:
                other = seen[desc["label"]]
                view_cluster_desc[other]["label"] += " I"
                desc["label"] += " II"
            seen[desc["label"]] = c_id

        n_eff = len(view_metric_names)  # after dedup
        print(f"  {vname:15s}  {len(cols):3d} metrics ({n_eff} effective)  "
              f"PC1+2={cumvar_v[1]*100:.1f}%  eff_dim={pr:.1f}  "
              f"geos: {', '.join(found)}")

        # Per-source relevance: fraction of this view's metrics where the
        # source ranks above the 90th percentile.  One-sided: only high
        # scores count.  Discounted by degeneracy factor for non-distributional
        # views (low-complexity sources get up to 70% penalty).
        from scipy.stats import rankdata as _rd
        _n = X_view.shape[0]
        _ranks = np.zeros_like(X_view)
        for _j in range(X_view.shape[1]):
            _col = np.nan_to_num(X_view[:, _j], nan=0.0)
            _ranks[:, _j] = (_rd(_col) / _n
                             if np.std(_col) > 1e-15 else 0.5)
        _count = np.sum(_ranks > 0.90, axis=1).astype(float)
        # Apply degeneracy discount (skip for Distributional — it IS the
        # complexity measure, so discounting there would be circular)
        if vname.lower() != 'distributional':
            _count = _count * _degeneracy_factor
        _cmax = _count.max()
        relevance = (_count / _cmax if _cmax > 1e-15
                     else np.zeros(len(_count)))

        view_results[vname] = {
            "coords": coords_v,
            "varexp": varexp_v,
            "labels": labels_v,
            "n_metrics": n_eff,
            "eff_dim": float(pr),
            "question": vinfo["question"],
            "detects": vinfo["detects"],
            "geometries": found,
            "axes": axes_info,
            "clusters": view_cluster_desc,
            "relevance": relevance.tolist(),
        }

    # Cross-view stability: for each source, how many other sources
    # does it SOMETIMES but not ALWAYS co-cluster with?
    n_views = len(view_results)
    n = len(names)
    co_matrix = np.zeros((n, n))
    for vr in view_results.values():
        lbl = vr["labels"]
        for i in range(n):
            for j in range(i + 1, n):
                if lbl[i] == lbl[j]:
                    co_matrix[i, j] += 1
                    co_matrix[j, i] += 1

    stability = np.zeros(n)
    for i in range(n):
        frac = co_matrix[i, :] / max(n_views, 1)
        stability[i] = np.sum((frac > 0) & (frac < 1))

    print(f"\n  Cross-view stability (lower = more robust identity):")
    order_stable = np.argsort(stability)
    for rank in range(min(5, n)):
        idx = order_stable[rank]
        print(f"    {names[idx]:35s} [{domains[names[idx]]:12s}]  instability={stability[idx]:.0f}")
    print(f"  ...")
    for rank in range(min(5, n)):
        idx = order_stable[-(rank + 1)]
        print(f"    {names[idx]:35s} [{domains[names[idx]]:12s}]  instability={stability[idx]:.0f}")

    return view_results, stability


# ==============================================================
# FIGURE
# ==============================================================

def make_figure(names, domains, coords, varexp, dist, clusters, labels, order, link):
    plt.rcParams.update({
        'figure.facecolor': '#181818', 'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444', 'axes.labelcolor': 'white',
        'text.color': 'white', 'xtick.color': '#cccccc', 'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 16), facecolor='#181818')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle("The Structure Atlas: How Exotic Geometry Organizes the World",
                 fontsize=15, fontweight='bold', color='white')

    # --- Panel 1: 2D Atlas (PC1 vs PC2) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#181818')

    for i, name in enumerate(names):
        domain = domains[name]
        color = DOMAIN_COLORS.get(domain, '#888888')
        ax1.scatter(coords[i, 0], coords[i, 1], c=color, s=80,
                    edgecolors='white', linewidths=0.5, zorder=3)
        ax1.annotate(name, (coords[i, 0], coords[i, 1]),
                     fontsize=6, color=color, ha='center', va='bottom',
                     xytext=(0, 6), textcoords='offset points')

    ax1.set_xlabel(f'PC1 ({varexp[0]*100:.1f}%)', fontsize=9)
    ax1.set_ylabel(f'PC2 ({varexp[1]*100:.1f}%)', fontsize=9)
    ax1.set_title("The Structure Atlas (PC1 vs PC2)", fontsize=11,
                  fontweight='bold')
    ax1.axhline(0, color='#333333', linewidth=0.5)
    ax1.axvline(0, color='#333333', linewidth=0.5)

    # --- Panel 2: PC1 vs PC3 ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#181818')

    for i, name in enumerate(names):
        domain = domains[name]
        color = DOMAIN_COLORS.get(domain, '#888888')
        ax2.scatter(coords[i, 0], coords[i, 2], c=color, s=80,
                    edgecolors='white', linewidths=0.5, zorder=3)
        ax2.annotate(name, (coords[i, 0], coords[i, 2]),
                     fontsize=6, color=color, ha='center', va='bottom',
                     xytext=(0, 6), textcoords='offset points')

    ax2.set_xlabel(f'PC1 ({varexp[0]*100:.1f}%)', fontsize=9)
    ax2.set_ylabel(f'PC3 ({varexp[2]*100:.1f}%)', fontsize=9)
    ax2.set_title("Structure Atlas (PC1 vs PC3)", fontsize=11,
                  fontweight='bold')
    ax2.axhline(0, color='#333333', linewidth=0.5)
    ax2.axvline(0, color='#333333', linewidth=0.5)

    # --- Panel 3: Distance matrix (clustered) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#181818')

    ordered_dist = dist[np.ix_(order, order)]
    im = ax3.imshow(ordered_dist, cmap='viridis', aspect='equal')
    ordered_names = [names[i] for i in order]

    ax3.set_xticks(range(len(names)))
    ax3.set_yticks(range(len(names)))
    ax3.set_xticklabels(ordered_names, rotation=90, fontsize=6)
    ax3.set_yticklabels(ordered_names, fontsize=6)

    # Color sidebar by domain
    for idx, i in enumerate(order):
        domain = domains[names[i]]
        color = DOMAIN_COLORS.get(domain, '#888888')
        ax3.add_patch(plt.Rectangle((-1.5, idx - 0.5), 1, 1,
                      color=color, clip_on=False))

    ax3.set_title("Rank-Normalized Distance (Hierarchical Order)", fontsize=11,
                  fontweight='bold')
    cb = fig.colorbar(im, ax=ax3, shrink=0.7, pad=0.02)
    cb.ax.tick_params(labelsize=6, colors='#cccccc')

    # --- Panel 4: Dendrogram ---
    from scipy.cluster.hierarchy import dendrogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#181818')

    # Color leaves by domain
    leaf_colors = {}
    for i, name in enumerate(names):
        leaf_colors[name] = DOMAIN_COLORS.get(domains[name], '#888888')

    def color_func(k):
        # Internal nodes get grey, leaves get domain color
        return '#666666'

    dend = dendrogram(link, labels=names, ax=ax4, orientation='right',
                      leaf_font_size=7, above_threshold_color='#666666',
                      color_threshold=0)

    # Color the leaf labels by domain
    ylbls = ax4.get_ymajorticklabels()
    for lbl in ylbls:
        name = lbl.get_text()
        if name in leaf_colors:
            lbl.set_color(leaf_colors[name])

    ax4.set_title("Structure Dendrogram", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Ward Distance (Rank-Normalized)", fontsize=8, color='#cccccc')
    ax4.tick_params(axis='x', colors='#cccccc', labelsize=7)

    # Legend
    handles = []
    for domain in sorted(DOMAIN_COLORS.keys()):
        handles.append(plt.Line2D([0], [0], marker='o', color='none',
                       markerfacecolor=DOMAIN_COLORS[domain],
                       markersize=8, label=domain))
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=8,
               frameon=True, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc', bbox_to_anchor=(0.5, 0.005))

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'structure_atlas.png')
    fig.savefig(outpath, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")


def make_figure_3d(names, domains, coords, varexp, labels):
    """3D phase space visualization: PC1 × PC2 × PC3 with stem lines."""
    from mpl_toolkits.mplot3d import Axes3D

    plt.rcParams.update({
        'figure.facecolor': '#181818', 'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444', 'axes.labelcolor': 'white',
        'text.color': 'white', 'xtick.color': '#cccccc', 'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(20, 18), facecolor='#181818')
    fig.suptitle("Structure Atlas: Phase Space (PC1 × PC2 × PC3)",
                 fontsize=16, fontweight='bold', color='white', y=0.97)

    pc1, pc2, pc3 = coords[:, 0], coords[:, 1], coords[:, 2]
    z_floor = pc3.min() - 1.0

    ax = fig.add_subplot(111, projection='3d', facecolor='#181818')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#252525')
    ax.yaxis.pane.set_edgecolor('#252525')
    ax.zaxis.pane.set_edgecolor('#252525')
    ax.grid(False)

    # Stem lines
    for i in range(len(names)):
        domain = domains[names[i]]
        color = DOMAIN_COLORS.get(domain, '#888888')
        ax.plot([pc1[i], pc1[i]], [pc2[i], pc2[i]], [z_floor, pc3[i]],
                color=color, alpha=0.18, linewidth=0.6)

    # Floor shadows
    for i in range(len(names)):
        domain = domains[names[i]]
        color = DOMAIN_COLORS.get(domain, '#888888')
        ax.scatter(pc1[i], pc2[i], z_floor, c=color, s=15,
                   alpha=0.12, marker='o', depthshade=False)

    # Points
    for i, name in enumerate(names):
        domain = domains[name]
        color = DOMAIN_COLORS.get(domain, '#888888')
        ax.scatter(pc1[i], pc2[i], pc3[i], c=color, s=120,
                   edgecolors='white', linewidths=0.5,
                   depthshade=True, alpha=0.92, zorder=5)

    # Labels: greedy placement — outliers first, fade overlapping ones
    centroid = np.array([pc1.mean(), pc2.mean(), pc3.mean()])
    dists_from_center = np.array([np.sqrt((pc1[i]-centroid[0])**2 +
                         (pc2[i]-centroid[1])**2 +
                         (pc3[i]-centroid[2])**2) for i in range(len(names))])
    label_order = np.argsort(dists_from_center)[::-1]

    labeled_positions = []
    for i in label_order:
        name = names[i]
        domain = domains[name]
        color = DOMAIN_COLORS.get(domain, '#888888')
        pos = np.array([pc1[i], pc2[i], pc3[i]])
        too_close = any(np.linalg.norm(pos - lp) < 0.9 for lp in labeled_positions)
        fontsize = 6.5 if not too_close else 4.5
        alpha = 0.95 if not too_close else 0.45
        ax.text(pc1[i], pc2[i], pc3[i] + 0.3, name,
                fontsize=fontsize, color=color, alpha=alpha,
                ha='left', va='bottom', zorder=6)
        labeled_positions.append(pos)

    ax.set_xlabel(f'PC1 ({varexp[0]*100:.1f}%)', fontsize=10, labelpad=12)
    ax.set_ylabel(f'PC2 ({varexp[1]*100:.1f}%)', fontsize=10, labelpad=12)
    ax.set_zlabel(f'PC3 ({varexp[2]*100:.1f}%)', fontsize=10, labelpad=12)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=20, azim=-50)

    # Legend
    handles = []
    for domain in sorted(DOMAIN_COLORS.keys()):
        handles.append(plt.Line2D([0], [0], marker='o', color='none',
                       markerfacecolor=DOMAIN_COLORS[domain],
                       markersize=9, label=domain))
    fig.legend(handles=handles, loc='lower center', ncol=len(DOMAIN_COLORS),
               fontsize=9, frameon=True, facecolor='#222222',
               edgecolor='#444444', labelcolor='#cccccc',
               bbox_to_anchor=(0.5, 0.02))

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'structure_atlas_3d.png')
    fig.savefig(outpath, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"3D figure saved: {outpath}")


def make_figure_techniques(names, domains, coords, varexp, surr_results, scale_results, scales):
    """D6+D7 figure: Surrogate decomposition and multi-scale filtration."""
    plt.rcParams.update({
        'figure.facecolor': '#181818', 'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444', 'axes.labelcolor': 'white',
        'text.color': 'white', 'xtick.color': '#cccccc', 'ytick.color': '#cccccc',
    })

    fig = plt.figure(figsize=(22, 18), facecolor='#181818')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25,
                          left=0.07, right=0.96, top=0.93, bottom=0.06)
    fig.suptitle("Structure Atlas: Surrogate Decomposition & Multi-Scale",
                 fontsize=15, fontweight='bold', color='white')

    # --- Panel 1: Atlas colored by sequential fraction ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#181818')

    seq_vals = [surr_results.get(name, {}).get('seq_fraction', 0) for name in names]
    seq_max = max(seq_vals) if seq_vals else 1.0
    if seq_max < 0.01:
        seq_max = 1.0

    import matplotlib.colors as mcolors
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=seq_max)

    for i, name in enumerate(names):
        seq = surr_results.get(name, {}).get('seq_fraction', 0)
        color = cmap(norm(seq))
        ax1.scatter(coords[i, 0], coords[i, 1], c=[color], s=80,
                    edgecolors='white', linewidths=0.4, zorder=3)
        ax1.annotate(name, (coords[i, 0], coords[i, 1]),
                     fontsize=5, color=color, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, shrink=0.7, pad=0.02)
    cb.set_label('Sequential Structure\n(cosine dist to shuffled)',
                 fontsize=8, color='#cccccc')
    cb.ax.tick_params(labelsize=7, colors='#cccccc')

    ax1.set_xlabel(f'PC1 ({varexp[0]*100:.1f}%)', fontsize=9)
    ax1.set_ylabel(f'PC2 ({varexp[1]*100:.1f}%)', fontsize=9)
    ax1.set_title("D6a: Sequential Structure Fraction", fontsize=11,
                  fontweight='bold')
    ax1.axhline(0, color='#333333', linewidth=0.5)
    ax1.axvline(0, color='#333333', linewidth=0.5)

    # --- Panel 2: Surrogate bar chart (top 20 by sequential fraction) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#181818')

    ranked = sorted(surr_results.items(), key=lambda x: -x[1]['seq_fraction'])
    top_n = min(25, len(ranked))
    bar_names = [r[0] for r in ranked[:top_n]]
    bar_shuf = [r[1]['shuffled_sig'] for r in ranked[:top_n]]
    bar_roll = [r[1]['rolled_sig'] for r in ranked[:top_n]]
    bar_rev = [r[1]['reversed_sig'] for r in ranked[:top_n]]

    x = np.arange(top_n)
    w = 0.25
    ax2.barh(x - w, bar_shuf, w, label='Shuffled', color='#e74c3c', alpha=0.85)
    ax2.barh(x, bar_roll, w, label='Rolled', color='#3498db', alpha=0.85)
    ax2.barh(x + w, bar_rev, w, label='Reversed', color='#2ecc71', alpha=0.85)

    ax2.set_yticks(x)
    ax2.set_yticklabels(bar_names, fontsize=6)
    ax2.set_xlabel('Metrics Disrupted by Surrogate', fontsize=9)
    ax2.set_title("D6b: Surrogate Distances (Top 25)", fontsize=11,
                  fontweight='bold')
    ax2.legend(fontsize=7, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc')
    ax2.invert_yaxis()

    # --- Panel 3: Multi-scale heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#181818')

    scale_names = sorted(scale_results.keys(),
                         key=lambda n: -max(scale_results[n].values()))
    matrix = np.array([[scale_results[n].get(s, 0) for s in scales]
                        for n in scale_names])

    im = ax3.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_yticks(range(len(scale_names)))
    ax3.set_yticklabels(scale_names, fontsize=7)
    ax3.set_xticks(range(len(scales)))
    ax3.set_xticklabels([str(s) for s in scales], fontsize=8)
    ax3.set_xlabel('Chunk Size (bytes)', fontsize=9)
    ax3.set_title("D7a: Significant Metrics vs White Noise by Scale",
                  fontsize=11, fontweight='bold')

    for i in range(len(scale_names)):
        for j in range(len(scales)):
            val = matrix[i, j]
            ax3.text(j, i, str(int(val)), ha='center', va='center',
                     fontsize=7, color='white' if val > matrix.max()*0.5 else '#cccccc')

    cb3 = fig.colorbar(im, ax=ax3, shrink=0.7, pad=0.02)
    cb3.set_label('Sig. metrics', fontsize=8, color='#cccccc')
    cb3.ax.tick_params(labelsize=7, colors='#cccccc')

    # --- Panel 4: Multi-scale line plot ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#181818')

    line_colors = {
        'chaos': '#e74c3c', 'number_theory': '#3498db', 'noise': '#95a5a6',
        'bearing': '#f39c12', 'medical': '#e84393', 'bio': '#1abc9c',
        'financial': '#fdcb6e', 'motion': '#6c5ce7', 'astro': '#fd79a8',
        'climate': '#74b9ff', 'speech': '#a29bfe', 'waveform': '#2ecc71',
    }

    _src_domain = {n: d for n, _, d in SOURCES}
    for name in scale_names:
        vals = [scale_results[name].get(s, 0) for s in scales]
        domain = _src_domain.get(name, 'noise')
        color = line_colors.get(domain, '#888888')
        ax4.plot(range(len(scales)), vals, 'o-', color=color, linewidth=1.5,
                 markersize=5, alpha=0.8, label=name)

    ax4.set_xticks(range(len(scales)))
    ax4.set_xticklabels([str(s) for s in scales], fontsize=8)
    ax4.set_xlabel('Chunk Size (bytes)', fontsize=9)
    ax4.set_ylabel('Significant Metrics vs White Noise', fontsize=9)
    ax4.set_title("D7b: Scale-Dependent Structure", fontsize=11,
                  fontweight='bold')
    ax4.legend(fontsize=5.5, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc', ncol=2, loc='upper left')
    ax4.grid(True, alpha=0.15, color='#555555')

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'structure_atlas_techniques.png')
    fig.savefig(outpath, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"Techniques figure saved: {outpath}")


def make_figure_views(names, domains, coords_full, varexp_full, labels_full,
                      view_results, stability):
    """6-panel figure: each geometric lens shows its own PCA projection."""
    plt.rcParams.update({
        'figure.facecolor': '#181818', 'axes.facecolor': '#181818',
        'axes.edgecolor': '#444444', 'axes.labelcolor': 'white',
        'text.color': 'white', 'xtick.color': '#cccccc', 'ytick.color': '#cccccc',
    })

    view_names = list(view_results.keys())
    n_views = len(view_names)
    ncols = 3
    nrows = (n_views + ncols - 1) // ncols
    fig = plt.figure(figsize=(24, 8 * nrows), facecolor='#181818')
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.32, wspace=0.25,
                           left=0.04, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle("Geometric Views: Same Data, Different Lenses",
                 fontsize=16, fontweight='bold', color='white')

    # Stability-based alpha: unstable sources get highlighted
    stab_norm = stability / max(stability.max(), 1)

    for vi, vname in enumerate(view_names):
        vr = view_results[vname]
        row, col = divmod(vi, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#181818')

        vc = vr["coords"]
        vv = vr["varexp"]
        vl = vr["labels"]
        n_clusters = len(set(vl))

        # Cluster-based color palette for this view
        cluster_cmap = plt.cm.Set2
        cluster_colors = {c: cluster_cmap(i / max(n_clusters - 1, 1))
                          for i, c in enumerate(sorted(set(vl)))}

        for i, name in enumerate(names):
            domain = domains[name]
            dcolor = DOMAIN_COLORS.get(domain, '#888888')
            # Ring color = domain, fill = view cluster
            ccolor = cluster_colors[vl[i]]
            alpha = 0.4 + 0.6 * stab_norm[i]  # unstable = more opaque
            ax.scatter(vc[i, 0], vc[i, 1], c=[ccolor], s=60,
                       edgecolors=dcolor, linewidths=1.0, alpha=alpha, zorder=3)
            ax.annotate(name, (vc[i, 0], vc[i, 1]),
                        fontsize=4.5, color=dcolor, alpha=max(alpha, 0.5),
                        ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')

        ax.set_xlabel(f'PC1 ({vv[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({vv[1]*100:.1f}%)', fontsize=8)
        ax.set_title(f"{vname} Lens ({vr['n_metrics']} metrics, "
                     f"eff_dim={vr['eff_dim']:.1f})",
                     fontsize=10, fontweight='bold')
        ax.axhline(0, color='#333333', linewidth=0.5)
        ax.axvline(0, color='#333333', linewidth=0.5)

        # Question as subtitle
        ax.text(0.5, -0.08, vr["question"], transform=ax.transAxes,
                fontsize=7, color='#999999', ha='center', style='italic')

    # Legend
    handles = []
    for domain in sorted(DOMAIN_COLORS.keys()):
        handles.append(plt.Line2D([0], [0], marker='o', color='none',
                       markerfacecolor=DOMAIN_COLORS[domain],
                       markersize=7, label=domain))
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=7,
               frameon=True, facecolor='#222222', edgecolor='#444444',
               labelcolor='#cccccc', bbox_to_anchor=(0.5, 0.002))

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'figures', 'structure_atlas_views.png')
    fig.savefig(outpath, dpi=180, facecolor='#181818', bbox_inches='tight')
    plt.close(fig)
    print(f"Views figure saved: {outpath}")


# ==============================================================
# CLUSTER LABELING
# ==============================================================

# Structural concepts mapped from metric signatures.
# Each entry: (metric_keyword, z_threshold, label)
# Checked in order; first match wins.
_SIGNATURE_RULES = [
    # Entropy / uniformity (low seq_fraction, high entropy metrics)
    (lambda z, seq: seq < 0.10, "Maximum Entropy"),
    # Chaotic: dominant lyapunov / var_scaling
    (lambda z, seq: z.get('lyapunov_max', 0) > 1.5 or z.get('var_scaling_ratio', 0) > 3.0,
     "Chaotic Dynamics"),
    # Deterministic: extreme Fisher det or path regularity
    (lambda z, seq: z.get('path_regularity', 0) > 3.0 or z.get('det_fisher', 0) > 5.0,
     "Deterministic Sequences"),
    # Intermittent/bursty: high elliptic or lightlike fraction
    (lambda z, seq: z.get('elliptic_fraction', 0) > 3.0 or z.get('lightlike_fraction', 0) > 2.0,
     "Intermittent Bursts"),
    # Discrete symbolic: high c3_energy or skew_mean
    (lambda z, seq: z.get('c3_energy', 0) > 2.5 or
     (z.get('skew_mean', 0) > 2.0 and z.get('total_winding', 0) < -2.0),
     "Discrete Symbolic"),
    # Smooth correlated: high graph density / hurst, low self_similarity
    (lambda z, seq: (z.get('mean_degree', 0) > 2.0 or z.get('graph_density', 0) > 2.0) and
     z.get('hurst_estimate', 0) > 1.5,
     "Smooth Correlated"),
    # Dense structured: high connectedness + hapax
    (lambda z, seq: z.get('connectedness', 0) > 1.5 and z.get('hapax_ratio', 0) > 1.0,
     "Dense Structured"),
    # Stochastic: moderate structure, noise-like
    (lambda z, seq: 0.30 <= seq <= 0.55, "Stochastic Processes"),
    # Weakly structured
    (lambda z, seq: seq < 0.30, "Weakly Structured"),
]


def _label_clusters(clusters, names, X, metric_names, domains, surr_results):
    """Label clusters using metric z-score signatures, not domain heuristics.

    For each cluster, compute per-metric z-scores vs all other sources.
    Then match against structural signature rules.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    metric_short = [m.split(':')[1] for m in metric_names]

    cluster_descriptions = {}
    for c_id in sorted(clusters.keys()):
        members = clusters[c_id]
        domain_counts = Counter(domains[m] for m in members)
        top_domains = [d for d, _ in domain_counts.most_common(3)]

        idxs = [name_to_idx[m] for m in members]
        others = [i for i in range(len(names)) if i not in set(idxs)]
        cluster_mean = X[idxs].mean(axis=0)
        other_mean = X[others].mean(axis=0)
        other_std = X[others].std(axis=0)
        other_std[other_std < 1e-10] = 1.0
        diff = (cluster_mean - other_mean) / other_std

        # Build z-score dict keyed by short metric name
        z_scores = {}
        for j, mn in enumerate(metric_short):
            # Keep the most extreme z-score if metric name repeats
            if mn not in z_scores or abs(diff[j]) > abs(z_scores[mn]):
                z_scores[mn] = float(diff[j])

        # Surrogate seq_fraction
        member_seqs = [surr_results[m]['seq_fraction']
                       for m in members if m in surr_results]
        avg_seq = np.mean(member_seqs) if member_seqs else 0.5

        # Match against rules
        label = "Mixed Structure"
        for rule_fn, rule_label in _SIGNATURE_RULES:
            try:
                if rule_fn(z_scores, avg_seq):
                    label = rule_label
                    break
            except (KeyError, TypeError):
                continue

        cluster_descriptions[c_id] = {
            "domains": top_domains,
            "size": len(members),
            "members": members,
            "label": label,
        }

    # Deduplicate labels
    seen_labels = {}
    for c_id, desc in cluster_descriptions.items():
        lbl = desc["label"]
        if lbl in seen_labels:
            other = seen_labels[lbl]
            # Disambiguate by top distinguishing metric
            for cid_dup, idx_list in [(other, [name_to_idx[m] for m in cluster_descriptions[other]["members"]]),
                                       (c_id, [name_to_idx[m] for m in members])]:
                oth = [i for i in range(len(names)) if i not in set(idx_list)]
                cm = X[idx_list].mean(axis=0)
                om = X[oth].mean(axis=0)
                os = X[oth].std(axis=0)
                os[os < 1e-10] = 1.0
                d = (cm - om) / os
                top_j = np.argmax(np.abs(d))
                suffix = metric_short[top_j].replace('_', ' ')
                cluster_descriptions[cid_dup]["label"] += f" ({suffix})"
        seen_labels[lbl] = c_id

    return cluster_descriptions


# ==============================================================
# MAIN
# ==============================================================

def main(tier='complete'):
    runner = Runner("Structure Atlas", mode="1d", data_size=DATA_SIZE,
                     cache=True, n_workers=8, tier=tier)

    profiles, profiles_std, domains, descriptions = direction_1(runner)
    names, X, coords, varexp, cumvar, Z = direction_2(profiles, runner.metric_names)
    direction_3(names, domains)
    neighbors, dist = direction_4(names, Z, domains)
    clusters, labels, order, link = direction_5(Z, names, domains)

    make_figure(names, domains, coords, varexp, dist, clusters, labels, order, link)
    make_figure_3d(names, domains, coords, varexp, labels)

    # D6: Surrogate decomposition
    surr_results = direction_6(runner, domains)

    # D7: Multi-scale filtration
    scale_results, scales = direction_7(runner, domains)

    make_figure_techniques(names, domains, coords, varexp, surr_results,
                           scale_results, scales)

    # D8: Geometric views — lenses discovered from geometry metadata
    from exotic_geometry_framework import GeometryAnalyzer as _GA
    _analyzer = _GA().add_all_geometries()
    view_lenses = _analyzer.view_lenses()
    geometry_catalog = _analyzer.geometry_catalog()

    view_results, stability = direction_8_views(
        names, X, runner.metric_names, domains, view_lenses)
    make_figure_views(names, domains, coords, varexp, labels,
                      view_results, stability)

    # Build descriptive cluster names from metric signatures
    cluster_descriptions = _label_clusters(
        clusters, names, X, runner.metric_names, domains, surr_results)

    def _build_axes_metadata(coords, names, domains, varexp):
        """Build axis labels that match the current PCA sign convention."""
        name_to_idx = {n: i for i, n in enumerate(names)}

        def _sign(anchor_low, anchor_high, pc_idx):
            """Return +1 if anchor_low has lower PC score, else -1."""
            lo = np.mean([coords[name_to_idx[n], pc_idx]
                          for n in anchor_low if n in name_to_idx])
            hi = np.mean([coords[name_to_idx[n], pc_idx]
                          for n in anchor_high if n in name_to_idx])
            return 1 if lo < hi else -1

        # PC1: White Noise should be at the "uniform" end
        s1 = _sign(["White Noise", "AES Encrypted"],
                    ["Sine Wave", "fBm (Persistent)"], 0)
        # PC2: DNA should be at the "symbolic" end
        s2 = _sign(["DNA Human", "Prime Gaps"],
                    ["Sine Wave", "Lorenz Attractor"], 1)
        # PC3: Fibonacci should be at the "formal" end
        s3 = _sign(["Fibonacci Word", "Thue-Morse"],
                    ["EEG Seizure", "Bearing Ball"], 2)

        def _ends(sign, low_label, high_label):
            return (low_label, high_label) if sign == 1 else (high_label, low_label)

        pc1_lo, pc1_hi = _ends(s1, "Uniform / high entropy", "Structured / peaked")
        pc2_lo, pc2_hi = _ends(s2, "Discrete / symbolic", "Continuous / oscillatory")
        pc3_lo, pc3_hi = _ends(s3, "Formal / deterministic", "Empirical / physical")

        return {
            "pc1": {
                "label": "PC1: Distributional Order",
                "variance": round(float(varexp[0]), 4),
                "description": f"{pc1_lo} at low end; {pc1_hi} at high end",
                "low": pc1_lo, "high": pc1_hi,
            },
            "pc2": {
                "label": "PC2: Spectral Character",
                "variance": round(float(varexp[1]), 4),
                "description": f"{pc2_lo} at low end; {pc2_hi} at high end",
                "low": pc2_lo, "high": pc2_hi,
            },
            "pc3": {
                "label": "PC3: Source Formality",
                "variance": round(float(varexp[2]), 4),
                "description": f"{pc3_lo} at low end; {pc3_hi} at high end",
                "low": pc3_lo, "high": pc3_hi,
            },
        }

    # Global relevance: fraction of all metrics where the source ranks
    # above the 90th percentile.  Same approach as per-view relevance.
    from scipy.stats import rankdata as _grd
    _gn = X.shape[0]
    _granks = np.zeros_like(X)
    for _j in range(X.shape[1]):
        _col = np.nan_to_num(X[:, _j], nan=0.0)
        _granks[:, _j] = (_grd(_col) / _gn
                          if np.std(_col) > 1e-15 else 0.5)
    _gcount = np.sum(_granks > 0.90, axis=1).astype(float)
    # Recompute degeneracy factor for global relevance (same logic as direction_8_views)
    _complexity_names_g = {'Cantor Set:coverage', 'Torus T^2:coverage'}
    _cidx_g = [j for j, mn in enumerate(runner.metric_names) if mn in _complexity_names_g]
    if _cidx_g:
        _cranks_g = np.column_stack([_grd(X[:, j]) / _gn for j in _cidx_g])
        _complexity_g = np.mean(_cranks_g, axis=1)
    else:
        _complexity_g = np.ones(len(X))
    _degeneracy_factor_g = np.clip(_complexity_g / 0.25, 0.3, 1.0)
    _gcount = _gcount * _degeneracy_factor_g  # discount low-complexity sources
    _gcmax = _gcount.max()
    global_relevance = (_gcount / _gcmax if _gcmax > 1e-15
                        else np.zeros(len(_gcount)))

    # Export enriched data for interactive 3D visualization
    atlas_data = {
        "sources": [
            {
                "name": names[i],
                "domain": domains[names[i]],
                "description": descriptions.get(names[i], ""),
                "pc": coords[i, :10].tolist(),
                "cluster": int(labels[i]),
                "relevance": round(float(global_relevance[i]), 4),
                "seq_fraction": round(surr_results[names[i]]['seq_fraction'], 3)
                    if names[i] in surr_results else None,
                "shuffled_sig": surr_results[names[i]]['shuffled_sig']
                    if names[i] in surr_results else None,
                "reversed_sig": surr_results[names[i]]['reversed_sig']
                    if names[i] in surr_results else None,
                "neighbors": [
                    {"name": nn_name, "distance": round(nn_dist, 3),
                     "cross_domain": domains[nn_name] != domains[names[i]]}
                    for nn_name, nn_dist in neighbors[names[i]]
                ] if names[i] in neighbors else [],
            }
            for i in range(len(names))
        ],
        "variance_explained": varexp[:10].tolist(),
        "axes": _build_axes_metadata(coords, names, domains, varexp),
        "domain_colors": DOMAIN_COLORS,
        "n_metrics": len(runner.metric_names),
        "n_geometries": len(geometry_catalog),
        "data_size": DATA_SIZE,
        "n_trials": N_TRIALS,
        "geometry_catalog": geometry_catalog,
        "clusters": {
            str(c_id): {
                "top_domains": desc["domains"],
                "size": desc["size"],
                "members": desc["members"],
                "label": desc["label"],
            }
            for c_id, desc in cluster_descriptions.items()
        },
        "metric_names": list(runner.metric_names),
        "profiles": X.tolist(),
        "pca_loadings": (np.linalg.svd(Z, full_matrices=False)[2]).tolist(),
        "explained_variance_ratio": varexp.tolist(),
        "views": {
            vname: {
                "question": vr["question"],
                "detects": vr["detects"],
                "geometries": vr["geometries"],
                "n_metrics": vr["n_metrics"],
                "eff_dim": vr["eff_dim"],
                "pc": vr["coords"][:, :5].tolist(),
                "variance_explained": vr["varexp"][:5].tolist(),
                "labels": vr["labels"].tolist(),
                "axes": vr.get("axes", {}),
                "relevance": vr.get("relevance", []),
                "clusters": {
                    str(c_id): {
                        "label": cd["label"],
                        "size": cd["size"],
                        "members": cd["members"],
                        "top_domains": cd["top_domains"],
                    }
                    for c_id, cd in vr.get("clusters", {}).items()
                },
            }
            for vname, vr in view_results.items()
        },
        "stability": {
            names[i]: round(float(stability[i]), 1)
            for i in range(len(names))
        },
    }
    atlas_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', 'figures', 'structure_atlas_data.json')
    with open(atlas_json_path, 'w') as f:
        _json.dump(atlas_data, f, indent=2)
    print(f"Atlas data exported: {atlas_json_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sources: {len(names)}")
    print(f"  Metrics: {len(runner.metric_names)}")
    print(f"  Effective dimensionality: {np.sum(varexp)**2 / np.sum(varexp**2):.1f}")
    print(f"  PC1+2 variance: {cumvar[1]*100:.1f}%")
    print(f"  Clusters: {len(clusters)}")

    # Cross-domain neighbors
    n_cross = 0
    for name in names:
        nn = neighbors[name]
        domain = domains[name]
        n_cross += sum(1 for n, _ in nn if domains[n] != domain)
    print(f"  Cross-domain nearest neighbors: {n_cross} / {len(names) * 3}")

    runner.close()


if __name__ == "__main__":
    main()
