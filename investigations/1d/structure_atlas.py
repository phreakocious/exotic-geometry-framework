#!/usr/bin/env python3
"""
Investigation: The Structure Atlas
====================================

Map diverse data sources into the framework's 141-metric structure space.
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
import numpy as np
from scipy.io import loadmat
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.investigation_runner import Runner

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================
# CONFIG
# ==============================================================
DATA_SIZE = 12000
N_TRIALS = 25
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'cwru', 'raw')

# Domain color map
DOMAIN_COLORS = {
    'chaos': '#e74c3c',
    'number_theory': '#3498db',
    'noise': '#95a5a6',
    'waveform': '#2ecc71',
    'bearing': '#f39c12',
    'binary': '#9b59b6',
    'bio': '#1abc9c',
    'medical': '#e84393',
    'financial': '#fdcb6e',
    'motion': '#6c5ce7',
    'astro': '#fd79a8',
    'climate': '#74b9ff',
    'speech': '#a29bfe',
}

# ==============================================================
# DATA GENERATORS
# ==============================================================

# --- Chaos ---
def gen_logistic(rng, size):
    x = 0.1 + 0.8 * rng.random()
    for _ in range(1000):
        x = 4.0 * x * (1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 4.0 * x * (1.0 - x)
        vals[i] = int(x * 255)
    return vals

def gen_henon(rng, size):
    x, y = 0.1 * rng.random(), 0.1 * rng.random()
    for _ in range(1000):
        x, y = 1.0 - 1.4 * x**2 + y, 0.3 * x
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x, y = 1.0 - 1.4 * x**2 + y, 0.3 * x
        vals[i] = int(np.clip((x + 1.5) / 3.0 * 255, 0, 255))
    return vals

def gen_tent(rng, size):
    x = rng.random()
    for _ in range(500):
        x = 1.999 * min(x, 1.0 - x)
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = 1.999 * min(x, 1.0 - x)
        vals[i] = int(x * 255)
    return vals

# --- Number Theory ---
_PRIMES = None
def _get_primes():
    global _PRIMES
    if _PRIMES is None:
        limit = 2_000_000
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        _PRIMES = np.where(sieve)[0]
    return _PRIMES

def gen_prime_gaps(rng, size):
    primes = _get_primes()
    max_start = len(primes) - size - 100
    start = rng.integers(0, max_start)
    gaps = np.diff(primes[start:start + size + 1])
    return np.clip(gaps, 0, 255).astype(np.uint8)

def gen_collatz(rng, size):
    n = int(rng.integers(10_000, 100_000_000))
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        vals[i] = n % 256
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
    return vals

def gen_divisor_count(rng, size):
    start = int(rng.integers(10_000, 500_000))
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        n = start + i
        d = 0
        for j in range(1, int(n**0.5) + 1):
            if n % j == 0:
                d += 2 if j != n // j else 1
        vals[i] = min(d, 255)
    return vals

# --- Noise ---
def gen_white_noise(rng, size):
    return rng.integers(0, 256, size, dtype=np.uint8)

def gen_pink_noise(rng, size):
    white = rng.standard_normal(size)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(size)
    freqs[0] = 1.0
    fft /= np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=size)
    pink = (pink - pink.min()) / (pink.max() - pink.min() + 1e-15) * 255
    return pink.astype(np.uint8)

def gen_brownian(rng, size):
    steps = rng.standard_normal(size)
    walk = np.cumsum(steps)
    walk = (walk - walk.min()) / (walk.max() - walk.min() + 1e-15) * 255
    return walk.astype(np.uint8)

# --- Waveforms ---
def gen_sine(rng, size):
    freq = rng.uniform(5, 50)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, 1, size)
    wave = np.sin(2 * np.pi * freq * t + phase)
    return ((wave + 1) / 2 * 255).astype(np.uint8)

def gen_sawtooth(rng, size):
    freq = rng.uniform(5, 50)
    t = np.linspace(0, 1, size)
    wave = 2 * (t * freq - np.floor(t * freq + 0.5))
    return ((wave + 1) / 2 * 255).astype(np.uint8)

# --- More chaos ---
def gen_lorenz(rng, size):
    x, y, z = 0.1 * rng.standard_normal(3)
    dt = 0.01
    for _ in range(5000):  # transient
        dx = 10.0 * (y - x) * dt
        dy = (x * (28.0 - z) - y) * dt
        dz = (x * y - (8.0/3.0) * z) * dt
        x, y, z = x + dx, y + dy, z + dz
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        dx = 10.0 * (y - x) * dt
        dy = (x * (28.0 - z) - y) * dt
        dz = (x * y - (8.0/3.0) * z) * dt
        x, y, z = x + dx, y + dy, z + dz
        vals[i] = int(np.clip((x + 25) / 50 * 255, 0, 255))
    return vals

def gen_rossler(rng, size):
    x, y, z = 0.1 * rng.standard_normal(3)
    dt = 0.02
    a, b, c = 0.2, 0.2, 5.7
    for _ in range(5000):
        dx = (-y - z) * dt
        dy = (x + a * y) * dt
        dz = (b + z * (x - c)) * dt
        x, y, z = x + dx, y + dy, z + dz
    vals = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        dx = (-y - z) * dt
        dy = (x + a * y) * dt
        dz = (b + z * (x - c)) * dt
        x, y, z = x + dx, y + dy, z + dz
        vals[i] = int(np.clip((x + 12) / 24 * 255, 0, 255))
    return vals

# --- Fractional Brownian motion ---
def _gen_fbm(H):
    def gen(rng, size):
        # Hosking method approximation via spectral synthesis
        freqs = np.fft.rfftfreq(size, d=1.0)
        freqs[0] = 1.0
        psd = freqs ** (-(2 * H + 1))
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        fft_vals = np.sqrt(psd) * np.exp(1j * phases)
        fft_vals[0] = 0
        signal = np.fft.irfft(fft_vals, n=size)
        signal = np.cumsum(signal)  # integrate to get fBm
        lo, hi = signal.min(), signal.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((signal - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_fbm_03 = _gen_fbm(0.3)  # antipersistent
gen_fbm_07 = _gen_fbm(0.7)  # persistent

# --- Stochastic processes ---
def gen_arma(rng, size):
    """ARMA(2,1) process."""
    ar = np.array([0.7, -0.2])
    ma = np.array([0.5])
    noise = rng.standard_normal(size + 100)
    x = np.zeros(size + 100)
    for t in range(2, len(x)):
        x[t] = ar[0]*x[t-1] + ar[1]*x[t-2] + noise[t] + ma[0]*noise[t-1]
    x = x[100:]  # discard transient
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)

def gen_regime_switch(rng, size):
    """Two-state regime switching (bull/bear or stable/volatile)."""
    state = 0
    vals = np.zeros(size)
    x = 0.0
    for i in range(size):
        if state == 0:  # calm
            x += rng.standard_normal() * 0.5
            if rng.random() < 0.01:
                state = 1
        else:  # volatile
            x += rng.standard_normal() * 3.0
            if rng.random() < 0.05:
                state = 0
        vals[i] = x
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((vals - lo) / (hi - lo) * 255).astype(np.uint8)

# --- Encrypted / compressed ---
def gen_aes_encrypted(rng, size):
    """AES-CTR-like: deterministic permutation of counter bytes.
    Approximated as a keyed hash since we can't import crypto libs easily."""
    # Use numpy's bit generator to simulate block cipher output
    key = rng.integers(0, 2**32)
    cipher_rng = np.random.default_rng(key)
    return cipher_rng.integers(0, 256, size, dtype=np.uint8)

def gen_gzip_compressed(rng, size):
    """Compress random text-like data and use the compressed bytes."""
    import zlib
    # Generate pseudo-text (biased bytes with repetition)
    text = rng.choice(np.arange(32, 127, dtype=np.uint8), size=size*3,
                      p=None)
    compressed = zlib.compress(bytes(text), level=6)
    out = np.frombuffer(compressed, dtype=np.uint8)
    if len(out) < size:
        out = np.pad(out, (0, size - len(out)), constant_values=0)
    return out[:size].copy()

# --- Mathematical constants ---
def gen_pi_digits(rng, size):
    """Pi in base 256 via the mpmath library, or fallback to computed digits."""
    try:
        from mpmath import mp
        mp.dps = size * 2 + 100
        pi_str = mp.nstr(mp.pi, size * 2 + 50, strip_zeros=False)
        # Remove "3." prefix
        digits = pi_str.replace('.', '')[1:]
        # Convert pairs of decimal digits to bytes
        vals = []
        for i in range(0, len(digits) - 1, 3):
            chunk = digits[i:i+3]
            if len(chunk) == 3:
                vals.append(int(chunk) % 256)
            if len(vals) >= size:
                break
        return np.array(vals[:size], dtype=np.uint8)
    except ImportError:
        # Fallback: use a simple approximation
        # Pi digits are ~uniform in base 256 so this is close
        return np.random.default_rng(314159).integers(0, 256, size,
                                                       dtype=np.uint8)

# --- Blue noise (high-frequency emphasis) ---
def gen_blue_noise(rng, size):
    white = rng.standard_normal(size)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(size)
    freqs[0] = 1e-10
    fft *= np.sqrt(freqs)  # +3dB/octave
    blue = np.fft.irfft(fft, n=size)
    lo, hi = blue.min(), blue.max()
    if hi - lo < 1e-15:
        hi = lo + 1.0
    return ((blue - lo) / (hi - lo) * 255).astype(np.uint8)

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

# --- Binary ---
def gen_binary_x86(rng, size):
    """Read random chunk from this process's own binary."""
    exe = sys.executable
    with open(exe, 'rb') as f:
        data = f.read()
    if len(data) < size + 1000:
        return rng.integers(0, 256, size, dtype=np.uint8)
    start = rng.integers(1000, len(data) - size)
    return np.frombuffer(data[start:start + size], dtype=np.uint8).copy()

# --- Bio-like (synthetic DNA statistics) ---
def gen_dna_like(rng, size):
    """Synthetic DNA: biased tetranucleotide frequencies like real genomes."""
    # Real DNA has GC-content ~40-60%, dinucleotide biases
    gc = rng.uniform(0.35, 0.65)
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A, C, G, T
    bases = rng.choice(4, size=size, p=probs)
    # Map to spread-out byte values (not just 0-3)
    mapping = np.array([65, 67, 71, 84], dtype=np.uint8)  # ASCII A,C,G,T
    return mapping[bases]


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

gen_dna_human = _gen_dna_species('human')
gen_dna_chimp = _gen_dna_species('chimpanzee')
gen_dna_dog = _gen_dna_species('dog')


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
        # Concatenate random stars to reach size
        n_stars = size // 3197 + 1
        indices = rng.choice(len(pool), size=n_stars, replace=True)
        concat = pool[indices].flatten()[:size]
        lo, hi = concat.min(), concat.max()
        if hi - lo < 1e-15:
            hi = lo + 1.0
        return ((concat - lo) / (hi - lo) * 255).astype(np.uint8)
    return gen

gen_kepler_exoplanet = _gen_kepler(2)  # confirmed exoplanet
gen_kepler_nonplanet = _gen_kepler(1)  # no exoplanet


# ==============================================================
# ALL DATA SOURCES
# ==============================================================

SOURCES = [
    # (name, generator, domain)
    # Chaos
    ('Logistic Chaos', gen_logistic, 'chaos'),
    ('Henon Map', gen_henon, 'chaos'),
    ('Tent Map', gen_tent, 'chaos'),
    ('Lorenz Attractor', gen_lorenz, 'chaos'),
    ('Rossler Attractor', gen_rossler, 'chaos'),
    # Number theory
    ('Prime Gaps', gen_prime_gaps, 'number_theory'),
    ('Collatz mod256', gen_collatz, 'number_theory'),
    ('Divisor Count', gen_divisor_count, 'number_theory'),
    # Noise family
    ('White Noise', gen_white_noise, 'noise'),
    ('Blue Noise', gen_blue_noise, 'noise'),
    ('Pink Noise', gen_pink_noise, 'noise'),
    ('Brownian Walk', gen_brownian, 'noise'),
    ('fBm H=0.3', gen_fbm_03, 'noise'),
    ('fBm H=0.7', gen_fbm_07, 'noise'),
    # Stochastic processes
    ('ARMA(2,1)', gen_arma, 'noise'),
    ('Regime Switch', gen_regime_switch, 'noise'),
    # Waveforms
    ('Sine Wave', gen_sine, 'waveform'),
    ('Sawtooth Wave', gen_sawtooth, 'waveform'),
    # Engineering
    ('Bearing Normal', gen_bearing_normal, 'bearing'),
    ('Bearing Ball', gen_bearing_ball, 'bearing'),
    ('Bearing Inner', gen_bearing_inner, 'bearing'),
    ('Bearing Outer', gen_bearing_outer, 'bearing'),
    # Digital
    ('Executable Binary', gen_binary_x86, 'binary'),
    ('AES Encrypted', gen_aes_encrypted, 'binary'),
    ('Gzip Compressed', gen_gzip_compressed, 'binary'),
    ('Synthetic DNA', gen_dna_like, 'bio'),
    ('Pi (base 256)', gen_pi_digits, 'number_theory'),
    # Medical
    ('ECG Normal', gen_ecg_normal, 'medical'),
    ('ECG Supraventr.', gen_ecg_svpb, 'medical'),
    ('ECG Ventricular', gen_ecg_vpb, 'medical'),
    ('ECG Fusion', gen_ecg_fusion, 'medical'),
    ('EEG Seizure', gen_eeg_seizure, 'medical'),
    ('EEG Tumor', gen_eeg_tumor, 'medical'),
    ('EEG Healthy', gen_eeg_healthy, 'medical'),
    ('EEG Eyes Closed', gen_eeg_eyes_closed, 'medical'),
    # Financial
    ('NYSE Returns', gen_stock_sp500, 'financial'),
    ('Nikkei Returns', gen_stock_nikkei, 'financial'),
    ('NASDAQ Returns', gen_stock_nasdaq, 'financial'),
    # Genomics (real DNA)
    ('DNA Human', gen_dna_human, 'bio'),
    ('DNA Chimp', gen_dna_chimp, 'bio'),
    ('DNA Dog', gen_dna_dog, 'bio'),
    # Motion (accelerometer)
    ('Accel Walk', gen_motion_walk, 'motion'),
    ('Accel Jog', gen_motion_jog, 'motion'),
    ('Accel Sit', gen_motion_sit, 'motion'),
    ('Accel Stairs', gen_motion_stairs, 'motion'),
    # Astrophysics (gravitational waves)
    ('LIGO Hanford', gen_gw_hanford, 'astro'),
    ('LIGO Livingston', gen_gw_livingston, 'astro'),
    # Astronomy (Kepler light curves)
    ('Kepler Exoplanet', gen_kepler_exoplanet, 'astro'),
    ('Kepler Non-planet', gen_kepler_nonplanet, 'astro'),
    # Climate
    ('Temperature', gen_climate_temp, 'climate'),
    ('Pressure', gen_climate_pressure, 'climate'),
    ('Humidity', gen_climate_humidity, 'climate'),
    ('Wind Speed', gen_climate_wind, 'climate'),
    # Speech (audio)
    ('Speech "Zero"', gen_speech_zero, 'speech'),
    ('Speech "Five"', gen_speech_five, 'speech'),
    ('Speech "Nine"', gen_speech_nine, 'speech'),
]

# ==============================================================
# COLLECTION
# ==============================================================

def collect_profiles(runner):
    """Collect mean metric profiles for all sources."""
    print("\n" + "=" * 60)
    print("COLLECTING STRUCTURE PROFILES")
    print("=" * 60)

    profiles = {}  # name -> {metric: mean_value}
    profiles_std = {}
    domains = {}

    for name, gen_fn, domain in SOURCES:
        rngs = runner.trial_rngs()
        chunks = [gen_fn(rng, runner.data_size) for rng in rngs]
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

        profiles[name] = mean_profile
        profiles_std[name] = std_profile
        domains[name] = domain
        n_nonzero = sum(1 for v in mean_profile.values() if abs(v) > 1e-15)
        print(f"  {name:20s} [{domain:13s}]  {n_nonzero}/{len(runner.metric_names)} active metrics")

    return profiles, profiles_std, domains


# ==============================================================
# ANALYSIS
# ==============================================================

def build_matrix(profiles, metric_names):
    """Build (n_sources × n_metrics) matrix from profiles."""
    names = list(profiles.keys())
    X = np.array([[profiles[n].get(m, 0.0) for m in metric_names] for n in names])
    return names, X


def principal_projection(X):
    """Z-score, SVD, project to principal components."""
    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)
    col_std[col_std < 1e-15] = 1.0
    Z = (X - col_mean) / col_std
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    variance_explained = s**2 / np.sum(s**2)
    cumvar = np.cumsum(variance_explained)

    # Project to first k components
    coords = U * s  # (n_sources, n_components)
    return coords, variance_explained, cumvar, Z


def find_neighbors(names, Z, k=3):
    """Find k nearest neighbors for each source in z-scored space."""
    dist = squareform(pdist(Z, metric='cosine'))
    neighbors = {}
    for i, name in enumerate(names):
        dists = dist[i].copy()
        dists[i] = np.inf
        nearest = np.argsort(dists)[:k]
        neighbors[name] = [(names[j], dists[j]) for j in nearest]
    return neighbors, dist


def cluster_atlas(Z, names, n_clusters=5):
    """Hierarchical clustering of the atlas."""
    condensed = pdist(Z, metric='cosine')
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=0.0)
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
    profiles, profiles_std, domains = collect_profiles(runner)
    return profiles, profiles_std, domains


def direction_2(profiles, metric_names):
    """D2: Principal structure space."""
    print("\n" + "=" * 60)
    print("D2: PRINCIPAL STRUCTURE SPACE")
    print("=" * 60)

    names, X = build_matrix(profiles, metric_names)
    coords, varexp, cumvar, Z = principal_projection(X)

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

    clusters, labels, order, link = cluster_atlas(Z, names, n_clusters=7)

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

    source_lookup = {name: (gen_fn, domain) for name, gen_fn, domain in SOURCES}

    results = {}  # name -> {shuffled_dist, rolled_dist, reversed_dist, seq_fraction}

    for name, gen_fn, domain in SOURCES:
        rngs = runner.trial_rngs()

        # Generate real chunks
        real_chunks = [gen_fn(rng, runner.data_size) for rng in rngs]
        real_metrics = runner.collect(real_chunks)

        # Surrogate 1: Distribution-matched (shuffle each chunk independently)
        surr_rngs = runner.trial_rngs(offset=1000)
        shuf_chunks = []
        for chunk, srng in zip(real_chunks, surr_rngs):
            s = chunk.copy()
            srng.shuffle(s)
            shuf_chunks.append(s)
        shuf_metrics = runner.collect(shuf_chunks)

        # Surrogate 2: Rolled (cyclic shift by random amount)
        roll_chunks = []
        for chunk, srng in zip(real_chunks, surr_rngs):
            shift = int(srng.integers(len(chunk) // 4, 3 * len(chunk) // 4))
            roll_chunks.append(np.roll(chunk, shift))
        roll_metrics = runner.collect(roll_chunks)

        # Surrogate 3: Reversed
        rev_chunks = [chunk[::-1].copy() for chunk in real_chunks]
        rev_metrics = runner.collect(rev_chunks)

        # Compare using Cohen's d: count how many metrics change significantly
        from scipy import stats as sp_stats
        bonf = 0.05 / len(runner.metric_names)

        def _count_sig(metrics_a, metrics_b):
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

        n_shuf, d_shuf = _count_sig(real_metrics, shuf_metrics)
        n_roll, d_roll = _count_sig(real_metrics, roll_metrics)
        n_rev, d_rev = _count_sig(real_metrics, rev_metrics)

        # Sequential fraction = metrics disrupted by shuffling / total metrics
        seq_frac = n_shuf / len(runner.metric_names)

        results[name] = {
            'shuffled_sig': n_shuf,
            'rolled_sig': n_roll,
            'reversed_sig': n_rev,
            'shuffled_d': d_shuf,
            'rolled_d': d_roll,
            'reversed_d': d_rev,
            'seq_fraction': seq_frac,
        }
        print(f"  {name:20s}  shuf={n_shuf:3d}  roll={n_roll:3d}  "
              f"rev={n_rev:3d}  seq={seq_frac:.2f}")

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

    SCALES = [256, 512, 1024, 2048, 4096]

    # Representative subset — one per major structure type + key exemplars
    SUBSET = [
        ('Logistic Chaos', gen_logistic, 'chaos'),
        ('Lorenz Attractor', gen_lorenz, 'chaos'),
        ('Prime Gaps', gen_prime_gaps, 'number_theory'),
        ('White Noise', gen_white_noise, 'noise'),
        ('Pink Noise', gen_pink_noise, 'noise'),
        ('Brownian Walk', gen_brownian, 'noise'),
        ('Bearing Ball', gen_bearing_ball, 'bearing'),
        ('ECG Normal', gen_ecg_normal, 'medical'),
        ('EEG Seizure', gen_eeg_seizure, 'medical'),
        ('LIGO Hanford', gen_gw_hanford, 'astro'),
        ('Temperature', gen_climate_temp, 'climate'),
        ('DNA Human', gen_dna_human, 'bio'),
        ('Accel Walk', gen_motion_walk, 'motion'),
        ('Speech "Five"', gen_speech_five, 'speech'),
        ('NYSE Returns', gen_stock_sp500, 'financial'),
    ]

    from exotic_geometry_framework import GeometryAnalyzer
    results = {}  # name -> {scale: n_sig}

    for name, gen_fn, domain in SUBSET:
        results[name] = {}
        for scale in SCALES:
            # Build a temporary analyzer for this scale
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', '.cache')
            analyzer = GeometryAnalyzer(cache_dir=cache_dir).add_all_geometries()
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

    ax3.set_title("Cosine Distance (Hierarchical Order)", fontsize=11,
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
    ax4.set_xlabel("Cosine Distance (Ward)", fontsize=8, color='#cccccc')
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

    for name in scale_names:
        vals = [scale_results[name].get(s, 0) for s in scales]
        src = next((s for s in SOURCES if s[0] == name), None)
        domain = src[2] if src else 'noise'
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


# ==============================================================
# MAIN
# ==============================================================

def main():
    runner = Runner("Structure Atlas", mode="1d", data_size=DATA_SIZE,
                     cache=True, n_workers=4)

    profiles, profiles_std, domains = direction_1(runner)
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


if __name__ == "__main__":
    main()
