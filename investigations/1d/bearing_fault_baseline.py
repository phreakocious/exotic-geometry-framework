#!/usr/bin/env python3
"""
Bearing Fault Baseline: Standard Signal Processing Features
============================================================

Same data, same statistics, but using the features that the bearing fault
diagnosis literature actually uses:

1. Time-domain: RMS, kurtosis, crest factor, skewness, peak-to-peak, shape factor
2. Frequency-domain: spectral centroid, spectral spread, spectral kurtosis,
   dominant frequency, band energy ratios
3. Envelope analysis: Hilbert demodulation RMS, kurtosis (industry standard
   for bearing fault detection — BPFO/BPFI show up here)
4. Wavelet energy: db4 decomposition energy per level

This is the fair comparison: same Cohen's d + Bonferroni pipeline, same
N_TRIALS=25, same chunk size, same data.
"""

import sys
import os
import numpy as np
from scipy.io import loadmat
from scipy import signal as sp_signal
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# ==============================================================
# CONFIG
# ==============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'cwru', 'raw')
DATA_SIZE = 2000
N_TRIALS = 25
ALPHA = 0.05

FILES = [
    ('Time_Normal_1_098.mat', 'X098_DE_time', 'Normal', 'normal', 0),
    ('B007_1_123.mat', 'X123_DE_time', 'Ball-7', 'ball', 7),
    ('B014_1_190.mat', 'X190_DE_time', 'Ball-14', 'ball', 14),
    ('B021_1_227.mat', 'X227_DE_time', 'Ball-21', 'ball', 21),
    ('IR007_1_110.mat', 'X110_DE_time', 'Inner-7', 'inner', 7),
    ('IR014_1_175.mat', 'X175_DE_time', 'Inner-14', 'inner', 14),
    ('IR021_1_214.mat', 'X214_DE_time', 'Inner-21', 'inner', 21),
    ('OR007_6_1_136.mat', 'X136_DE_time', 'Outer-7', 'outer', 7),
    ('OR014_6_1_202.mat', 'X202_DE_time', 'Outer-14', 'outer', 14),
    ('OR021_6_1_239.mat', 'X239_DE_time', 'Outer-21', 'outer', 21),
]

# ==============================================================
# STANDARD FEATURE EXTRACTION
# ==============================================================

def time_domain_features(x):
    """Classic time-domain vibration features."""
    return {
        'rms': np.sqrt(np.mean(x**2)),
        'kurtosis': sp_stats.kurtosis(x, fisher=True),
        'skewness': sp_stats.skew(x),
        'crest_factor': np.max(np.abs(x)) / np.sqrt(np.mean(x**2)) if np.mean(x**2) > 0 else 0,
        'peak_to_peak': np.max(x) - np.min(x),
        'shape_factor': np.sqrt(np.mean(x**2)) / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0,
        'impulse_factor': np.max(np.abs(x)) / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0,
        'margin_factor': np.max(np.abs(x)) / (np.mean(np.sqrt(np.abs(x))))**2 if np.mean(np.sqrt(np.abs(x))) > 0 else 0,
        'std': np.std(x),
        'variance': np.var(x),
    }


def frequency_domain_features(x, fs=48000):
    """Standard spectral features."""
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    psd = np.abs(np.fft.rfft(x))**2
    psd_norm = psd / (np.sum(psd) + 1e-15)

    centroid = np.sum(freqs * psd_norm)
    spread = np.sqrt(np.sum((freqs - centroid)**2 * psd_norm))

    # Spectral kurtosis (4th moment)
    spec_kurt = np.sum((freqs - centroid)**4 * psd_norm) / (spread**4 + 1e-15)

    # Band energies (divide spectrum into 4 bands)
    n_bins = len(freqs)
    q = n_bins // 4
    band_energies = [np.sum(psd[i*q:(i+1)*q]) for i in range(4)]
    total_e = sum(band_energies) + 1e-15

    # Dominant frequency
    dom_freq = freqs[np.argmax(psd[1:]) + 1]

    # Spectral entropy
    spec_ent = -np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0]))

    return {
        'spectral_centroid': centroid,
        'spectral_spread': spread,
        'spectral_kurtosis': spec_kurt,
        'dominant_freq': dom_freq,
        'spectral_entropy': spec_ent,
        'band_energy_ratio_1': band_energies[0] / total_e,
        'band_energy_ratio_2': band_energies[1] / total_e,
        'band_energy_ratio_3': band_energies[2] / total_e,
        'band_energy_ratio_4': band_energies[3] / total_e,
    }


def envelope_features(x, fs=48000):
    """Envelope analysis via Hilbert transform — the industry standard
    for bearing fault detection."""
    analytic = sp_signal.hilbert(x)
    envelope = np.abs(analytic)

    # Envelope spectrum
    env_freqs = np.fft.rfftfreq(len(envelope), 1/fs)
    env_psd = np.abs(np.fft.rfft(envelope - np.mean(envelope)))**2
    env_psd_norm = env_psd / (np.sum(env_psd) + 1e-15)

    env_centroid = np.sum(env_freqs * env_psd_norm)
    env_dom = env_freqs[np.argmax(env_psd[1:]) + 1]

    return {
        'envelope_rms': np.sqrt(np.mean(envelope**2)),
        'envelope_kurtosis': sp_stats.kurtosis(envelope, fisher=True),
        'envelope_skewness': sp_stats.skew(envelope),
        'envelope_crest': np.max(envelope) / np.sqrt(np.mean(envelope**2)) if np.mean(envelope**2) > 0 else 0,
        'envelope_centroid': env_centroid,
        'envelope_dom_freq': env_dom,
        'envelope_entropy': -np.sum(env_psd_norm[env_psd_norm > 0] *
                                     np.log2(env_psd_norm[env_psd_norm > 0])),
    }


def wavelet_features(x):
    """Wavelet decomposition energy (db4, 5 levels)."""
    try:
        import pywt
        coeffs = pywt.wavedec(x, 'db4', level=5)
        energies = [np.sum(c**2) for c in coeffs]
        total = sum(energies) + 1e-15
        return {f'wavelet_energy_L{i}': e / total
                for i, e in enumerate(energies)}
    except ImportError:
        # Fallback: simple multi-scale energy via decimation
        features = {}
        current = x.copy()
        for level in range(5):
            # Low-pass decimation approximation
            if len(current) < 4:
                break
            b, a = sp_signal.butter(4, 0.5)
            filtered = sp_signal.filtfilt(b, a, current)
            detail = current[:len(filtered)] - filtered
            features[f'multiscale_energy_L{level}'] = np.sum(detail**2)
            current = filtered[::2]
        total = sum(features.values()) + 1e-15
        return {k: v / total for k, v in features.items()}


def extract_all_features(x):
    """Extract all standard features from a raw vibration chunk."""
    features = {}
    features.update(time_domain_features(x))
    features.update(frequency_domain_features(x))
    features.update(envelope_features(x))
    features.update(wavelet_features(x))
    return features


# ==============================================================
# DATA + STATS (identical pipeline to exotic geometry)
# ==============================================================

def load_signal(filename, key):
    path = os.path.join(DATA_DIR, filename)
    return loadmat(path)[key].flatten()


def get_chunks(signal, rng, n_trials=N_TRIALS, chunk_size=DATA_SIZE):
    """Extract random non-overlapping chunks (RAW float, not uint8)."""
    n_available = len(signal) // chunk_size
    indices = rng.choice(n_available, size=min(n_trials, n_available), replace=False)
    chunks = []
    for idx in indices:
        start = idx * chunk_size
        chunks.append(signal[start:start + chunk_size])
    return chunks


def collect_features(chunks):
    """Extract features from all chunks, return {metric: [values]}."""
    all_features = [extract_all_features(c) for c in chunks]
    keys = sorted(all_features[0].keys())
    return {k: [f[k] for f in all_features] for k in keys}


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps


def compare(data_a, data_b, metric_names, bonf_alpha):
    sig = 0
    findings = []
    for m in metric_names:
        a = np.array(data_a.get(m, []))
        b = np.array(data_b.get(m, []))
        if len(a) < 3 or len(b) < 3:
            continue
        d = cohens_d(a, b)
        if not np.isfinite(d):
            continue
        _, p = sp_stats.ttest_ind(a, b, equal_var=False)
        if p < bonf_alpha and abs(d) > 0.8:
            sig += 1
            findings.append((m, d, p))
    findings.sort(key=lambda x: -abs(x[1]))
    return sig, findings


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("BASELINE: Standard Signal Processing Features")
    print("=" * 60)

    # Discover metric names from a dummy extraction
    dummy = extract_all_features(np.random.randn(DATA_SIZE))
    metric_names = sorted(dummy.keys())
    n_metrics = len(metric_names)
    bonf_alpha = ALPHA / n_metrics
    print(f"  Features: {n_metrics} (Bonferroni alpha = {bonf_alpha:.1e})")

    # Load all conditions
    rng = np.random.default_rng(42)
    all_metrics = {}
    for fname, de_key, label, ftype, sev in FILES:
        signal = load_signal(fname, de_key)
        chunks = get_chunks(signal, rng)
        all_metrics[label] = collect_features(chunks)
        print(f"  {label}: {len(chunks)} chunks")

    # ---- D1: Fault Type Detection (pooled) ----
    print("\n" + "=" * 60)
    print("D1: FAULT TYPE DETECTION")
    print("=" * 60)

    type_groups = {
        'Normal': ['Normal'],
        'Ball': ['Ball-7', 'Ball-14', 'Ball-21'],
        'Inner': ['Inner-7', 'Inner-14', 'Inner-21'],
        'Outer': ['Outer-7', 'Outer-14', 'Outer-21'],
    }

    pooled = {}
    for tname, labels in type_groups.items():
        combined = {m: [] for m in metric_names}
        for lab in labels:
            for m in metric_names:
                combined[m].extend(all_metrics[lab].get(m, []))
        # Trim to 25 per metric
        pooled[tname] = {m: combined[m][:N_TRIALS] for m in metric_names}

    type_names = ['Normal', 'Ball', 'Inner', 'Outer']
    print(f"\n  Pairwise comparisons:")
    for i in range(len(type_names)):
        for j in range(i + 1, len(type_names)):
            n_sig, findings = compare(pooled[type_names[i]], pooled[type_names[j]],
                                       metric_names, bonf_alpha)
            top = f"  top: {findings[0][0]} d={findings[0][1]:+.1f}" if findings else ""
            print(f"  {type_names[i]:8s} vs {type_names[j]:8s} = {n_sig:3d} sig{top}")

    # ---- D2: Severity Grading ----
    print("\n" + "=" * 60)
    print("D2: SEVERITY GRADING")
    print("=" * 60)

    for fault_type in ['ball', 'inner', 'outer']:
        group = [(f[2], f[4]) for f in FILES if f[3] == fault_type]
        print(f"\n  {fault_type.upper()} fault severity:")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                n_sig, findings = compare(all_metrics[group[i][0]],
                                           all_metrics[group[j][0]],
                                           metric_names, bonf_alpha)
                top = f"  top: {findings[0][0]} d={findings[0][1]:+.1f}" if findings else ""
                print(f"    {group[i][1]}mil vs {group[j][1]}mil: {n_sig:3d} sig{top}")

    # ---- D3: Early Fault Detection ----
    print("\n" + "=" * 60)
    print("D3: EARLY FAULT DETECTION (Normal vs 0.007\")")
    print("=" * 60)

    for f in FILES:
        if f[4] == 7:
            n_sig, findings = compare(all_metrics['Normal'], all_metrics[f[2]],
                                       metric_names, bonf_alpha)
            top = findings[0] if findings else None
            print(f"  Normal vs {f[2]:10s}: {n_sig:3d} sig", end="")
            if top:
                print(f"  Best: {top[0]} (d={top[1]:.2f})")
            else:
                print()

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_metrics} standard features vs 141 exotic geometry metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
