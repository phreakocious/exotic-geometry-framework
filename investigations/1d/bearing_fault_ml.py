#!/usr/bin/env python3
"""
Bearing Fault ML Comparison: Standard vs Exotic vs Combined
============================================================

Quick sklearn comparison: do exotic geometry features add discriminative
information on top of standard signal processing features?

3 feature sets × 2 tasks × 5-fold stratified CV:
  - Standard (31 features): RMS, kurtosis, envelope, spectral, wavelet
  - Exotic (141 features): all 1D exotic geometry metrics
  - Combined (172 features): both

Tasks:
  - 4-class fault type (Normal/Ball/Inner/Outer — pooled severity)
  - 10-class full (Normal + 3 types × 3 severities)
"""

import sys
import os
import numpy as np
from scipy.io import loadmat
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from exotic_geometry_framework import GeometryAnalyzer

# ==============================================================
# CONFIG
# ==============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'cwru', 'raw')
DATA_SIZE = 2000
N_CHUNKS = 50  # per recording — more samples for ML

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
# STANDARD FEATURES (same as baseline)
# ==============================================================

def time_domain_features(x):
    return {
        'rms': np.sqrt(np.mean(x**2)),
        'kurtosis': sp_stats.kurtosis(x, fisher=True),
        'skewness': sp_stats.skew(x),
        'crest_factor': np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-15),
        'peak_to_peak': np.max(x) - np.min(x),
        'shape_factor': np.sqrt(np.mean(x**2)) / (np.mean(np.abs(x)) + 1e-15),
        'impulse_factor': np.max(np.abs(x)) / (np.mean(np.abs(x)) + 1e-15),
        'margin_factor': np.max(np.abs(x)) / (np.mean(np.sqrt(np.abs(x)))**2 + 1e-15),
        'std': np.std(x),
        'variance': np.var(x),
    }

def frequency_domain_features(x, fs=48000):
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    psd = np.abs(np.fft.rfft(x))**2
    psd_norm = psd / (np.sum(psd) + 1e-15)
    centroid = np.sum(freqs * psd_norm)
    spread = np.sqrt(np.sum((freqs - centroid)**2 * psd_norm))
    spec_kurt = np.sum((freqs - centroid)**4 * psd_norm) / (spread**4 + 1e-15)
    n_bins = len(freqs)
    q = n_bins // 4
    band_energies = [np.sum(psd[i*q:(i+1)*q]) for i in range(4)]
    total_e = sum(band_energies) + 1e-15
    dom_freq = freqs[np.argmax(psd[1:]) + 1]
    spec_ent = -np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0]))
    return {
        'spectral_centroid': centroid, 'spectral_spread': spread,
        'spectral_kurtosis': spec_kurt, 'dominant_freq': dom_freq,
        'spectral_entropy': spec_ent,
        'band_energy_ratio_1': band_energies[0]/total_e,
        'band_energy_ratio_2': band_energies[1]/total_e,
        'band_energy_ratio_3': band_energies[2]/total_e,
        'band_energy_ratio_4': band_energies[3]/total_e,
    }

def envelope_features(x, fs=48000):
    analytic = sp_signal.hilbert(x)
    envelope = np.abs(analytic)
    env_freqs = np.fft.rfftfreq(len(envelope), 1/fs)
    env_psd = np.abs(np.fft.rfft(envelope - np.mean(envelope)))**2
    env_psd_norm = env_psd / (np.sum(env_psd) + 1e-15)
    env_centroid = np.sum(env_freqs * env_psd_norm)
    env_dom = env_freqs[np.argmax(env_psd[1:]) + 1]
    return {
        'envelope_rms': np.sqrt(np.mean(envelope**2)),
        'envelope_kurtosis': sp_stats.kurtosis(envelope, fisher=True),
        'envelope_skewness': sp_stats.skew(envelope),
        'envelope_crest': np.max(envelope) / (np.sqrt(np.mean(envelope**2)) + 1e-15),
        'envelope_centroid': env_centroid, 'envelope_dom_freq': env_dom,
        'envelope_entropy': -np.sum(env_psd_norm[env_psd_norm > 0] *
                                     np.log2(env_psd_norm[env_psd_norm > 0])),
    }

def wavelet_features(x):
    features = {}
    current = x.copy()
    for level in range(5):
        if len(current) < 4:
            break
        b, a = sp_signal.butter(4, 0.5)
        filtered = sp_signal.filtfilt(b, a, current)
        detail = current[:len(filtered)] - filtered
        features[f'multiscale_energy_L{level}'] = np.sum(detail**2)
        current = filtered[::2]
    total = sum(features.values()) + 1e-15
    return {k: v / total for k, v in features.items()}

def extract_standard(x):
    f = {}
    f.update(time_domain_features(x))
    f.update(frequency_domain_features(x))
    f.update(envelope_features(x))
    f.update(wavelet_features(x))
    return f

# ==============================================================
# EXOTIC GEOMETRY FEATURES
# ==============================================================

_analyzer = GeometryAnalyzer().add_all_geometries()

def extract_exotic(x_uint8):
    """Extract exotic geometry features from uint8 chunk."""
    result = _analyzer.analyze(x_uint8)
    features = {}
    for r in result.results:
        for mn, mv in r.metrics.items():
            features[f"{r.geometry_name}:{mn}"] = mv if np.isfinite(mv) else 0.0
    return features

# ==============================================================
# DATA LOADING
# ==============================================================

def load_all():
    """Load all data, extract both feature sets."""
    rng = np.random.default_rng(42)

    std_features = []
    exo_features = []
    labels_10 = []  # 10-class
    labels_4 = []   # 4-class fault type

    for fname, de_key, label, ftype, sev in FILES:
        path = os.path.join(DATA_DIR, fname)
        signal = loadmat(path)[de_key].flatten()

        n_available = len(signal) // DATA_SIZE
        indices = rng.choice(n_available, size=min(N_CHUNKS, n_available),
                             replace=False)

        for idx in indices:
            start = idx * DATA_SIZE
            chunk_raw = signal[start:start + DATA_SIZE]

            # Standard features on raw float
            std_features.append(extract_standard(chunk_raw))

            # Exotic on uint8
            lo, hi = chunk_raw.min(), chunk_raw.max()
            if hi - lo < 1e-15:
                hi = lo + 1.0
            chunk_u8 = ((chunk_raw - lo) / (hi - lo) * 255).astype(np.uint8)
            exo_features.append(extract_exotic(chunk_u8))

            labels_10.append(label)
            labels_4.append(ftype)

        print(f"  {label}: {min(N_CHUNKS, n_available)} chunks extracted")

    # Convert to arrays
    std_keys = sorted(std_features[0].keys())
    exo_keys = sorted(exo_features[0].keys())

    X_std = np.array([[f[k] for k in std_keys] for f in std_features])
    X_exo = np.array([[f.get(k, 0.0) for k in exo_keys] for f in exo_features])
    X_both = np.hstack([X_std, X_exo])

    y_10 = np.array(labels_10)
    y_4 = np.array(labels_4)

    # Clean NaN/inf
    X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_exo = np.nan_to_num(X_exo, nan=0.0, posinf=0.0, neginf=0.0)
    X_both = np.nan_to_num(X_both, nan=0.0, posinf=0.0, neginf=0.0)

    return X_std, X_exo, X_both, y_10, y_4, std_keys, exo_keys

# ==============================================================
# ML EVALUATION
# ==============================================================

def evaluate(X, y, name, n_splits=5):
    """Stratified k-fold CV with Random Forest."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42,
                                       n_jobs=-1, max_depth=None))
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"  {name:20s}: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%  "
          f"(folds: {', '.join(f'{s*100:.0f}%' for s in scores)})")
    return scores.mean()


def main():
    print("=" * 65)
    print("BEARING FAULT ML: Standard vs Exotic vs Combined")
    print("=" * 65)
    print(f"  Chunks per recording: {N_CHUNKS}")
    print(f"  Chunk size: {DATA_SIZE}")
    print()

    X_std, X_exo, X_both, y_10, y_4, std_keys, exo_keys = load_all()

    print(f"\n  Samples: {len(y_10)}")
    print(f"  Standard features: {X_std.shape[1]}")
    print(f"  Exotic features: {X_exo.shape[1]}")
    print(f"  Combined features: {X_both.shape[1]}")

    # ---- 4-class fault type ----
    print(f"\n{'='*65}")
    print("TASK 1: 4-CLASS FAULT TYPE (Normal / Ball / Inner / Outer)")
    print(f"{'='*65}")
    s4_std = evaluate(X_std, y_4, "Standard (31)")
    s4_exo = evaluate(X_exo, y_4, "Exotic (141)")
    s4_both = evaluate(X_both, y_4, "Combined (172)")

    # ---- 10-class full ----
    print(f"\n{'='*65}")
    print("TASK 2: 10-CLASS FULL (type + severity)")
    print(f"{'='*65}")
    s10_std = evaluate(X_std, y_10, "Standard (31)")
    s10_exo = evaluate(X_exo, y_10, "Exotic (141)")
    s10_both = evaluate(X_both, y_10, "Combined (172)")

    # ---- Feature importance (combined model, 10-class) ----
    print(f"\n{'='*65}")
    print("TOP 15 FEATURES (Combined RF, 10-class)")
    print(f"{'='*65}")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipe.fit(X_both, y_10)
    importances = pipe.named_steps['clf'].feature_importances_
    all_keys = std_keys + exo_keys
    ranked = sorted(zip(all_keys, importances), key=lambda x: -x[1])
    for i, (name, imp) in enumerate(ranked[:15]):
        source = "STD" if name in std_keys else "EXO"
        print(f"  {i+1:2d}. [{source}] {name:45s} {imp:.4f}")

    # Count std vs exotic in top-N
    for topn in [10, 15, 20, 30]:
        n_std = sum(1 for k, _ in ranked[:topn] if k in std_keys)
        n_exo = topn - n_std
        print(f"  Top {topn}: {n_std} standard, {n_exo} exotic")

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  4-class:  Std={s4_std*100:.1f}%  Exotic={s4_exo*100:.1f}%  Combined={s4_both*100:.1f}%")
    print(f"  10-class: Std={s10_std*100:.1f}%  Exotic={s10_exo*100:.1f}%  Combined={s10_both*100:.1f}%")
    delta_4 = s4_both - s4_std
    delta_10 = s10_both - s10_std
    print(f"  Exotic adds: {delta_4*100:+.1f}pp (4-class), {delta_10*100:+.1f}pp (10-class)")


if __name__ == "__main__":
    main()
