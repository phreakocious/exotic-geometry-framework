"""
3I/ATLAS SETI Investigation
============================
Geometric analysis of GBT radio telescope data from interstellar
object 3I/ATLAS. Compares ON-target vs OFF-target observations
using the exotic geometry framework.

Data: Breakthrough Listen GBT part-0002 (spectral-line product)
Portal: https://bldata.berkeley.edu/ATLAS/GB_ATLAS/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import hashlib
import html.parser
import re
import urllib.request
from collections import defaultdict

import h5py
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'seti_3i')
CACHE_DIR = os.path.join(DATA_DIR, '.cache')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
PORTAL_BASE = "https://bldata.berkeley.edu/ATLAS/GB_ATLAS"
DATA_SIZE = 2000
ALPHA = 0.05
N_SUBBANDS = 4
N_WINDOWS = 4

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def to_uint8(x):
    """Percentile-clip float array to uint8."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi - lo < 1e-10:
        return np.full(len(x), 128, dtype=np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# RFI mitigation
# ---------------------------------------------------------------------------
def sigma_clip_spectrogram(spec, sigma=3.0):
    """Replace pixels > sigma*std from channel median with channel median."""
    cleaned = spec.copy()
    med = np.median(spec, axis=0)
    std = np.std(spec, axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    mask = np.abs(spec - med) > sigma * std
    cleaned[mask] = np.broadcast_to(med, spec.shape)[mask]
    return cleaned

# ---------------------------------------------------------------------------
# 1D Extraction
# ---------------------------------------------------------------------------
def extract_time_series(spec, n_subbands=N_SUBBANDS, target_len=DATA_SIZE):
    """Extract band-integrated time series from spectrogram.

    Parameters
    ----------
    spec : ndarray, shape (n_time, n_freq)
    n_subbands : int -- number of frequency sub-bands
    target_len : int -- minimum required length

    Returns
    -------
    list of uint8 arrays, one per sub-band. Empty if time axis too short.
    """
    n_time, n_freq = spec.shape
    if n_time < target_len:
        return []
    band_width = n_freq // n_subbands
    result = []
    for i in range(n_subbands):
        f_lo = i * band_width
        f_hi = (i + 1) * band_width if i < n_subbands - 1 else n_freq
        ts = spec[:, f_lo:f_hi].sum(axis=1)
        result.append(to_uint8(ts[:target_len]))
    return result


def extract_spectral_profiles(spec, n_windows=N_WINDOWS, target_len=DATA_SIZE):
    """Extract time-integrated spectral profiles from spectrogram.

    Parameters
    ----------
    spec : ndarray, shape (n_time, n_freq)
    n_windows : int -- number of time windows
    target_len : int -- target output length

    Returns
    -------
    list of uint8 arrays, one per time window.
    """
    n_time, n_freq = spec.shape
    win_size = n_time // n_windows
    result = []
    for i in range(n_windows):
        t_lo = i * win_size
        t_hi = (i + 1) * win_size if i < n_windows - 1 else n_time
        profile = spec[t_lo:t_hi, :].sum(axis=0)
        # Rebin to target_len using contiguous block averaging
        if len(profile) >= target_len:
            trim = (len(profile) // target_len) * target_len
            rebinned = profile[:trim].reshape(target_len, -1).mean(axis=1)
            result.append(to_uint8(rebinned))
        else:
            x_old = np.linspace(0, 1, len(profile))
            x_new = np.linspace(0, 1, target_len)
            result.append(to_uint8(np.interp(x_new, x_old, profile)))
    return result

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def cohens_d(a, b):
    """Pooled-std Cohen's d."""
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    ps = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if ps < 1e-15:
        diff = np.mean(a) - np.mean(b)
        return 0.0 if abs(diff) < 1e-15 else np.sign(diff) * float('inf')
    return (np.mean(a) - np.mean(b)) / ps
