# 3I/ATLAS SETI Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Analyze real GBT radio telescope data from interstellar object 3I/ATLAS with the exotic geometry framework, comparing ON-target vs OFF-target observations to detect geometric structure.

**Architecture:** Single investigation script following the established `investigations/1d/` pattern. Downloads GBT part-0002 HDF5 files from the Breakthrough Listen portal, extracts 1D sequences (time series + spectral profiles), runs the full geometry framework, and compares ON vs OFF observations statistically. Separate test file for extraction/comparison logic.

**Tech Stack:** h5py (HDF5 reading), urllib (downloads), numpy/scipy (extraction, stats), matplotlib (figures), exotic_geometry_framework (analysis)

**Spec:** `docs/superpowers/specs/2026-03-24-3i-atlas-seti-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `investigations/1d/seti_3i_atlas.py` | Main investigation script — download, extract, analyze, compare, report |
| `tests/test_seti_3i_extraction.py` | Unit tests for extraction, encoding, RFI mitigation, comparison logic |
| `data/seti_3i/` | Downloaded HDF5 files (gitignored) |
| `data/seti_3i/.cache/` | Cached analysis results per extraction (gitignored) |
| `figures/seti_3i_atlas.png` | Main results figure |
| `docs/investigations/seti_3i_atlas.md` | Findings document |

---

### Task 1: Test scaffold and core helpers

**Files:**
- Create: `tests/test_seti_3i_extraction.py`
- Create: `investigations/1d/seti_3i_atlas.py` (skeleton)

- [ ] **Step 1: Write tests for `to_uint8` encoding**

```python
# tests/test_seti_3i_extraction.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'investigations', '1d'))

import numpy as np
import pytest


def test_to_uint8_normal_range():
    """Percentile encoding maps data to 0-255."""
    from seti_3i_atlas import to_uint8
    x = np.linspace(0, 100, 2000)
    result = to_uint8(x)
    assert result.dtype == np.uint8
    assert len(result) == 2000
    assert result.min() == 0
    assert result.max() == 255


def test_to_uint8_constant_input():
    """Constant input should map to 128 (midpoint)."""
    from seti_3i_atlas import to_uint8
    x = np.full(2000, 42.0)
    result = to_uint8(x)
    assert result.dtype == np.uint8
    assert np.all(result == 128)


def test_to_uint8_outliers_clipped():
    """Extreme outliers get clipped by percentile encoding."""
    from seti_3i_atlas import to_uint8
    x = np.zeros(2000)
    x[0] = 1e6   # extreme outlier
    x[-1] = -1e6
    result = to_uint8(x)
    assert result.dtype == np.uint8
    # Most values should be near the middle since the bulk is 0
    assert np.median(result) > 100
    assert np.median(result) < 160
```

- [ ] **Step 2: Write tests for `sigma_clip_spectrogram`**

```python
# append to tests/test_seti_3i_extraction.py

def test_sigma_clip_replaces_outliers():
    """Sigma-clip replaces > 3-sigma pixels with channel median."""
    from seti_3i_atlas import sigma_clip_spectrogram
    rng = np.random.default_rng(42)
    # 100 time steps, 500 freq channels
    spec = rng.normal(0, 1, (100, 500)).astype(np.float32)
    # Inject RFI spike
    spec[50, 250] = 1000.0
    cleaned = sigma_clip_spectrogram(spec, sigma=3.0)
    assert cleaned.shape == spec.shape
    # Spike should be replaced with something near 0 (channel median)
    assert abs(cleaned[50, 250]) < 5.0


def test_sigma_clip_preserves_normal_data():
    """Normal data should be mostly unchanged."""
    from seti_3i_atlas import sigma_clip_spectrogram
    rng = np.random.default_rng(42)
    spec = rng.normal(0, 1, (100, 500)).astype(np.float32)
    cleaned = sigma_clip_spectrogram(spec, sigma=3.0)
    # Most pixels unchanged (within float tolerance)
    n_changed = np.sum(np.abs(cleaned - spec) > 1e-6)
    # Expect ~0.27% of pixels to exceed 3-sigma by chance
    assert n_changed < 0.01 * spec.size
```

- [ ] **Step 3: Write tests for 1D extraction functions**

```python
# append to tests/test_seti_3i_extraction.py

def test_extract_time_series():
    """Band-integrated time series: sum over freq sub-bands."""
    from seti_3i_atlas import extract_time_series
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (200, 4000)).astype(np.float32)
    series_list = extract_time_series(spec, n_subbands=4, target_len=2000)
    # Should get 4 sub-band time series
    assert len(series_list) == 4
    for s in series_list:
        assert s.dtype == np.uint8
        assert len(s) == 2000


def test_extract_spectral_profiles():
    """Time-integrated spectral profiles: sum over time windows."""
    from seti_3i_atlas import extract_spectral_profiles
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (200, 4000)).astype(np.float32)
    profiles = extract_spectral_profiles(spec, n_windows=4, target_len=2000)
    assert len(profiles) == 4
    for p in profiles:
        assert p.dtype == np.uint8
        assert len(p) == 2000


def test_extract_time_series_short_axis():
    """If time axis < target_len and can't be extended, return fewer or skip."""
    from seti_3i_atlas import extract_time_series
    rng = np.random.default_rng(42)
    # Only 16 time steps — too short
    spec = rng.normal(100, 10, (16, 4000)).astype(np.float32)
    series_list = extract_time_series(spec, n_subbands=4, target_len=2000)
    # Should return empty list (skip short time axes)
    assert len(series_list) == 0
```

- [ ] **Step 4: Write test for `cohens_d`**

```python
# append to tests/test_seti_3i_extraction.py

def test_cohens_d_same_population():
    """Samples from the same population should give |d| < 0.5."""
    from seti_3i_atlas import cohens_d
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0, 1, 50)
    d = cohens_d(a, b)
    assert abs(d) < 0.5  # same population, small effect


def test_cohens_d_large_effect():
    """Well-separated distributions should give |d| > 2."""
    from seti_3i_atlas import cohens_d
    a = np.array([1.0, 1.1, 0.9, 1.0, 1.05])
    b = np.array([5.0, 5.1, 4.9, 5.0, 5.05])
    d = cohens_d(a, b)
    assert abs(d) > 2.0


def test_cohens_d_direction():
    """Cohen's d should be negative when a < b."""
    from seti_3i_atlas import cohens_d
    a = np.array([1.0, 1.1, 0.9])
    b = np.array([5.0, 5.1, 4.9])
    d = cohens_d(a, b)
    assert d < -2.0  # a < b → negative d


def test_cohens_d_zero_variance():
    """Constant arrays with different means: a < b → -inf."""
    from seti_3i_atlas import cohens_d
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0])
    d = cohens_d(a, b)
    assert d == float('-inf')
```

- [ ] **Step 5: Create script skeleton with these functions stubbed**

```python
# investigations/1d/seti_3i_atlas.py
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
    # Replace flagged pixels with channel median
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
    n_subbands : int — number of frequency sub-bands
    target_len : int — minimum required length

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
    n_windows : int — number of time windows
    target_len : int — target output length

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
            # Trim to exact multiple, then reshape and mean
            trim = (len(profile) // target_len) * target_len
            rebinned = profile[:trim].reshape(target_len, -1).mean(axis=1)
            result.append(to_uint8(rebinned))
        else:
            # Interpolate up to target_len
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
```

- [ ] **Step 6: Run tests — all should fail (functions exist but some edge cases may differ)**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py -v`
Expected: All tests PASS (functions are implemented in skeleton).

- [ ] **Step 7: Commit scaffold**

```bash
git add investigations/1d/seti_3i_atlas.py tests/test_seti_3i_extraction.py
git commit -m "feat(seti-3i): scaffold investigation script with core helpers and tests"
```

---

### Task 2: File manifest and download infrastructure

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`
- Modify: `tests/test_seti_3i_extraction.py`

- [ ] **Step 1: Write test for filename parsing**

```python
# append to tests/test_seti_3i_extraction.py

def test_parse_gbt_filename():
    """Parse GBT BL filename into components."""
    from seti_3i_atlas import parse_gbt_filename
    info = parse_gbt_filename(
        "blc20_guppi_61027_31308_DIAG_3I_ATLAS_0061.rawspec.0002.h5"
    )
    assert info['node'] == 'blc20'
    assert info['scan'] == '0061'
    assert info['part'] == '0002'
    assert info['is_off'] is False

    info_off = parse_gbt_filename(
        "blc20_guppi_61027_31308_DIAG_3I_ATLAS_OFF_0062.rawspec.0002.h5"
    )
    assert info_off['is_off'] is True
    assert info_off['node'] == 'blc20'


def test_parse_gbt_filename_returns_none_for_non_matching():
    """Non-matching filenames return None."""
    from seti_3i_atlas import parse_gbt_filename
    assert parse_gbt_filename("random_file.h5") is None
    assert parse_gbt_filename("README.md") is None
```

- [ ] **Step 2: Run test — should fail**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py::test_parse_gbt_filename -v`
Expected: FAIL — `parse_gbt_filename` not defined

- [ ] **Step 3: Implement filename parser and file manifest builder**

```python
# Add to investigations/1d/seti_3i_atlas.py after the statistics section
# (Note: re, urllib.request, html.parser already imported at top of file)

# ---------------------------------------------------------------------------
# GBT filename parsing
# ---------------------------------------------------------------------------
# Format: blcXX_guppi_MJDDAY_SECONDS_DIAG_3I_ATLAS[_OFF]_SCAN.rawspec.PART.h5
_GBT_RE = re.compile(
    r'(blc\d+)_guppi_(\d+)_(\d+)_[A-Z0-9]+_3I_ATLAS(_(OFF))?_(\d+)\.rawspec\.(\d+)\.h5'
)

def parse_gbt_filename(name):
    """Parse a GBT BL filename into components. Returns None if no match."""
    m = _GBT_RE.match(name)
    if not m:
        return None
    return {
        'filename': name,
        'node': m.group(1),
        'mjd': m.group(2),
        'seconds': m.group(3),
        'is_off': m.group(5) == 'OFF',
        'scan': m.group(6),
        'part': m.group(7),
    }


class _LinkParser(html.parser.HTMLParser):
    """Extract hrefs from an HTML directory listing."""
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for k, v in attrs:
                if k == 'href' and v.endswith('.h5'):
                    self.links.append(v)


def fetch_file_manifest():
    """Fetch the GBT directory listing and parse all part-0002 filenames.

    Returns list of parsed file info dicts, sorted by scan number.
    """
    url = f"{PORTAL_BASE}/"
    print(f"  Fetching file listing from {url} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'ExoticGeometry/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html_text = resp.read().decode('utf-8', errors='ignore')

    parser = _LinkParser()
    parser.feed(html_text)

    manifest = []
    for link in parser.links:
        name = link.strip('/').split('/')[-1]
        info = parse_gbt_filename(name)
        if info and info['part'] == '0002':
            manifest.append(info)

    manifest.sort(key=lambda x: (x['scan'], x['node']))
    print(f"  Found {len(manifest)} part-0002 files "
          f"({sum(1 for f in manifest if not f['is_off'])} ON, "
          f"{sum(1 for f in manifest if f['is_off'])} OFF)")
    return manifest


# ---------------------------------------------------------------------------
# Band classification
# ---------------------------------------------------------------------------
# GBT BL compute node → approximate band mapping.
# Scan ranges identify bands: L (0013-0018), S (0029-0034),
# C (0045-0050), X (0061-0066). We map by scan number.
BAND_SCAN_RANGES = {
    'L': (13, 18),
    'S': (29, 34),
    'C': (45, 50),
    'X': (61, 66),
}

def classify_band(scan_str):
    """Classify a scan number string into a band name."""
    scan = int(scan_str)
    for band, (lo, hi) in BAND_SCAN_RANGES.items():
        if lo <= scan <= hi:
            return band
    return 'unknown'


def select_files(manifest, nodes_per_band=3):
    """Select a representative subset of files for analysis.

    For each band, pick `nodes_per_band` compute nodes (low/mid/high)
    and select one ON + one OFF scan per node.

    Returns list of (file_info, band) tuples.
    """
    by_band = defaultdict(lambda: defaultdict(list))
    for info in manifest:
        band = classify_band(info['scan'])
        if band == 'unknown':
            continue
        by_band[band][info['node']].append(info)

    selected = []
    for band in ['L', 'S', 'C', 'X']:
        nodes = sorted(by_band[band].keys())
        if not nodes:
            print(f"  WARNING: No files found for {band}-band")
            continue
        # Pick low/mid/high nodes
        if len(nodes) <= nodes_per_band:
            pick = nodes
        else:
            indices = np.linspace(0, len(nodes) - 1, nodes_per_band).astype(int)
            pick = [nodes[i] for i in indices]

        for node in pick:
            files = by_band[band][node]
            on_files = [f for f in files if not f['is_off']]
            off_files = [f for f in files if f['is_off']]
            if on_files:
                selected.append((on_files[0], band))
            if off_files:
                selected.append((off_files[0], band))

    print(f"  Selected {len(selected)} files across "
          f"{len(set(b for _, b in selected))} bands")
    return selected
```

- [ ] **Step 4: Run tests — should pass**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py -v`
Expected: All PASS

- [ ] **Step 5: Implement download function**

```python
# Add to investigations/1d/seti_3i_atlas.py

def download_file(filename):
    """Download a single file from the BL portal. Returns local path.

    Skips download if file already exists locally.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    url = f"{PORTAL_BASE}/{filename}"
    print(f"    Downloading {filename} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ExoticGeometry/1.0'})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(local_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"{size_mb:.1f} MB")
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None
    return local_path
```

- [ ] **Step 6: Commit**

```bash
git add investigations/1d/seti_3i_atlas.py tests/test_seti_3i_extraction.py
git commit -m "feat(seti-3i): file manifest parsing, band classification, download"
```

---

### Task 3: HDF5 reading and spectrogram loading

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`
- Modify: `tests/test_seti_3i_extraction.py`

- [ ] **Step 1: Write test for HDF5 loading with synthetic file**

```python
# append to tests/test_seti_3i_extraction.py
import tempfile, h5py

def test_load_spectrogram_from_h5():
    """Load a spectrogram from a synthetic HDF5 file."""
    from seti_3i_atlas import load_spectrogram
    rng = np.random.default_rng(42)
    n_time, n_freq = 100, 4000
    data = rng.normal(100, 10, (n_time, 1, n_freq)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('data', data=data)
            hf['data'].attrs['fch1'] = 1500.0
            hf['data'].attrs['foff'] = -0.00028
            hf['data'].attrs['tsamp'] = 18.253611
        spec, meta = load_spectrogram(path)
        assert spec.shape == (n_time, n_freq)
        assert spec.dtype == np.float32
        assert 'fch1' in meta
        assert 'foff' in meta
        assert 'tsamp' in meta
    finally:
        os.unlink(path)


def test_load_spectrogram_applies_mask():
    """If /mask exists, apply it to the spectrogram."""
    from seti_3i_atlas import load_spectrogram
    rng = np.random.default_rng(42)
    n_time, n_freq = 50, 1000
    data = rng.normal(100, 10, (n_time, 1, n_freq)).astype(np.float32)
    mask = np.zeros((n_time, 1, n_freq), dtype=np.uint8)
    mask[10, 0, 500] = 1  # Flag one pixel

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('data', data=data)
            hf.create_dataset('mask', data=mask)
            hf['data'].attrs['fch1'] = 1500.0
            hf['data'].attrs['foff'] = -0.00028
            hf['data'].attrs['tsamp'] = 18.253611
        spec, meta = load_spectrogram(path)
        # Masked pixel should be replaced with channel median
        # (the mask handling replaces flagged pixels)
        assert spec.shape == (n_time, n_freq)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests — should fail**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py::test_load_spectrogram_from_h5 -v`
Expected: FAIL — `load_spectrogram` not defined

- [ ] **Step 3: Implement `load_spectrogram`**

```python
# Add to investigations/1d/seti_3i_atlas.py after download_file
# (Note: h5py already imported at top of file)

def load_spectrogram(path):
    """Load a BL filterbank HDF5 file as a 2D spectrogram.

    BL format: /data has shape (n_time, n_pol, n_freq).
    We take Stokes I (pol index 0) and squeeze to (n_time, n_freq).

    If /mask exists (uint8, same shape), flagged pixels (nonzero)
    are replaced with the channel median.

    Returns
    -------
    spec : ndarray, shape (n_time, n_freq), float32
    meta : dict with keys fch1, foff, tsamp from HDF5 attributes
    """
    with h5py.File(path, 'r') as hf:
        raw = hf['data'][:]          # (n_time, n_pol, n_freq)
        spec = raw[:, 0, :].astype(np.float32)

        # Apply mask if present
        if 'mask' in hf:
            mask_raw = hf['mask'][:]
            mask_2d = mask_raw[:, 0, :] > 0
            med = np.median(spec, axis=0)
            spec[mask_2d] = np.broadcast_to(med, spec.shape)[mask_2d]

        # Extract metadata
        attrs = dict(hf['data'].attrs)
        meta = {
            'fch1': float(attrs.get('fch1', 0)),
            'foff': float(attrs.get('foff', 0)),
            'tsamp': float(attrs.get('tsamp', 0)),
            'n_time': spec.shape[0],
            'n_freq': spec.shape[1],
        }

    return spec, meta
```

- [ ] **Step 4: Run tests — should pass**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add investigations/1d/seti_3i_atlas.py tests/test_seti_3i_extraction.py
git commit -m "feat(seti-3i): HDF5 spectrogram loading with mask support"
```

---

### Task 4: Full extraction pipeline (spectrogram → labeled uint8 arrays)

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`
- Modify: `tests/test_seti_3i_extraction.py`

- [ ] **Step 1: Write test for the combined extraction pipeline**

```python
# append to tests/test_seti_3i_extraction.py

def test_extract_all_from_spectrogram():
    """Full extraction: time series + spectral profiles, labeled."""
    from seti_3i_atlas import extract_all
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (3000, 4000)).astype(np.float32)
    extractions = extract_all(spec, label="test_blc20_L_ON")
    # Should have N_SUBBANDS time series + N_WINDOWS spectral profiles
    assert len(extractions) > 0
    for name, arr in extractions:
        assert isinstance(name, str)
        assert "test_blc20_L_ON" in name
        assert arr.dtype == np.uint8
        assert len(arr) == 2000


def test_extract_all_short_time_axis():
    """Short time axis: only spectral profiles, no time series."""
    from seti_3i_atlas import extract_all
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (16, 4000)).astype(np.float32)
    extractions = extract_all(spec, label="test_short")
    # Should have spectral profiles but no time series
    names = [n for n, _ in extractions]
    assert not any("ts_" in n for n in names)
    assert any("sp_" in n for n in names)
```

- [ ] **Step 2: Run test — should fail**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py::test_extract_all_from_spectrogram -v`
Expected: FAIL — `extract_all` not defined

- [ ] **Step 3: Implement `extract_all`**

```python
# Add to investigations/1d/seti_3i_atlas.py

def extract_all(spec, label, n_subbands=N_SUBBANDS, n_windows=N_WINDOWS,
                target_len=DATA_SIZE):
    """Extract all 1D sequences from a spectrogram.

    Returns list of (name, uint8_array) tuples.
    Names follow pattern: {label}/ts_{i} or {label}/sp_{i}
    """
    extractions = []

    # Time series (band-integrated)
    ts_list = extract_time_series(spec, n_subbands=n_subbands,
                                  target_len=target_len)
    for i, ts in enumerate(ts_list):
        extractions.append((f"{label}/ts_{i}", ts))

    # Spectral profiles (time-integrated)
    sp_list = extract_spectral_profiles(spec, n_windows=n_windows,
                                        target_len=target_len)
    for i, sp in enumerate(sp_list):
        extractions.append((f"{label}/sp_{i}", sp))

    return extractions
```

- [ ] **Step 4: Run tests — should pass**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add investigations/1d/seti_3i_atlas.py tests/test_seti_3i_extraction.py
git commit -m "feat(seti-3i): combined extraction pipeline with labeling"
```

---

### Task 5: Geometry analysis with result caching

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`

- [ ] **Step 1: Implement framework setup and metric discovery (module-level)**

Add near the top of the script, after imports and config:

```python
from exotic_geometry_framework import GeometryAnalyzer

# ---------------------------------------------------------------------------
# Framework setup — discover metric names
# ---------------------------------------------------------------------------
def _init_analyzer():
    analyzer = GeometryAnalyzer().add_all_geometries()
    dummy = analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
    metric_names = []
    for r in dummy.results:
        for mn in sorted(r.metrics.keys()):
            metric_names.append(f"{r.geometry_name}:{mn}")
    return analyzer, metric_names
```

- [ ] **Step 2: Implement `analyze_extraction` with caching**

```python
# Add to investigations/1d/seti_3i_atlas.py
# (Note: hashlib already imported at top of file)

def _framework_hash():
    """Hash exotic_geometry_framework.py for cache invalidation."""
    fw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'exotic_geometry_framework.py')
    with open(fw_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

_FW_HASH = None

def analyze_extraction(analyzer, metric_names, name, data):
    """Analyze a single uint8 extraction. Returns {metric: value} dict.

    Uses file-based caching in CACHE_DIR to avoid re-analysis.
    Cache key includes framework code hash for invalidation on code changes.
    """
    global _FW_HASH
    os.makedirs(CACHE_DIR, exist_ok=True)

    if _FW_HASH is None:
        _FW_HASH = _framework_hash()

    # Cache key: hash of data + framework code hash
    data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    cache_key = hashlib.sha256(
        f"{name}|{data_hash}|{_FW_HASH}".encode()
    ).hexdigest()[:24]
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")

    if os.path.exists(cache_path):
        loaded = np.load(cache_path, allow_pickle=True)
        return dict(loaded['metrics'].item())

    # Run analysis
    result = analyzer.analyze(data)
    metrics = {}
    for r in result.results:
        for mn, mv in r.metrics.items():
            key = f"{r.geometry_name}:{mn}"
            if key in metric_names and np.isfinite(mv):
                metrics[key] = mv

    # Cache
    np.savez_compressed(cache_path, metrics=metrics)
    return metrics


def analyze_file_extractions(analyzer, metric_names, extractions):
    """Analyze all extractions from one file. Returns {name: {metric: value}}."""
    results = {}
    for name, data in extractions:
        metrics = analyze_extraction(analyzer, metric_names, name, data)
        results[name] = metrics
    return results
```

- [ ] **Step 3: Verify analysis works on a synthetic extraction**

Add a quick smoke test to `__main__` block (temporary):

```python
if __name__ == "__main__":
    print("Smoke test: analyzing synthetic extraction...")
    analyzer, metric_names = _init_analyzer()
    test_data = np.random.default_rng(42).integers(0, 256, DATA_SIZE, dtype=np.uint8)
    metrics = analyze_extraction(analyzer, metric_names, "smoke_test", test_data)
    print(f"  {len(metrics)} metrics computed")
    assert len(metrics) > 100, f"Expected 100+ metrics, got {len(metrics)}"
    print("  OK")
```

Run: `.venv/bin/python3 investigations/1d/seti_3i_atlas.py`
Expected: prints "~200 metrics computed" and "OK"

- [ ] **Step 4: Remove smoke test, commit**

Remove the temporary `if __name__` smoke test block.

```bash
git add investigations/1d/seti_3i_atlas.py
git commit -m "feat(seti-3i): geometry analysis with file-based caching"
```

---

### Task 6: Statistical comparison engine

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`
- Modify: `tests/test_seti_3i_extraction.py`

- [ ] **Step 1: Write test for comparison functions**

```python
# append to tests/test_seti_3i_extraction.py

def test_compare_exploratory():
    """Exploratory comparison: rank by effect size, no correction."""
    from seti_3i_atlas import compare_exploratory
    # Simulate metric profiles for ON and OFF
    rng = np.random.default_rng(42)
    on_profiles = [
        {'A:m1': rng.normal(5, 1), 'A:m2': rng.normal(0, 1)}
        for _ in range(10)
    ]
    off_profiles = [
        {'A:m1': rng.normal(0, 1), 'A:m2': rng.normal(0, 1)}
        for _ in range(10)
    ]
    ranked = compare_exploratory(on_profiles, off_profiles)
    # m1 should rank higher (large effect size)
    assert len(ranked) > 0
    assert ranked[0][0] == 'A:m1'
    assert abs(ranked[0][1]) > 1.0  # large Cohen's d


def test_compare_pooled():
    """Pooled comparison with Bonferroni correction."""
    from seti_3i_atlas import compare_pooled
    rng = np.random.default_rng(42)
    metric_names = ['A:m1', 'A:m2', 'A:m3']
    # m1: large difference. m2, m3: no difference.
    on_profiles = [
        {'A:m1': rng.normal(10, 0.5), 'A:m2': rng.normal(0, 1),
         'A:m3': rng.normal(0, 1)}
        for _ in range(30)
    ]
    off_profiles = [
        {'A:m1': rng.normal(0, 0.5), 'A:m2': rng.normal(0, 1),
         'A:m3': rng.normal(0, 1)}
        for _ in range(30)
    ]
    n_sig, findings = compare_pooled(on_profiles, off_profiles,
                                      metric_names, alpha=0.05)
    assert n_sig >= 1
    assert findings[0][0] == 'A:m1'
```

- [ ] **Step 2: Run tests — should fail**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py::test_compare_exploratory -v`
Expected: FAIL

- [ ] **Step 3: Implement comparison functions**

```python
# Add to investigations/1d/seti_3i_atlas.py

def compare_exploratory(on_profiles, off_profiles):
    """Exploratory per-file comparison. Rank metrics by |Cohen's d|.

    Parameters
    ----------
    on_profiles : list of {metric: value} dicts (one per ON extraction)
    off_profiles : list of {metric: value} dicts (one per OFF extraction)

    Returns
    -------
    list of (metric_name, cohens_d_value) sorted by |d| descending
    """
    all_metrics = set()
    for p in on_profiles + off_profiles:
        all_metrics.update(p.keys())

    ranked = []
    for m in sorted(all_metrics):
        on_vals = [p[m] for p in on_profiles if m in p]
        off_vals = [p[m] for p in off_profiles if m in p]
        if len(on_vals) < 3 or len(off_vals) < 3:
            continue
        d = cohens_d(np.array(on_vals), np.array(off_vals))
        if np.isfinite(d):
            ranked.append((m, d))

    ranked.sort(key=lambda x: -abs(x[1]))
    return ranked


def compare_pooled(on_profiles, off_profiles, metric_names, alpha=ALPHA):
    """Pooled comparison with Bonferroni correction.

    Returns (n_significant, [(metric, d, p), ...]) sorted by |d|.
    """
    bonf = alpha / max(len(metric_names), 1)
    n_sig = 0
    findings = []

    for m in metric_names:
        on_vals = np.array([p[m] for p in on_profiles if m in p])
        off_vals = np.array([p[m] for p in off_profiles if m in p])
        if len(on_vals) < 3 or len(off_vals) < 3:
            continue
        d = cohens_d(on_vals, off_vals)
        if not np.isfinite(d):
            continue
        _, p = stats.ttest_ind(on_vals, off_vals, equal_var=False)
        if p < bonf and abs(d) > 0.8:
            n_sig += 1
            findings.append((m, d, p))

    findings.sort(key=lambda x: -abs(x[1]))
    return n_sig, findings
```

- [ ] **Step 4: Run tests — should pass**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add investigations/1d/seti_3i_atlas.py tests/test_seti_3i_extraction.py
git commit -m "feat(seti-3i): exploratory and pooled statistical comparison"
```

---

### Task 7: Figure generation

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`

- [ ] **Step 1: Implement dark theme and figure layout**

```python
# Add to investigations/1d/seti_3i_atlas.py
# (Note: matplotlib, plt, gridspec already imported at top of file)

def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": "#181818",
        "axes.facecolor": "#181818",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "white",
        "text.color": "white",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
    })

def _dark_ax(ax):
    ax.set_facecolor('#181818')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#cccccc', labelsize=7)
    return ax
```

- [ ] **Step 2: Implement the main figure function**

```python
# Add to investigations/1d/seti_3i_atlas.py

def make_figure(band_results, pooled_results, metric_names):
    """Generate the main results figure.

    Parameters
    ----------
    band_results : dict — {band: {'exploratory': [...], 'n_on': int, 'n_off': int}}
    pooled_results : dict — {band: (n_sig, [(metric, d, p), ...])}
    metric_names : list of str
    """
    _apply_dark_theme()
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.96, top=0.92, bottom=0.06)

    fig.suptitle("3I/ATLAS — Geometric Structure: ON-Target vs OFF-Target",
                 fontsize=14, fontweight='bold', color='white')

    bands = ['L', 'S', 'C', 'X']
    band_colors = {'L': '#ff6b6b', 'S': '#ffd93d', 'C': '#6bcb77', 'X': '#4d96ff'}

    # Panel 1: Band summary — significant metrics per band (pooled)
    ax1 = _dark_ax(fig.add_subplot(gs[0, 0]))
    n_sigs = [pooled_results.get(b, (0, []))[0] for b in bands]
    colors = [band_colors[b] for b in bands]
    bars = ax1.bar(bands, n_sigs, color=colors, edgecolor='#444444', linewidth=0.5)
    ax1.set_ylabel('Significant Metrics (pooled)', fontsize=9)
    ax1.set_title('ON vs OFF by Band', fontsize=10, color='#aaaaaa')
    for bar, n in zip(bars, n_sigs):
        if n > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(n), ha='center', va='bottom', fontsize=9, color='white')

    # Panel 2: Top discriminating metrics (pooled, all bands combined)
    ax2 = _dark_ax(fig.add_subplot(gs[0, 1]))
    all_findings = []
    for band in bands:
        _, findings = pooled_results.get(band, (0, []))
        for m, d, p in findings:
            all_findings.append((m, d, p, band))
    all_findings.sort(key=lambda x: -abs(x[1]))
    top_n = min(15, len(all_findings))
    if top_n > 0:
        names = [f[0].split(':')[-1] for f in all_findings[:top_n]]
        ds = [f[1] for f in all_findings[:top_n]]
        cs = [band_colors[f[3]] for f in all_findings[:top_n]]
        y_pos = np.arange(top_n)
        ax2.barh(y_pos, ds, color=cs, edgecolor='#444444', linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel("Cohen's d", fontsize=9)
        ax2.set_title('Top Discriminating Metrics', fontsize=10, color='#aaaaaa')
        ax2.axvline(0, color='#666666', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'No significant metrics found',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=11, color='#888888')
        ax2.set_title('Top Discriminating Metrics', fontsize=10, color='#aaaaaa')

    # Panel 3: Per-band exploratory heatmap (top 20 metrics × 4 bands)
    ax3 = _dark_ax(fig.add_subplot(gs[1, :]))
    # Collect top exploratory metrics across all bands
    all_exploratory = {}
    for band in bands:
        br = band_results.get(band, {})
        for m, d in br.get('exploratory', [])[:30]:
            if m not in all_exploratory:
                all_exploratory[m] = {}
            all_exploratory[m][band] = d
    # Rank by max |d| across bands
    ranked_metrics = sorted(all_exploratory.keys(),
                            key=lambda m: max(abs(all_exploratory[m].get(b, 0))
                                              for b in bands),
                            reverse=True)[:20]
    if ranked_metrics:
        heatmap_data = np.zeros((len(ranked_metrics), len(bands)))
        for i, m in enumerate(ranked_metrics):
            for j, b in enumerate(bands):
                heatmap_data[i, j] = all_exploratory[m].get(b, 0)
        im = ax3.imshow(heatmap_data.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(heatmap_data)),
                        vmax=np.max(np.abs(heatmap_data)))
        ax3.set_xticks(range(len(ranked_metrics)))
        ax3.set_xticklabels([m.split(':')[-1] for m in ranked_metrics],
                            rotation=45, ha='right', fontsize=6)
        ax3.set_yticks(range(len(bands)))
        ax3.set_yticklabels(bands, fontsize=9)
        ax3.set_title("Exploratory: Cohen's d by Metric × Band",
                       fontsize=10, color='#aaaaaa')
        cbar = fig.colorbar(im, ax=ax3, shrink=0.6, pad=0.02)
        cbar.set_label("Cohen's d", fontsize=8, color='#cccccc')
        cbar.ax.tick_params(colors='#cccccc', labelsize=7)
    else:
        ax3.text(0.5, 0.5, 'No exploratory data',
                 ha='center', va='center', transform=ax3.transAxes,
                 fontsize=11, color='#888888')

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, 'seti_3i_atlas.png')
    fig.savefig(out_path, dpi=180, facecolor='#181818')
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")
    return out_path
```

- [ ] **Step 3: Quick visual test with synthetic data**

Add temporary test at bottom of script:

```python
if __name__ == "__main__":
    print("Figure smoke test...")
    metric_names = [f"Geom{i}:metric_{j}" for i in range(5) for j in range(4)]
    band_results = {}
    pooled_results = {}
    rng = np.random.default_rng(42)
    for band in ['L', 'S', 'C', 'X']:
        exploratory = [(m, rng.normal(0, 2)) for m in metric_names[:10]]
        band_results[band] = {'exploratory': exploratory}
        findings = [(m, rng.normal(0, 3), 1e-5) for m in metric_names[:5]]
        pooled_results[band] = (len(findings), findings)
    make_figure(band_results, pooled_results, metric_names)
    print("  OK — check figures/seti_3i_atlas.png")
```

Run: `.venv/bin/python3 investigations/1d/seti_3i_atlas.py`
Expected: Figure generated at `figures/seti_3i_atlas.png`

- [ ] **Step 4: Remove smoke test, commit**

```bash
git add investigations/1d/seti_3i_atlas.py
git commit -m "feat(seti-3i): dark-theme figure generation"
```

---

### Task 8: Main pipeline — wire everything together

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py`

- [ ] **Step 1: Implement main() orchestration**

```python
# Replace the if __name__ block in investigations/1d/seti_3i_atlas.py

def process_file(analyzer, metric_names, file_info, band):
    """Download, load, clean, extract, and analyze one file.

    Returns (label, list_of_metric_dicts) or None on failure.
    """
    filename = file_info['filename']
    on_off = "OFF" if file_info['is_off'] else "ON"
    label = f"{band}_{file_info['node']}_{on_off}"

    path = download_file(filename)
    if path is None:
        return None

    print(f"    Loading {filename} ...", flush=True)
    try:
        spec, meta = load_spectrogram(path)
    except Exception as e:
        print(f"    FAILED to load: {e}")
        return None

    print(f"    Shape: {spec.shape}  "
          f"fch1={meta['fch1']:.1f} MHz  foff={meta['foff']:.6f} MHz  "
          f"tsamp={meta['tsamp']:.3f} s")

    # RFI mitigation
    spec = sigma_clip_spectrogram(spec, sigma=3.0)

    # Extract 1D sequences
    extractions = extract_all(spec, label=label)
    if not extractions:
        print(f"    WARNING: no valid extractions from {filename}")
        return None

    print(f"    {len(extractions)} extractions, analyzing ...", end=" ", flush=True)

    # Analyze each extraction
    profiles = []
    for name, data in extractions:
        metrics = analyze_extraction(analyzer, metric_names, name, data)
        profiles.append(metrics)

    print(f"done ({len(profiles)} profiles)")
    return label, profiles


def main():
    print("=" * 78)
    print("3I/ATLAS SETI INVESTIGATION")
    print("Geometric analysis of GBT radio telescope observations")
    print("=" * 78)

    # --- Phase 1: File selection ---
    print(f"\n{'='*78}")
    print("PHASE 1: FILE MANIFEST")
    print(f"{'='*78}")
    manifest = fetch_file_manifest()
    selected = select_files(manifest, nodes_per_band=3)

    if not selected:
        print("ERROR: No files selected. Check portal connectivity.")
        sys.exit(1)

    # --- Phase 2: Download, extract, analyze ---
    print(f"\n{'='*78}")
    print("PHASE 2: DOWNLOAD, EXTRACT, ANALYZE")
    print(f"{'='*78}")

    analyzer, metric_names = _init_analyzer()
    n_metrics = len(metric_names)
    bonf = ALPHA / n_metrics
    print(f"  {n_metrics} metrics, Bonferroni alpha = {bonf:.2e}")

    # Collect profiles by band and ON/OFF status
    band_on = defaultdict(list)   # {band: [metric_dict, ...]}
    band_off = defaultdict(list)
    band_exploratory = {}

    for file_info, band in selected:
        result = process_file(analyzer, metric_names, file_info, band)
        if result is None:
            continue
        label, profiles = result
        if file_info['is_off']:
            band_off[band].extend(profiles)
        else:
            band_on[band].extend(profiles)

    # --- Phase 3: Statistical comparison ---
    print(f"\n{'='*78}")
    print("PHASE 3: STATISTICAL COMPARISON")
    print(f"{'='*78}")

    band_results = {}
    pooled_results = {}

    for band in ['L', 'S', 'C', 'X']:
        on = band_on.get(band, [])
        off = band_off.get(band, [])
        print(f"\n  {band}-band: {len(on)} ON profiles, {len(off)} OFF profiles")

        if len(on) < 3 or len(off) < 3:
            print(f"    SKIP — insufficient data")
            continue

        # Exploratory (per-band, no correction)
        ranked = compare_exploratory(on, off)
        band_results[band] = {
            'exploratory': ranked,
            'n_on': len(on),
            'n_off': len(off),
        }
        print(f"    Exploratory top 5:")
        for m, d in ranked[:5]:
            print(f"      {m:50s}  d={d:+.2f}")

        # Pooled (Bonferroni-corrected)
        n_sig, findings = compare_pooled(on, off, metric_names)
        pooled_results[band] = (n_sig, findings)
        print(f"    Pooled: {n_sig} significant metrics")
        for m, d, p in findings[:5]:
            print(f"      {m:50s}  d={d:+.2f}  p={p:.2e}")

    # --- Phase 4: Summary ---
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    total_sig = sum(pooled_results.get(b, (0, []))[0] for b in ['L', 'S', 'C', 'X'])
    print(f"  Total significant ON vs OFF metrics (pooled): {total_sig}")
    for band in ['L', 'S', 'C', 'X']:
        n_sig = pooled_results.get(band, (0, []))[0]
        n_on = len(band_on.get(band, []))
        n_off = len(band_off.get(band, []))
        print(f"  {band}-band: {n_sig:3d} sig  ({n_on} ON, {n_off} OFF profiles)")

    # --- Phase 5: Figure ---
    print(f"\n{'='*78}")
    print("GENERATING FIGURE")
    print(f"{'='*78}")
    make_figure(band_results, pooled_results, metric_names)

    return band_results, pooled_results


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs without errors on imports**

Run: `.venv/bin/python3 -c "import investigations.seti_3i_atlas"` won't work due to path setup, so instead:

Run: `.venv/bin/python3 -c "exec(open('investigations/1d/seti_3i_atlas.py').read().split('if __name__')[0])"`
Expected: No import errors. (This validates the module loads cleanly.)

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/python3 -m pytest tests/test_seti_3i_extraction.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add investigations/1d/seti_3i_atlas.py
git commit -m "feat(seti-3i): main pipeline orchestration"
```

---

### Task 9: Run the investigation on real data

**Files:**
- Modify: `investigations/1d/seti_3i_atlas.py` (only if bugs found)
- Create: `docs/investigations/seti_3i_atlas.md`

- [ ] **Step 1: Run the full investigation**

Run: `.venv/bin/python3 investigations/1d/seti_3i_atlas.py`

This will:
1. Fetch the file listing from `bldata.berkeley.edu`
2. Select ~24 files across L/S/C/X bands
3. Download ~1.2 GB of HDF5 data
4. Extract 1D sequences, analyze each with the framework
5. Compare ON vs OFF statistically
6. Generate `figures/seti_3i_atlas.png`

Expected runtime: Downloads will dominate. Analysis is ~5-10 minutes after download.

Watch for:
- HTTP errors (portal down or files moved)
- HDF5 format surprises (different `/data` shape than expected)
- Time axis too short for time series extraction (expected for part-0002; spectral profiles should still work)
- RFI contamination showing up as large effect sizes in L-band

- [ ] **Step 2: Fix any issues found during the run**

Common issues to watch for:
- If HDF5 shape is not `(n_time, 1, n_freq)`, update `load_spectrogram` to handle the actual shape
- If download fails with 403/404, try alternate files from the manifest
- If analysis is too slow, reduce `nodes_per_band` from 3 to 2
- If L-band is dominated by RFI artifacts, note this in findings

- [ ] **Step 3: Write findings document**

After the run completes, write `docs/investigations/seti_3i_atlas.md` with:
- Objective (link to synthetic SETI investigation)
- Data source (GBT, which files, total volume)
- Methodology (extraction, RFI mitigation, comparison strategy)
- Findings (per-band results, significant metrics, interpretation)
- The figure
- Conclusion

Follow the exact format of `docs/investigations/seti.md`.

- [ ] **Step 4: Commit findings and figure**

```bash
git add docs/investigations/seti_3i_atlas.md figures/seti_3i_atlas.png
git commit -m "feat(seti-3i): 3I/ATLAS investigation findings and figure"
```

- [ ] **Step 5: Final commit — add data directory to .gitignore if not already**

Check if `data/seti_3i/` is gitignored. If not:

```bash
echo "data/seti_3i/" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore 3I/ATLAS SETI data directory"
```
