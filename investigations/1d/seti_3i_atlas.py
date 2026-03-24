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

from exotic_geometry_framework import GeometryAnalyzer

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
# Task 4: Full extraction pipeline
# ---------------------------------------------------------------------------
def extract_all(spec, label, n_subbands=N_SUBBANDS, n_windows=N_WINDOWS,
                target_len=DATA_SIZE):
    extractions = []
    ts_list = extract_time_series(spec, n_subbands=n_subbands, target_len=target_len)
    for i, ts in enumerate(ts_list):
        extractions.append((f"{label}/ts_{i}", ts))
    sp_list = extract_spectral_profiles(spec, n_windows=n_windows, target_len=target_len)
    for i, sp in enumerate(sp_list):
        extractions.append((f"{label}/sp_{i}", sp))
    return extractions

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

# ---------------------------------------------------------------------------
# Task 5: Geometry analysis with result caching
# ---------------------------------------------------------------------------
def _init_analyzer():
    analyzer = GeometryAnalyzer().add_all_geometries()
    dummy = analyzer.analyze(np.random.default_rng(0).integers(0, 256, 200, dtype=np.uint8))
    metric_names = []
    for r in dummy.results:
        for mn in sorted(r.metrics.keys()):
            metric_names.append(f"{r.geometry_name}:{mn}")
    return analyzer, metric_names


def _framework_hash():
    """Hash exotic_geometry_framework.py for cache invalidation."""
    fw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'exotic_geometry_framework.py')
    with open(fw_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


_FW_HASH = None


def analyze_extraction(analyzer, metric_names, name, data):
    """Analyze a single uint8 extraction. Returns {metric: value} dict."""
    global _FW_HASH
    os.makedirs(CACHE_DIR, exist_ok=True)
    if _FW_HASH is None:
        _FW_HASH = _framework_hash()
    data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    cache_key = hashlib.sha256(
        f"{name}|{data_hash}|{_FW_HASH}".encode()
    ).hexdigest()[:24]
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    if os.path.exists(cache_path):
        loaded = np.load(cache_path, allow_pickle=True)
        return dict(loaded['metrics'].item())
    result = analyzer.analyze(data)
    metrics = {}
    for r in result.results:
        for mn, mv in r.metrics.items():
            key = f"{r.geometry_name}:{mn}"
            if key in metric_names and np.isfinite(mv):
                metrics[key] = mv
    np.savez_compressed(cache_path, metrics=metrics)
    return metrics


def analyze_file_extractions(analyzer, metric_names, extractions):
    """Analyze all extractions from one file. Returns {name: {metric: value}}."""
    results = {}
    for name, data in extractions:
        metrics = analyze_extraction(analyzer, metric_names, name, data)
        results[name] = metrics
    return results


# ---------------------------------------------------------------------------
# File manifest and download infrastructure
# ---------------------------------------------------------------------------
_GBT_RE = re.compile(
    r'(blc\d+)_guppi_(\d+)_(\d+)_[A-Z0-9]+_3I_ATLAS(_(OFF))?_(\d+)\.rawspec\.(\d+)\.h5'
)

def parse_gbt_filename(name):
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
    """Extract .h5 hrefs from an HTML directory listing."""
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, val in attrs:
                if attr == 'href' and val and val.endswith('.h5'):
                    self.links.append(val)


def fetch_file_manifest():
    """Fetch the portal directory listing and return parsed file info dicts."""
    req = urllib.request.Request(
        PORTAL_BASE + '/',
        headers={'User-Agent': 'ExoticGeometry/1.0'}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html_bytes = resp.read()
    parser = _LinkParser()
    parser.feed(html_bytes.decode('utf-8', errors='replace'))
    manifest = []
    for link in parser.links:
        info = parse_gbt_filename(os.path.basename(link))
        if info is not None:
            manifest.append(info)
    return manifest


BAND_SCAN_RANGES = {
    'L': (13, 18), 'S': (29, 34), 'C': (45, 50), 'X': (61, 66),
}

def classify_band(scan_str):
    scan = int(scan_str)
    for band, (lo, hi) in BAND_SCAN_RANGES.items():
        if lo <= scan <= hi:
            return band
    return 'unknown'


def select_files(manifest, nodes_per_band=3):
    """Pick representative ON/OFF pairs per band.

    Groups by band and node. For each band, selects up to nodes_per_band
    nodes and returns their ON and OFF file info dicts.
    """
    # Group by (band, node)
    groups = defaultdict(list)
    for info in manifest:
        band = classify_band(info['scan'])
        if band == 'unknown':
            continue
        groups[(band, info['node'])].append(info)

    selected = []
    bands_seen = defaultdict(set)
    for (band, node), files in sorted(groups.items()):
        if len(bands_seen[band]) >= nodes_per_band:
            continue
        bands_seen[band].add(node)
        selected.extend(files)
    return selected


def download_file(filename):
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
                    chunk = resp.read(1024 * 1024)
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

# ---------------------------------------------------------------------------
# HDF5 reading and spectrogram loading
# ---------------------------------------------------------------------------
def load_spectrogram(path):
    with h5py.File(path, 'r') as hf:
        raw = hf['data'][:]
        spec = raw[:, 0, :].astype(np.float32)
        if 'mask' in hf:
            mask_raw = hf['mask'][:]
            mask_2d = mask_raw[:, 0, :] > 0
            med = np.median(spec, axis=0)
            spec[mask_2d] = np.broadcast_to(med, spec.shape)[mask_2d]
        attrs = dict(hf['data'].attrs)
        meta = {
            'fch1': float(attrs.get('fch1', 0)),
            'foff': float(attrs.get('foff', 0)),
            'tsamp': float(attrs.get('tsamp', 0)),
            'n_time': spec.shape[0],
            'n_freq': spec.shape[1],
        }
    return spec, meta


# ---------------------------------------------------------------------------
# Task 6: Statistical comparison engine
# ---------------------------------------------------------------------------
def compare_exploratory(on_profiles, off_profiles):
    """Exploratory per-file comparison. Rank metrics by |Cohen's d|."""
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


# ---------------------------------------------------------------------------
# Task 7: Figure generation
# ---------------------------------------------------------------------------
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

def make_figure(band_results, pooled_results, metric_names):
    _apply_dark_theme()
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.96, top=0.92, bottom=0.06)
    fig.suptitle("3I/ATLAS — Geometric Structure: ON-Target vs OFF-Target",
                 fontsize=14, fontweight='bold', color='white')
    bands = ['L', 'S', 'C', 'X']
    band_colors = {'L': '#ff6b6b', 'S': '#ffd93d', 'C': '#6bcb77', 'X': '#4d96ff'}

    # Panel 1: Band summary bar chart
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

    # Panel 2: Top discriminating metrics (horizontal bar)
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

    # Panel 3: Exploratory heatmap (bottom, spanning both columns)
    ax3 = _dark_ax(fig.add_subplot(gs[1, :]))
    all_exploratory = {}
    for band in bands:
        br = band_results.get(band, {})
        for m, d in br.get('exploratory', [])[:30]:
            if m not in all_exploratory:
                all_exploratory[m] = {}
            all_exploratory[m][band] = d
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
        ax3.set_title("Exploratory: Cohen's d by Metric x Band",
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

