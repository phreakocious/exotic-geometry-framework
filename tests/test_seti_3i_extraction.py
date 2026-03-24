import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'investigations', '1d'))

import tempfile
import h5py
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
    x[0] = 1e6
    x[-1] = -1e6
    result = to_uint8(x)
    assert result.dtype == np.uint8
    assert np.median(result) > 100
    assert np.median(result) < 160


def test_sigma_clip_replaces_outliers():
    """Sigma-clip replaces > 3-sigma pixels with channel median."""
    from seti_3i_atlas import sigma_clip_spectrogram
    rng = np.random.default_rng(42)
    spec = rng.normal(0, 1, (100, 500)).astype(np.float32)
    spec[50, 250] = 1000.0
    cleaned = sigma_clip_spectrogram(spec, sigma=3.0)
    assert cleaned.shape == spec.shape
    assert abs(cleaned[50, 250]) < 5.0


def test_sigma_clip_preserves_normal_data():
    """Normal data should be mostly unchanged."""
    from seti_3i_atlas import sigma_clip_spectrogram
    rng = np.random.default_rng(42)
    spec = rng.normal(0, 1, (100, 500)).astype(np.float32)
    cleaned = sigma_clip_spectrogram(spec, sigma=3.0)
    n_changed = np.sum(np.abs(cleaned - spec) > 1e-6)
    assert n_changed < 0.01 * spec.size


def test_extract_time_series():
    """Band-integrated time series: sum over freq sub-bands."""
    from seti_3i_atlas import extract_time_series
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (3000, 4000)).astype(np.float32)
    series_list = extract_time_series(spec, n_subbands=4, target_len=2000)
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
    """If time axis < target_len, return empty list."""
    from seti_3i_atlas import extract_time_series
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (16, 4000)).astype(np.float32)
    series_list = extract_time_series(spec, n_subbands=4, target_len=2000)
    assert len(series_list) == 0


def test_cohens_d_same_population():
    """Samples from the same population should give |d| < 0.5."""
    from seti_3i_atlas import cohens_d
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0, 1, 50)
    d = cohens_d(a, b)
    assert abs(d) < 0.5


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
    assert d < -2.0


def test_cohens_d_zero_variance():
    """Constant arrays with different means: a < b -> -inf."""
    from seti_3i_atlas import cohens_d
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0])
    d = cohens_d(a, b)
    assert d == float('-inf')


# ---------------------------------------------------------------------------
# File manifest and download infrastructure
# ---------------------------------------------------------------------------
def test_parse_gbt_filename():
    from seti_3i_atlas import parse_gbt_filename
    info = parse_gbt_filename("blc20_guppi_61027_31308_DIAG_3I_ATLAS_0061.rawspec.0002.h5")
    assert info['node'] == 'blc20'
    assert info['scan'] == '0061'
    assert info['part'] == '0002'
    assert info['is_off'] is False
    info_off = parse_gbt_filename("blc20_guppi_61027_31308_DIAG_3I_ATLAS_OFF_0062.rawspec.0002.h5")
    assert info_off['is_off'] is True
    assert info_off['node'] == 'blc20'


def test_parse_gbt_filename_returns_none_for_non_matching():
    from seti_3i_atlas import parse_gbt_filename
    assert parse_gbt_filename("random_file.h5") is None
    assert parse_gbt_filename("README.md") is None


# ---------------------------------------------------------------------------
# HDF5 reading and spectrogram loading
# ---------------------------------------------------------------------------
def test_load_spectrogram_from_h5():
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
    finally:
        os.unlink(path)


def test_load_spectrogram_applies_mask():
    from seti_3i_atlas import load_spectrogram
    rng = np.random.default_rng(42)
    n_time, n_freq = 50, 1000
    data = rng.normal(100, 10, (n_time, 1, n_freq)).astype(np.float32)
    mask = np.zeros((n_time, 1, n_freq), dtype=np.uint8)
    mask[10, 0, 500] = 1
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
        assert spec.shape == (n_time, n_freq)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------
def test_extract_all_from_spectrogram():
    from seti_3i_atlas import extract_all
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (3000, 4000)).astype(np.float32)
    extractions = extract_all(spec, label="test_blc20_L_ON")
    assert len(extractions) > 0
    for name, arr in extractions:
        assert isinstance(name, str)
        assert "test_blc20_L_ON" in name
        assert arr.dtype == np.uint8
        assert len(arr) == 2000


def test_extract_all_short_time_axis():
    from seti_3i_atlas import extract_all
    rng = np.random.default_rng(42)
    spec = rng.normal(100, 10, (16, 4000)).astype(np.float32)
    extractions = extract_all(spec, label="test_short")
    names = [n for n, _ in extractions]
    assert not any("ts_" in n for n in names)
    assert any("sp_" in n for n in names)


# ---------------------------------------------------------------------------
# Task 6: Statistical comparison engine
# ---------------------------------------------------------------------------
def test_compare_exploratory():
    """Exploratory comparison: rank by effect size, no correction."""
    from seti_3i_atlas import compare_exploratory
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
    assert len(ranked) > 0
    assert ranked[0][0] == 'A:m1'
    assert abs(ranked[0][1]) > 1.0


def test_compare_pooled():
    """Pooled comparison with Bonferroni correction."""
    from seti_3i_atlas import compare_pooled
    rng = np.random.default_rng(42)
    metric_names = ['A:m1', 'A:m2', 'A:m3']
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
