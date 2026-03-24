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
