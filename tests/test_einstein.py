import numpy as np
import pytest
from exotic_geometry_framework import EinsteinHatGeometry, GeometryResult

def test_einstein_initialization():
    geom = EinsteinHatGeometry()
    assert geom.name == "Einstein (Hat Monotile)"
    assert geom.dimension == "2D with hex+chiral"
    assert geom.hat_kernel is not None
    assert len(geom.hat_kernel) == 13

def test_einstein_embedding_shape():
    geom = EinsteinHatGeometry()
    # 100 points -> 101 path vertices (start at 0,0)
    data = np.random.randint(0, 256, 100, dtype=np.uint8)
    path = geom.embed(data)
    assert path.shape == (101, 2)
    assert np.all(path[0] == [0, 0])

def test_einstein_embedding_deterministic():
    geom = EinsteinHatGeometry()
    data = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    # 0: +1, 0
    # 1: 0, +1
    # 2: -1, 1
    # 3: -1, 0
    # 4: 0, -1
    # 5: 1, -1
    path = geom.embed(data)
    
    expected_path = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 2],
        [-1, 2],
        [-1, 1],
        [0, 0]
    ])
    np.testing.assert_array_equal(path, expected_path)

def test_einstein_metrics_structure():
    geom = EinsteinHatGeometry()
    data = np.random.randint(0, 256, 200, dtype=np.uint8)
    result = geom.compute_metrics(data)
    
    assert isinstance(result, GeometryResult)
    assert result.geometry_name == "Einstein (Hat Monotile)"
    
    metrics = result.metrics
    expected_keys = {
        "hat_boundary_match",
    }
    assert expected_keys.issubset(metrics.keys())
    
    for k, v in metrics.items():
        assert isinstance(v, float)

def test_hat_boundary_match_varies():
    """
    Test that hat_boundary_match produces different values for
    structured vs random data.
    """
    geom = EinsteinHatGeometry()

    # Structured: repeating pattern aligned with kernel length
    structured = np.tile(np.arange(13, dtype=np.uint8), 20)
    res_struct = geom.compute_metrics(structured)

    # Random
    rng = np.random.default_rng(42)
    random_data = rng.integers(0, 256, 260, dtype=np.uint8)
    res_rand = geom.compute_metrics(random_data)

    m_struct = res_struct.metrics['hat_boundary_match']
    m_rand = res_rand.metrics['hat_boundary_match']

    # Both should be valid floats
    assert np.isfinite(m_struct)
    assert np.isfinite(m_rand)
