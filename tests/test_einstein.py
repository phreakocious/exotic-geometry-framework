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
        "inflation_similarity",
        "chirality",
        "hex_balance"
    }
    assert expected_keys.issubset(metrics.keys())
    
    for k, v in metrics.items():
        assert isinstance(v, float)

def test_chirality_sensitivity():
    """
    Test that 'chirality' metric distinguishes between clockwise 
    and counter-clockwise loops.
    """
    geom = EinsteinHatGeometry()
    
    # Counter-clockwise hexagon: 0, 1, 2, 3, 4, 5
    # (East, SE, SW, West, NW, NE) -> Standard math angle increasing?
    # Let's check coordinates from previous test:
    # 0,0 -> 1,0 -> 1,1 -> 0,2 -> -1,2 -> -1,1 -> 0,0
    # This traces a hexagon.
    # Area calculation in compute_metrics: sum(q[:-1] * r[1:] - r[:-1] * q[1:])
    # This is standard shoelace formula (signed area).
    
    ccw_data = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    # Clockwise: 0, 5, 4, 3, 2, 1
    cw_data = np.array([0, 5, 4, 3, 2, 1], dtype=np.uint8)
    
    res_ccw = geom.compute_metrics(ccw_data)
    res_cw = geom.compute_metrics(cw_data)
    
    # Chirality should be opposite sign (approx)
    # Note: metrics might be normalized or scaled, but sign should flip 
    # if it's a signed area. The implementation divides by length etc.
    
    c_ccw = res_ccw.metrics['chirality']
    c_cw = res_cw.metrics['chirality']
    
    # Check if they have opposite signs or at least different values
    # The shoelace formula is antisymmetric.
    
    # However, the loop might not be closed exactly or might be translated.
    # But for a closed loop 0..5, it is closed.
    
    assert c_ccw != c_cw
    # One should be positive, one negative (or at least significantly different)
    assert np.sign(c_ccw) != np.sign(c_cw)
