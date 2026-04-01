import numpy as np
import pytest
from exotic_geometry_framework import PenroseGeometry, GeometryResult

def test_penrose_initialization():
    geom = PenroseGeometry()
    assert geom.name == "Penrose (Quasicrystal)"
    assert np.isclose(geom.PHI, (1 + np.sqrt(5)) / 2)

def test_penrose_embedding():
    geom = PenroseGeometry()
    # 50 pairs -> 50 points of dimension 7 (5 projections + 2 original)
    data = np.random.randint(0, 256, 100, dtype=np.uint8)
    embedded = geom.embed(data)

    # Check shape: N/2 points, 7 coords each (5 projected + x + y)
    assert embedded.shape == (50, 7)

def test_penrose_metrics():
    geom = PenroseGeometry()
    data = np.random.randint(0, 256, 200, dtype=np.uint8)
    result = geom.compute_metrics(data)

    assert isinstance(result, GeometryResult)
    metrics = result.metrics

    expected = {
        "long_range_order",
        "algebraic_tower",
    }
    assert expected == set(metrics.keys())

def test_fibonacci_detection():
    """
    Test that Penrose geometry gives higher 'long_range_order'
    for a Fibonacci-structured signal than for white noise.
    """
    geom = PenroseGeometry()
    rng = np.random.default_rng(42)

    # 1. White noise
    noise = rng.integers(0, 256, 1000, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)

    # 2. Fibonacci structure (quasiperiodic)
    s = "1"
    seq = "0"
    while len(seq) < 1000:
        new_seq = seq + s
        s = seq
        seq = new_seq

    fib_data = np.array([255 if c == '1' else 0 for c in seq[:1000]], dtype=np.uint8)
    res_fib = geom.compute_metrics(fib_data)

    # Fibonacci sequence is THE example of 1D quasicrystal with ratio Phi.
    assert res_fib.metrics['long_range_order'] > res_noise.metrics['long_range_order']

    # Algebraic tower tests phi-based spectral scaling
    assert res_fib.metrics['algebraic_tower'] > res_noise.metrics['algebraic_tower']
