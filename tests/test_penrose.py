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
    # Use a Fibonacci sequence to trigger quasicrystal detection?
    # Fibonacci word: 0 -> 01, 1 -> 0
    # A B A B A ...
    # Let's just use random data for structure check first
    data = np.random.randint(0, 256, 200, dtype=np.uint8)
    result = geom.compute_metrics(data)
    
    assert isinstance(result, GeometryResult)
    metrics = result.metrics
    
    expected = {
        "fivefold_symmetry",
        "peak_sharpness",
        "index_diversity",
        "long_range_order"
    }
    assert expected.issubset(metrics.keys())
    
    # Check ranges
    assert 0.0 <= metrics["fivefold_symmetry"] <= 1.0
    assert 0.0 <= metrics["peak_sharpness"] <= 1.0

def test_fibonacci_detection():
    """
    Test that Penrose geometry gives higher 'fivefold_symmetry' or 'long_range_order'
    for a Fibonacci-structured signal than for white noise.
    """
    geom = PenroseGeometry()
    rng = np.random.default_rng(42)
    
    # 1. White noise
    noise = rng.integers(0, 256, 1000, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    # 2. Fibonacci structure (quasiperiodic)
    # Generate Fibonacci word of 0s and 1s, map to 0 and 255
    s = "1"
    seq = "0"
    while len(seq) < 1000:
        new_seq = seq + s
        s = seq
        seq = new_seq
    
    fib_data = np.array([255 if c == '1' else 0 for c in seq[:1000]], dtype=np.uint8)
    res_fib = geom.compute_metrics(fib_data)
    
    # Expect higher structure scores for Fibonacci
    # Specifically 'index_diversity' (subword complexity stability) 
    # and 'long_range_order' (ACF self-similarity).
    # 'fivefold_symmetry' checks spectral self-similarity at ratio Phi.
    
    print(f"Noise LRO: {res_noise.metrics['long_range_order']}")
    print(f"Fib LRO: {res_fib.metrics['long_range_order']}")
    
    # Fibonacci sequence is THE example of 1D quasicrystal with ratio Phi.
    # It should beat noise significantly.
    assert res_fib.metrics['long_range_order'] > res_noise.metrics['long_range_order']
    
    # Subword complexity for Fibonacci grows linearly (low diff variation -> high metric)
    # Random grows exponentially (high diff variation -> low metric)
    assert res_fib.metrics['index_diversity'] > res_noise.metrics['index_diversity']
