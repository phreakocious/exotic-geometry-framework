import numpy as np
import pytest
from exotic_geometry_framework import DecagonalGeometry, SeptagonalGeometry, DodecagonalGeometry

def generate_quasi_signal(ratio, size=1000):
    """
    Generate a synthetic signal with spectral peaks at 1, 1/ratio, ... 1/ratio^4.
    """
    t = np.arange(size)
    signal = np.zeros(size)
    f = 0.4 # Start high
    for i in range(5):
        signal += np.cos(2*np.pi * (f * (ratio**-i)) * t)
        
    return ((signal + 5) / 10 * 255).astype(np.uint8)

def test_decagonal_detection():
    """
    Decagonal (10-fold) should detect Golden Ratio scaling.
    """
    geom = DecagonalGeometry()
    
    # 1. Phi-scaled signal
    phi = geom.PHI
    signal_phi = generate_quasi_signal(phi)
    res_phi = geom.compute_metrics(signal_phi)
    
    # 2. Noise
    noise = np.random.randint(0, 256, 1000, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    score_phi = res_phi.metrics['tenfold_symmetry']
    score_noise = res_noise.metrics['tenfold_symmetry']
    
    print(f"Decagonal Symmetry - Phi: {score_phi:.3f}, Noise: {score_noise:.3f}")
    
    assert score_phi > score_noise, "Should detect 10-fold/Phi spectral ratio"
    assert score_phi > 0.1

def test_septagonal_detection():
    """
    Septagonal (7-fold) should detect Rho (2.247...) scaling.
    """
    geom = SeptagonalGeometry()
    
    # 1. Rho-scaled signal
    rho = geom.RATIO
    signal_rho = generate_quasi_signal(rho)
    res_rho = geom.compute_metrics(signal_rho)
    
    # 2. Noise
    noise = np.random.randint(0, 256, 1000, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    score_rho = res_rho.metrics['sevenfold_symmetry']
    score_noise = res_noise.metrics['sevenfold_symmetry']
    
    print(f"Septagonal Symmetry - Rho: {score_rho:.3f}, Noise: {score_noise:.3f}")
    
    assert score_rho > score_noise, "Should detect 7-fold/Rho spectral ratio"
    assert score_rho > 0.1

def test_dodecagonal_detection():
    """
    Dodecagonal (12-fold) should detect 2+sqrt(3) scaling.
    """
    geom = DodecagonalGeometry()
    
    # 1. Ratio-scaled signal
    ratio = geom.RATIO
    signal_ratio = generate_quasi_signal(ratio)
    res_ratio = geom.compute_metrics(signal_ratio)
    
    # 2. Noise
    noise = np.random.randint(0, 256, 1000, dtype=np.uint8)
    res_noise = geom.compute_metrics(noise)
    
    score = res_ratio.metrics['twelvefold_symmetry']
    score_noise = res_noise.metrics['twelvefold_symmetry']
    
    print(f"Dodecagonal Symmetry - Ratio: {score:.3f}, Noise: {score_noise:.3f}")
    
    assert score > score_noise, "Should detect 12-fold spectral ratio"
    assert score > 0.1

def test_dodecagonal_embedding():
    """
    Verify embedding dimensionality.
    """
    geom = DodecagonalGeometry()
    data = np.random.randint(0, 256, 100, dtype=np.uint8)
    embedded = geom.embed(data)
    # 50 points, 14 coords (12 projected + x + y)
    assert embedded.shape == (50, 14)

def test_cross_sensitivity():
    """
    Verify that 7-fold geometry does NOT strongly detect 12-fold signal, and vice versa.
    This ensures specificity.
    """
    geom7 = SeptagonalGeometry()
    rho = geom7.RATIO # ~2.247
    
    # Generate 12-fold signal (ratio ~3.732)
    ratio12 = 2 + np.sqrt(3)
    signal_12 = generate_quasi_signal(ratio12)
    
    res_cross = geom7.compute_metrics(signal_12)
    score_cross = res_cross.metrics['sevenfold_symmetry']
    
    # Generate 7-fold signal
    signal_7 = generate_quasi_signal(rho)
    res_correct = geom7.compute_metrics(signal_7)
    score_correct = res_correct.metrics['sevenfold_symmetry']
    
    print(f"7-fold Detector - On 7-fold data: {score_correct:.3f}, On 12-fold data: {score_cross:.3f}")
    
    assert score_correct > score_cross, "Geometry should be specific to its own ratio"
